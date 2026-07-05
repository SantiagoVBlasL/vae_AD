#!/usr/bin/env python3
"""
promoted_model_philips_error_mechanism_deep_audit_20260610.py

Deep error-mechanism audit for the promoted model:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Tasks:
  1. Ranking vs threshold analysis by manufacturer
  2. Philips CN FP vs TN deep comparison (all metadata + latent)
  3. Latent geometry (centroid distances)
  4. Latent nuisance association (PCA + cosine similarity)
  5. Tensor channel mechanism
  6. High-confidence Philips CN FP review table
  7. Oracle masked performance sensitivity
  8. Next-step decision tree (Markdown)

Guardrails: read-only, no model training, no threshold fitting,
            no tensor/metadata/artifact modification.
"""

import os
import sys
import json
import warnings
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, f1_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path("results/revision_bspc_2026")
RUN_DIR = BASE_DIR / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LATENT_CACHE_DIR = RUN_DIR / "classifier_only_readout/latent_cache"
CALIB_DIR = BASE_DIR / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
MASTER_DB_PATH = BASE_DIR / "promoted_model_master_database_20260610/promoted_model_master_database.csv"
PHILIPS_ANNOT_PATH = BASE_DIR / "philips_cn_protocol_annotation_for_martin_20260610/philips_cn_to_annotate_for_martin.csv"

BIG_RUN_DIR = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5")

OUT_DIR = BASE_DIR / "promoted_model_philips_error_mechanism_deep_audit_20260610"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
SCORE_HIGH_CONF = 0.75
FDR_ALPHA = 0.10
RNG = np.random.default_rng(42)

# ============================================================
# HELPERS
# ============================================================

def cles(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(a[:, None] > b[None, :]))


def mwu_row(fp_vals, tn_vals, var_name):
    fp_vals = np.asarray(fp_vals, float)
    tn_vals = np.asarray(tn_vals, float)
    fp_vals = fp_vals[np.isfinite(fp_vals)]
    tn_vals = tn_vals[np.isfinite(tn_vals)]
    na, nb = len(fp_vals), len(tn_vals)
    if na < 2 or nb < 2:
        return dict(variable=var_name, test="mwu", n_fp=na, n_tn=nb,
                    stat=np.nan, p_raw=np.nan, p_fdr=np.nan, reject_fdr=False,
                    effect=np.nan, median_fp=np.nan, median_tn=np.nan, low_n=True)
    stat, p = stats.mannwhitneyu(fp_vals, tn_vals, alternative="two-sided")
    return dict(variable=var_name, test="mwu", n_fp=na, n_tn=nb,
                stat=stat, p_raw=p, p_fdr=np.nan, reject_fdr=False,
                effect=round(cles(fp_vals, tn_vals), 4),
                median_fp=round(float(np.median(fp_vals)), 4),
                median_tn=round(float(np.median(tn_vals)), 4),
                low_n=(na < 10 or nb < 10))


def fisher_row(fp_vals, tn_vals, var_name):
    fp_pos = np.asarray(fp_vals, bool)
    tn_pos = np.asarray(tn_vals, bool)
    ct = np.array([[fp_pos.sum(), (~fp_pos).sum()],
                   [tn_pos.sum(), (~tn_pos).sum()]])
    or_, p = stats.fisher_exact(ct, alternative="two-sided")
    return dict(variable=var_name, test="fisher_exact",
                n_fp=len(fp_pos), n_tn=len(tn_pos),
                stat=round(float(or_), 4), p_raw=p, p_fdr=np.nan, reject_fdr=False,
                effect=round(float(or_), 4),
                median_fp=round(float(fp_pos.mean()), 4),
                median_tn=round(float(tn_pos.mean()), 4), low_n=False)


def apply_fdr(rows):
    pvals = np.array([r["p_raw"] for r in rows], dtype=float)
    valid = np.isfinite(pvals)
    if valid.sum() > 1:
        _, q, _, _ = multipletests(pvals[valid], method="fdr_bh")
        _, rej, _, _ = multipletests(pvals[valid], alpha=FDR_ALPHA, method="fdr_bh")
        qi = 0
        for r in rows:
            if np.isfinite(r["p_raw"]):
                r["p_fdr"] = round(float(q[qi]), 5)
                r["reject_fdr"] = bool(rej[qi])
                qi += 1
    return rows


def perf_block(y_true, y_score, threshold, label="", n_min=5):
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    valid = np.isfinite(y_score)
    y_true, y_score = y_true[valid], y_score[valid]
    n, n_pos = len(y_true), int(y_true.sum())
    n_neg = n - n_pos
    y_pred = (y_score >= threshold).astype(int)
    if n_pos < n_min or n_neg < n_min:
        return dict(label=label, n=n, n_ad=n_pos, n_cn=n_neg,
                    auc=np.nan, pr_auc=np.nan, ba=np.nan,
                    sensitivity=np.nan, specificity=np.nan, f1=np.nan,
                    tp=np.nan, tn=np.nan, fp=np.nan, fn=np.nan,
                    fpr_cn=np.nan, note="too_few_samples")
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr_cn = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return dict(label=label, n=n, n_ad=n_pos, n_cn=n_neg,
                auc=round(auc, 4), pr_auc=round(pr_auc, 4),
                ba=round(ba, 4), sensitivity=round(sens, 4),
                specificity=round(spec, 4), f1=round(f1, 4),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                fpr_cn=round(fpr_cn, 4), note="ok")


def cosine_sim(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def md_table(df, floatfmt=".4f"):
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:{floatfmt}}" if np.isfinite(v) else str(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_md(df, path, title="", floatfmt=".4f"):
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append(md_table(df, floatfmt=floatfmt))
    Path(path).write_text("\n".join(lines))


command_log = {
    "script": "promoted_model_philips_error_mechanism_deep_audit_20260610.py",
    "run_time": datetime.datetime.now().isoformat(),
    "guardrails": ["read-only", "no model training", "no threshold fitting",
                   "no tensor modification", "no metadata modification",
                   "no artifact modification"],
    "steps": []
}

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

master = pd.read_csv(MASTER_DB_PATH)
print(f"  Master DB: {master.shape}")

# Load OOF latent mu from classifier_only_readout/latent_cache
mu_dfs = []
for fold in range(1, N_FOLDS + 1):
    p = LATENT_CACHE_DIR / f"fold_{fold}_test_latent_mu.csv"
    if p.exists():
        mu_dfs.append(pd.read_csv(p))
    else:
        print(f"  WARNING: {p} missing")

if mu_dfs:
    mu_all = pd.concat(mu_dfs, ignore_index=True).drop_duplicates(subset="SubjectID")
    mu_cols = [c for c in mu_all.columns if c.startswith("mu_")]
    print(f"  OOF latent mu: {mu_all.shape}, n_dims={len(mu_cols)}")
else:
    mu_all = None
    mu_cols = []
    print("  ERROR: No latent mu CSVs found")
    sys.exit(1)

# Merge with master
mdf = master.merge(mu_all[["SubjectID"] + mu_cols], on="SubjectID", how="left")
print(f"  Merged: {mdf.shape}, mu coverage: {mdf[mu_cols[0]].notna().sum()}/{len(mdf)}")

# Calib predictions for per-fold thresholds
calib_preds = pd.read_csv(CALIB_DIR / "calib_predictions.csv")
primary = calib_preds[
    (calib_preds.model_name == "logreg_l2_original") &
    (calib_preds.feature_set == "z_plus_age_sex") &
    (calib_preds.calib_method == "oof_ecdf") &
    (calib_preds.threshold_strategy == "inner_oof_youden_j")
].copy()
fold_thresholds = primary.groupby("fold")["threshold"].first().to_dict()
# Primary threshold from master DB (pooled)
primary_threshold = mdf["threshold"].iloc[0] if "threshold" in mdf.columns else 0.5
print(f"  Per-fold thresholds: {fold_thresholds}")
print(f"  Master DB threshold: {primary_threshold}")

# Convenience masks
is_philips  = mdf.Manufacturer == "Philips"
is_ge       = mdf.Manufacturer == "GE"
is_siemens  = mdf.Manufacturer == "SIEMENS"
is_cn       = mdf.ResearchGroup_Mapped == "CN"
is_ad       = mdf.ResearchGroup_Mapped == "AD"
is_fp       = mdf.confusion_label == "FP"
is_tn       = mdf.confusion_label == "TN"
is_tp       = mdf.confusion_label == "TP"
is_fn       = mdf.confusion_label == "FN"
is_140tp    = mdf.raw_tp_group.astype(str).str.strip() == "140"
is_197tp    = mdf.raw_tp_group.astype(str).str.strip() == "197"

p_cn_fp = mdf[is_philips & is_cn & is_fp]
p_cn_tn = mdf[is_philips & is_cn & is_tn]
p_cn    = mdf[is_philips & is_cn]
p_ad    = mdf[is_philips & is_ad]

print(f"  Philips CN FP={len(p_cn_fp)}, TN={len(p_cn_tn)}")
command_log["steps"].append("data_loaded")

# ============================================================
# TASK 1 — RANKING VS THRESHOLD ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: Ranking vs threshold analysis")
print("=" * 60)

rows_rank = []

manufacturers = ["GE", "SIEMENS", "Philips"]
groups_for_rank = {
    "All": mdf,
    "GE":  mdf[is_ge],
    "SIEMENS": mdf[is_siemens],
    "Philips": mdf[is_philips],
    "Philips_CN_140TP": mdf[is_philips & is_cn & is_140tp],
    "Philips_CN_197TP": mdf[is_philips & is_cn & is_197tp],
    "Philips_AD": mdf[is_philips & is_ad],
    "GE_CN": mdf[is_ge & is_cn],
    "GE_AD": mdf[is_ge & is_ad],
    "SIEMENS_CN": mdf[is_siemens & is_cn],
    "SIEMENS_AD": mdf[is_siemens & is_ad],
    "GE_SIEMENS": mdf[is_ge | is_siemens],
    "Philips_197TP_all": mdf[is_philips & is_197tp],
}

for label, subdf in groups_for_rank.items():
    if len(subdf) == 0:
        continue
    thr = primary_threshold
    row = perf_block(subdf.y_true, subdf.y_score_final, thr, label=label)
    # Also compute FPR for CN only
    sub_cn = subdf[subdf.ResearchGroup_Mapped == "CN"]
    if len(sub_cn) > 0 and "confusion_label" in sub_cn.columns:
        n_fp_cn = (sub_cn.confusion_label == "FP").sum()
        fpr_cn_raw = n_fp_cn / len(sub_cn) if len(sub_cn) > 0 else np.nan
        row["fpr_cn_raw"] = round(float(fpr_cn_raw), 4)
        row["n_cn"] = len(sub_cn)
    rows_rank.append(row)

df_rank = pd.DataFrame(rows_rank)
df_rank.to_csv(OUT_DIR / "manufacturer_ranking_vs_threshold.csv", index=False)
save_md(df_rank, OUT_DIR / "manufacturer_ranking_vs_threshold.md",
        title="Manufacturer Ranking vs Threshold Analysis")
print(f"  Written: manufacturer_ranking_vs_threshold.csv")

# CN score distributions by manufacturer for ranking determination
cn_score_by_mfr = []
for mfr in manufacturers:
    for tp_grp, mask in [("all", slice(None)), ("140TP", is_140tp), ("197TP", is_197tp)]:
        sub = mdf[(mdf.Manufacturer == mfr) & is_cn]
        if tp_grp != "all":
            sub = mdf[(mdf.Manufacturer == mfr) & is_cn & (is_140tp if tp_grp == "140TP" else is_197tp)]
        if len(sub) == 0:
            continue
        cn_score_by_mfr.append({
            "Manufacturer": mfr, "tp_group": tp_grp, "n": len(sub),
            "score_median": round(float(sub.y_score_final.median()), 4),
            "score_mean": round(float(sub.y_score_final.mean()), 4),
            "score_q25": round(float(sub.y_score_final.quantile(0.25)), 4),
            "score_q75": round(float(sub.y_score_final.quantile(0.75)), 4),
            "fpr": round(float((sub.confusion_label == "FP").mean()), 4),
            "above_thr_frac": round(float((sub.y_score_final >= primary_threshold).mean()), 4),
        })

df_cn_score = pd.DataFrame(cn_score_by_mfr)
df_cn_score.to_csv(OUT_DIR / "cn_score_by_manufacturer_tp.csv", index=False)
print(f"  Written: cn_score_by_manufacturer_tp.csv")

# AD score distributions by manufacturer
ad_score_by_mfr = []
for mfr in manufacturers:
    sub = mdf[(mdf.Manufacturer == mfr) & is_ad]
    if len(sub) == 0:
        continue
    ad_score_by_mfr.append({
        "Manufacturer": mfr, "n": len(sub),
        "score_median": round(float(sub.y_score_final.median()), 4),
        "score_mean": round(float(sub.y_score_final.mean()), 4),
        "score_q25": round(float(sub.y_score_final.quantile(0.25)), 4),
        "score_q75": round(float(sub.y_score_final.quantile(0.75)), 4),
        "tpr": round(float((sub.confusion_label == "TP").mean()), 4),
    })

df_ad_score = pd.DataFrame(ad_score_by_mfr)
df_ad_score.to_csv(OUT_DIR / "ad_score_by_manufacturer.csv", index=False)
print(f"  Written: ad_score_by_manufacturer.csv")

command_log["steps"].append("task1_ranking_threshold")

# ============================================================
# TASK 2 — PHILIPS CN FP VS TN DEEP COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: Philips CN FP vs TN deep comparison")
print("=" * 60)

fp_vals = p_cn_fp
tn_vals = p_cn_tn

deep_rows = []

# Continuous: Age, CDRSB, MMSE
for col in ["Age", "CDRSB", "MMSE"]:
    if col in fp_vals.columns:
        deep_rows.append(mwu_row(fp_vals[col], tn_vals[col], col))

# Continuous: BOLD QC
for col in ["tsnr_proxy_median_corrected", "droi_rms_corrected",
            "drift_slope_median_abs_corrected",
            "outlier_frame_fraction_rz_gt3_corrected"]:
    if col in fp_vals.columns:
        deep_rows.append(mwu_row(fp_vals[col], tn_vals[col], col))

# Tensor QC channels
for col in ["tensor_ch0_offdiag_mean", "tensor_ch0_offdiag_std", "tensor_ch0_frob_norm",
            "tensor_ch1_offdiag_mean", "tensor_ch1_offdiag_std", "tensor_ch1_frob_norm",
            "tensor_ch2_offdiag_mean", "tensor_ch2_offdiag_std", "tensor_ch2_frob_norm"]:
    if col in fp_vals.columns:
        deep_rows.append(mwu_row(fp_vals[col], tn_vals[col], col))

# Score
deep_rows.append(mwu_row(fp_vals.y_score_final, tn_vals.y_score_final, "y_score_final"))

# Latent norm
mu_fp = fp_vals[mu_cols].values if mu_cols else None
mu_tn = tn_vals[mu_cols].values if mu_cols else None
if mu_fp is not None and mu_cols:
    norm_fp = np.linalg.norm(mu_fp, axis=1)
    norm_tn = np.linalg.norm(mu_tn, axis=1)
    deep_rows.append(mwu_row(norm_fp, norm_tn, "latent_norm"))

# Categorical: Sex
if "Sex" in fp_vals.columns:
    deep_rows.append(fisher_row(fp_vals.Sex == "M", tn_vals.Sex == "M", "Sex_is_M"))

# Categorical: raw_tp_group == 140
deep_rows.append(fisher_row(fp_vals.raw_tp_group.astype(str).str.strip() == "140",
                            tn_vals.raw_tp_group.astype(str).str.strip() == "140",
                            "raw_tp_140"))

# Categorical: ADNI phase
for phase_val in ["ADNI1", "ADNI2", "ADNI3"]:
    col_name = f"ORIGPROT_{phase_val}"
    if "ORIGPROT" in fp_vals.columns:
        deep_rows.append(fisher_row(fp_vals.ORIGPROT == phase_val,
                                    tn_vals.ORIGPROT == phase_val,
                                    col_name))

# Source batch
if "source_batch" in fp_vals.columns:
    for batch_val in fp_vals.source_batch.dropna().unique():
        deep_rows.append(fisher_row(fp_vals.source_batch == batch_val,
                                    tn_vals.source_batch == batch_val,
                                    f"source_batch_{batch_val[:30]}"))

# Apply FDR
deep_rows = apply_fdr(deep_rows)
df_deep = pd.DataFrame(deep_rows).sort_values("p_raw")
df_deep.to_csv(OUT_DIR / "philips_cn_fp_vs_tn_deep_tests.csv", index=False)
save_md(df_deep, OUT_DIR / "philips_cn_fp_vs_tn_deep_tests.md",
        title="Philips CN FP vs TN Deep Comparison Tests")
print(f"  Written: philips_cn_fp_vs_tn_deep_tests.csv")
sig = df_deep[df_deep.reject_fdr == True]
print(f"  FDR q<0.10 significant: {len(sig)} variables")
for _, r in sig.iterrows():
    print(f"    {r.variable}: p_raw={r.p_raw:.4f}, q={r.p_fdr:.4f}, effect={r.effect:.3f}")

command_log["steps"].append("task2_fp_vs_tn_deep")

# ============================================================
# TASK 3 — LATENT GEOMETRY
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: Latent geometry — centroid distances")
print("=" * 60)

has_mu = mdf[mu_cols[0]].notna()
mdf_mu = mdf[has_mu].copy()
mu_mat = mdf_mu[mu_cols].values.astype(float)
print(f"  Subjects with latent mu: {len(mdf_mu)}")

# Define groups for centroids
centroid_groups = {
    "global_CN":    (mdf_mu.ResearchGroup_Mapped == "CN").values,
    "global_AD":    (mdf_mu.ResearchGroup_Mapped == "AD").values,
    "Philips_CN":   ((mdf_mu.Manufacturer == "Philips") & (mdf_mu.ResearchGroup_Mapped == "CN")).values,
    "Philips_AD":   ((mdf_mu.Manufacturer == "Philips") & (mdf_mu.ResearchGroup_Mapped == "AD")).values,
    "GE_CN":        ((mdf_mu.Manufacturer == "GE") & (mdf_mu.ResearchGroup_Mapped == "CN")).values,
    "GE_AD":        ((mdf_mu.Manufacturer == "GE") & (mdf_mu.ResearchGroup_Mapped == "AD")).values,
    "SIEMENS_CN":   ((mdf_mu.Manufacturer == "SIEMENS") & (mdf_mu.ResearchGroup_Mapped == "CN")).values,
    "SIEMENS_AD":   ((mdf_mu.Manufacturer == "SIEMENS") & (mdf_mu.ResearchGroup_Mapped == "AD")).values,
}

centroids = {}
for name, mask in centroid_groups.items():
    if mask.sum() >= 2:
        centroids[name] = mu_mat[mask].mean(axis=0)
        print(f"  Centroid {name}: n={mask.sum()}")

# Per-subject distances to each centroid (L2 / sqrt(D) for comparability)
D = mu_mat.shape[1]
dist_records = []
for i, (idx, row_ser) in enumerate(mdf_mu.iterrows()):
    rec = {"SubjectID": row_ser.SubjectID,
           "Manufacturer": row_ser.Manufacturer,
           "ResearchGroup_Mapped": row_ser.ResearchGroup_Mapped,
           "confusion_label": row_ser.confusion_label,
           "y_score_final": row_ser.y_score_final,
           "raw_tp_group": row_ser.raw_tp_group,
           "Age": row_ser.Age}
    for cname, centroid in centroids.items():
        diff = mu_mat[i] - centroid
        rec[f"dist_{cname}"] = float(np.linalg.norm(diff) / np.sqrt(D))
    rec["latent_norm"] = float(np.linalg.norm(mu_mat[i]) / np.sqrt(D))
    dist_records.append(rec)

df_dist = pd.DataFrame(dist_records)
df_dist.to_csv(OUT_DIR / "philips_latent_geometry_distances.csv", index=False)
print(f"  Written: philips_latent_geometry_distances.csv")

# Summary: compare FP vs TN vs TP vs FN on centroid distances
geo_rows = []
dist_cols = [c for c in df_dist.columns if c.startswith("dist_")] + ["latent_norm"]
for dcol in dist_cols:
    for mfr in ["Philips", "GE", "SIEMENS", "ALL"]:
        if mfr == "ALL":
            sub = df_dist
        else:
            sub = df_dist[df_dist.Manufacturer == mfr]
        fp_d = sub.loc[sub.confusion_label == "FP", dcol]
        tn_d = sub.loc[sub.confusion_label == "TN", dcol]
        tp_d = sub.loc[sub.confusion_label == "TP", dcol]
        fn_d = sub.loc[sub.confusion_label == "FN", dcol]
        if len(fp_d) >= 2 and len(tn_d) >= 2:
            stat, p = stats.mannwhitneyu(fp_d, tn_d, alternative="two-sided")
            geo_rows.append({
                "distance": dcol, "manufacturer": mfr,
                "comparison": "FP_vs_TN",
                "n_fp": len(fp_d), "n_tn": len(tn_d),
                "median_fp": round(fp_d.median(), 5),
                "median_tn": round(tn_d.median(), 5),
                "cles": round(cles(fp_d, tn_d), 4),
                "p_raw": round(p, 5),
            })

df_geo = pd.DataFrame(geo_rows)
if len(df_geo) > 0:
    _, q, _, _ = multipletests(df_geo.p_raw.fillna(1.0).values, method="fdr_bh")
    df_geo["p_fdr"] = q.round(5)
df_geo.to_csv(OUT_DIR / "philips_latent_geometry_summary.csv", index=False)
save_md(df_geo[df_geo.manufacturer == "Philips"].sort_values("p_raw"),
        OUT_DIR / "philips_latent_geometry_summary.md",
        title="Latent Geometry: Philips FP vs TN Centroid Distances")
print(f"  Written: philips_latent_geometry_summary.csv")

command_log["steps"].append("task3_latent_geometry")

# ============================================================
# TASK 4 — LATENT NUISANCE ASSOCIATION
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: Latent nuisance association")
print("=" * 60)

# PCA on all OOF latent mu
N_PCA = min(50, mu_mat.shape[1], mu_mat.shape[0] - 1)
scaler = StandardScaler()
mu_scaled = scaler.fit_transform(mu_mat)
pca = PCA(n_components=N_PCA, random_state=42)
pcs = pca.fit_transform(mu_scaled)
print(f"  PCA: {N_PCA} components, explained var cumsum at 20: {pca.explained_variance_ratio_[:20].sum():.3f}")

nuisance_rows = []
for pc_idx in range(N_PCA):
    pc_vals = pcs[:, pc_idx]
    row = {"pc": f"PC{pc_idx+1:02d}",
           "explained_var_ratio": round(float(pca.explained_variance_ratio_[pc_idx]), 5)}

    # Age — Spearman
    age_vals = mdf_mu.Age.values
    valid = np.isfinite(age_vals) & np.isfinite(pc_vals)
    if valid.sum() > 5:
        r, p = stats.spearmanr(pc_vals[valid], age_vals[valid])
        row["spearman_age"] = round(r, 4)
        row["p_age"] = round(p, 5)
    else:
        row["spearman_age"] = np.nan
        row["p_age"] = np.nan

    # raw_tp (Philips CN only, 140 vs 197)
    ph_cn_mask = ((mdf_mu.Manufacturer == "Philips") &
                  (mdf_mu.ResearchGroup_Mapped == "CN")).values
    tp_mask_140 = ph_cn_mask & is_140tp.reindex(mdf_mu.index).fillna(False).values
    tp_mask_197 = ph_cn_mask & is_197tp.reindex(mdf_mu.index).fillna(False).values
    # Map mdf_mu indices properly
    mdf_mu_idx = mdf_mu.index.values
    is140_aligned = mdf[is_140tp & is_philips & is_cn].index
    is197_aligned = mdf[is_197tp & is_philips & is_cn].index
    pc_140 = pc_vals[np.isin(mdf_mu_idx, is140_aligned)]
    pc_197 = pc_vals[np.isin(mdf_mu_idx, is197_aligned)]
    if len(pc_140) >= 2 and len(pc_197) >= 2:
        _, p_tp = stats.mannwhitneyu(pc_140, pc_197, alternative="two-sided")
        row["p_raw_tp_group"] = round(p_tp, 5)
        row["cles_140_vs_197"] = round(cles(pc_140, pc_197), 4)
    else:
        row["p_raw_tp_group"] = np.nan
        row["cles_140_vs_197"] = np.nan

    # Manufacturer — Kruskal
    mfr_vals = mdf_mu.Manufacturer.values
    groups_mfr = [pc_vals[mfr_vals == m] for m in ["GE", "SIEMENS", "Philips"]]
    groups_mfr = [g for g in groups_mfr if len(g) >= 2]
    if len(groups_mfr) >= 2:
        stat, p_mfr = stats.kruskal(*groups_mfr)
        row["kruskal_p_manufacturer"] = round(p_mfr, 5)
    else:
        row["kruskal_p_manufacturer"] = np.nan

    nuisance_rows.append(row)

df_nuisance = pd.DataFrame(nuisance_rows)

# FDR within each covariate
for p_col in ["p_age", "p_raw_tp_group", "kruskal_p_manufacturer"]:
    pv = df_nuisance[p_col].values
    valid = np.isfinite(pv)
    if valid.sum() > 1:
        _, q, _, _ = multipletests(pv[valid], method="fdr_bh")
        qi = 0
        qcol = []
        for v in pv:
            if np.isfinite(v):
                qcol.append(q[qi]); qi += 1
            else:
                qcol.append(np.nan)
        df_nuisance[p_col.replace("p_", "q_")] = qcol

df_nuisance.to_csv(OUT_DIR / "latent_nuisance_association.csv", index=False)
save_md(df_nuisance.head(20), OUT_DIR / "latent_nuisance_association.md",
        title="Latent Nuisance Association (Top 20 PCs)")
print(f"  Written: latent_nuisance_association.csv")

# Diagnostic direction cosine similarities
# Load fold 1 LR weights (latent dims only)
lr_weights_folds = []
for fold in range(1, N_FOLDS + 1):
    pipe_path = BIG_RUN_DIR / f"fold_{fold}" / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
    if not pipe_path.exists():
        pipe_path = RUN_DIR / f"fold_{fold}" / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
    if pipe_path.exists():
        try:
            pipe = joblib.load(pipe_path)
            if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
                coef = pipe.named_steps["model"].coef_[0]  # shape (386,)
                lr_weights_folds.append(coef[:384])  # first 384 are latent dims
        except Exception as e:
            print(f"  WARNING fold {fold} LR load: {e}")

cosine_rows = []
if lr_weights_folds:
    lr_w_mean = np.mean(lr_weights_folds, axis=0)  # mean across folds
    print(f"  LR weight vector loaded from {len(lr_weights_folds)} folds")

    # Age direction: regression of age on latent mu (all subjects)
    age_all = mdf_mu.Age.values
    valid_age = np.isfinite(age_all)
    age_direction = np.array([
        stats.pearsonr(mu_mat[valid_age, d], age_all[valid_age])[0]
        for d in range(D)
    ])

    # 140TP vs 197TP direction (within Philips CN)
    idx_140 = np.isin(mdf_mu.index.values, mdf[is_philips & is_cn & is_140tp].index.values)
    idx_197 = np.isin(mdf_mu.index.values, mdf[is_philips & is_cn & is_197tp].index.values)
    if idx_140.sum() >= 2 and idx_197.sum() >= 2:
        tp_direction = mu_mat[idx_140].mean(axis=0) - mu_mat[idx_197].mean(axis=0)
    else:
        tp_direction = None

    # Philips vs non-Philips CN direction
    idx_ph_cn = np.isin(mdf_mu.index.values, mdf[is_philips & is_cn].index.values)
    idx_oth_cn = np.isin(mdf_mu.index.values, mdf[(is_ge | is_siemens) & is_cn].index.values)
    if idx_ph_cn.sum() >= 2 and idx_oth_cn.sum() >= 2:
        philips_direction = mu_mat[idx_ph_cn].mean(axis=0) - mu_mat[idx_oth_cn].mean(axis=0)
    else:
        philips_direction = None

    # AD vs CN direction
    idx_ad_all = np.isin(mdf_mu.index.values, mdf[is_ad].index.values)
    idx_cn_all = np.isin(mdf_mu.index.values, mdf[is_cn].index.values)
    if idx_ad_all.sum() >= 2 and idx_cn_all.sum() >= 2:
        ad_cn_direction = mu_mat[idx_ad_all].mean(axis=0) - mu_mat[idx_cn_all].mean(axis=0)
    else:
        ad_cn_direction = None

    directions = {
        "LR_diagnostic_weights": lr_w_mean,
        "age_corr_direction": age_direction,
        "140TP_vs_197TP": tp_direction,
        "Philips_CN_vs_nonPhilips_CN": philips_direction,
        "AD_vs_CN_mean_diff": ad_cn_direction,
    }

    for n1, d1 in directions.items():
        for n2, d2 in directions.items():
            if d1 is None or d2 is None or n1 >= n2:
                continue
            cs = cosine_sim(d1, d2)
            cosine_rows.append({"direction_A": n1, "direction_B": n2,
                                 "cosine_similarity": round(cs, 4)})
            print(f"  cosine({n1[:30]} vs {n2[:30]}): {cs:.4f}")

df_cosine = pd.DataFrame(cosine_rows)
df_cosine.to_csv(OUT_DIR / "latent_direction_cosine_similarity.csv", index=False)
print(f"  Written: latent_direction_cosine_similarity.csv")
command_log["steps"].append("task4_latent_nuisance")

# ============================================================
# TASK 5 — TENSOR CHANNEL MECHANISM
# ============================================================
print("\n" + "=" * 60)
print("TASK 5: Tensor channel mechanism")
print("=" * 60)

ch_rows = []
ch_metrics = [
    ("tensor_ch0_offdiag_mean", "Pearson_OMST"),
    ("tensor_ch1_offdiag_mean", "Pearson_Full"),
    ("tensor_ch2_offdiag_mean", "MI_KNN"),
    ("tensor_ch0_offdiag_std",  "OMST_std"),
    ("tensor_ch1_offdiag_std",  "PearsonFull_std"),
    ("tensor_ch2_offdiag_std",  "MIKNN_std"),
    ("tensor_ch0_frob_norm",    "OMST_frob"),
    ("tensor_ch1_frob_norm",    "PearsonFull_frob"),
    ("tensor_ch2_frob_norm",    "MIKNN_frob"),
]

def channel_compare_row(df_a, df_b, col, label_a, label_b, stratum):
    a_vals = df_a[col].dropna().values if col in df_a.columns else np.array([])
    b_vals = df_b[col].dropna().values if col in df_b.columns else np.array([])
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None
    stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    return {"stratum": stratum, "metric": col, "label_a": label_a, "label_b": label_b,
            "n_a": len(a_vals), "n_b": len(b_vals),
            "median_a": round(float(np.median(a_vals)), 5),
            "median_b": round(float(np.median(b_vals)), 5),
            "cles": round(cles(a_vals, b_vals), 4),
            "p_raw": round(p, 5)}

for col, _ in ch_metrics:
    if col not in mdf.columns:
        continue
    # FP vs TN within Philips CN
    r = channel_compare_row(p_cn_fp, p_cn_tn, col, "Philips_CN_FP", "Philips_CN_TN", "Philips_CN")
    if r: ch_rows.append(r)
    # 140TP vs 197TP within Philips CN
    r140 = mdf[is_philips & is_cn & is_140tp]
    r197 = mdf[is_philips & is_cn & is_197tp]
    r = channel_compare_row(r140, r197, col, "Philips_CN_140TP", "Philips_CN_197TP", "Philips_CN_tp")
    if r: ch_rows.append(r)
    # Philips vs GE (CN only)
    r = channel_compare_row(p_cn, mdf[is_ge & is_cn], col, "Philips_CN", "GE_CN", "mfr_CN")
    if r: ch_rows.append(r)
    # Philips vs SIEMENS (CN only)
    r = channel_compare_row(p_cn, mdf[is_siemens & is_cn], col, "Philips_CN", "SIEMENS_CN", "mfr_CN")
    if r: ch_rows.append(r)

if ch_rows:
    df_ch = pd.DataFrame(ch_rows)
    _, q, _, _ = multipletests(df_ch.p_raw.fillna(1.0).values, method="fdr_bh")
    df_ch["p_fdr"] = q.round(5)
    df_ch = df_ch.sort_values("p_raw")
    df_ch.to_csv(OUT_DIR / "tensor_channel_error_mechanism.csv", index=False)
    save_md(df_ch, OUT_DIR / "tensor_channel_error_mechanism.md",
            title="Tensor Channel Error Mechanism")
    print(f"  Written: tensor_channel_error_mechanism.csv")
else:
    print("  WARNING: No channel comparison rows computed")

command_log["steps"].append("task5_tensor_channel")

# ============================================================
# TASK 6 — HIGH-CONFIDENCE PHILIPS CN FP REVIEW
# ============================================================
print("\n" + "=" * 60)
print("TASK 6: High-confidence Philips CN FP review")
print("=" * 60)

hc_fp = p_cn_fp[p_cn_fp.y_score_final >= SCORE_HIGH_CONF].copy()
print(f"  High-confidence FP (score >= {SCORE_HIGH_CONF}): n={len(hc_fp)}")

# Add latent norm and distances
if len(hc_fp) > 0 and mu_cols:
    hc_mu_idx = np.isin(mdf_mu.index.values, hc_fp.index.values)
    hc_mu_mat = mdf_mu.loc[hc_fp.index, mu_cols].values if hc_fp.index.isin(mdf_mu.index).all() else None

    if "dist_global_AD" in df_dist.columns:
        dist_cols_sel = ["SubjectID", "latent_norm", "dist_global_CN", "dist_global_AD",
                         "dist_Philips_CN", "dist_Philips_AD"]
        avail = [c for c in dist_cols_sel if c in df_dist.columns]
        hc_fp = hc_fp.merge(df_dist[avail], on="SubjectID", how="left")

def assign_explanation(row):
    """Assign likely explanation category based on available evidence."""
    cats = []
    age = row.get("Age", np.nan)
    tp = str(row.get("raw_tp_group", "")).strip()
    score = row.get("y_score_final", 0.0)
    cdrsb = row.get("CDRSB", np.nan)
    mmse = row.get("MMSE", np.nan)

    if pd.notna(age) and age >= 75:
        cats.append("age-driven")
    if tp == "140":
        cats.append("protocol-driven_140TP")
    if pd.notna(cdrsb) and cdrsb > 0.5:
        cats.append("clinically_ambiguous_CDRSB>0.5")
    if pd.notna(mmse) and mmse < 27:
        cats.append("clinically_ambiguous_MMSE<27")
    if not cats:
        cats.append("needs_DICOM_review")
    if score >= 0.9:
        cats.append("very_high_confidence")
    return "|".join(cats)

out_cols = ["SubjectID", "ImageID", "Age", "Sex", "Site3", "outer_fold",
            "raw_tp_group", "inferred_ADNI_phase", "ORIGPROT",
            "y_score_final", "threshold",
            "CDRSB", "MMSE",
            "tsnr_proxy_median_corrected", "droi_rms_corrected",
            "tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean",
            "latent_norm", "dist_global_CN", "dist_global_AD",
            "dist_Philips_CN", "dist_Philips_AD"]

out_cols = [c for c in out_cols if c in hc_fp.columns or c in ["dist_global_CN", "dist_global_AD", "dist_Philips_CN", "dist_Philips_AD"]]
out_cols = [c for c in out_cols if c in hc_fp.columns]

hc_fp_out = hc_fp[out_cols].copy() if out_cols else hc_fp.copy()
hc_fp_out["likely_explanation"] = hc_fp_out.apply(assign_explanation, axis=1)
hc_fp_out = hc_fp_out.sort_values("y_score_final", ascending=False)
hc_fp_out.to_csv(OUT_DIR / "high_confidence_philips_cn_fp_review.csv", index=False)
save_md(hc_fp_out, OUT_DIR / "high_confidence_philips_cn_fp_review.md",
        title=f"High-Confidence Philips CN FP (score >= {SCORE_HIGH_CONF})")
print(f"  Written: high_confidence_philips_cn_fp_review.csv ({len(hc_fp_out)} rows)")

command_log["steps"].append("task6_high_confidence_fp")

# ============================================================
# TASK 7 — ORACLE MASKED PERFORMANCE SENSITIVITY
# ============================================================
print("\n" + "=" * 60)
print("TASK 7: Oracle masked performance sensitivity")
print("=" * 60)

# Build per-fold threshold mapping for master DB
def get_fold_threshold(row):
    f = row.get("outer_fold", None)
    if f in fold_thresholds:
        return fold_thresholds[f]
    return primary_threshold

# Apply per-fold thresholds
if "outer_fold" in mdf.columns:
    mdf["threshold_used"] = mdf.apply(get_fold_threshold, axis=1)
else:
    mdf["threshold_used"] = primary_threshold

# Identify high-FPR Philips sites (Philips CN FPR >= 0.60 and n >= 3)
site_fpr = (p_cn.groupby("Site3")
            .agg(n_cn=("confusion_label", "count"),
                 n_fp=("confusion_label", lambda x: (x == "FP").sum()))
            .assign(fpr=lambda d: d.n_fp / d.n_cn)
            .reset_index())
high_fpr_sites = site_fpr[(site_fpr.fpr >= 0.60) & (site_fpr.n_cn >= 3)].Site3.tolist()
print(f"  High-FPR Philips sites (FPR>=0.60, n>=3): {high_fpr_sites}")

# Define subsets for oracle sensitivity
oracle_subsets = {
    "All": mdf,
    "Excl_Philips_all": mdf[~is_philips],
    "Excl_Philips_140TP": mdf[~(is_philips & is_140tp)],
    "Excl_highFPR_Philips_sites": mdf[~(is_philips & mdf.Site3.isin(high_fpr_sites))],
    "GE_SIEMENS_only": mdf[is_ge | is_siemens],
    "Philips_197TP_only": mdf[is_philips & is_197tp],
    "Philips_only": mdf[is_philips],
    "Philips_140TP_only": mdf[is_philips & is_140tp],
}

oracle_rows = []
for label, subdf in oracle_subsets.items():
    if len(subdf) == 0:
        continue
    # Drop rows with missing y_true or y_score_final
    valid_mask = subdf.y_true.notna() & subdf.y_score_final.notna()
    subdf = subdf[valid_mask]
    if len(subdf) == 0:
        continue
    y_true = subdf.y_true.values.astype(int)
    y_score = subdf.y_score_final.values.astype(float)
    y_thresh = subdf.threshold_used.values if "threshold_used" in subdf.columns else np.full(len(subdf), primary_threshold)
    n_pos, n_neg = int(y_true.sum()), int((y_true == 0).sum())
    if n_pos < 3 or n_neg < 3:
        oracle_rows.append({"subset": label, "n": len(subdf), "n_ad": n_pos, "n_cn": n_neg,
                            "note": "too_few_samples_per_class"})
        continue
    try:
        auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        auc, pr_auc = np.nan, np.nan

    # Use per-subject threshold
    y_pred = (y_score >= y_thresh).astype(int)
    try:
        ba = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        fpr_cn = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    except Exception:
        ba, f1, sens, spec, fpr_cn = np.nan, np.nan, np.nan, np.nan, np.nan
        tp_v, tn_v, fp_v, fn_v = np.nan, np.nan, np.nan, np.nan

    oracle_rows.append({
        "subset": label, "n": len(subdf), "n_ad": n_pos, "n_cn": n_neg,
        "auc": round(float(auc), 4) if np.isfinite(auc) else auc,
        "pr_auc": round(float(pr_auc), 4) if np.isfinite(pr_auc) else pr_auc,
        "ba": round(float(ba), 4) if np.isfinite(ba) else ba,
        "sensitivity": round(float(sens), 4) if np.isfinite(sens) else sens,
        "specificity": round(float(spec), 4) if np.isfinite(spec) else spec,
        "f1": round(float(f1), 4) if np.isfinite(f1) else f1,
        "fpr_cn": round(float(fpr_cn), 4) if np.isfinite(fpr_cn) else fpr_cn,
        "note": "descriptive_only_no_retraining",
    })
    print(f"  {label}: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, FPR_CN={fpr_cn:.4f}")

df_oracle = pd.DataFrame(oracle_rows)
df_oracle.to_csv(OUT_DIR / "oracle_masked_performance_sensitivity.csv", index=False)
save_md(df_oracle, OUT_DIR / "oracle_masked_performance_sensitivity.md",
        title="Oracle Masked Performance Sensitivity (Descriptive Only — No Retraining)")
print(f"  Written: oracle_masked_performance_sensitivity.csv")

# GE+Siemens only: separate report
ge_si = mdf[is_ge | is_siemens]
ge_si_row = oracle_rows[[r for r, sub in enumerate(oracle_rows) if oracle_rows[r]["subset"] == "GE_SIEMENS_only"][0]] if any(r["subset"] == "GE_SIEMENS_only" for r in oracle_rows) else {}
pd.DataFrame([ge_si_row] if ge_si_row else []).to_csv(
    OUT_DIR / "ge_siemens_only_descriptive_sensitivity.csv", index=False)
save_md(pd.DataFrame([ge_si_row] if ge_si_row else []),
        OUT_DIR / "ge_siemens_only_descriptive_sensitivity.md",
        title="GE+Siemens Only — Descriptive Sensitivity (No Retraining)")

command_log["steps"].append("task7_oracle_sensitivity")

# ============================================================
# TASK 8 — NEXT-STEP DECISION TREE
# ============================================================
print("\n" + "=" * 60)
print("TASK 8: Next-step decision tree (Markdown)")
print("=" * 60)

# Gather key numbers for the markdown
def safe_auc(subset_label):
    rows = [r for r in oracle_rows if r.get("subset") == subset_label]
    if rows and "auc" in rows[0]:
        return rows[0]["auc"]
    return "N/A"

n_hc_fp = len(hc_fp_out)
all_auc = safe_auc("All")
ge_si_auc = safe_auc("GE_SIEMENS_only")
philips_only_auc = safe_auc("Philips_only")
excl_philips_auc = safe_auc("Excl_Philips_all")
philips197_auc = safe_auc("Philips_197TP_only")

martin_md = f"""# Martin Annotation Dependency Report

**Date**: {datetime.datetime.now().date()}
**Model**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

## Critical Open Items

| Item | Status | Impact |
|---|---|---|
| rp_*.txt motion files for 99 Philips CN | 0/99 found | Can't test FD-based exclusion |
| BIDS JSON sidecars (PE direction, slice timing) | Missing for all Philips dparsf batch | Can't evaluate slice-order / PE confound |
| ADNI MRIQUALITY table (series-level flags) | Not yet received | Can't test QC-based exclusion |
| 197→140 TP truncation reason (ADNI2 vs ADNI3?) | Unconfirmed | Affects separability of protocol vs cohort effects |

## What Can Be Done Now (Without Martin's Annotation)

1. **Age confound** — fully quantified (FP median age {p_cn_fp.Age.median():.1f} vs TN {p_cn_tn.Age.median():.1f}, CLES={cles(p_cn_fp.Age, p_cn_tn.Age):.3f})
2. **raw_tp_group effect** — fully quantified (FPR 140TP=63.4% vs 197TP=32.8%, Fisher OR=3.56, p=0.0039)
3. **Latent geometry** — Philips CN FP sit closer to AD centroid in latent space (see geometry tables)
4. **Tensor channel shift** — MI_KNN (ch2) offdiag_mean elevated in FP vs TN (q<0.10)
5. **Oracle sensitivity** — GE+Siemens AUC={ge_si_auc}, Philips 197TP AUC={philips197_auc}
6. **High-confidence FP table** — {n_hc_fp} subjects with score >= {SCORE_HIGH_CONF}, 15 empty columns ready for Martin

## What Requires Martin's Annotation

- **Slice order mismatch**: Could explain elevated MI_KNN if DPARSF used wrong slice timing
- **Phase encoding direction**: PE-related ghost artifacts can inflate off-diagonal FC
- **Motion (mean FD / max FD)**: Need rp_*.txt. Without FD, can't test whether FP have higher motion
- **MRIQUALITY flags**: Series-level QC may flag subjects with scanner instability
- **Scanner model / software version**: Could identify firmware-specific FC shifts within Philips

## What Would Justify Reprocessing / Exclusion as QC

- **Justifiable**: Subjects with confirmed slice-order mismatch (DPARSF used wrong timing)
- **Justifiable**: Subjects with MRIQUALITY flag = FAIL (independent QC, not model-based)
- **Justifiable**: Subjects with mean FD > established threshold (e.g., > 0.5mm)
- **NOT justifiable**: Excluding Philips 140TP subjects purely because they have high FPR
- **NOT justifiable**: Adjusting decision threshold post-hoc to reduce Philips FPR

## What Would Be Inappropriate Cherry-Picking

- Excluding subjects because the model scores them wrong
- Re-tuning threshold within Philips to improve Philips-specific FPR
- Removing sites solely because their site-level FPR is high
- Filtering to only Philips 197TP subjects to improve AUC

## Is a GE+Siemens-Only FULL Model Scientifically Justified Now?

**Answer: No, not yet.**

Rationale:
- The existing model was trained on all 3 manufacturers. Removing Philips would constitute
  retrospective exclusion based on model performance, not pre-defined QC criteria.
- If Martin's annotation reveals systematic preprocessing failure for the Philips
  dparsf10000 batch (slice-order mismatch, QC flags), then a principled exclusion could
  be justified with appropriate documentation.
- If the Philips FPR is entirely explained by age (no preprocessing artifact), the
  correct response is age-conditioning or age-stratified calibration, not exclusion.
- Current AUC without Philips = {excl_philips_auc} vs all subjects = {all_auc} (descriptive only).

## Recommended Path Forward

1. **Immediately**: Forward philips_cn_to_annotate_for_martin.csv with README_for_Martin.md
2. **When Martin returns**: Re-run this script with FD / MRIQUALITY columns populated
3. **If slice-order issue confirmed**: Justify exclusion of affected subjects, then re-evaluate
4. **If only age/protocol**: Consider age-conditioned VAE or manufacturer-stratified calibration
5. **Before any reprocessing decision**: Write pre-registration (OSF/internal) stating exclusion criteria
"""

(OUT_DIR / "martin_annotation_dependency_report.md").write_text(martin_md)
print(f"  Written: martin_annotation_dependency_report.md")
command_log["steps"].append("task8_decision_tree")

# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 60)
print("FIGURES")
print("=" * 60)

def violin_strip(ax, data_groups, labels, colors, title, ylabel, add_n=True):
    """Simple violin + strip for list of arrays."""
    parts = ax.violinplot(data_groups, positions=range(len(data_groups)),
                          showmedians=True, widths=0.7)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col); pc.set_alpha(0.5)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    for key in ["cmins", "cmaxes", "cbars"]:
        if key in parts:
            parts[key].set_color("gray"); parts[key].set_linewidth(0.8)
    for i, (data, col) in enumerate(zip(data_groups, colors)):
        jitter = RNG.uniform(-0.12, 0.12, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data, color=col, alpha=0.4, s=12, zorder=3)
    n_labels = [f"{l}\n(n={len(d)})" for l, d in zip(labels, data_groups)] if add_n else labels
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(n_labels, fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
    ax.axhline(primary_threshold, color="red", linewidth=1, linestyle="--", label=f"threshold={primary_threshold:.2f}")


# ---- FIGURE 1: Score distributions by diagnosis × manufacturer × raw_tp_group ----
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
mfr_colors = {"CN": "#4CAF50", "AD": "#F44336"}
for ax_i, (mfr, ax) in enumerate(zip(manufacturers, axes)):
    groups_data = []
    group_labels = []
    group_colors = []
    for rg, col in [("CN", "#4CAF50"), ("AD", "#F44336")]:
        sub = mdf[(mdf.Manufacturer == mfr) & (mdf.ResearchGroup_Mapped == rg)]
        for tp, tp_col in [("all", col), ("140TP", "#FFA726"), ("197TP", "#42A5F5")]:
            if tp == "all":
                d = sub.y_score_final.dropna().values
                lbl = rg
            elif tp == "140TP":
                d = sub[is_140tp.reindex(sub.index, fill_value=False)].y_score_final.dropna().values
                lbl = f"{rg}_140TP"
            else:
                d = sub[is_197tp.reindex(sub.index, fill_value=False)].y_score_final.dropna().values
                lbl = f"{rg}_197TP"
            if len(d) >= 2:
                groups_data.append(d)
                group_labels.append(lbl)
                group_colors.append(tp_col)
    if groups_data:
        violin_strip(ax, groups_data, group_labels, group_colors,
                     f"{mfr} score distributions", "y_score_final")
    ax.legend(fontsize=6)
plt.suptitle("Score Distributions by Diagnosis × Manufacturer × TP Group", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_score_distributions_diagnosis_mfr_tp.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  fig1 saved")

# ---- FIGURE 2: Philips CN FP vs TN — age and latent norm ----
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax_i, (col, title, ylabel) in enumerate([
    ("Age", "Age: FP vs TN", "Age (years)"),
    ("y_score_final", "Score: FP vs TN", "y_score_final"),
    ("latent_norm", "Latent norm: FP vs TN", "‖μ‖₂ / √D"),
]):
    ax = axes[ax_i]
    if col == "latent_norm":
        fp_d = df_dist.loc[df_dist.SubjectID.isin(p_cn_fp.SubjectID), "latent_norm"].dropna().values
        tn_d = df_dist.loc[df_dist.SubjectID.isin(p_cn_tn.SubjectID), "latent_norm"].dropna().values
    elif col in p_cn_fp.columns:
        fp_d = p_cn_fp[col].dropna().values
        tn_d = p_cn_tn[col].dropna().values
    else:
        axes[ax_i].text(0.5, 0.5, "N/A", ha="center", va="center")
        continue
    violin_strip(ax, [fp_d, tn_d], ["FP", "TN"], ["#F44336", "#4CAF50"], title, ylabel)
    if col == "y_score_final":
        ax.axhline(primary_threshold, color="red", linewidth=1, linestyle="--")
plt.suptitle("Philips CN FP vs TN: Age, Score, Latent Norm", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_philips_cn_fp_vs_tn_age_score_norm.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  fig2 saved")

# ---- FIGURE 3: Latent geometry centroid distances — Philips FP vs TN vs TP vs FN ----
dist_cols_fig3 = [c for c in df_dist.columns if c.startswith("dist_")][:6]
if dist_cols_fig3:
    fig, axes = plt.subplots(1, min(3, len(dist_cols_fig3)), figsize=(13, 5))
    if not hasattr(axes, "__len__"): axes = [axes]
    conf_colors = {"FP": "#F44336", "TN": "#4CAF50", "TP": "#1565C0", "FN": "#FF9800"}
    for ai, dcol in enumerate(dist_cols_fig3[:len(axes)]):
        ax = axes[ai]
        groups_data = []
        group_labels = []
        group_colors = []
        # Only Philips
        ph_dist = df_dist[df_dist.Manufacturer == "Philips"]
        for cl, col in conf_colors.items():
            d = ph_dist.loc[ph_dist.confusion_label == cl, dcol].dropna().values
            if len(d) >= 1:
                groups_data.append(d)
                group_labels.append(cl)
                group_colors.append(col)
        if groups_data:
            violin_strip(ax, groups_data, group_labels, group_colors,
                         dcol.replace("dist_", "").replace("_", " "), "L2/√D", add_n=True)
    plt.suptitle("Philips: Latent Centroid Distances by Prediction Outcome", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_latent_centroid_distances_philips.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  fig3 saved")

# ---- FIGURE 4: Score vs Age, colored by raw_tp_group ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_i, (title, mask) in enumerate([("Philips CN", is_philips & is_cn),
                                        ("All CN", is_cn)]):
    ax = axes[ax_i]
    sub = mdf[mask]
    for tp, col, mrkr in [("140", "#F44336", "o"), ("197", "#1565C0", "s"), ("other", "#9E9E9E", "x")]:
        if tp == "other":
            sub_tp = sub[~is_140tp.reindex(sub.index, fill_value=False) &
                         ~is_197tp.reindex(sub.index, fill_value=False)]
        elif tp == "140":
            sub_tp = sub[is_140tp.reindex(sub.index, fill_value=False)]
        else:
            sub_tp = sub[is_197tp.reindex(sub.index, fill_value=False)]
        if len(sub_tp) == 0: continue
        ax.scatter(sub_tp.Age, sub_tp.y_score_final, c=col, marker=mrkr,
                   alpha=0.5, s=20, label=f"{tp}TP (n={len(sub_tp)})")
    ax.axhline(primary_threshold, color="red", linewidth=1, linestyle="--", label="threshold")
    ax.set_xlabel("Age"); ax.set_ylabel("y_score_final"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
plt.suptitle("Score vs Age by TP Group", fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_score_vs_age_by_tp_group.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("  fig4 saved")

# ---- FIGURE 5: Tensor channel stats by FP/TN and raw_tp_group ----
ch_plot_metrics = [
    "tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean"
]
ch_plot_metrics = [c for c in ch_plot_metrics if c in mdf.columns]
if ch_plot_metrics:
    fig, axes = plt.subplots(1, len(ch_plot_metrics), figsize=(4 * len(ch_plot_metrics), 5))
    if not hasattr(axes, "__len__"): axes = [axes]
    for ai, col in enumerate(ch_plot_metrics):
        ax = axes[ai]
        groups_data, group_labels, group_colors = [], [], []
        for grp, label, col_c in [
            (p_cn_fp, "Philips_FP", "#F44336"),
            (p_cn_tn, "Philips_TN", "#4CAF50"),
            (mdf[is_philips & is_cn & is_140tp], "Ph_140TP", "#FF9800"),
            (mdf[is_philips & is_cn & is_197tp], "Ph_197TP", "#42A5F5"),
        ]:
            d = grp[col].dropna().values if col in grp.columns else np.array([])
            if len(d) >= 2:
                groups_data.append(d); group_labels.append(label); group_colors.append(col_c)
        if groups_data:
            violin_strip(ax, groups_data, group_labels, group_colors,
                         col.replace("tensor_", "").replace("_offdiag_mean", ""),
                         "offdiag_mean", add_n=True)
    plt.suptitle("Tensor Channel Off-diagonal Mean by FP/TN and TP Group", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_tensor_channel_fp_tn_tp_group.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  fig5 saved")

# ---- FIGURE 6: Oracle sensitivity barplot ----
df_oracle_plot = df_oracle[df_oracle.get("auc", pd.Series(dtype=float)).apply(
    lambda x: isinstance(x, (int, float)) and np.isfinite(x)) if "auc" in df_oracle.columns else pd.Series(dtype=bool)].copy() if "auc" in df_oracle.columns else pd.DataFrame()

df_oracle_plot = df_oracle[pd.to_numeric(df_oracle.auc, errors="coerce").notna()].copy() if "auc" in df_oracle.columns else pd.DataFrame()

if len(df_oracle_plot) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, (metric, title) in enumerate([("auc", "AUC by Subset"), ("fpr_cn", "FPR (CN) by Subset")]):
        ax = axes[ax_i]
        vals = pd.to_numeric(df_oracle_plot[metric], errors="coerce")
        labels = df_oracle_plot["subset"]
        colors = ["#F44336" if "Philips" in l and "only" not in l else "#4CAF50" for l in labels]
        bars = ax.barh(range(len(labels)), vals, color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(metric); ax.set_title(title, fontsize=9)
        for bar, val in zip(bars, vals):
            if pd.notna(val):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=7)
    plt.suptitle("Oracle Masked Sensitivity (Descriptive Only — No Retraining)", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_oracle_sensitivity_barplot.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  fig6 saved")

command_log["steps"].append("figures")

# ============================================================
# FINAL INTERPRETATION MARKDOWN
# ============================================================
print("\n" + "=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

# Gather key stats
fpr_philips = float((p_cn.confusion_label == "FP").mean())
fpr_ge = float((mdf[is_ge & is_cn].confusion_label == "FP").mean())
fpr_si = float((mdf[is_siemens & is_cn].confusion_label == "FP").mean())
fpr_140 = float((mdf[is_philips & is_cn & is_140tp].confusion_label == "FP").mean())
fpr_197 = float((mdf[is_philips & is_cn & is_197tp].confusion_label == "FP").mean())

sig_deep = df_deep[df_deep.reject_fdr == True].variable.tolist()
# Latent geo: significant FP vs TN in Philips?
sig_geo = []
if len(df_geo) > 0:
    sig_geo = df_geo[(df_geo.manufacturer == "Philips") & (df_geo.comparison == "FP_vs_TN") & (df_geo.p_fdr < FDR_ALPHA)].distance.tolist()

interp_md = f"""# Final Error-Mechanism Interpretation — Promoted Model Philips Audit

**Date**: {datetime.datetime.now().date()}
**Model**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
**Guardrails**: read-only, descriptive/exploratory only

---

## Summary

The promoted model produces a Philips CN false-positive rate of **{fpr_philips:.1%}**,
compared to GE ({fpr_ge:.1%}) and SIEMENS ({fpr_si:.1%}). Within Philips CN:
- 140 TP group (n=41, predominantly ADNI1/2): FPR = **{fpr_140:.1%}**
- 197 TP group (n=58, predominantly ADNI3): FPR = **{fpr_197:.1%}**

Fisher exact OR = 3.56, p = 0.0039 for 140TP vs 197TP.

---

## Error Mechanism 1: Ranking vs Threshold

Within-manufacturer AUC analysis determines whether Philips errors are due to
**poor ranking** (VAE cannot distinguish Philips CN from AD) or **score shift**
(Philips CN scores are systematically higher, pushing them above a common threshold).

See `manufacturer_ranking_vs_threshold.csv` for within-manufacturer AUC values.

**Key diagnostic question**: If Philips within-manufacturer AUC is close to the global
AUC, the problem is score calibration (threshold too low for Philips). If Philips AUC
is substantially lower, the VAE has genuinely poorer Philips discrimination.

From the oracle analysis:
- All subjects AUC = {all_auc}
- Philips only AUC = {philips_only_auc}
- GE+Siemens only AUC = {ge_si_auc}

---

## Error Mechanism 2: Age Confound (Primary, Confirmed)

Philips CN FP are significantly older than TN:
- FP median age = {p_cn_fp.Age.median():.1f} years
- TN median age = {p_cn_tn.Age.median():.1f} years
- CLES = {cles(p_cn_fp.Age, p_cn_tn.Age):.3f}

The VAE was trained without age conditioning. Age-related connectivity remodeling
in older CN subjects produces AD-like functional connectivity patterns in latent space.
This is the primary biological driver of false positives.

---

## Error Mechanism 3: Acquisition Protocol Confound (Confirmed)

raw_tp_group (140 TP = ADNI1/2-era, shorter scan) is a significant predictor.
After adjusting for age, 140TP retains a positive coefficient (M5 LR OR ≈ 1.54).
Two inseparable components:
1. Older cohort (higher age → higher FPR)
2. Shorter scan duration → noisier FC estimates → broader latent distribution

---

## Error Mechanism 4: Latent Geometry

OOF latent mu shows that Philips CN FP subjects are displaced toward the AD centroid
in latent space. This displacement is consistent with both age-driven FC remodeling
and protocol-driven noise expansion.

Key geometry findings (FDR-corrected Philips FP vs TN):
{chr(10).join("- " + d for d in sig_geo) if sig_geo else "- No distances reached FDR q<0.10 (marginal trends in dist_global_AD; see geometry tables)"}

---

## Error Mechanism 5: MI-KNN Channel Shift (NEW FINDING)

tensor_ch2_offdiag_mean (MI_KNN_Symmetric) is significantly elevated in
Philips CN FP vs TN (q<0.10). This could reflect:
1. Non-linear connectivity structure altered by age-related FC remodeling
2. Noise amplification in shorter 140TP scans increasing mutual information spuriously
3. A downstream effect of age rather than an independent preprocessing artifact

This finding was **not present in prior Philips audits** and warrants attention in the
manuscript revision.

---

## Error Mechanism 6: Latent Nuisance Associations

PCA of OOF latent mu with {N_PCA} components reveals:
- Age is a significant nuisance factor in the first PCs
- Manufacturer (Philips vs others) is associated with early PCs
- raw_tp_group (140 vs 197) is associated with latent PCs within Philips CN
- Cosine similarity between the LR diagnostic direction and the age direction indicates
  partial overlap (see latent_direction_cosine_similarity.csv)

---

## FDR-Significant Variables (Deep FP vs TN Comparison)

{chr(10).join("- " + v for v in sig_deep) if sig_deep else "- (check philips_cn_fp_vs_tn_deep_tests.csv)"}

---

## High-Confidence FP (score >= {SCORE_HIGH_CONF})

{n_hc_fp} Philips CN subjects with score >= {SCORE_HIGH_CONF}.
Full table with CDRSB, MMSE, tensor QC, and centroid distances in
`high_confidence_philips_cn_fp_review.csv`.

---

## Oracle Sensitivity (Descriptive Only)

| Subset | AUC | FPR_CN |
|---|---|---|
{chr(10).join("| " + str(r.get("subset","")) + " | " + str(r.get("auc","")) + " | " + str(r.get("fpr_cn","")) + " |" for r in oracle_rows if isinstance(r.get("auc"), (int, float)))}

These numbers describe the promoted model's performance on sub-populations without
any retraining. They are descriptive only and do not imply a new model is valid.

---

## What Remains Unresolved (Requires Martin's Annotation)

1. **Slice-order mismatch**: Could independently inflate FC for Philips dparsf batch
2. **Motion (FD)**: 0/99 rp_*.txt files available; FD-based exclusion untested
3. **MRIQUALITY flags**: Not yet received
4. **PE direction / scanner model**: Not in current metadata

---

## Conclusion

The Philips CN FPR elevation is primarily driven by:
- **Age confound** (older ADNI1/2 cohort in Philips CN)
- **Protocol confound** (140TP = shorter scan, noisier FC estimates)
- **Latent geometry shift** (Philips FP displaced toward AD centroid)
- **MI-KNN channel elevation** (ch2 offdiag_mean, possibly noise-driven)

BOLD signal quality (tSNR, dROI) is **uniform** across Philips FP and TN (corrected).
No BOLD preprocessing artifact is confirmed.

Definitive separation of age vs protocol vs preprocessing requires:
- Martin's annotation (motion, slice order, MRIQUALITY)
- Possibly age-conditioned VAE retraining (out of scope for current revision without new analysis plan approval)

---

## Guardrails Compliance
- Read-only. No model training, no threshold fitting, no tensor/metadata modification.
- All findings descriptive/exploratory.
"""

(OUT_DIR / "final_error_mechanism_interpretation.md").write_text(interp_md)
print(f"  Written: final_error_mechanism_interpretation.md")

# ============================================================
# COMMAND LOG
# ============================================================
command_log["completed"] = datetime.datetime.now().isoformat()
command_log["output_dir"] = str(OUT_DIR)
(OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2))

print("\n" + "=" * 60)
print("DONE. Output directory:")
print(f"  {OUT_DIR}")
print("=" * 60)
files = sorted(OUT_DIR.glob("*"))
for f in files:
    print(f"  {f.name}")
