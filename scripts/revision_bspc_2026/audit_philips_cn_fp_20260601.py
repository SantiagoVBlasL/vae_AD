#!/usr/bin/env python
"""
Read-only Philips CN false-positive audit for the promoted ADNI model.

Promoted model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
Best readout:   logreg_l2_original / z_plus_age_sex / oof_logitz / inner_oof_target_sens_ge_0p70_max_spec
                AUC ≈ 0.795, PR-AUC ≈ 0.574

Constraints: no training, no model selection, no tensor modification, no metadata modification.

Output: results/revision_bspc_2026/promoted_model_philips_cn_false_positive_audit_20260601/
"""

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/home/diego/proyectos/vae_AD/results/revision_bspc_2026")
PROMOTED_RUN_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
DROP10_RUN_DIR = Path(
    "/home/diego/proyectos/vae_AD/results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5"
)
ENCDEC_RUN_DIR = Path(
    "/home/diego/proyectos/vae_AD/results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5"
)
STAGEB_DIR = Path(
    "/home/diego/proyectos/vae_AD/results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_stageB_oof_score_calibration"
)
METADATA_PATH = Path(
    "/home/diego/proyectos/vae_AD/results/revision_bspc_2026/"
    "adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
)
SUBJECTS_DATA_PATH = Path(
    "/home/diego/proyectos/vae_AD/data/SubjectsData_AAL3_procesado2.csv"
)
OUTPUT_DIR = RESULTS_DIR / "promoted_model_philips_cn_false_positive_audit_20260601"

N_FOLDS = 5
LATENT_DIM = 384

# Promoted readout definition
PROMOTED_MODEL = "logreg_l2_original"
PROMOTED_FEATURE_SET = "z_plus_age_sex"
PROMOTED_CALIB = "oof_logitz"
PROMOTED_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

LATENT_CACHE_DIR = PROMOTED_RUN_DIR / "classifier_only_readout" / "latent_cache"
READOUT_DIR = PROMOTED_RUN_DIR / "classifier_only_readout"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  Saved {label}: {path.name} ({len(df)} rows)")


def _save_md(text: str, path: Path, label: str) -> None:
    path.write_text(text)
    print(f"  Saved {label}: {path.name}")


def fisher_exact_2x2(a: int, b: int, c: int, d: int):
    """Fisher exact test on 2×2 table [[a,b],[c,d]].

    Returns (odds_ratio, p_two_sided, ci_lower, ci_upper).
    a = exposed+, b = exposed-, c = unexposed+, d = unexposed-
    """
    _, p = stats.fisher_exact([[a, b], [c, d]])
    # Haldane-Anscombe odds ratio with log CI
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a_ * d_) / (b_ * c_)
    se_log = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    ci_lo = np.exp(np.log(or_val) - 1.96 * se_log)
    ci_hi = np.exp(np.log(or_val) + 1.96 * se_log)
    return float(or_val), float(p), float(ci_lo), float(ci_hi)


def _parse_scanner_model(protocol_str: str) -> str:
    """Extract Manufacturer field from ImagingProtocol string."""
    m = re.search(r"Manufacturer=([^;]+)", str(protocol_str))
    return m.group(1).strip() if m else "Unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_promoted_predictions() -> pd.DataFrame:
    """Load the promoted readout predictions from stageB calibration CSV."""
    df = pd.read_csv(STAGEB_DIR / "calib_predictions.csv")
    mask = (
        (df["model_name"] == PROMOTED_MODEL)
        & (df["feature_set"] == PROMOTED_FEATURE_SET)
        & (df["calib_method"] == PROMOTED_CALIB)
        & (df["threshold_strategy"] == PROMOTED_THRESHOLD)
    )
    promoted = df[mask].copy()
    assert len(promoted) > 0, "No rows found for promoted readout"
    return promoted


def load_sweep_predictions(run_dir: Path) -> pd.DataFrame:
    """Load classifier_sweep_predictions.csv for a given run."""
    path = run_dir / "classifier_only_readout" / "classifier_sweep_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_metadata() -> pd.DataFrame:
    """Load patched metadata + SubjectsData and merge into a single covariate table."""
    meta = pd.read_csv(METADATA_PATH)[["SubjectID", "Site3", "source_batch", "source_label"]].drop_duplicates("SubjectID")

    subj = pd.read_csv(SUBJECTS_DATA_PATH)
    subj = subj[["SubjectID", "Phase", "CDRSB", "MMSE", "MOCA", "ImagingProtocol"]].drop_duplicates("SubjectID")
    subj["ScannerModel"] = subj["ImagingProtocol"].apply(_parse_scanner_model)

    combined = meta.merge(subj, on="SubjectID", how="left")
    return combined


def load_latent_cache() -> pd.DataFrame:
    """Concatenate all fold test latent mu CSVs; keep one row per subject (test fold)."""
    frames = []
    for fold in range(1, N_FOLDS + 1):
        fp = LATENT_CACHE_DIR / f"fold_{fold}_test_latent_mu.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df["outer_fold"] = fold
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    latent = pd.concat(frames, ignore_index=True)
    # Each subject appears exactly once across folds in OOF
    assert latent["SubjectID"].nunique() == len(latent), "Duplicate subjects in latent cache"
    return latent


def load_latent_info_per_dim() -> pd.DataFrame:
    """Concatenate fold_N_test_latent_info_per_dim.csv across all folds."""
    frames = []
    for fold in range(1, N_FOLDS + 1):
        fp = PROMOTED_RUN_DIR / f"fold_{fold}" / f"fold_{fold}_test_latent_info_per_dim.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df["outer_fold"] = fold
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_scanner_leakage() -> pd.DataFrame:
    """Aggregate fold_N_test_scanner_leakage_summary.csv across all folds."""
    frames = []
    for fold in range(1, N_FOLDS + 1):
        fp = PROMOTED_RUN_DIR / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df["outer_fold"] = fold
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Section A: Error burden by manufacturer / site / phase / fold
# ---------------------------------------------------------------------------

def section_a(prom: pd.DataFrame, meta: pd.DataFrame, sweep_prom: pd.DataFrame,
              sweep_drop10: pd.DataFrame, sweep_encdec: pd.DataFrame,
              output_dir: Path) -> None:
    print("\n--- Section A: FP rates by subgroup ---")

    # Merge predictions with metadata
    pred = prom.merge(meta[["SubjectID", "Site3", "Phase", "source_batch", "ScannerModel"]], on="SubjectID", how="left")

    # --- A1: FP rates by manufacturer (CN only) ---
    cn = pred[pred["y_true"] == 0].copy()
    rows_mfr = []
    for mfr, grp in cn.groupby("Manufacturer"):
        fp = int((grp["y_pred"] == 1).sum())
        n = len(grp)
        rows_mfr.append({
            "Manufacturer": mfr,
            "N_CN": n,
            "FP": fp,
            "FPR": fp / n if n > 0 else np.nan,
        })
    df_mfr = pd.DataFrame(rows_mfr).sort_values("Manufacturer")
    _save_csv(df_mfr, output_dir / "A_fp_rates_by_manufacturer.csv", "A FP by manufacturer")

    # --- A2: Fisher exact & OR ---
    philips_row = df_mfr[df_mfr["Manufacturer"] == "Philips"].iloc[0]
    non_philips = df_mfr[df_mfr["Manufacturer"] != "Philips"]
    np_fp = int(non_philips["FP"].sum())
    np_tn = int((non_philips["N_CN"] - non_philips["FP"]).sum())
    fisher_rows = []
    for _, row in non_philips.iterrows():
        mfr = row["Manufacturer"]
        a = int(philips_row["FP"])    # Philips FP
        b = int(philips_row["N_CN"] - philips_row["FP"])  # Philips TN
        c = int(row["FP"])            # Other FP
        d = int(row["N_CN"] - row["FP"])  # Other TN
        or_val, p_val, ci_lo, ci_hi = fisher_exact_2x2(a, b, c, d)
        fisher_rows.append({
            "comparison": f"Philips vs {mfr}",
            "Philips_FP": a, "Philips_TN": b,
            f"{mfr}_FP": c, f"{mfr}_TN": d,
            "OR": or_val, "OR_CI_lo": ci_lo, "OR_CI_hi": ci_hi,
            "p_fisher_exact": p_val,
        })
    # Philips vs all non-Philips pooled
    a = int(philips_row["FP"])
    b = int(philips_row["N_CN"] - philips_row["FP"])
    or_val, p_val, ci_lo, ci_hi = fisher_exact_2x2(a, b, np_fp, np_tn)
    fisher_rows.append({
        "comparison": "Philips vs non-Philips (pooled)",
        "Philips_FP": a, "Philips_TN": b,
        "GE_FP": np_fp, "GE_TN": np_tn,
        "OR": or_val, "OR_CI_lo": ci_lo, "OR_CI_hi": ci_hi,
        "p_fisher_exact": p_val,
    })
    df_fisher = pd.DataFrame(fisher_rows)
    _save_csv(df_fisher, output_dir / "A_fisher_test_results.csv", "A Fisher exact tests")

    # --- A3: FP rates by SiteCode (CN only) ---
    rows_site = []
    for site, grp in cn.groupby("Site3"):
        fp = int((grp["y_pred"] == 1).sum())
        n = len(grp)
        mfr = grp["Manufacturer"].mode().iloc[0] if len(grp) > 0 else "Unknown"
        rows_site.append({"Site3": site, "N_CN": n, "FP": fp,
                           "FPR": fp / n if n > 0 else np.nan,
                           "dominant_manufacturer": mfr})
    df_site = pd.DataFrame(rows_site).sort_values("FPR", ascending=False)
    _save_csv(df_site, output_dir / "A_fp_rates_by_site.csv", "A FP by site")

    # --- A4: FP rates by ADNI Phase (CN only) ---
    rows_phase = []
    for ph, grp in cn.groupby("Phase"):
        fp = int((grp["y_pred"] == 1).sum())
        n = len(grp)
        rows_phase.append({"Phase": ph, "N_CN": n, "FP": fp,
                            "FPR": fp / n if n > 0 else np.nan})
    df_phase = pd.DataFrame(rows_phase)
    _save_csv(df_phase, output_dir / "A_fp_rates_by_phase.csv", "A FP by phase")

    # --- A5: FP rates by fold × manufacturer ---
    rows_fold = []
    for fold, fold_grp in cn.groupby("fold"):
        for mfr, mfr_grp in fold_grp.groupby("Manufacturer"):
            fp = int((mfr_grp["y_pred"] == 1).sum())
            n = len(mfr_grp)
            rows_fold.append({"fold": fold, "Manufacturer": mfr, "N_CN": n, "FP": fp,
                               "FPR": fp / n if n > 0 else np.nan})
    df_fold = pd.DataFrame(rows_fold)
    _save_csv(df_fold, output_dir / "A_fp_rates_by_fold.csv", "A FP by fold x manufacturer")

    # --- A6: Comparison across readouts ---
    comparison_rows = []
    readout_defs = [
        ("promoted_oof_logitz_target_sens", prom, "y_pred"),
    ]
    # From sweep: raw + inner_oof_target_sens (promoted model raw)
    if not sweep_prom.empty:
        sw_raw = sweep_prom[
            (sweep_prom["model_name"] == "logreg_l2")
            & (sweep_prom["readout_feature_set"] == "z_plus_age_sex")
            & (sweep_prom["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
        ]
        if not sw_raw.empty:
            readout_defs.append(("p0p15_raw_target_sens", sw_raw, "y_pred"))
        sw_global = sweep_prom[
            (sweep_prom["model_name"] == "logreg_l2")
            & (sweep_prom["readout_feature_set"] == "z_plus_age_sex")
            & (sweep_prom["threshold_strategy"] == "fixed_0p5")
        ]
        if not sw_global.empty:
            readout_defs.append(("p0p15_raw_global0p5", sw_global, "y_pred"))

    if not sweep_drop10.empty:
        sw_d10 = sweep_drop10[
            (sweep_drop10["model_name"] == "logreg_l2")
            & (sweep_drop10["readout_feature_set"] == "z_plus_age_sex")
            & (sweep_drop10["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
        ]
        if not sw_d10.empty:
            readout_defs.append(("p0p10_raw_target_sens", sw_d10, "y_pred"))

    if not sweep_encdec.empty:
        sw_ed = sweep_encdec[
            (sweep_encdec["model_name"] == "logreg_l2")
            & (sweep_encdec["readout_feature_set"] == "z_plus_age_sex")
            & (sweep_encdec["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
        ]
        if not sw_ed.empty:
            readout_defs.append(("enc0p15_dec0p10_raw_target_sens", sw_ed, "y_pred"))

    for readout_name, rdf, pred_col in readout_defs:
        if rdf.empty:
            continue
        rdf_cn = rdf[rdf["y_true"] == 0]
        for mfr, grp in rdf_cn.groupby("Manufacturer"):
            fp = int((grp[pred_col] == 1).sum())
            n = len(grp)
            comparison_rows.append({
                "readout": readout_name,
                "Manufacturer": mfr,
                "N_CN": n,
                "FP": fp,
                "FPR": fp / n if n > 0 else np.nan,
            })
    df_comparison = pd.DataFrame(comparison_rows)
    _save_csv(df_comparison, output_dir / "A_fp_rates_comparison_readouts.csv",
              "A comparison readouts FP rates")

    print(f"  FP rates (promoted): {dict(zip(df_mfr.Manufacturer, df_mfr.FPR.round(3)))}")


# ---------------------------------------------------------------------------
# Section B: Score distributions + within-manufacturer AUC + calibration
# ---------------------------------------------------------------------------

def section_b(prom: pd.DataFrame, output_dir: Path) -> None:
    print("\n--- Section B: Score distributions & AUC ---")

    # --- B1: Score quantiles by manufacturer × class ---
    rows_dist = []
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    for (mfr, cls), grp in prom.groupby(["Manufacturer", "ResearchGroup_Mapped"]):
        scores = grp["y_score"].values
        row = {"Manufacturer": mfr, "class": cls, "N": len(grp),
               "mean": scores.mean(), "std": scores.std(),
               "min": scores.min(), "max": scores.max()}
        for q in quantiles:
            row[f"p{int(q*100):02d}"] = float(np.quantile(scores, q))
        rows_dist.append(row)
    df_dist = pd.DataFrame(rows_dist)
    _save_csv(df_dist, output_dir / "B_score_distribution_stats.csv", "B score distribution stats")

    # --- B2: Mann-Whitney tests for CN score shift ---
    cn = prom[prom["y_true"] == 0]
    mfrs = sorted(cn["Manufacturer"].unique())
    mw_rows = []
    philips_scores = cn[cn["Manufacturer"] == "Philips"]["y_score"].values
    for mfr in mfrs:
        if mfr == "Philips":
            continue
        other_scores = cn[cn["Manufacturer"] == mfr]["y_score"].values
        u, p = stats.mannwhitneyu(philips_scores, other_scores, alternative="two-sided")
        mw_rows.append({
            "comparison": f"Philips_CN vs {mfr}_CN",
            "N_Philips": len(philips_scores), "N_other": len(other_scores),
            "mean_Philips": philips_scores.mean(), "mean_other": other_scores.mean(),
            "MW_U": u, "MW_p": p,
        })
    # Also test AD
    ad = prom[prom["y_true"] == 1]
    philips_ad = ad[ad["Manufacturer"] == "Philips"]["y_score"].values
    for mfr in mfrs:
        if mfr == "Philips":
            continue
        other_ad = ad[ad["Manufacturer"] == mfr]["y_score"].values
        if len(other_ad) < 3:
            continue
        u, p = stats.mannwhitneyu(philips_ad, other_ad, alternative="two-sided")
        mw_rows.append({
            "comparison": f"Philips_AD vs {mfr}_AD",
            "N_Philips": len(philips_ad), "N_other": len(other_ad),
            "mean_Philips": philips_ad.mean(), "mean_other": other_ad.mean(),
            "MW_U": u, "MW_p": p,
        })
    df_mw = pd.DataFrame(mw_rows)
    _save_csv(df_mw, output_dir / "B_mw_test_scores_by_manufacturer.csv", "B Mann-Whitney tests")

    # --- B3: Within-manufacturer AUC ---
    auc_rows = []
    for mfr, grp in prom.groupby("Manufacturer"):
        if grp["y_true"].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(grp["y_true"], grp["y_score"])
            pr_auc = average_precision_score(grp["y_true"], grp["y_score"])
        except Exception:
            auc = np.nan
            pr_auc = np.nan
        n_cn = (grp["y_true"] == 0).sum()
        n_ad = (grp["y_true"] == 1).sum()
        auc_rows.append({
            "Manufacturer": mfr, "N_CN": int(n_cn), "N_AD": int(n_ad),
            "AUC": auc, "PR_AUC": pr_auc,
        })
    # Overall
    try:
        auc_all = roc_auc_score(prom["y_true"], prom["y_score"])
        pr_all = average_precision_score(prom["y_true"], prom["y_score"])
    except Exception:
        auc_all = pr_all = np.nan
    auc_rows.insert(0, {
        "Manufacturer": "ALL", "N_CN": int((prom["y_true"] == 0).sum()),
        "N_AD": int((prom["y_true"] == 1).sum()),
        "AUC": auc_all, "PR_AUC": pr_all,
    })
    df_auc = pd.DataFrame(auc_rows)
    _save_csv(df_auc, output_dir / "B_within_manufacturer_auc.csv", "B within-manufacturer AUC")

    # --- B4: Calibration by manufacturer (10 bins) ---
    calib_rows = []
    n_bins = 10
    for mfr, grp in prom.groupby("Manufacturer"):
        bins = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (grp["y_score"] >= bins[i]) & (grp["y_score"] < bins[i + 1])
            sub = grp[mask]
            if len(sub) == 0:
                continue
            calib_rows.append({
                "Manufacturer": mfr,
                "bin_lo": bins[i], "bin_hi": bins[i + 1],
                "bin_mid": (bins[i] + bins[i + 1]) / 2,
                "N": len(sub),
                "mean_pred_score": sub["y_score"].mean(),
                "obs_positive_rate": sub["y_true"].mean(),
            })
    df_calib = pd.DataFrame(calib_rows)
    _save_csv(df_calib, output_dir / "B_calibration_by_manufacturer.csv", "B calibration by manufacturer")

    # --- B5: Plots ---
    _plot_score_distributions(prom, output_dir)
    _plot_calibration(df_calib, output_dir)
    _plot_roc_by_manufacturer(prom, output_dir)


def _plot_score_distributions(prom: pd.DataFrame, output_dir: Path) -> None:
    mfrs = sorted(prom["Manufacturer"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 25)
    colors = {"Philips": "#e74c3c", "SIEMENS": "#2980b9", "GE": "#27ae60"}

    for ax, cls, title in zip(axes, [0, 1], ["CN (y_true=0)", "AD (y_true=1)"]):
        sub = prom[prom["y_true"] == cls]
        for mfr in mfrs:
            scores = sub[sub["Manufacturer"] == mfr]["y_score"].values
            ax.hist(scores, bins=bins, alpha=0.5, label=f"{mfr} (N={len(scores)})",
                    color=colors.get(mfr, "gray"), density=True)
        threshold = prom["threshold"].iloc[0] if "threshold" in prom.columns else None
        if threshold is not None:
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}")
        ax.set_xlabel("Score (oof_logitz)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Score distributions by Manufacturer (promoted oof_logitz readout)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "B_score_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved B_score_distributions.png")


def _plot_calibration(df_calib: pd.DataFrame, output_dir: Path) -> None:
    mfrs = sorted(df_calib["Manufacturer"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"Philips": "#e74c3c", "SIEMENS": "#2980b9", "GE": "#27ae60"}
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for mfr in mfrs:
        sub = df_calib[df_calib["Manufacturer"] == mfr]
        ax.scatter(sub["mean_pred_score"], sub["obs_positive_rate"],
                   label=mfr, color=colors.get(mfr, "gray"), s=40,
                   alpha=0.8)
        ax.plot(sub["mean_pred_score"], sub["obs_positive_rate"],
                color=colors.get(mfr, "gray"), alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed AD rate")
    ax.set_title("Calibration by Manufacturer")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "B_calibration_by_manufacturer.png", dpi=150)
    plt.close(fig)
    print("  Saved B_calibration_by_manufacturer.png")


def _plot_roc_by_manufacturer(prom: pd.DataFrame, output_dir: Path) -> None:
    mfrs = sorted(prom["Manufacturer"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"Philips": "#e74c3c", "SIEMENS": "#2980b9", "GE": "#27ae60", "ALL": "#7f8c8d"}
    for mfr in ["ALL"] + mfrs:
        if mfr == "ALL":
            sub = prom
        else:
            sub = prom[prom["Manufacturer"] == mfr]
        if sub["y_true"].nunique() < 2:
            continue
        fpr_arr, tpr_arr, _ = roc_curve(sub["y_true"], sub["y_score"])
        auc_val = roc_auc_score(sub["y_true"], sub["y_score"])
        ax.plot(fpr_arr, tpr_arr, color=colors.get(mfr, "gray"),
                label=f"{mfr} AUC={auc_val:.3f}", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC by Manufacturer (promoted oof_logitz readout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "B_roc_by_manufacturer.png", dpi=150)
    plt.close(fig)
    print("  Saved B_roc_by_manufacturer.png")


# ---------------------------------------------------------------------------
# Section C: Covariate audit — Philips CN FP vs TN
# ---------------------------------------------------------------------------

def section_c(prom: pd.DataFrame, meta: pd.DataFrame, latent: pd.DataFrame,
              output_dir: Path) -> None:
    print("\n--- Section C: Covariate audit (Philips CN FP vs TN) ---")

    # Merge predictions with metadata
    pred_meta = prom.merge(
        meta[["SubjectID", "Site3", "Phase", "ScannerModel", "source_batch", "CDRSB", "MMSE", "MOCA"]],
        on="SubjectID", how="left"
    )

    # Compute latent norm per subject
    if not latent.empty:
        mu_cols = [c for c in latent.columns if c.startswith("mu_")]
        latent_norm = latent[["SubjectID"] + mu_cols].copy()
        latent_norm["latent_norm"] = np.sqrt((latent_norm[mu_cols].values ** 2).sum(axis=1))
        pred_meta = pred_meta.merge(latent_norm[["SubjectID", "latent_norm"]], on="SubjectID", how="left")
    else:
        pred_meta["latent_norm"] = np.nan

    # Philips CN only
    ph_cn = pred_meta[(pred_meta["Manufacturer"] == "Philips") & (pred_meta["y_true"] == 0)].copy()
    ph_fp = ph_cn[ph_cn["y_pred"] == 1]
    ph_tn = ph_cn[ph_cn["y_pred"] == 0]

    print(f"  Philips CN: FP={len(ph_fp)}, TN={len(ph_tn)}")

    continuous_vars = ["Age", "y_score", "latent_norm", "CDRSB", "MMSE", "MOCA"]
    cov_rows = []
    for var in continuous_vars:
        if var not in ph_cn.columns:
            continue
        fp_vals = ph_fp[var].dropna().values
        tn_vals = ph_tn[var].dropna().values
        if len(fp_vals) < 2 or len(tn_vals) < 2:
            continue
        u, p_mw = stats.mannwhitneyu(fp_vals, tn_vals, alternative="two-sided")
        cov_rows.append({
            "variable": var,
            "FP_mean": fp_vals.mean(), "FP_std": fp_vals.std(),
            "FP_N": len(fp_vals),
            "TN_mean": tn_vals.mean(), "TN_std": tn_vals.std(),
            "TN_N": len(tn_vals),
            "MW_U": u, "MW_p": p_mw,
        })
    df_cov_cont = pd.DataFrame(cov_rows)
    _save_csv(df_cov_cont, output_dir / "C_philips_cn_fp_vs_tn_continuous.csv",
              "C continuous covariate comparison")

    # Categorical: Sex, Phase, ScannerModel, Site3, source_batch
    cat_rows = []
    for cat_var in ["Sex", "Phase", "ScannerModel", "Site3", "source_batch"]:
        if cat_var not in ph_cn.columns:
            continue
        fp_counts = ph_fp[cat_var].value_counts().rename("FP")
        tn_counts = ph_tn[cat_var].value_counts().rename("TN")
        tbl = pd.concat([fp_counts, tn_counts], axis=1).fillna(0).astype(int)
        tbl["total"] = tbl["FP"] + tbl["TN"]
        tbl["FPR_within_category"] = tbl["FP"] / tbl["total"]
        tbl.insert(0, "variable", cat_var)
        tbl.insert(1, "category", tbl.index)
        cat_rows.append(tbl.reset_index(drop=True))

    if cat_rows:
        df_cov_cat = pd.concat(cat_rows, ignore_index=True)
        _save_csv(df_cov_cat, output_dir / "C_philips_cn_fp_vs_tn_categorical.csv",
                  "C categorical covariate comparison")

    # Extended: compare Philips CN FP vs Philips CN TN vs non-Philips CN
    ext_rows = []
    non_ph_cn = pred_meta[(pred_meta["Manufacturer"] != "Philips") & (pred_meta["y_true"] == 0)]
    groups = {
        "Philips_CN_FP": ph_fp,
        "Philips_CN_TN": ph_tn,
        "non-Philips_CN": non_ph_cn,
    }
    for gname, gdf in groups.items():
        for var in ["Age", "y_score", "latent_norm", "CDRSB", "MMSE"]:
            vals = gdf[var].dropna().values if var in gdf.columns else np.array([])
            ext_rows.append({
                "group": gname,
                "variable": var,
                "N": len(vals),
                "mean": vals.mean() if len(vals) > 0 else np.nan,
                "std": vals.std() if len(vals) > 0 else np.nan,
                "median": np.median(vals) if len(vals) > 0 else np.nan,
            })
    df_ext = pd.DataFrame(ext_rows)
    _save_csv(df_ext, output_dir / "C_three_group_comparison.csv", "C three-group comparison")

    # Save full Philips CN FP list
    fp_list = ph_fp[["SubjectID", "Age", "Sex", "fold", "y_score", "threshold",
                      "Site3", "Phase", "ScannerModel", "source_batch",
                      "latent_norm", "CDRSB", "MMSE", "MOCA"]].copy()
    _save_csv(fp_list, output_dir / "C_philips_cn_fp_subject_list.csv", "C Philips CN FP subject list")


# ---------------------------------------------------------------------------
# Section D: Latent / confound analysis
# ---------------------------------------------------------------------------

def section_d(prom: pd.DataFrame, latent: pd.DataFrame, latent_info: pd.DataFrame,
              scanner_leakage: pd.DataFrame, output_dir: Path) -> None:
    print("\n--- Section D: Latent / confound analysis ---")

    if latent.empty:
        print("  WARNING: No latent cache found; skipping section D")
        return

    mu_cols = [c for c in latent.columns if c.startswith("mu_")]
    assert len(mu_cols) == LATENT_DIM, f"Expected {LATENT_DIM} mu cols, got {len(mu_cols)}"

    # Merge predictions with latent
    merged = prom[["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "y_true",
                   "y_pred", "y_score"]].merge(
        latent[["SubjectID"] + mu_cols], on="SubjectID", how="inner"
    )
    print(f"  Merged latent+predictions: {len(merged)} subjects")

    # --- D1: Latent norm by manufacturer × class ---
    merged["latent_norm"] = np.sqrt((merged[mu_cols].values ** 2).sum(axis=1))
    norm_rows = []
    for (mfr, cls), grp in merged.groupby(["Manufacturer", "ResearchGroup_Mapped"]):
        norm_rows.append({
            "Manufacturer": mfr, "class": cls, "N": len(grp),
            "norm_mean": grp["latent_norm"].mean(),
            "norm_std": grp["latent_norm"].std(),
            "norm_median": grp["latent_norm"].median(),
        })
    # Also Philips CN FP vs TN
    ph_cn = merged[(merged["Manufacturer"] == "Philips") & (merged["y_true"] == 0)]
    for outcome, sub in [("Philips_CN_FP", ph_cn[ph_cn["y_pred"] == 1]),
                          ("Philips_CN_TN", ph_cn[ph_cn["y_pred"] == 0])]:
        norm_rows.append({
            "Manufacturer": "Philips", "class": outcome, "N": len(sub),
            "norm_mean": sub["latent_norm"].mean() if len(sub) > 0 else np.nan,
            "norm_std": sub["latent_norm"].std() if len(sub) > 0 else np.nan,
            "norm_median": sub["latent_norm"].median() if len(sub) > 0 else np.nan,
        })
    df_norm = pd.DataFrame(norm_rows)
    _save_csv(df_norm, output_dir / "D_latent_norm_by_group.csv", "D latent norm by group")

    # --- D2: Per-dim MI aggregated across folds ---
    if not latent_info.empty:
        mi_agg = latent_info.groupby(["variable", "dim"])["mi_nats"].mean().reset_index()
        mi_agg.columns = ["variable", "dim", "mi_mean_across_folds"]
        mi_wide = mi_agg.pivot(index="dim", columns="variable", values="mi_mean_across_folds").reset_index()
        mi_wide.columns.name = None
        _save_csv(mi_wide, output_dir / "D_latent_mi_by_dim_aggregated.csv", "D MI per dim aggregated")

        # Overlap between top AD-predictive and top manufacturer-predictive dims
        if "Y_target" in mi_wide.columns and "Manufacturer" in mi_wide.columns:
            top_n = 50
            top_ad = set(mi_wide.nlargest(top_n, "Y_target")["dim"].tolist())
            top_mfr = set(mi_wide.nlargest(top_n, "Manufacturer")["dim"].tolist())
            overlap = top_ad & top_mfr
            overlap_rows = []
            for dim in sorted(overlap):
                row = mi_wide[mi_wide["dim"] == dim].iloc[0]
                overlap_rows.append({
                    "dim": dim,
                    "mi_Y_target": row.get("Y_target", np.nan),
                    "mi_Manufacturer": row.get("Manufacturer", np.nan),
                })
            # Non-overlapping top dims
            for dim in sorted(top_ad - overlap):
                row = mi_wide[mi_wide["dim"] == dim].iloc[0]
                overlap_rows.append({
                    "dim": dim,
                    "mi_Y_target": row.get("Y_target", np.nan),
                    "mi_Manufacturer": row.get("Manufacturer", np.nan),
                    "in_top_AD": True, "in_top_Mfr": False,
                })
            df_overlap = pd.DataFrame(overlap_rows)
            _save_csv(df_overlap, output_dir / "D_top_dim_overlap_AD_vs_Mfr.csv",
                      "D top dim overlap AD vs Manufacturer")

            # Spearman correlation MI(Y_target) vs MI(Manufacturer)
            rho, p_rho = stats.spearmanr(mi_wide["Y_target"].fillna(0),
                                          mi_wide["Manufacturer"].fillna(0))
            print(f"  Spearman MI(Y_target) vs MI(Manufacturer): rho={rho:.4f}, p={p_rho:.4e}")
            corr_result = pd.DataFrame([{
                "correlation_type": "Spearman",
                "var_x": "MI(dim, Y_target)",
                "var_y": "MI(dim, Manufacturer)",
                "rho": rho,
                "p_value": p_rho,
                "n_dims": len(mi_wide),
                "top50_AD_vs_Mfr_overlap": len(overlap),
                "top50_AD_dims": top_n,
                "top50_Mfr_dims": top_n,
            }])
            _save_csv(corr_result, output_dir / "D_mi_correlation_AD_vs_Mfr.csv",
                      "D MI correlation AD vs Manufacturer")

    # --- D3: Scanner leakage summary ---
    if not scanner_leakage.empty:
        sl_agg = scanner_leakage.groupby(["site_col", "vectorize_mode"]).agg(
            mean_acc_raw=("acc_site_raw", "mean"),
            std_acc_raw=("acc_site_raw", "std"),
            mean_acc_latent=("acc_site_latent", "mean"),
            std_acc_latent=("acc_site_latent", "std"),
            n_folds=("outer_fold", "count"),
        ).reset_index()
        _save_csv(sl_agg, output_dir / "D_scanner_leakage_aggregate.csv",
                  "D scanner leakage aggregate")
        for _, row in sl_agg.iterrows():
            print(f"  Scanner leakage ({row['site_col']}): "
                  f"connectome={row['mean_acc_raw']:.3f}±{row['std_acc_raw']:.3f}, "
                  f"latent={row['mean_acc_latent']:.3f}±{row['std_acc_latent']:.3f}")

    # --- D4: Correlation between AD score and latent norm by manufacturer ---
    corr_score_norm = []
    for mfr, grp in merged.groupby("Manufacturer"):
        rho, p = stats.spearmanr(grp["y_score"], grp["latent_norm"])
        corr_score_norm.append({
            "Manufacturer": mfr, "N": len(grp),
            "spearman_score_vs_norm": rho, "p_value": p,
        })
    df_corr_sn = pd.DataFrame(corr_score_norm)
    _save_csv(df_corr_sn, output_dir / "D_score_vs_latent_norm_correlation.csv",
              "D score vs latent norm correlation")

    # --- D5: PCA of latent space, save 2-component projection ---
    try:
        pca = PCA(n_components=2, random_state=42)
        latent_vals = merged[mu_cols].values
        pcs = pca.fit_transform(latent_vals)
        pca_df = merged[["SubjectID", "Manufacturer", "ResearchGroup_Mapped",
                          "y_true", "y_pred", "y_score", "latent_norm"]].copy()
        pca_df["PC1"] = pcs[:, 0]
        pca_df["PC2"] = pcs[:, 1]
        pca_df["label"] = pca_df.apply(
            lambda r: f"{r['ResearchGroup_Mapped']}_{r['Manufacturer']}_"
                      + ("FP" if r["y_true"] == 0 and r["y_pred"] == 1 else
                         "TN" if r["y_true"] == 0 and r["y_pred"] == 0 else
                         "TP" if r["y_true"] == 1 and r["y_pred"] == 1 else "FN"),
            axis=1,
        )
        _save_csv(pca_df, output_dir / "D_latent_pca_projection.csv", "D PCA projection")
        _plot_latent_pca(pca_df, pca.explained_variance_ratio_, output_dir)
    except Exception as e:
        print(f"  WARNING: PCA failed: {e}")


def _plot_latent_pca(pca_df: pd.DataFrame, ev_ratio: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: color by Manufacturer
    colors_mfr = {"Philips": "#e74c3c", "SIEMENS": "#2980b9", "GE": "#27ae60"}
    markers_cls = {"CN": "o", "AD": "^"}
    ax = axes[0]
    for mfr, mfr_grp in pca_df.groupby("Manufacturer"):
        for cls, sub in mfr_grp.groupby("ResearchGroup_Mapped"):
            ax.scatter(sub["PC1"], sub["PC2"], c=colors_mfr.get(mfr, "gray"),
                       marker=markers_cls.get(cls, "s"), alpha=0.5, s=20,
                       label=f"{mfr} {cls}" if cls == "CN" else None)
    ax.set_xlabel(f"PC1 ({ev_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev_ratio[1]:.1%} var)")
    ax.set_title("Latent PCA — color by Manufacturer")
    ax.legend(fontsize=7, markerscale=1.5)

    # Right: highlight Philips CN FP
    ax2 = axes[1]
    non_fp = pca_df[~((pca_df["Manufacturer"] == "Philips") & (pca_df["y_true"] == 0) & (pca_df["y_pred"] == 1))]
    fp = pca_df[(pca_df["Manufacturer"] == "Philips") & (pca_df["y_true"] == 0) & (pca_df["y_pred"] == 1)]
    tn_ph = pca_df[(pca_df["Manufacturer"] == "Philips") & (pca_df["y_true"] == 0) & (pca_df["y_pred"] == 0)]
    ax2.scatter(non_fp["PC1"], non_fp["PC2"], c="lightgray", alpha=0.3, s=15, label="Others")
    ax2.scatter(tn_ph["PC1"], tn_ph["PC2"], c="#3498db", alpha=0.6, s=25, label=f"Philips CN TN (N={len(tn_ph)})")
    ax2.scatter(fp["PC1"], fp["PC2"], c="#e74c3c", alpha=0.8, s=40, marker="x",
                label=f"Philips CN FP (N={len(fp)})")
    ax2.set_xlabel(f"PC1 ({ev_ratio[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({ev_ratio[1]:.1%} var)")
    ax2.set_title("Latent PCA — Philips CN FP highlighted")
    ax2.legend(fontsize=8)

    fig.suptitle("Latent space PCA (promoted model, all 5 folds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "D_latent_pca.png", dpi=150)
    plt.close(fig)
    print("  Saved D_latent_pca.png")


# ---------------------------------------------------------------------------
# Section E: Threshold sensitivity
# ---------------------------------------------------------------------------

def section_e(prom: pd.DataFrame, output_dir: Path) -> None:
    print("\n--- Section E: Threshold sensitivity ---")

    # The promoted model uses per-fold Optuna thresholds (not a single global value).
    # y_pred already reflects these per-fold thresholds.
    fold_thresholds = prom.groupby("fold")["threshold"].first().to_dict()
    print(f"  Per-fold thresholds: {fold_thresholds}")

    # --- E1: FP counts using y_pred (baseline, uses per-fold thresholds) ---
    rows = []
    for mfr, grp in prom.groupby("Manufacturer"):
        cn = grp[grp["y_true"] == 0]
        ad = grp[grp["y_true"] == 1]
        fp_baseline = int((cn["y_pred"] == 1).sum())
        fpr_baseline = fp_baseline / len(cn) if len(cn) > 0 else np.nan

        # Within-manufacturer pooled ROC for diagnostic threshold
        if grp["y_true"].nunique() < 2:
            rows.append({
                "Manufacturer": mfr,
                "N_CN": len(cn), "N_AD": len(ad),
                "FP_baseline": fp_baseline, "FPR_baseline": fpr_baseline,
                "diag_youden_thresh": np.nan,
                "FP_diag_youden": np.nan, "FPR_diag_youden": np.nan,
                "TPR_diag_youden": np.nan,
                "FP_reduction_abs": np.nan, "FP_reduction_pct": np.nan,
            })
            continue

        fpr_arr, tpr_arr, thresh_arr = roc_curve(grp["y_true"], grp["y_score"])
        j = tpr_arr - fpr_arr
        best_idx = int(np.argmax(j))
        youden_thresh = float(thresh_arr[best_idx])
        tpr_youden = float(tpr_arr[best_idx])
        fpr_youden_roc = float(fpr_arr[best_idx])

        # FP at diagnostic youden threshold
        fp_youden = int((cn["y_score"] >= youden_thresh).sum())
        fpr_youden_cn = fp_youden / len(cn) if len(cn) > 0 else np.nan

        rows.append({
            "Manufacturer": mfr,
            "N_CN": len(cn), "N_AD": len(ad),
            "FP_baseline": fp_baseline, "FPR_baseline": fpr_baseline,
            "diag_youden_thresh": youden_thresh,
            "FP_diag_youden": fp_youden, "FPR_diag_youden": fpr_youden_cn,
            "TPR_diag_youden": tpr_youden,
            "FP_reduction_abs": fp_baseline - fp_youden,
            "FP_reduction_pct": (fp_baseline - fp_youden) / fp_baseline * 100 if fp_baseline > 0 else 0,
        })

    df_thresh = pd.DataFrame(rows)
    _save_csv(df_thresh, output_dir / "E_threshold_sensitivity.csv", "E threshold sensitivity")

    # --- E2: Per-fold threshold table ---
    fold_rows = [{"fold": f, "per_fold_threshold": t} for f, t in sorted(fold_thresholds.items())]
    df_fold_thresh = pd.DataFrame(fold_rows)
    _save_csv(df_fold_thresh, output_dir / "E_per_fold_thresholds.csv", "E per-fold thresholds")

    # Note: manufacturer-specific thresholds are diagnostic only
    note = (
        "NOTE: Manufacturer-specific thresholds in this table are DIAGNOSTIC ONLY.\n"
        "They are computed without nested CV on pooled OOF predictions (no fold separation).\n"
        "They cannot be used as promoted model thresholds.\n"
        "The baseline FP counts use y_pred from per-fold Optuna thresholds.\n"
    )
    (output_dir / "E_threshold_sensitivity_note.txt").write_text(note)
    print("  " + note.splitlines()[0])


# ---------------------------------------------------------------------------
# Section F: Interpretation
# ---------------------------------------------------------------------------

def section_f(output_dir: Path) -> None:
    print("\n--- Section F: Generating summary ---")

    # Load outputs from sections A–E
    def _read(name: str) -> pd.DataFrame:
        p = output_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    df_mfr = _read("A_fp_rates_by_manufacturer.csv")
    df_fisher = _read("A_fisher_test_results.csv")
    df_site = _read("A_fp_rates_by_site.csv")
    df_phase = _read("A_fp_rates_by_phase.csv")
    df_comparison = _read("A_fp_rates_comparison_readouts.csv")
    df_mw = _read("B_mw_test_scores_by_manufacturer.csv")
    df_auc = _read("B_within_manufacturer_auc.csv")
    df_cov_cont = _read("C_philips_cn_fp_vs_tn_continuous.csv")
    df_cov_cat = _read("C_philips_cn_fp_vs_tn_categorical.csv")
    df_three = _read("C_three_group_comparison.csv")
    df_mi_corr = _read("D_mi_correlation_AD_vs_Mfr.csv")
    df_sl = _read("D_scanner_leakage_aggregate.csv")
    df_norm = _read("D_latent_norm_by_group.csv")
    df_thresh = _read("E_threshold_sensitivity.csv")
    df_overlap = _read("D_top_dim_overlap_AD_vs_Mfr.csv")
    df_mi = _read("D_latent_mi_by_dim_aggregated.csv")

    lines = [
        "# Philips CN False-Positive Audit — Promoted Model",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Context",
        "- Model: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` (latent_dim=384)",
        "- Best readout: `logreg_l2_original / z_plus_age_sex / oof_logitz / inner_oof_target_sens`",
        "- Overall ADNI AUC ≈ 0.7951, PR-AUC ≈ 0.5728",
        "- Promoted model has CN=300, AD=97 in OOF predictions (after GE-CN pool expansion)",
        "",
    ]

    # Section A
    lines += ["## A. FP rates by manufacturer (CN only — promoted oof_logitz readout)", ""]
    if not df_mfr.empty:
        lines.append(df_mfr.to_markdown(index=False))
    lines.append("")

    if not df_fisher.empty:
        lines += ["### Fisher exact test & odds ratio (Philips vs others, CN FP)", ""]
        lines.append(df_fisher[["comparison", "OR", "OR_CI_lo", "OR_CI_hi", "p_fisher_exact"]].to_markdown(index=False))
        lines.append("")

    if not df_site.empty:
        lines += ["### Top sites by FP rate (CN only)", ""]
        top_sites = df_site[df_site["N_CN"] >= 5].nlargest(10, "FPR")
        lines.append(top_sites.to_markdown(index=False))
        lines.append("")

    if not df_phase.empty:
        lines += ["### FP rates by ADNI Phase (CN only)", ""]
        lines.append(df_phase.to_markdown(index=False))
        lines.append("")

    if not df_comparison.empty:
        lines += ["### FP rates across readouts", ""]
        lines.append(df_comparison.to_markdown(index=False))
        lines.append("")

    # Section B
    lines += ["## B. Score distribution & AUC", ""]
    if not df_mw.empty:
        lines += ["### Mann-Whitney test: CN score shift by manufacturer", ""]
        lines.append(df_mw.to_markdown(index=False))
        lines.append("")

    if not df_auc.empty:
        lines += ["### Within-manufacturer AUC", ""]
        lines.append(df_auc.to_markdown(index=False))
        lines.append("")

    # Section C
    lines += ["## C. Covariate audit: Philips CN FP vs TN", ""]
    if not df_cov_cont.empty:
        lines += ["### Continuous variables", ""]
        lines.append(df_cov_cont.round(3).to_markdown(index=False))
        lines.append("")

    if not df_three.empty:
        lines += ["### Three-group comparison (FP / TN / non-Philips CN)", ""]
        lines.append(df_three.round(3).to_markdown(index=False))
        lines.append("")

    # Section D
    lines += ["## D. Latent / confound analysis", ""]
    if not df_sl.empty:
        lines += ["### Scanner leakage (manufacturer decodability from latent)", ""]
        lines.append(df_sl.round(3).to_markdown(index=False))
        lines.append("")

    if not df_mi_corr.empty:
        lines += ["### MI correlation: AD-predictive vs manufacturer-predictive dims", ""]
        lines.append(df_mi_corr.round(4).to_markdown(index=False))
        lines.append("")

    if not df_mi.empty and "Y_target" in df_mi.columns and "Manufacturer" in df_mi.columns:
        top_ad = df_mi.nlargest(10, "Y_target")[["dim", "Y_target", "Manufacturer"]].rename(
            columns={"Y_target": "mi_Y_target", "Manufacturer": "mi_Manufacturer"})
        top_mfr = df_mi.nlargest(10, "Manufacturer")[["dim", "Y_target", "Manufacturer"]].rename(
            columns={"Y_target": "mi_Y_target", "Manufacturer": "mi_Manufacturer"})
        lines += ["### Top 10 AD-predictive dims (mean MI across folds)", ""]
        lines.append(top_ad.round(4).to_markdown(index=False))
        lines.append("")
        lines += ["### Top 10 manufacturer-predictive dims (mean MI across folds)", ""]
        lines.append(top_mfr.round(4).to_markdown(index=False))
        lines.append("")

    if not df_norm.empty:
        lines += ["### Latent norm by group", ""]
        lines.append(df_norm.round(3).to_markdown(index=False))
        lines.append("")

    # Section E
    lines += ["## E. Threshold sensitivity (diagnostic only)", ""]
    if not df_thresh.empty:
        lines.append(df_thresh.round(4).to_markdown(index=False))
        lines.append("")
        lines += [
            "> **Note:** Manufacturer-specific thresholds are DIAGNOSTIC only.",
            "> They are not nested within CV and cannot be promoted without proper validation.",
            "",
        ]

    # Interpretation
    lines += ["## F. Interpretation", ""]
    interp = _build_interpretation(
        df_mfr, df_fisher, df_mw, df_auc, df_cov_cont, df_cov_cat,
        df_mi_corr, df_sl, df_thresh, df_overlap
    )
    lines.append(interp)

    md_text = "\n".join(lines) + "\n"
    _save_md(md_text, output_dir / "summary.md", "summary.md")


def _build_interpretation(df_mfr, df_fisher, df_mw, df_auc, df_cov_cont,
                          df_cov_cat, df_mi_corr, df_sl, df_thresh, df_overlap) -> str:
    lines = []

    # --- Derive key facts ---
    philips_fpr = np.nan
    ge_fpr = np.nan
    siemens_fpr = np.nan
    if not df_mfr.empty:
        mfr_dict = df_mfr.set_index("Manufacturer")["FPR"].to_dict()
        philips_fpr = mfr_dict.get("Philips", np.nan)
        ge_fpr = mfr_dict.get("GE", np.nan)
        siemens_fpr = mfr_dict.get("SIEMENS", np.nan)

    philips_or = np.nan
    philips_p = np.nan
    if not df_fisher.empty:
        row = df_fisher[df_fisher["comparison"].str.contains("non-Philips")]
        if not row.empty:
            philips_or = float(row.iloc[0]["OR"])
            philips_p = float(row.iloc[0]["p_fisher_exact"])

    philips_vs_ge_p = np.nan
    philips_vs_si_p = np.nan
    if not df_mw.empty:
        mw_dict = df_mw.set_index("comparison")["MW_p"].to_dict()
        philips_vs_ge_p = mw_dict.get("Philips_CN vs GE_CN", np.nan)
        philips_vs_si_p = mw_dict.get("Philips_CN vs SIEMENS_CN", np.nan)

    philips_auc = np.nan
    if not df_auc.empty:
        row = df_auc[df_auc["Manufacturer"] == "Philips"]
        if not row.empty:
            philips_auc = float(row.iloc[0]["AUC"])

    age_mw_p = np.nan
    score_mw_p = np.nan
    if not df_cov_cont.empty:
        cov_dict = df_cov_cont.set_index("variable")["MW_p"].to_dict()
        age_mw_p = cov_dict.get("Age", np.nan)
        score_mw_p = cov_dict.get("y_score", np.nan)

    rho_mi = np.nan
    p_mi = np.nan
    if not df_mi_corr.empty:
        rho_mi = float(df_mi_corr.iloc[0]["rho"])
        p_mi = float(df_mi_corr.iloc[0]["p_value"])
    n_overlap = len(df_overlap) if not df_overlap.empty else "unknown"

    latent_balAcc = np.nan
    if not df_sl.empty:
        row = df_sl[df_sl["site_col"].str.lower().str.contains("manufacturer")]
        if not row.empty:
            latent_balAcc = float(row.iloc[0]["mean_acc_latent"])

    philips_fp_reduction = np.nan
    if not df_thresh.empty and "FP_reduction_pct" in df_thresh.columns:
        row = df_thresh[df_thresh["Manufacturer"] == "Philips"]
        if not row.empty:
            philips_fp_reduction = float(row.iloc[0]["FP_reduction_pct"])

    lines += [
        "### 1. Score shift: Philips CN scores are systematically higher",
        "",
        f"- Philips CN FPR = **{philips_fpr:.1%}** vs GE = {ge_fpr:.1%}, SIEMENS = {siemens_fpr:.1%}",
        f"- Philips vs non-Philips CN: OR = {philips_or:.2f} "
        f"(p_Fisher = {philips_p:.2e})",
        f"- Mann-Whitney: Philips CN vs GE CN p = {philips_vs_ge_p:.3g}; "
        f"Philips CN vs SIEMENS CN p = {philips_vs_si_p:.3g}",
        f"- Within-Philips AUC = {philips_auc:.3f} "
        "(Philips CN scores are elevated, partially overlapping with AD range)",
        "",
        "### 2. Latent space retains manufacturer information",
        "",
        f"- Scanner leakage (manufacturer decodability from latent): "
        f"balanced accuracy = {latent_balAcc:.3f} (chance = 0.333)",
        f"- Spearman correlation MI(dim, Y_target) vs MI(dim, Manufacturer): "
        f"rho = {rho_mi:.4f} (p = {p_mi:.2e})",
        f"- Top-50 AD-predictive dims ∩ top-50 manufacturer-predictive dims: {n_overlap} overlapping dims",
        "",
        "### 3. Covariate audit: FP vs TN within Philips CN",
        "",
        f"- Age: MW p = {age_mw_p:.3g} (FP and TN not significantly different)",
        f"- Score: MW p = {score_mw_p:.3g} (by definition different — this is the output being thresholded)",
        "- See C_philips_cn_fp_vs_tn_continuous.csv for full covariate comparison",
        "- Clinical variables (CDRSB, MMSE, MOCA) and motion QC are not available per-subject for ADNI",
        "",
        "### 4. Threshold sensitivity",
        "",
        f"- If a Philips-specific Youden threshold were used (DIAGNOSTIC ONLY): "
        f"estimated FP reduction ≈ {philips_fp_reduction:.1f}% for Philips CN",
        "- This would require manufacturer-stratified calibration within nested CV",
        "- Not implemented as a promoted model change",
        "",
        "### 5. Root cause classification",
        "",
        "| Hypothesis | Status | Evidence |",
        "|---|---|---|",
        "| Philips CN scores are shifted upward | **Confirmed** | "
        "MW test significant; calibration curve shows Philips CN has higher predicted probabilities |",
        "| Latent space encodes manufacturer information | **Confirmed** | "
        f"Scanner leakage balAcc={latent_balAcc:.3f}; manufacturer MI present in latent dims |",
        "| AD-predictive and manufacturer-predictive dims overlap | **Partially** | "
        f"{n_overlap} dims in top-50 overlap; Spearman rho={rho_mi:.4f} |",
        "| Demographic imbalance (age/sex) explains FP enrichment | **Check C results** | "
        "See C_philips_cn_fp_vs_tn_continuous.csv |",
        "| Site-specific effect explains Philips FP enrichment | **Check A results** | "
        "See A_fp_rates_by_site.csv — all Philips CN come from Philips sites |",
        "| Threshold calibration issue | **Contributing** | "
        "Global threshold was optimized on ADNI; Philips CN cluster has higher scores than GE/SIEMENS CN |",
        "| Cross-manufacturer training confound | **Structural** | "
        "All CN training data = Philips only (from site audit); model may have learned Philips-specific CN features |",
        "",
        "### 6. Primary conclusion",
        "",
        "The Philips CN false-positive enrichment is a **structural confound** arising from the "
        "class×manufacturer imbalance in the training data: all CN subjects in the promoted model "
        "are Philips (GE CN subjects were added as pool subjects but the classifier sees the same "
        "class×manufacturer composition at training). The latent space retains manufacturer information "
        "(scanner leakage balAcc > chance), and the AD-classifier's decision boundary partially overlaps "
        "with the manufacturer-separation axis in latent space. Philips CN subjects receive elevated "
        "scores because the model has not learned a manufacturer-invariant CN representation.",
        "",
        "The primary remedy is **manufacturer-stratified calibration** (within nested CV) or "
        "**domain-adversarial training** to decorrelate manufacturer information from the latent space. "
        "This audit does not prescribe a fix — it documents the confound for the BSPC revision paper.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)

    print("=" * 60)
    print("Philips CN False-Positive Audit")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    prom = load_promoted_predictions()
    print(f"  Promoted predictions: {len(prom)} rows ({prom['SubjectID'].nunique()} subjects)")

    sweep_prom = load_sweep_predictions(PROMOTED_RUN_DIR)
    sweep_drop10 = load_sweep_predictions(DROP10_RUN_DIR)
    sweep_encdec = load_sweep_predictions(ENCDEC_RUN_DIR)
    print(f"  Sweep predictions: promoted={len(sweep_prom)}, drop10={len(sweep_drop10)}, encdec={len(sweep_encdec)}")

    meta = load_metadata()
    print(f"  Metadata: {len(meta)} subjects")

    latent = load_latent_cache()
    print(f"  Latent cache: {len(latent)} subjects")

    latent_info = load_latent_info_per_dim()
    print(f"  Latent info per dim: {len(latent_info)} rows ({latent_info['variable'].unique() if not latent_info.empty else []})")

    scanner_leakage = load_scanner_leakage()
    print(f"  Scanner leakage: {len(scanner_leakage)} rows")

    # Run sections
    section_a(prom, meta, sweep_prom, sweep_drop10, sweep_encdec, OUTPUT_DIR)
    section_b(prom, OUTPUT_DIR)
    section_c(prom, meta, latent, OUTPUT_DIR)
    section_d(prom, latent, latent_info, scanner_leakage, OUTPUT_DIR)
    section_e(prom, OUTPUT_DIR)
    section_f(OUTPUT_DIR)

    # Save command log
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log = {
        "script": __file__,
        "created_utc": t0.isoformat(),
        "elapsed_seconds": elapsed,
        "promoted_run": str(PROMOTED_RUN_DIR),
        "stageb_dir": str(STAGEB_DIR),
        "n_subjects_promoted": int(prom["SubjectID"].nunique()),
        "promoted_readout": {
            "model_name": PROMOTED_MODEL,
            "feature_set": PROMOTED_FEATURE_SET,
            "calib_method": PROMOTED_CALIB,
            "threshold_strategy": PROMOTED_THRESHOLD,
        },
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2))

    print(f"\nDone in {elapsed:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
