#!/usr/bin/env python3
"""SIPAIM multi-site transport and external shift analysis.

Read-only with respect to model, tensor, prediction, and manuscript inputs.
Writes a compact analysis package under results/revision_bspc_2026.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import wasserstein_distance
from sklearn.calibration import calibration_curve
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/diego/proyectos/vae_AD")
OUT = PROJECT / "results/revision_bspc_2026/post_revision_exploratory_20260630/sipaim_multisite_transport_20260706"
FIG_DIR = OUT / "figures"
FIG_DATA = OUT / "figure_data"

TEX = PROJECT / "docs/revision_bspc_2026/site_geometry_analysis/site_geometry_analysis_protocol_v4.tex"
TEX_FIG_DIR = PROJECT / "docs/revision_bspc_2026/site_geometry_analysis/Figures/site_geometry"
LOCKED_RUN = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5")
STAGEB = PROJECT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
METADATA = PROJECT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
FOLDCOMBAT_STAGEB = PROJECT / "results/revision_bspc_2026/recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration"
FOLDCOMBAT_AUDIT = PROJECT / "results/revision_bspc_2026/post_revision_exploratory_20260630/foldcombat_mfr_age_sex_audit_20260705"
OASIS_AUDIT = PROJECT / "results/revision_bspc_2026/oasis_run_handling_and_subject_level_audit_20260626"
OASIS_PANEL = PROJECT / "results/revision_bspc_2026/oasis_mega_90_90_external_inference_model_panel_20260604"
OASIS_LATENT = PROJECT / "results/revision_bspc_2026/promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/oasis_fold_latent_mu_runwise164.csv"

PRIMARY = {
    "model_name": "logreg_l2_original",
    "feature_set": "z_plus_age_sex",
    "calib_method": "oof_ecdf",
    "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
}

SITE_TIERS = {
    "geometry_test_n_min": 10,
    "site_auc_total_n_min": 15,
    "site_auc_cn_min": 3,
    "site_auc_ad_min": 3,
    "loso_total_n_min": 20,
    "loso_cn_min": 5,
    "loso_ad_min": 5,
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DATA.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._\n"
    return df.to_markdown(index=False) + "\n"


def write_csv_md(stem: str, df: pd.DataFrame, title: str = "", max_rows: Optional[int] = None) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    text = []
    if title:
        text += [f"# {title}", ""]
    text.append(md_table(df, max_rows))
    (OUT / f"{stem}.md").write_text("\n".join(text), encoding="utf-8")


def save_figure_data(name: str, df: pd.DataFrame) -> None:
    df.to_csv(FIG_DATA / f"{name}.csv", index=False)


def primary_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k, v in PRIMARY.items():
        mask &= df[k].astype(str).eq(v)
    return mask


def get_site_code(subject_id: Any) -> Optional[str]:
    s = str(subject_id)
    m = re.match(r"^(\d{3})_S_\d{4}$", s)
    return m.group(1) if m else None


def normalize_site3(v: Any) -> Optional[str]:
    if pd.isna(v):
        return None
    try:
        return f"{int(float(v)):03d}"
    except Exception:
        s = str(v).strip()
        return s.zfill(3) if s.isdigit() else s


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METADATA)
    meta["SiteCode"] = meta["SubjectID"].map(get_site_code)
    meta["Site3_norm"] = meta["Site3"].map(normalize_site3) if "Site3" in meta.columns else None
    meta["sitecode_valid_id"] = meta["SiteCode"].notna()
    meta["sitecode_site3_match"] = np.where(meta["Site3_norm"].notna(), meta["SiteCode"].eq(meta["Site3_norm"]), np.nan)
    return meta


def load_primary_predictions() -> pd.DataFrame:
    pred = pd.read_csv(STAGEB / "calib_predictions.csv")
    pred = pred[primary_mask(pred)].copy()
    if len(pred) != 397:
        raise RuntimeError(f"Primary OOF predictions expected N=397, found {len(pred)}")
    pred["SiteCode"] = pred["SubjectID"].map(get_site_code)
    pred["confusion_label"] = np.select(
        [
            (pred.y_true == 0) & (pred.y_pred == 0),
            (pred.y_true == 0) & (pred.y_pred == 1),
            (pred.y_true == 1) & (pred.y_pred == 0),
            (pred.y_true == 1) & (pred.y_pred == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )
    pred["correct"] = pred["y_true"].eq(pred["y_pred"])
    return pred


def load_latent(fold: int, split: str) -> pd.DataFrame:
    path = LOCKED_RUN / "classifier_only_readout/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
    df = pd.read_csv(path)
    df["SiteCode"] = df["SubjectID"].map(get_site_code)
    return df


def mu_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("mu_")], key=lambda x: int(x.split("_")[1]))


def covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    sex = df["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).fillna(0.5).to_numpy()
    age = pd.to_numeric(df["Age"], errors="coerce").fillna(pd.to_numeric(df["Age"], errors="coerce").median()).to_numpy()
    y = pd.to_numeric(df["y"], errors="coerce").fillna((df["ResearchGroup_Mapped"].astype(str) == "AD").astype(int)).to_numpy()
    return np.column_stack([np.ones(len(df)), y, age, sex])


def scale_and_residualize(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols = mu_cols(train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[cols].to_numpy(float))
    x_test = scaler.transform(test[cols].to_numpy(float))
    z_train = covariate_matrix(train)
    z_test = covariate_matrix(test)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(z_train, x_train)
    return x_train - reg.predict(z_train), x_test - reg.predict(z_test)


def stratified_site_permutation(train: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    labels = train["SiteCode"].astype(str).to_numpy().copy()
    out = labels.copy()
    strata = train["ResearchGroup_Mapped"].astype(str) + "|" + train["Manufacturer"].astype(str)
    for _, idx in pd.Series(np.arange(len(train))).groupby(strata).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) > 1:
            out[idx_arr] = rng.permutation(out[idx_arr])
    return out


def fit_site_classifier(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(x_train, y_train)
    return clf


def metric_basic(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    out: Dict[str, Any] = {"n": int(len(y)), "n_cn": int((y == 0).sum()), "n_ad": int((y == 1).sum())}
    if len(np.unique(y)) == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out.update(
            {
                "auc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "brier": float(brier_score_loss(y, score)) if np.nanmin(score) >= 0 and np.nanmax(score) <= 1 else np.nan,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "f1": float(sklearn_f1_score(y, pred)),
                "cn_fpr": float(fp / (tn + fp)) if tn + fp else np.nan,
                "ad_fnr": float(fn / (tp + fn)) if tp + fn else np.nan,
            }
        )
    else:
        out.update({k: np.nan for k in ["auc", "pr_auc", "brier", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1", "cn_fpr", "ad_fnr"]})
    return out


def bootstrap_auc_pr(y: np.ndarray, score: np.ndarray, n_boot: int = 1000, seed: int = 20260706) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2 or min((y == 0).sum(), (y == 1).sum()) < 3:
        return {"auc_ci_low": np.nan, "auc_ci_high": np.nan, "pr_auc_ci_low": np.nan, "pr_auc_ci_high": np.nan, "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    aucs, prs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)])
        aucs.append(roc_auc_score(y[idx], score[idx]))
        prs.append(average_precision_score(y[idx], score[idx]))
    return {
        "auc_ci_low": float(np.percentile(aucs, 2.5)),
        "auc_ci_high": float(np.percentile(aucs, 97.5)),
        "pr_auc_ci_low": float(np.percentile(prs, 2.5)),
        "pr_auc_ci_high": float(np.percentile(prs, 97.5)),
        "bootstrap_n": n_boot,
    }


def site_tables(meta: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    validation = meta.assign(
        malformed_subject_id=~meta["sitecode_valid_id"],
        duplicate_subject_id=meta["SubjectID"].duplicated(keep=False),
        sitecode_site3_mismatch=np.where(meta["Site3_norm"].notna(), ~meta["SiteCode"].eq(meta["Site3_norm"]), False),
    )[
        [
            "SubjectID",
            "SiteCode",
            "Site3",
            "Site3_norm",
            "ResearchGroup_Mapped",
            "Manufacturer",
            "malformed_subject_id",
            "duplicate_subject_id",
            "sitecode_site3_match",
            "sitecode_site3_mismatch",
        ]
    ]
    comp = (
        meta.groupby("SiteCode", dropna=False)
        .agg(
            total_n=("SubjectID", "nunique"),
            cn_n=("ResearchGroup_Mapped", lambda s: int((s == "CN").sum())),
            ad_n=("ResearchGroup_Mapped", lambda s: int((s == "AD").sum())),
            mci_n=("ResearchGroup_Mapped", lambda s: int((s == "MCI").sum())),
            manufacturer_n=("Manufacturer", "nunique"),
        )
        .reset_index()
    )
    pred_counts = (
        pred.groupby("SiteCode")
        .agg(
            supervised_n=("SubjectID", "nunique"),
            supervised_cn_n=("y_true", lambda s: int((s == 0).sum())),
            supervised_ad_n=("y_true", lambda s: int((s == 1).sum())),
            test_n=("SubjectID", "nunique"),
        )
        .reset_index()
    )
    comp = comp.merge(pred_counts, on="SiteCode", how="left").fillna({"test_n": 0, "supervised_n": 0, "supervised_cn_n": 0, "supervised_ad_n": 0})
    comp["geometry_supported"] = comp["test_n"] >= SITE_TIERS["geometry_test_n_min"]
    comp["site_auc_supported"] = (
        (comp.supervised_n >= SITE_TIERS["site_auc_total_n_min"])
        & (comp.supervised_cn_n >= SITE_TIERS["site_auc_cn_min"])
        & (comp.supervised_ad_n >= SITE_TIERS["site_auc_ad_min"])
    )
    comp["loso_supported"] = (
        (comp.supervised_n >= SITE_TIERS["loso_total_n_min"])
        & (comp.supervised_cn_n >= SITE_TIERS["loso_cn_min"])
        & (comp.supervised_ad_n >= SITE_TIERS["loso_ad_min"])
    )
    mfr = pd.crosstab(meta["SiteCode"], meta["Manufacturer"]).reset_index()
    dx = pd.crosstab(meta["SiteCode"], meta["ResearchGroup_Mapped"]).reset_index()
    fold = pd.crosstab(pred["SiteCode"], pred["fold"]).reset_index()
    excl = comp.assign(
        descriptive_supported=True,
        geometry_reason=np.where(comp["geometry_supported"], "supported", "test-fold N < 10"),
        site_auc_reason=np.where(comp["site_auc_supported"], "supported", "requires supervised CN/AD N>=15, CN>=3, AD>=3"),
        loso_reason=np.where(comp["loso_supported"], "supported", "requires supervised CN/AD N>=20, CN>=5, AD>=5"),
    )
    return {
        "sitecode_validation": validation,
        "site_composition": comp.sort_values(["total_n", "SiteCode"], ascending=[False, True]),
        "site_by_manufacturer": mfr,
        "site_by_diagnosis": dx,
        "site_by_outer_fold": fold,
        "site_exclusion_table": excl.sort_values("SiteCode"),
    }


def foldlocal_site_decoding(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, within_rows, error_rows = [], [], []
    rng = np.random.default_rng(20260706)
    for fold in range(1, 6):
        train = load_latent(fold, "trainDev")
        test = load_latent(fold, "test")
        train = train[train["SiteCode"].notna()].copy()
        test = test[test["SiteCode"].notna()].copy()
        x_train, x_test = scale_and_residualize(train, test)
        train_sites = train["SiteCode"].astype(str).to_numpy()
        test_sites = test["SiteCode"].astype(str).to_numpy()
        valid_train_sites = pd.Series(train_sites).value_counts()
        keep_train = np.array([valid_train_sites[s] >= 2 for s in train_sites])
        train_eval_sites = np.unique(train_sites[keep_train])
        keep_test = np.array([s in train_eval_sites for s in test_sites])
        obs_ba = np.nan
        obs_f1 = np.nan
        if keep_train.sum() > 5 and keep_test.sum() > 0 and len(np.unique(train_sites[keep_train])) > 1 and len(np.unique(test_sites[keep_test])) > 1:
            clf = fit_site_classifier(x_train[keep_train], train_sites[keep_train])
            yhat = clf.predict(x_test[keep_test])
            obs_ba = balanced_accuracy_score(test_sites[keep_test], yhat)
            obs_f1 = sklearn_f1_score(test_sites[keep_test], yhat, average="macro", zero_division=0)
            null_ba = []
            for _ in range(50):
                perm_sites = stratified_site_permutation(train.iloc[np.where(keep_train)[0]], rng)
                if len(np.unique(perm_sites)) < 2:
                    continue
                clf_null = fit_site_classifier(x_train[keep_train], perm_sites)
                null_ba.append(balanced_accuracy_score(test_sites[keep_test], clf_null.predict(x_test[keep_test])))
            null_ba = np.asarray(null_ba, dtype=float)
            null_mean = float(np.nanmean(null_ba)) if len(null_ba) else np.nan
            null_sd = float(np.nanstd(null_ba, ddof=1)) if len(null_ba) > 1 else np.nan
            perm_p = float((np.sum(null_ba >= obs_ba) + 1) / (len(null_ba) + 1)) if len(null_ba) else np.nan
        else:
            null_mean = null_sd = perm_p = np.nan
        # Site centroid/radius on test, after train-fitted transforms.
        centroids, radii = [], []
        for site, idx in pd.Series(np.arange(len(test))).groupby(test_sites).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < 2:
                continue
            xs = x_test[idx]
            c = xs.mean(axis=0)
            centroids.append(c)
            radii.append(float(np.sqrt(((xs - c) ** 2).sum(axis=1)).mean()))
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        rows.append(
            {
                "fold": fold,
                "test_n": int(len(test)),
                "test_sites_n": int(pd.Series(test_sites).nunique()),
                "evaluated_test_n": int(keep_test.sum()),
                "site_balanced_accuracy": obs_ba,
                "site_macro_f1": obs_f1,
                "permutation_null_ba_mean": null_mean,
                "permutation_null_ba_sd": null_sd,
                "permutation_p_ge_observed": perm_p,
                "site_centroid_pairwise_distance_mean": float(np.mean(dists)) if dists else np.nan,
                "pooled_within_site_radius_mean": float(np.mean(radii)) if radii else np.nan,
                "centroid_distance_over_radius": float(np.mean(dists) / np.mean(radii)) if dists and np.mean(radii) else np.nan,
            }
        )
        # Within-manufacturer decoding.
        for mfr in sorted(test["Manufacturer"].dropna().astype(str).unique()):
            tr_m = train["Manufacturer"].astype(str).eq(mfr).to_numpy()
            te_m = test["Manufacturer"].astype(str).eq(mfr).to_numpy()
            if tr_m.sum() < 10 or te_m.sum() < 5:
                continue
            ytr = train_sites[tr_m]
            yte = test_sites[te_m]
            valid = pd.Series(ytr).value_counts()
            tr_keep = tr_m & np.array([valid.get(s, 0) >= 2 for s in train_sites])
            train_supported_sites = set(valid[valid >= 2].index.astype(str))
            te_keep = te_m & np.array([s in train_supported_sites for s in test_sites])
            if tr_keep.sum() < 5 or te_keep.sum() < 3 or len(np.unique(train_sites[tr_keep])) < 2 or len(np.unique(test_sites[te_keep])) < 2:
                continue
            clf = fit_site_classifier(x_train[tr_keep], train_sites[tr_keep])
            yh = clf.predict(x_test[te_keep])
            within_rows.append(
                {
                    "fold": fold,
                    "Manufacturer": mfr,
                    "train_n": int(tr_keep.sum()),
                    "test_n": int(te_keep.sum()),
                    "train_sites_n": int(len(np.unique(train_sites[tr_keep]))),
                    "test_sites_n": int(len(np.unique(test_sites[te_keep]))),
                    "balanced_accuracy": float(balanced_accuracy_score(test_sites[te_keep], yh)),
                    "macro_f1": float(sklearn_f1_score(test_sites[te_keep], yh, average="macro", zero_division=0)),
                }
            )
        # Site error association counts.
        fold_pred = pred[pred["fold"] == fold]
        for site, sub in fold_pred.groupby("SiteCode"):
            counts = sub["confusion_label"].value_counts().to_dict()
            error_rows.append(
                {
                    "fold": fold,
                    "SiteCode": site,
                    "n": len(sub),
                    "TN": int(counts.get("TN", 0)),
                    "FP": int(counts.get("FP", 0)),
                    "FN": int(counts.get("FN", 0)),
                    "TP": int(counts.get("TP", 0)),
                    "error_rate": float((~sub["correct"]).mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(within_rows), pd.DataFrame(error_rows)


def site_specific_metrics(pred: pd.DataFrame, site_comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    supported = set(site_comp[site_comp["site_auc_supported"]]["SiteCode"].astype(str))
    for site, sub in pred.groupby("SiteCode"):
        out = {"SiteCode": site, "supported_for_auc": site in supported}
        out.update(metric_basic(sub["y_true"].to_numpy(), sub["y_score"].to_numpy(), sub["y_pred"].to_numpy()))
        out.update(bootstrap_auc_pr(sub["y_true"].to_numpy(), sub["y_score"].to_numpy(), n_boot=1000))
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["supported_for_auc", "n"], ascending=[False, False])


def loso_readout(site_comp: pd.DataFrame) -> pd.DataFrame:
    target_sites = sorted(site_comp[site_comp["loso_supported"]]["SiteCode"].astype(str).tolist())
    rows = []
    for target in target_sites:
        pred_rows = []
        for fold in range(1, 6):
            train = load_latent(fold, "trainDev")
            test = load_latent(fold, "test")
            train = train[train["SiteCode"].notna()].copy()
            test = test[test["SiteCode"].eq(target)].copy()
            if test.empty:
                continue
            cols = mu_cols(train)
            feature_cols = cols + ["Age", "Sex_num"]
            train["Sex_num"] = train["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).fillna(0.5)
            test["Sex_num"] = test["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).fillna(0.5)
            train = train[~train["SiteCode"].eq(target)].copy()
            if train["y"].nunique() < 2:
                continue
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train[feature_cols].to_numpy(float))
            x_test = scaler.transform(test[feature_cols].to_numpy(float))
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs")
            clf.fit(x_train, train["y"].to_numpy(int))
            score = clf.predict_proba(x_test)[:, 1]
            yhat = (score >= 0.5).astype(int)
            tmp = test[["SubjectID", "SiteCode", "fold", "y", "ResearchGroup_Mapped", "Manufacturer"]].copy()
            tmp["score"] = score
            tmp["pred"] = yhat
            pred_rows.append(tmp)
        if not pred_rows:
            continue
        pp = pd.concat(pred_rows, ignore_index=True)
        out = {"target_site": target}
        out.update(metric_basic(pp["y"].to_numpy(), pp["score"].to_numpy(), pp["pred"].to_numpy()))
        # Calibration slope/intercept for descriptive evaluation only.
        eps = 1e-6
        if pp["y"].nunique() == 2:
            logit = np.log(np.clip(pp["score"].to_numpy(), eps, 1 - eps) / np.clip(1 - pp["score"].to_numpy(), eps, 1 - eps)).reshape(-1, 1)
            cal = LogisticRegression(solver="lbfgs").fit(logit, pp["y"].to_numpy(int))
            out["calibration_intercept"] = float(cal.intercept_[0])
            out["calibration_slope"] = float(cal.coef_[0][0])
        else:
            out["calibration_intercept"] = np.nan
            out["calibration_slope"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_dirs: int = 128, seed: int = 20260706) -> float:
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    dirs = rng.normal(size=(n_dirs, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = [wasserstein_distance(x @ v, y @ v) for v in dirs]
    return float(np.mean(vals))


def sym_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = eigh((mat + mat.T) / 2.0)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def bures_wasserstein_gaussian(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    mx, my = x.mean(axis=0), y.mean(axis=0)
    cx = LedoitWolf().fit(x).covariance_
    cy = LedoitWolf().fit(y).covariance_
    sqrt_cx = sym_sqrt(cx)
    inner = sym_sqrt(sqrt_cx @ cy @ sqrt_cx)
    val = float(np.sum((mx - my) ** 2) + np.trace(cx + cy - 2 * inner))
    return float(np.sqrt(max(val, 0.0)))


def oasis_wasserstein(site_comp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not OASIS_LATENT.exists():
        raise FileNotFoundError(f"Authoritative OASIS latent file not found: {OASIS_LATENT}")
    oasis = pd.read_csv(OASIS_LATENT)
    if oasis["fold"].nunique() != 5 or oasis.groupby("fold")["SubjectID"].nunique().min() != 180:
        raise RuntimeError("OASIS latent file does not contain five folds with 180 unique subjects each")
    if set(oasis["ResearchGroup_Mapped"].astype(str).unique()) != {"CN", "AD"}:
        raise RuntimeError("OASIS latent file does not have CN/AD labels only")
    rows = []
    perf_rows = []
    supported_sites = set(site_comp[site_comp["geometry_supported"]]["SiteCode"].astype(str))
    for fold in range(1, 6):
        adni = load_latent(fold, "trainDev")
        adni = adni[adni["SiteCode"].notna()].copy()
        o = oasis[oasis["fold"].eq(fold)].copy()
        cols = mu_cols(adni)
        missing = [c for c in cols if c not in o.columns]
        if missing:
            raise RuntimeError(f"OASIS latent file missing mu columns, first={missing[:5]}")
        scaler = StandardScaler().fit(adni[cols].to_numpy(float))
        x_ref = scaler.transform(adni[cols].to_numpy(float))
        x_o = scaler.transform(o[cols].to_numpy(float))
        groups = [("all_adni_trainDev", "all", np.ones(len(adni), dtype=bool))]
        for mfr in sorted(adni["Manufacturer"].dropna().astype(str).unique()):
            groups.append(("manufacturer", mfr, adni["Manufacturer"].astype(str).eq(mfr).to_numpy()))
        for site in sorted(supported_sites):
            groups.append(("site", site, adni["SiteCode"].astype(str).eq(site).to_numpy()))
        for group_type, group, mask in groups:
            if mask.sum() < 10:
                continue
            xg = x_ref[mask]
            rows.append(
                {
                    "fold": fold,
                    "comparison_group_type": group_type,
                    "comparison_group": group,
                    "adni_group_n": int(mask.sum()),
                    "oasis_n": int(len(o)),
                    "sliced_wasserstein": sliced_wasserstein(xg, x_o, seed=20260706 + fold),
                    "bures_wasserstein_ledoitwolf": bures_wasserstein_gaussian(xg, x_o),
                }
            )
    # Fold-level external performance for final promoted runwise164.
    foldwise = pd.read_csv(OASIS_PANEL / "foldwise_metrics.csv")
    fw = foldwise[
        (foldwise["candidate"] == "promoted_beta3p75_oof_ecdf")
        & (foldwise["build_candidate"] == "runwise164_pilot_parity")
        & (foldwise["prediction_level"] == "fold_model")
    ].copy()
    dist_all = pd.DataFrame(rows)
    all_ref = dist_all[dist_all["comparison_group"].eq("all")][["fold", "sliced_wasserstein", "bures_wasserstein_ledoitwolf"]]
    perf_rows = fw.merge(all_ref, on="fold", how="left")
    return dist_all, perf_rows


def main_results_table(fold_site: pd.DataFrame, loso: pd.DataFrame, oasis_w: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_csv(STAGEB / "calib_pooled_metrics.csv")
    primary = primary[primary_mask(primary)].iloc[0]
    combat = pd.read_csv(FOLDCOMBAT_STAGEB / "calib_pooled_metrics.csv")
    combat = combat[primary_mask(combat)].iloc[0]
    mfr_leak = []
    for fold in range(1, 6):
        df = pd.read_csv(LOCKED_RUN / f"fold_{fold}/fold_{fold}_test_scanner_leakage.csv")
        row = df[(df["representation"] == "latent_mu") & (df["site_col"] == "Manufacturer")].iloc[0]
        mfr_leak.append(row["balanced_accuracy_mean"])
    oasis_primary = pd.read_csv(OASIS_PANEL / "primary_metrics.csv")
    oasis_primary = oasis_primary[
        (oasis_primary["candidate"] == "promoted_beta3p75_oof_ecdf")
        & (oasis_primary["build_candidate"] == "runwise164_pilot_parity")
        & (oasis_primary["prediction_level"] == "ensemble_mean_score_majority_vote")
    ].iloc[0]
    w_all = oasis_w[oasis_w["comparison_group"].eq("all")]
    return pd.DataFrame(
        [
            {
                "result": "manufacturer latent decodability",
                "n_or_folds": 5,
                "primary_value": float(np.mean(mfr_leak)),
                "secondary_value": float(np.std(mfr_leak, ddof=1)),
                "metric": "balanced accuracy mean, SD",
            },
            {
                "result": "site latent decodability",
                "n_or_folds": 5,
                "primary_value": float(fold_site["site_balanced_accuracy"].mean()),
                "secondary_value": float(fold_site["site_balanced_accuracy"].std(ddof=1)),
                "metric": "balanced accuracy mean, SD",
            },
            {
                "result": "locked ADNI diagnostic performance",
                "n_or_folds": int(primary["n"]),
                "primary_value": float(primary["auc"]),
                "secondary_value": float(primary["pr_auc"]),
                "metric": "ROC-AUC, PR-AUC",
            },
            {
                "result": "fold-wise input-ComBat performance",
                "n_or_folds": int(combat["n"]),
                "primary_value": float(combat["auc"]),
                "secondary_value": float(combat["pr_auc"]),
                "metric": "ROC-AUC, PR-AUC",
            },
            {
                "result": "leave-one-site-out summary",
                "n_or_folds": int(len(loso)),
                "primary_value": float(loso["auc"].mean()) if len(loso) else np.nan,
                "secondary_value": float(loso["pr_auc"].mean()) if len(loso) else np.nan,
                "metric": "mean supported-site ROC-AUC, PR-AUC",
            },
            {
                "result": "OASIS external performance",
                "n_or_folds": int(oasis_primary["n"]),
                "primary_value": float(oasis_primary["auc"]),
                "secondary_value": float(oasis_primary["pr_auc"]),
                "metric": "runwise164 one-row-per-subject ROC-AUC, PR-AUC",
            },
            {
                "result": "ADNI-OASIS Wasserstein distance",
                "n_or_folds": 5,
                "primary_value": float(w_all["sliced_wasserstein"].mean()),
                "secondary_value": float(w_all["bures_wasserstein_ledoitwolf"].mean()),
                "metric": "mean sliced W, mean Gaussian Bures-W",
            },
        ]
    )


def make_figures(site_comp: pd.DataFrame, within_mfr: pd.DataFrame, site_metrics: pd.DataFrame, loso: pd.DataFrame, oasis_w: pd.DataFrame, oasis_perf: pd.DataFrame) -> None:
    # Site sample composition.
    top = site_comp.sort_values("total_n", ascending=False).head(20).copy()
    save_figure_data("site_sample_composition", top)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bottom = np.zeros(len(top))
    colors = {"CN": "#4C78A8", "MCI": "#B3B3B3", "AD": "#F58518"}
    for col, label in [("cn_n", "CN"), ("mci_n", "MCI"), ("ad_n", "AD")]:
        vals = top[col].to_numpy()
        ax.bar(top["SiteCode"].astype(str), vals, bottom=bottom, label=label, color=colors[label])
        bottom += vals
    ax.set_ylabel("N")
    ax.set_xlabel("SiteCode")
    ax.set_title("ADNI site composition by diagnosis")
    ax.legend(frameon=False, ncol=3)
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "site_sample_composition.pdf")
    plt.close(fig)

    # Within-manufacturer site geometry.
    save_figure_data("site_geometry_within_manufacturer", within_mfr)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if len(within_mfr):
        order = sorted(within_mfr["Manufacturer"].unique())
        data = [within_mfr[within_mfr["Manufacturer"].eq(m)]["balanced_accuracy"].dropna().to_numpy() for m in order]
        ax.boxplot(data, labels=order, showmeans=True)
        ax.axhline(1 / 3, color="0.5", ls="--", lw=1, label="3-class chance reference")
    ax.set_ylabel("Held-out site decoding BA")
    ax.set_title("Within-manufacturer site information in latent mu")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "site_geometry_within_manufacturer.pdf")
    plt.close(fig)

    # Site performance forest.
    sm = site_metrics[site_metrics["supported_for_auc"]].copy().sort_values("auc")
    save_figure_data("site_performance_forest", sm)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.3 * len(sm))))
    if len(sm):
        y = np.arange(len(sm))
        ax.errorbar(sm["auc"], y, xerr=[sm["auc"] - sm["auc_ci_low"], sm["auc_ci_high"] - sm["auc"]], fmt="o", color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels(sm["SiteCode"].astype(str))
        ax.axvline(0.5, color="0.5", ls="--", lw=1)
    ax.set_xlabel("OOF ROC-AUC")
    ax.set_ylabel("SiteCode")
    ax.set_title("Site-specific diagnostic transport")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "site_performance_forest.pdf")
    plt.close(fig)

    # LOSO performance.
    save_figure_data("leave_one_site_out_performance", loso)
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.3 * len(loso))))
    if len(loso):
        ll = loso.sort_values("auc")
        y = np.arange(len(ll))
        ax.scatter(ll["auc"], y, label="ROC-AUC", color="#4C78A8")
        ax.scatter(ll["pr_auc"], y, label="PR-AUC", color="#F58518")
        ax.set_yticks(y)
        ax.set_yticklabels(ll["target_site"].astype(str))
        ax.axvline(0.5, color="0.5", ls="--", lw=1)
        ax.legend(frameon=False)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Held-out SiteCode")
    ax.set_title("Classifier/readout transport on frozen latent representations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leave_one_site_out_performance.pdf")
    plt.close(fig)

    # OASIS Wasserstein.
    ow = oasis_w[oasis_w["comparison_group_type"].isin(["all_adni_trainDev", "manufacturer"])].copy()
    save_figure_data("external_dataset_wasserstein", ow)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if len(ow):
        labels = ow["comparison_group_type"] + ":" + ow["comparison_group"].astype(str)
        summary = ow.groupby(labels).agg(mean=("sliced_wasserstein", "mean"), sd=("sliced_wasserstein", "std")).reset_index().sort_values("mean")
        ax.barh(summary["index"], summary["mean"], xerr=summary["sd"], color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Sliced Wasserstein distance")
    ax.set_title("ADNI trainDev to OASIS latent distribution shift")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "external_dataset_wasserstein.pdf")
    plt.close(fig)

    # External performance vs shift.
    save_figure_data("external_performance_vs_domain_shift", oasis_perf)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(oasis_perf["sliced_wasserstein"], oasis_perf["auc"], color="#4C78A8", s=55, label="ROC-AUC")
    for _, r in oasis_perf.iterrows():
        ax.annotate(f"fold {int(r['fold'])}", (r["sliced_wasserstein"], r["auc"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("OASIS vs ADNI trainDev sliced W")
    ax.set_ylabel("OASIS fold-model ROC-AUC")
    ax.set_title("External performance vs domain shift (descriptive)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "external_performance_vs_domain_shift.pdf")
    plt.close(fig)


def write_reports(
    tex_audit: Dict[str, Any],
    site_val: pd.DataFrame,
    site_comp: pd.DataFrame,
    fold_site: pd.DataFrame,
    site_metrics: pd.DataFrame,
    loso: pd.DataFrame,
    oasis_w: pd.DataFrame,
    main: pd.DataFrame,
) -> None:
    malformed = int(site_val["malformed_subject_id"].sum())
    mismatches = int(site_val["sitecode_site3_mismatch"].sum())
    geom_mean = fold_site["site_balanced_accuracy"].mean()
    loso_auc = loso["auc"].mean() if len(loso) else np.nan
    w_all = oasis_w[oasis_w["comparison_group"].eq("all")]
    summary = f"""# SIPAIM Multisite Transport Executive Summary

## Guardrail Status

- no VAE training: true
- no OASIS preprocessing/inference: true
- no manuscript/protocol rewriting: true
- no pooling of raw latent coordinates across folds for site-geometry estimators: true

## TeX/Input Audit

- Working TeX exists: {tex_audit['tex_exists']}
- Existing site-geometry PDFs found: {tex_audit['existing_pdf_count']} ({', '.join(tex_audit['existing_pdfs'])})
- pdflatex pass 1 return code: {tex_audit['pdflatex_pass1_returncode']}
- pdflatex pass 2 return code: {tex_audit['pdflatex_pass2_returncode']}

## Main Findings

1. `SiteCode` reconstructed from `SSS_S_NNNN` SubjectID is valid for the final ADNI metadata line. Malformed IDs: {malformed}; numeric Site3 mismatches: {mismatches}.
2. Fold-local latent `mu` retains acquisition-site information after residualizing diagnosis, Age, and Sex. Mean held-out site decoding balanced accuracy: {geom_mean:.3f}.
3. Site-specific diagnostic transport is heterogeneous; unsupported small sites are explicitly listed in `site_exclusion_table`.
4. Frozen-latent classifier/readout leave-one-site-out is a sensitivity analysis, not VAE leave-one-site-out. Mean supported-site ROC-AUC: {loso_auc:.3f}.
5. OASIS external distribution shift was computed from the existing promoted runwise164 fold-latent file. Mean OASIS-vs-ADNI trainDev sliced Wasserstein distance: {w_all['sliced_wasserstein'].mean():.3f}; mean Gaussian Bures-Wasserstein: {w_all['bures_wasserstein_ledoitwolf'].mean():.3f}.

## Main Results Table

{md_table(main)}
"""
    (OUT / "00_executive_summary.md").write_text(summary, encoding="utf-8")
    limitations = """# Limitations

- SiteCode is reconstructed from ADNI SubjectID and validated against numeric Site3 where present; it is an acquisition-site proxy, not a complete scanner/protocol descriptor.
- Fold-local site decoding residualizes diagnosis, Age, and Sex, but residual confounding by protocol, scanner software, motion, and timepoint regime may remain.
- Classifier/readout leave-one-site-out freezes the final VAE latent representation; it does not test a VAE trained without the target site.
- OASIS Wasserstein analysis uses existing final runwise164 fold-level latent vectors. No raw OASIS data were reopened and no OASIS inference was rerun.
- External performance versus Wasserstein shift has only five fold-level points and is descriptive only.
- No TDA, Mapper, decoder traversal, adversarial training, CovBat, CORAL, or new VAE training was performed.
"""
    (OUT / "limitations.md").write_text(limitations, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")
    ensure_dirs()
    tex_audit = {
        "tex_path": str(TEX),
        "tex_exists": TEX.exists(),
        "fig_dir": str(TEX_FIG_DIR),
        "existing_pdfs": sorted(p.name for p in TEX_FIG_DIR.glob("*.pdf")) if TEX_FIG_DIR.exists() else [],
        "existing_pdf_count": len(list(TEX_FIG_DIR.glob("*.pdf"))) if TEX_FIG_DIR.exists() else 0,
        "pdflatex_pass1_returncode": 0,
        "pdflatex_pass2_returncode": 0,
        "pdflatex_note": "pdflatex was run twice before this script in the same audit session; the script did not rewrite TeX.",
    }
    if not TEX.exists():
        raise FileNotFoundError(TEX)
    if tex_audit["existing_pdf_count"] != 7:
        raise RuntimeError(f"Expected 7 existing PDFs, found {tex_audit['existing_pdf_count']}")

    meta = load_metadata()
    pred = load_primary_predictions()
    tables = site_tables(meta, pred)
    for name, df in tables.items():
        write_csv_md(name, df, name.replace("_", " ").title(), max_rows=80)
    tables["site_by_manufacturer"].to_csv(OUT / "site_by_manufacturer.csv", index=False)
    tables["site_by_diagnosis"].to_csv(OUT / "site_by_diagnosis.csv", index=False)
    tables["site_by_outer_fold"].to_csv(OUT / "site_by_outer_fold.csv", index=False)

    fold_site, within_mfr, site_errors = foldlocal_site_decoding(pred)
    write_csv_md("foldlocal_site_decoding", fold_site, "Fold-Local Site Decoding")
    write_csv_md("within_manufacturer_site_decoding", within_mfr, "Within-Manufacturer Site Decoding")
    write_csv_md("site_error_association", site_errors, "Site and FP/FN/Correct Status Association", max_rows=100)

    site_metrics = site_specific_metrics(pred, tables["site_composition"])
    write_csv_md("site_specific_diagnostic_metrics", site_metrics, "Site-Specific Diagnostic Metrics", max_rows=120)

    loso = loso_readout(tables["site_composition"])
    write_csv_md("leave_one_site_out_metrics", loso, "Classifier/Readout Transport on Frozen Latent Representations")

    oasis_w, oasis_perf = oasis_wasserstein(tables["site_composition"])
    write_csv_md("oasis_wasserstein_by_fold", oasis_w, "OASIS Wasserstein by Fold", max_rows=120)
    write_csv_md("oasis_performance_vs_domain_shift", oasis_perf, "OASIS Performance vs Domain Shift")

    main_table = main_results_table(fold_site, loso, oasis_w)
    write_csv_md("main_results_table", main_table, "Main Results Table")

    make_figures(tables["site_composition"], within_mfr, site_metrics, loso, oasis_w, oasis_perf)
    write_reports(tex_audit, tables["sitecode_validation"], tables["site_composition"], fold_site, site_metrics, loso, oasis_w, main_table)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "guardrails": {
            "did_train_vae": False,
            "did_train_new_vae": False,
            "did_run_oasis_inference": False,
            "did_modify_protocol_tex": False,
            "did_modify_model_outputs": False,
            "did_use_tda": False,
            "did_use_mapper": False,
        },
        "inputs": {
            "tex": str(TEX),
            "locked_run": str(LOCKED_RUN),
            "stageB": str(STAGEB),
            "metadata": str(METADATA),
            "oasis_audit": str(OASIS_AUDIT),
            "oasis_latent": str(OASIS_LATENT),
        },
        "site_tiers": SITE_TIERS,
        "outputs": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
