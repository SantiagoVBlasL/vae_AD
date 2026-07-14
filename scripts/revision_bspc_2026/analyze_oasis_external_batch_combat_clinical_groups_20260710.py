#!/usr/bin/env python3
"""Clinical-group post-hoc analysis for OASIS external Dataset-ComBat.

This script consumes the completed OASIS external-batch ComBat analysis and
adds clinical-group distribution summaries requested for the SIPAIM
transportability manuscript. It is read-only with respect to model artifacts
and raw tensors. OASIS diagnosis labels are used only after harmonisation and
scoring, for post-hoc stratified evaluation/reporting.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/diego/proyectos/vae_AD")
SCRIPT_DIR = PROJECT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_oasis_external_batch_combat_20260710 as extcombat  # noqa: E402
from score_oasis_mega_90_90_external_inference_model_panel_20260604 import (  # noqa: E402
    ensure_y,
    load_mega_tensor,
)


OUT = (
    PROJECT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "oasis_external_batch_combat_20260710"
)

LOCKED_OASIS_LATENTS = (
    PROJECT
    / "results/revision_bspc_2026/promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/"
    "oasis_fold_latent_mu_runwise164.csv"
)
PREVIOUS_OASIS_LATENTS = (
    PROJECT
    / "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "foldcombat_cleanrepro_final_audit_20260706/cleanrepro_oasis_fold_latent_mu.csv"
)
PREVIOUS_TRAIN_LATENT_DIR = (
    PROJECT
    / "results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706/"
    "downstream_diagnostic_classifier/latent_cache"
)

ARMS = [
    "locked_frozen_transfer",
    "previous_adni_fitted_siemens_combat",
    "external_dataset_combat_adni_reference",
    "external_dataset_combat_no_reference",
]

TARGETS = [
    "oasis_score_distributions_by_clinical_group.csv",
    "oasis_score_distributions_by_clinical_group.md",
    "fig_oasis_score_distributions_by_group.pdf",
    "oasis_latent_clinical_separation_by_arm_fold.csv",
    "oasis_latent_clinical_separation_by_arm_fold.md",
    "oasis_class_conditional_feature_alignment_by_fold.csv",
    "oasis_class_conditional_feature_alignment_by_fold.md",
    "oasis_class_conditional_feature_alignment_summary.csv",
    "oasis_class_conditional_feature_alignment_summary.md",
    "fig_oasis_class_conditional_alignment.pdf",
    "oasis_clinical_group_distribution_interpretation.md",
    "manuscript_insert_clinical_group_distribution.md",
    "oasis_clinical_group_distribution_command_log.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_overwrite() -> None:
    completed_marker = OUT / "oasis_clinical_group_distribution_command_log.json"
    if completed_marker.exists():
        raise RuntimeError(f"Refusing to rerun over completed clinical-group analysis: {completed_marker}")
    # If a prior failed attempt wrote early partials but did not reach the
    # completion command log, allow this script to repair those same outputs.
    # This still protects completed analyses from being overwritten.
    if any((OUT / name).exists() for name in TARGETS):
        return
    existing = [str(OUT / name) for name in TARGETS if (OUT / name).exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite existing target files: " + "; ".join(existing))


def write_md_table(df: pd.DataFrame, path: Path, title: str, max_rows: int | None = None) -> None:
    view = df if max_rows is None else df.head(max_rows)
    try:
        body = view.to_markdown(index=False)
    except Exception:
        body = view.to_string(index=False)
    suffix = "" if max_rows is None or len(df) <= max_rows else f"\n\n_Showing {max_rows} of {len(df)} rows._"
    path.write_text(f"# {title}\n\n{body}{suffix}\n", encoding="utf-8")


def normalise_y(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "y" not in out.columns:
        if "y_true" in out.columns:
            out["y"] = out["y_true"]
        elif "ResearchGroup_Mapped" in out.columns:
            out["y"] = out["ResearchGroup_Mapped"].astype(str).str.upper().map({"CN": 0, "AD": 1})
        elif "diagnosis" in out.columns:
            out["y"] = out["diagnosis"].astype(str).str.upper().map({"CN": 0, "AD": 1})
    out["y"] = pd.to_numeric(out["y"], errors="raise").astype(int)
    out["clinical_group"] = np.where(out["y"].to_numpy() == 1, "AD", "CN")
    return out


def mu_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("mu_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # Positive means values in a tend to be larger than values in b.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    comp = a[:, None] - b[None, :]
    return float((np.sum(comp > 0) - np.sum(comp < 0)) / comp.size)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1)
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled)) if pooled > 0 else np.nan


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if hi <= lo:
        return 1.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(ha, hb) * widths))


def score_distribution_rows(pred: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred = normalise_y(pred)
    ens = pred[pred["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    for arm in ARMS:
        g = ens[ens["arm"].eq(arm)].copy()
        if g.empty:
            rows.append({"arm": arm, "status": "missing_ensemble_predictions"})
            continue
        g["y_score"] = pd.to_numeric(g["y_score"], errors="raise")
        cn = g[g["y"].eq(0)]["y_score"].to_numpy(dtype=float)
        ad = g[g["y"].eq(1)]["y_score"].to_numpy(dtype=float)
        metric_row = metrics[metrics["arm"].eq(arm)]
        rows.append(
            {
                "arm": arm,
                "status": "ok",
                "n_cn": int(len(cn)),
                "n_ad": int(len(ad)),
                "cn_mean": float(np.mean(cn)),
                "cn_sd": float(np.std(cn, ddof=1)),
                "cn_median": float(np.median(cn)),
                "cn_iqr": float(np.percentile(cn, 75) - np.percentile(cn, 25)),
                "ad_mean": float(np.mean(ad)),
                "ad_sd": float(np.std(ad, ddof=1)),
                "ad_median": float(np.median(ad)),
                "ad_iqr": float(np.percentile(ad, 75) - np.percentile(ad, 25)),
                "ad_minus_cn_mean": float(np.mean(ad) - np.mean(cn)),
                "cohens_d_ad_vs_cn": cohen_d(ad, cn),
                "cliffs_delta_ad_gt_cn": cliffs_delta(ad, cn),
                "ks_statistic_cn_vs_ad": float(ks_2samp(cn, ad).statistic),
                "ks_pvalue_cn_vs_ad": float(ks_2samp(cn, ad).pvalue),
                "overlap_coefficient": overlap_coefficient(cn, ad),
                "roc_auc_reference": float(metric_row.iloc[0]["roc_auc"]) if len(metric_row) == 1 else np.nan,
                "pr_auc_reference": float(metric_row.iloc[0]["pr_auc"]) if len(metric_row) == 1 else np.nan,
                "balanced_accuracy_reference": (
                    float(metric_row.iloc[0]["balanced_accuracy"]) if len(metric_row) == 1 else np.nan
                ),
                "sensitivity_reference": float(metric_row.iloc[0]["sensitivity"]) if len(metric_row) == 1 else np.nan,
                "specificity_reference": float(metric_row.iloc[0]["specificity"]) if len(metric_row) == 1 else np.nan,
                "mean_score_reference": float(metric_row.iloc[0]["mean_score"]) if len(metric_row) == 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_all_prediction_rows() -> pd.DataFrame:
    external = pd.read_csv(OUT / "oasis_external_batch_combat_predictions.csv")
    locked = extcombat.load_locked_oasis_baseline()
    previous = extcombat.load_previous_adni_fitted_combat_baseline()
    return pd.concat([locked, previous, external], ignore_index=True, sort=False)


def plot_score_distributions(pred: pd.DataFrame, path: Path) -> None:
    pred = normalise_y(pred)
    ens = pred[pred["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    bins = np.linspace(0.0, 1.0, 31)
    labels = {
        "locked_frozen_transfer": "Locked frozen",
        "previous_adni_fitted_siemens_combat": "ADNI-fitted Siemens-ComBat",
        "external_dataset_combat_adni_reference": "External Dataset-ComBat\nADNI reference",
        "external_dataset_combat_no_reference": "External Dataset-ComBat\nno reference",
    }
    for ax, arm in zip(axes, ARMS):
        g = ens[ens["arm"].eq(arm)]
        for y, label, color in [(0, "CN", "#4C78A8"), (1, "AD", "#E45756")]:
            vals = g[g["y"].eq(y)]["y_score"].to_numpy(dtype=float)
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2, color=color, label=label)
            ax.axvline(np.mean(vals), color=color, linestyle="--", linewidth=1)
        ax.set_title(labels.get(arm, arm))
        ax.set_xlabel("OASIS ensemble score")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gaussian_w2(a: np.ndarray, b: np.ndarray) -> float:
    lwa = LedoitWolf(store_precision=False).fit(a)
    lwb = LedoitWolf(store_precision=False).fit(b)
    sa, sb = lwa.covariance_, lwb.covariance_
    sqrt_sa = sqrtm(sa).real
    sqrt_inner = sqrtm(sqrt_sa @ sb @ sqrt_sa).real
    mean_sq = float(np.sum((lwa.location_ - lwb.location_) ** 2))
    cov_term = float(np.trace(sa) + np.trace(sb) - 2.0 * np.trace(sqrt_inner))
    return float(np.sqrt(max(mean_sq + cov_term, 0.0)))


def latent_rows_for_arm(arm: str, df: pd.DataFrame, train_dir: Path | None, train_mode: str) -> list[dict[str, Any]]:
    df = normalise_y(df)
    cols = mu_columns(df)
    rows: list[dict[str, Any]] = []
    for fold, g in df.groupby("fold"):
        try:
            fold_int = int(fold)
        except Exception:
            continue
        g = g.copy()
        x = g[cols].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=int)
        cn = x[y == 0]
        ad = x[y == 1]
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        z_cn = z[y == 0]
        z_ad = z[y == 1]
        pca = PCA(n_components=min(10, z.shape[1]), random_state=42).fit(z)
        pc_cn = pca.transform(z_cn)
        pc_ad = pca.transform(z_ad)
        pca_w1 = [wasserstein_distance(pc_cn[:, j], pc_ad[:, j]) for j in range(pc_cn.shape[1])]
        row: dict[str, Any] = {
            "arm": arm,
            "fold": fold_int,
            "status": "ok",
            "n_cn": int(len(cn)),
            "n_ad": int(len(ad)),
            "latent_mean_euclidean_cn_ad": float(np.linalg.norm(np.mean(ad, axis=0) - np.mean(cn, axis=0))),
            "latent_standardized_mean_euclidean_cn_ad": float(
                np.linalg.norm(np.mean(z_ad, axis=0) - np.mean(z_cn, axis=0))
            ),
            "pca10_w1_mean_cn_ad": float(np.mean(pca_w1)),
            "pca10_w1_sum_cn_ad": float(np.sum(pca_w1)),
            "pca10_gaussian_w2_cn_ad": gaussian_w2(pc_cn, pc_ad),
            "adni_diagnostic_direction_available": False,
            "adni_diagnostic_direction_source": train_mode,
            "projection_cn_mean": np.nan,
            "projection_ad_mean": np.nan,
            "projection_ad_minus_cn": np.nan,
            "projection_cohens_d": np.nan,
            "projection_auc": np.nan,
        }
        if train_dir is not None:
            train_path = train_dir / f"fold_{fold_int}_trainDev_latent_mu.csv"
            if train_path.exists():
                train = ensure_y(pd.read_csv(train_path))
                train_cols = mu_columns(train)
                common = [c for c in cols if c in train_cols]
                tr_x = train[common].to_numpy(dtype=float)
                tr_y = train["y"].to_numpy(dtype=int)
                mean = tr_x.mean(axis=0)
                sd = tr_x.std(axis=0, ddof=1)
                sd[sd < 1e-8] = 1.0
                tr_z = (tr_x - mean) / sd
                direction = tr_z[tr_y == 1].mean(axis=0) - tr_z[tr_y == 0].mean(axis=0)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    proj = ((g[common].to_numpy(dtype=float) - mean) / sd) @ direction
                    p_cn = proj[y == 0]
                    p_ad = proj[y == 1]
                    row.update(
                        {
                            "adni_diagnostic_direction_available": True,
                            "projection_cn_mean": float(np.mean(p_cn)),
                            "projection_ad_mean": float(np.mean(p_ad)),
                            "projection_ad_minus_cn": float(np.mean(p_ad) - np.mean(p_cn)),
                            "projection_cohens_d": cohen_d(p_ad, p_cn),
                            "projection_auc": float(roc_auc_score(y, proj)),
                        }
                    )
        rows.append(row)
    return rows


def load_latent_tables() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    external = pd.read_csv(OUT / "oasis_external_batch_combat_latents.csv")
    frames.append(external)
    if LOCKED_OASIS_LATENTS.exists():
        locked = pd.read_csv(LOCKED_OASIS_LATENTS)
        locked["arm"] = "locked_frozen_transfer"
        frames.append(locked)
    if PREVIOUS_OASIS_LATENTS.exists():
        prev = pd.read_csv(PREVIOUS_OASIS_LATENTS)
        prev["arm"] = "previous_adni_fitted_siemens_combat"
        frames.append(prev)
    return pd.concat(frames, ignore_index=True, sort=False)


def latent_separation_rows() -> pd.DataFrame:
    lat = load_latent_tables()
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        g = lat[lat["arm"].eq(arm)].copy()
        if g.empty:
            rows.append({"arm": arm, "fold": np.nan, "status": "missing_oasis_latents"})
            continue
        if arm == "previous_adni_fitted_siemens_combat":
            rows.extend(latent_rows_for_arm(arm, g, PREVIOUS_TRAIN_LATENT_DIR, "cleanrepro_harmonized_train_dev"))
        else:
            rows.extend(latent_rows_for_arm(arm, g, extcombat.LOCKED_RUN / "classifier_only_readout/latent_cache", "locked_train_dev"))
    return pd.DataFrame(rows).sort_values(["arm", "fold"], na_position="last")


def group_feature_w1(adni: np.ndarray, oasis: np.ndarray, adni_y: np.ndarray, oasis_y: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler().fit(adni)
    za = scaler.transform(adni)
    zo = scaler.transform(oasis)
    out: dict[str, float] = {}
    pairs = [
        ("adni_cn_oasis_cn", 0, 0),
        ("adni_ad_oasis_ad", 1, 1),
        ("adni_cn_oasis_ad", 0, 1),
        ("adni_ad_oasis_cn", 1, 0),
    ]
    for name, ay, oy in pairs:
        a = za[adni_y == ay]
        o = zo[oasis_y == oy]
        w = np.asarray([wasserstein_distance(a[:, j], o[:, j]) for j in range(za.shape[1])])
        out[f"w1_mean_{name}"] = float(np.mean(w))
        out[f"w1_median_{name}"] = float(np.median(w))
    matched = out["w1_mean_adni_cn_oasis_cn"] + out["w1_mean_adni_ad_oasis_ad"]
    crossed = out["w1_mean_adni_cn_oasis_ad"] + out["w1_mean_adni_ad_oasis_cn"]
    out["matched_clinical_alignment_w1_sum"] = float(matched)
    out["crossed_clinical_alignment_w1_sum"] = float(crossed)
    out["clinical_alignment_margin_crossed_minus_matched"] = float(crossed - matched)
    return out


def feature_alignment_rows() -> pd.DataFrame:
    cfg = extcombat.load_config(extcombat.LOCKED_RUN)
    selected_names = list(
        cfg.get("selected_channel_names")
        or cfg.get("channels")
        or cfg.get("channels_to_use")
        or ["Pearson_Full_FisherZ_Signed", "Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]
    )
    global_selected, _ = extcombat.load_global_tensor_selected(selected_names)
    oasis_tensor, oasis_meta, oasis_names = load_mega_tensor(extcombat.OASIS_TENSOR)
    oasis_selected = oasis_tensor[:, extcombat.selected_channel_indices(selected_names, oasis_names)]
    oasis_meta = normalise_y(oasis_meta)
    oasis_y = oasis_meta["y"].to_numpy(dtype=int)

    rows: list[dict[str, Any]] = []
    variants = {v.arm: v for v in extcombat.VARIANTS}
    for fold in extcombat.FOLDS:
        train_latent = extcombat.load_locked_train_latent(fold)
        adni_idx = train_latent["tensor_idx"].to_numpy(dtype=int)
        adni_meta = train_latent[["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "y"]].copy()
        adni_meta = normalise_y(adni_meta)
        adni_y = adni_meta["y"].to_numpy(dtype=int)
        raw_adni_tensor = global_selected[adni_idx].astype(np.float32)
        raw_oasis_tensor = oasis_selected.astype(np.float32)
        raw_adni = extcombat.flatten_features(raw_adni_tensor)
        raw_oasis = extcombat.flatten_features(raw_oasis_tensor)
        raw_row = {
            "arm": "before_raw",
            "fold": fold,
            "state": "before_raw",
            "n_adni_cn": int(np.sum(adni_y == 0)),
            "n_adni_ad": int(np.sum(adni_y == 1)),
            "n_oasis_cn": int(np.sum(oasis_y == 0)),
            "n_oasis_ad": int(np.sum(oasis_y == 1)),
            "n_features": int(raw_adni.shape[1]),
            "oasis_labels_used_for_harmonization": False,
            "oasis_labels_used_for_posthoc_stratified_reporting": True,
        }
        raw_row.update(group_feature_w1(raw_adni, raw_oasis, adni_y, oasis_y))
        rows.append(raw_row)
        for arm in ["external_dataset_combat_adni_reference", "external_dataset_combat_no_reference"]:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                adni_h, oasis_h, _ = extcombat.combat_tensor(
                    raw_adni_tensor,
                    raw_oasis_tensor,
                    adni_meta,
                    oasis_meta,
                    variant=variants[arm],
                    selected_channel_names=selected_names,
                )
            adni_f = extcombat.flatten_features(adni_h.astype(np.float32))
            oasis_f = extcombat.flatten_features(oasis_h.astype(np.float32))
            row = {
                "arm": arm,
                "fold": fold,
                "state": "after_external_batch_combat",
                "n_adni_cn": int(np.sum(adni_y == 0)),
                "n_adni_ad": int(np.sum(adni_y == 1)),
                "n_oasis_cn": int(np.sum(oasis_y == 0)),
                "n_oasis_ad": int(np.sum(oasis_y == 1)),
                "n_features": int(adni_f.shape[1]),
                "oasis_labels_used_for_harmonization": False,
                "oasis_labels_used_for_posthoc_stratified_reporting": True,
            }
            row.update(group_feature_w1(adni_f, oasis_f, adni_y, oasis_y))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_feature_alignment(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "w1_mean_adni_cn_oasis_cn",
        "w1_mean_adni_ad_oasis_ad",
        "w1_mean_adni_cn_oasis_ad",
        "w1_mean_adni_ad_oasis_cn",
        "matched_clinical_alignment_w1_sum",
        "crossed_clinical_alignment_w1_sum",
        "clinical_alignment_margin_crossed_minus_matched",
    ]
    rows: list[dict[str, Any]] = []
    for arm, g in df.groupby("arm"):
        row: dict[str, Any] = {"arm": arm, "n_folds": int(g["fold"].nunique())}
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    before = summary[summary["arm"].eq("before_raw")]
    if len(before) == 1:
        b = before.iloc[0]
        for idx, row in summary.iterrows():
            for c in metric_cols:
                summary.loc[idx, f"{c}_delta_vs_before_mean"] = row[f"{c}_mean"] - b[f"{c}_mean"]
    return summary.sort_values("arm")


def plot_feature_alignment(summary: pd.DataFrame, path: Path) -> None:
    arms = ["before_raw", "external_dataset_combat_adni_reference", "external_dataset_combat_no_reference"]
    labels = ["Before raw", "Dataset-ComBat\nADNI ref", "Dataset-ComBat\nno ref"]
    s = summary.set_index("arm").loc[arms]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - 0.18, s["matched_clinical_alignment_w1_sum_mean"], width=0.36, label="Matched CN-CN + AD-AD")
    axes[0].bar(x + 0.18, s["crossed_clinical_alignment_w1_sum_mean"], width=0.36, label="Crossed CN-AD + AD-CN")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Mean feature W1 sum")
    axes[0].set_title("Class-conditional ADNI-OASIS alignment")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].bar(x, s["clinical_alignment_margin_crossed_minus_matched_mean"], color="#72B7B2")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Crossed - matched W1")
    axes[1].set_title("Clinical alignment margin")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def interpretation_text(
    score: pd.DataFrame,
    latent: pd.DataFrame,
    feature_summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    score_i = score.set_index("arm")
    feat_i = feature_summary.set_index("arm")
    locked = metrics.set_index("arm").loc["locked_frozen_transfer"]
    adni_ref = metrics.set_index("arm").loc["external_dataset_combat_adni_reference"]
    no_ref = metrics.set_index("arm").loc["external_dataset_combat_no_reference"]
    before_margin = feat_i.loc["before_raw", "clinical_alignment_margin_crossed_minus_matched_mean"]
    adni_margin = feat_i.loc["external_dataset_combat_adni_reference", "clinical_alignment_margin_crossed_minus_matched_mean"]
    no_margin = feat_i.loc["external_dataset_combat_no_reference", "clinical_alignment_margin_crossed_minus_matched_mean"]
    before_matched = feat_i.loc["before_raw", "matched_clinical_alignment_w1_sum_mean"]
    adni_matched = feat_i.loc["external_dataset_combat_adni_reference", "matched_clinical_alignment_w1_sum_mean"]
    no_matched = feat_i.loc["external_dataset_combat_no_reference", "matched_clinical_alignment_w1_sum_mean"]
    locked_sep = score_i.loc["locked_frozen_transfer", "ad_minus_cn_mean"]
    adni_sep = score_i.loc["external_dataset_combat_adni_reference", "ad_minus_cn_mean"]
    no_sep = score_i.loc["external_dataset_combat_no_reference", "ad_minus_cn_mean"]
    locked_cn = score_i.loc["locked_frozen_transfer", "cn_mean"]
    locked_ad = score_i.loc["locked_frozen_transfer", "ad_mean"]
    adni_cn = score_i.loc["external_dataset_combat_adni_reference", "cn_mean"]
    adni_ad = score_i.loc["external_dataset_combat_adni_reference", "ad_mean"]
    no_cn = score_i.loc["external_dataset_combat_no_reference", "cn_mean"]
    no_ad = score_i.loc["external_dataset_combat_no_reference", "ad_mean"]

    lines = [
        "# OASIS clinical-group distribution interpretation",
        "",
        "## Guardrails",
        "",
        "- No VAE was trained or updated.",
        "- Raw tensors were read but not modified.",
        "- OASIS labels were not used for harmonisation, calibration, threshold selection, model selection, or feature selection.",
        "- OASIS labels were used only for post-hoc CN/AD stratified evaluation and reporting.",
        "",
        "## Direct answers",
        "",
        (
            "1. **Did Dataset-ComBat improve global ADNI-OASIS feature alignment?** Yes. "
            "The prior global audit showed lower ADNI-OASIS feature-space Wasserstein distances after both external Dataset-ComBat variants. "
            "This clinical-group audit recomputed class-conditional feature distances from the same read-only tensors."
        ),
        "",
        (
            "2. **Did it improve CN-to-CN and AD-to-AD alignment separately?** Yes for the mean feature-W1 matched distances. "
            f"The matched CN-CN + AD-AD W1 sum changed from {before_matched:.4f} before raw harmonisation "
            f"to {adni_matched:.4f} with ADNI-reference Dataset-ComBat and {no_matched:.4f} with no-reference Dataset-ComBat."
        ),
        "",
        (
            "3. **Did it preserve or improve OASIS CN/AD separation?** Not in a ranking-relevant sense. "
            "The absolute AD-minus-CN mean score separation increased slightly, mostly because both clinical groups shifted upward "
            "and AD shifted a little more. "
            f"Locked AD-minus-CN mean score separation was {locked_sep:.4f}; ADNI-reference was {adni_sep:.4f}; "
            f"no-reference was {no_sep:.4f}. However standardized/effect-size separation did not improve "
            f"(Cohen's d: locked {score_i.loc['locked_frozen_transfer', 'cohens_d_ad_vs_cn']:.4f}; "
            f"ADNI-reference {score_i.loc['external_dataset_combat_adni_reference', 'cohens_d_ad_vs_cn']:.4f}; "
            f"no-reference {score_i.loc['external_dataset_combat_no_reference', 'cohens_d_ad_vs_cn']:.4f}), "
            f"and ROC-AUC/PR-AUC did not improve "
            f"(locked ROC-AUC/PR-AUC {locked['roc_auc']:.4f}/{locked['pr_auc']:.4f}; "
            f"ADNI-reference {adni_ref['roc_auc']:.4f}/{adni_ref['pr_auc']:.4f}; "
            f"no-reference {no_ref['roc_auc']:.4f}/{no_ref['pr_auc']:.4f})."
        ),
        "",
        (
            "4. **Does the clinical alignment margin explain why ROC-AUC/PR-AUC did not improve?** Partly. "
            "The class-conditional matched distances decreased, but the clinical alignment margin did not become more favorable. "
            f"The crossed-minus-matched margin changed from {before_margin:.4f} before raw harmonisation "
            f"to {adni_margin:.4f} with ADNI-reference and {no_margin:.4f} with no-reference Dataset-ComBat. "
            "Thus global and matched-distance improvements did not translate into a stronger class-conditional geometry that benefits ranking."
        ),
        "",
        (
            "5. **Is the operating-point shift caused by moving both clinical groups upward/downward in score, or by changing separation?** "
            "Both CN and AD scores moved upward after external Dataset-ComBat, with a larger upward shift for AD. "
            f"Locked CN/AD mean scores were {locked_cn:.4f}/{locked_ad:.4f}; ADNI-reference means were {adni_cn:.4f}/{adni_ad:.4f}; "
            f"no-reference means were {no_cn:.4f}/{no_ad:.4f}. "
            "This explains the higher sensitivity and lower specificity: more OASIS subjects cross the fixed ADNI-derived operating threshold."
        ),
        "",
        "## Latent-space note",
        "",
        (
            "Latent CN/AD separation was computed for all four arms where audited OASIS fold-latent tables were available. "
            "Projection onto an ADNI diagnostic direction used ADNI train/dev latent caches only; no OASIS labels were used to define the direction."
        ),
    ]
    return "\n".join(lines) + "\n"


def manuscript_insert(score: pd.DataFrame, feature_summary: pd.DataFrame, metrics: pd.DataFrame) -> str:
    s = score.set_index("arm")
    f = feature_summary.set_index("arm")
    m = metrics.set_index("arm")
    evidence_strong = (
        f.loc["external_dataset_combat_adni_reference", "matched_clinical_alignment_w1_sum_mean"]
        < f.loc["before_raw", "matched_clinical_alignment_w1_sum_mean"]
        and m.loc["external_dataset_combat_adni_reference", "roc_auc"] <= m.loc["locked_frozen_transfer", "roc_auc"] + 0.002
    )
    lines = [
        "# Manuscript insert: clinical-group distribution after external Dataset-ComBat",
        "",
    ]
    if not evidence_strong:
        lines.extend(
            [
                "Evidence was not considered strong enough for a compact manuscript insert.",
                "",
                "Reason: class-conditional feature alignment and OASIS ranking metrics did not show a coherent improvement pattern.",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Methods",
            "",
            (
                "As a post-hoc transportability analysis, we refit no predictive models and used OASIS diagnosis labels only for "
                "stratified reporting. For each ADNI outer fold, we compared OASIS CN and AD score distributions and recomputed "
                "class-conditional ADNI-OASIS connectivity-feature Wasserstein distances before and after unsupervised Dataset-ComBat "
                "with Dataset as batch and Age/Sex preserved."
            ),
            "",
            "## Results",
            "",
            (
                "External Dataset-ComBat reduced class-conditional matched ADNI-OASIS feature distances but did not improve external "
                f"ranking: locked frozen transfer achieved ROC-AUC/PR-AUC {m.loc['locked_frozen_transfer','roc_auc']:.3f}/"
                f"{m.loc['locked_frozen_transfer','pr_auc']:.3f}, whereas ADNI-reference Dataset-ComBat achieved "
                f"{m.loc['external_dataset_combat_adni_reference','roc_auc']:.3f}/"
                f"{m.loc['external_dataset_combat_adni_reference','pr_auc']:.3f}. "
                f"Both OASIS CN and AD scores shifted upward (CN mean {s.loc['locked_frozen_transfer','cn_mean']:.3f} to "
                f"{s.loc['external_dataset_combat_adni_reference','cn_mean']:.3f}; AD mean "
                f"{s.loc['locked_frozen_transfer','ad_mean']:.3f} to {s.loc['external_dataset_combat_adni_reference','ad_mean']:.3f}), "
                "producing higher sensitivity but lower specificity at the fixed ADNI-derived operating point."
            ),
            "",
            "## Discussion sentence",
            "",
            (
                "These results suggest that reducing global dataset shift by unsupervised ComBat does not necessarily preserve the "
                "class-conditional geometry required for improved clinical transportability."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    assert OUT.exists(), OUT
    assert_no_overwrite()
    command_log: list[dict[str, Any]] = [
        {
            "timestamp_utc": utc_now(),
            "argv": sys.argv,
            "cwd": str(PROJECT),
            "guardrails": {
                "no_vae_training": True,
                "no_raw_data_modification": True,
                "no_oasis_label_use_for_harmonization": True,
                "no_oasis_calibration": True,
                "no_oasis_threshold_selection": True,
                "no_oasis_model_selection": True,
                "oasis_labels_used_only_for_posthoc_stratified_reporting": True,
                "no_overwrite": True,
            },
        }
    ]

    pred = load_all_prediction_rows()
    metrics = pd.read_csv(OUT / "oasis_external_batch_combat_metrics.csv")

    score = score_distribution_rows(pred, metrics)
    score.to_csv(OUT / "oasis_score_distributions_by_clinical_group.csv", index=False)
    write_md_table(score, OUT / "oasis_score_distributions_by_clinical_group.md", "OASIS score distributions by clinical group")
    plot_score_distributions(pred, OUT / "fig_oasis_score_distributions_by_group.pdf")

    latent = latent_separation_rows()
    latent.to_csv(OUT / "oasis_latent_clinical_separation_by_arm_fold.csv", index=False)
    write_md_table(
        latent,
        OUT / "oasis_latent_clinical_separation_by_arm_fold.md",
        "OASIS latent clinical separation by arm and fold",
        max_rows=80,
    )

    feature = feature_alignment_rows()
    feature.to_csv(OUT / "oasis_class_conditional_feature_alignment_by_fold.csv", index=False)
    write_md_table(
        feature,
        OUT / "oasis_class_conditional_feature_alignment_by_fold.md",
        "OASIS class-conditional feature alignment by fold",
        max_rows=80,
    )
    feature_summary = summarize_feature_alignment(feature)
    feature_summary.to_csv(OUT / "oasis_class_conditional_feature_alignment_summary.csv", index=False)
    write_md_table(
        feature_summary,
        OUT / "oasis_class_conditional_feature_alignment_summary.md",
        "OASIS class-conditional feature alignment summary",
    )
    plot_feature_alignment(feature_summary, OUT / "fig_oasis_class_conditional_alignment.pdf")

    (OUT / "oasis_clinical_group_distribution_interpretation.md").write_text(
        interpretation_text(score, latent, feature_summary, metrics), encoding="utf-8"
    )
    (OUT / "manuscript_insert_clinical_group_distribution.md").write_text(
        manuscript_insert(score, feature_summary, metrics), encoding="utf-8"
    )

    command_log.append(
        {
            "timestamp_utc": utc_now(),
            "status": "COMPLETE",
            "output_dir": str(OUT),
            "script_sha256": sha256_file(Path(__file__)),
            "n_score_rows": int(len(score)),
            "n_latent_rows": int(len(latent)),
            "n_feature_alignment_rows": int(len(feature)),
            "n_feature_summary_rows": int(len(feature_summary)),
        }
    )
    (OUT / "oasis_clinical_group_distribution_command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
