#!/usr/bin/env python3
"""
LOSO inner-CV overfitting check.

For each held-out-site LOSO model, run a 5-fold stratified inner-CV over the
training pool to obtain genuine OOF (out-of-fold) performance on that pool.
This answers Martín's question: does the high apparent train AUC (0.92-0.96)
reflect real generalisation within the training distribution, or in-sample
overfitting?

The CV uses:
  - The tuned C from optuna_best_trial_logreg.json (already optimised — not re-tuned here)
  - A fresh LogisticRegression + _AutoPreprocessor per fold (new instances, not saved)
  - 5-fold StratifiedKFold stratified on ResearchGroup_Mapped (CN/AD), random_state=42

Read-only: VAE checkpoint, norm_params, train_subjects, optuna_best_trial_logreg.
New temporary LR instances fitted per fold (not saved as model artifacts).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("loso_inner_cv_check")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.interpretability.interpret_fold import (
    apply_normalization_params,
    clean_state_dict,
)
from betavae_xai.models.classifiers import _AutoPreprocessor
from betavae_xai.models.convolutional_vae import ConvolutionalVAE

# ── Constants ─────────────────────────────────────────────────────────────────
LOSO_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/eligible_site_holdout_fullmatched_20260624"
)
TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
OUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/loso_inner_cv_overfitting_check_20260625"
)
SITES = ["site_130", "site_006", "site_035", "site_135"]

# VAE architecture — from run_config.json (same for all 4 LOSO models)
CHANNELS = [1, 0, 2]
LATENT_DIM = 384
IMAGE_SIZE = 131
N_CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Apparent train AUC from prior audit (read-only reference values)
APPARENT_TRAIN_AUC = {
    "site_130": 0.9264,
    "site_006": 0.9397,
    "site_035": 0.9566,
    "site_135": 0.9150,
}
HELDOUT_AUC = {
    "site_130": 0.7875,
    "site_006": 0.7778,
    "site_035": 0.5733,
    "site_135": 0.6000,
}


def _vae_kwargs() -> dict:
    return dict(
        input_channels=len(CHANNELS),
        latent_dim=LATENT_DIM,
        image_size=IMAGE_SIZE,
        dropout_rate=0.15,
        use_layernorm_fc=False,
        num_conv_layers_encoder=4,
        decoder_type="convtranspose",
        intermediate_fc_dim_config="quarter",
        final_activation="tanh",
        num_groups=16,
    )


def _load_vae(ckpt_path: Path, device: torch.device) -> ConvolutionalVAE:
    vae = ConvolutionalVAE(**_vae_kwargs()).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    vae.load_state_dict(clean_state_dict(sd))
    vae.eval()
    return vae


def _encode_subjects(
    vae: ConvolutionalVAE,
    tensor_all: np.ndarray,
    norm_params: list,
    tensor_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    tens = tensor_all[tensor_indices][:, CHANNELS, :, :]
    tens = apply_normalization_params(tens, norm_params)
    mus = []
    with torch.no_grad():
        for start in range(0, len(tens), batch_size):
            batch = torch.from_numpy(tens[start : start + batch_size]).float().to(device)
            mu, _ = vae.encode(batch)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def _build_feature_df(mu: np.ndarray, meta_df: pd.DataFrame) -> pd.DataFrame:
    lat_cols = [f"latent_{i}" for i in range(LATENT_DIM)]
    df = pd.DataFrame(mu, columns=lat_cols, index=meta_df.index)
    df["Age"] = meta_df["Age"].values
    df["Sex"] = meta_df["Sex"].values
    return df


def _metrics_from_oof(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    bacc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "balanced_acc": round(float(bacc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "n_CN": int((y_true == 0).sum()),
        "n_AD": int((y_true == 1).sum()),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def _run_inner_cv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    tuned_C: float,
    site: str,
) -> tuple[dict, pd.DataFrame]:
    """5-fold stratified inner-CV; returns metrics dict and OOF predictions DataFrame."""
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    oof_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df, y), start=1):
        X_tr = X_df.iloc[train_idx].copy()
        X_val = X_df.iloc[val_idx].copy()
        y_tr = y[train_idx]

        # Fresh pipeline per fold — NOT saved, OOF predictions only
        pipe = Pipeline(
            steps=[
                ("preprocess", _AutoPreprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        C=tuned_C,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=20000,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]

        subjects = X_df.index[val_idx].tolist()
        for subj, prob, true_label in zip(subjects, proba, y[val_idx]):
            oof_records.append(
                {
                    "site": site,
                    "fold": fold_idx,
                    "SubjectID": subj,
                    "y_true": int(true_label),
                    "y_prob": float(prob),
                }
            )
        log.info(
            "  fold %d/%d: val n=%d (CN=%d AD=%d)",
            fold_idx,
            N_CV_FOLDS,
            len(val_idx),
            int((y[val_idx] == 0).sum()),
            int((y[val_idx] == 1).sum()),
        )

    oof_df = pd.DataFrame(oof_records)
    y_true_all = oof_df["y_true"].values
    y_prob_all = oof_df["y_prob"].values
    metrics = _metrics_from_oof(y_true_all, y_prob_all)
    return metrics, oof_df


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TENSOR_PATH.exists():
        raise FileNotFoundError(f"Tensor not found: {TENSOR_PATH}")

    log.info("Loading global tensor from %s ...", TENSOR_PATH)
    tensor_all = np.load(TENSOR_PATH)["global_tensor_data"]
    log.info("Tensor shape: %s", tensor_all.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    inner_cv_rows = []
    three_way_rows = []
    all_oof_dfs = []

    for site in SITES:
        site_dir = LOSO_DIR / site
        log.info("=== %s ===", site)

        train_csv = site_dir / "train_subjects.csv"
        optuna_json = site_dir / "optuna_best_trial_logreg.json"
        ckpt = site_dir / f"vae_model_{site}.pt"
        norm_path = site_dir / "vae_norm_params.joblib"

        train_df = pd.read_csv(train_csv)
        with open(optuna_json) as f:
            optuna_data = json.load(f)
        tuned_C = float(optuna_data["best_params"]["model__C"])
        log.info("  tuned C=%.8g  n_train=%d", tuned_C, len(train_df))

        norm_params = joblib.load(norm_path)
        vae = _load_vae(ckpt, device)

        train_tensor_idx = train_df["tensor_idx"].values.astype(int)
        log.info("  Encoding %d training subjects...", len(train_tensor_idx))
        mu_train = _encode_subjects(vae, tensor_all, norm_params, train_tensor_idx, device)
        train_df = train_df.set_index("SubjectID")
        X_df = _build_feature_df(mu_train, train_df)

        # Binary label: AD=1, CN=0
        label_map = {"AD": 1, "CN": 0}
        y = train_df["ResearchGroup_Mapped"].map(label_map).values
        cn_count = int((y == 0).sum())
        ad_count = int((y == 1).sum())
        log.info("  Training pool: CN=%d  AD=%d", cn_count, ad_count)

        log.info("  Running %d-fold inner CV (stratified, random_state=%d)...",
                 N_CV_FOLDS, CV_RANDOM_STATE)
        metrics, oof_df = _run_inner_cv(X_df, y, tuned_C, site)
        all_oof_dfs.append(oof_df)

        inner_cv_rows.append(
            {
                "site": site,
                "train_n_CN": cn_count,
                "train_n_AD": ad_count,
                "tuned_C": tuned_C,
                "inner_cv_oof_auc": metrics["auc"],
                "inner_cv_oof_pr_auc": metrics["pr_auc"],
                "inner_cv_oof_balanced_acc": metrics["balanced_acc"],
                "inner_cv_oof_sensitivity": metrics["sensitivity"],
                "inner_cv_oof_specificity": metrics["specificity"],
            }
        )
        three_way_rows.append(
            {
                "site": site,
                "apparent_train_auc": APPARENT_TRAIN_AUC[site],
                "inner_cv_oof_auc": metrics["auc"],
                "heldout_site_auc": HELDOUT_AUC[site],
                "apparent_vs_innercv_gap": round(
                    APPARENT_TRAIN_AUC[site] - metrics["auc"], 4
                ),
                "innercv_vs_heldout_gap": round(
                    metrics["auc"] - HELDOUT_AUC[site], 4
                ),
            }
        )
        log.info(
            "  => inner_cv_oof_auc=%.4f  apparent_train_auc=%.4f  heldout_auc=%.4f",
            metrics["auc"],
            APPARENT_TRAIN_AUC[site],
            HELDOUT_AUC[site],
        )

    # ── Write outputs ──────────────────────────────────────────────────────────
    inner_cv_df = pd.DataFrame(inner_cv_rows)
    three_way_df = pd.DataFrame(three_way_rows)
    oof_all_df = pd.concat(all_oof_dfs, ignore_index=True)

    inner_cv_df.to_csv(OUT_DIR / "inner_cv_oof_metrics_by_site.csv", index=False)
    three_way_df.to_csv(
        OUT_DIR / "three_way_comparison_apparent_vs_innercv_vs_heldout.csv", index=False
    )
    oof_all_df.to_csv(OUT_DIR / "fold_level_inner_cv_predictions.csv", index=False)

    _write_md_table(inner_cv_df, OUT_DIR / "inner_cv_oof_metrics_by_site.md",
                    title="LOSO Inner-CV OOF Metrics by Site",
                    notes=(
                        "5-fold stratified inner-CV over training pool using tuned C "
                        "(from optuna_best_trial_logreg.json). "
                        "Fresh LogisticRegression per fold — not saved. "
                        "These are genuine OOF metrics, not apparent (in-sample)."
                    ))
    _write_three_way_md(three_way_df, OUT_DIR / "three_way_comparison_apparent_vs_innercv_vs_heldout.md")
    _write_interpretation(three_way_df, inner_cv_df, OUT_DIR / "final_interpretation.md")

    elapsed = round(time.time() - t0, 1)
    cmd_log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_label": "loso_inner_cv_overfitting_check_20260625",
        "elapsed_seconds": elapsed,
        "sites_processed": SITES,
        "cv_folds": N_CV_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_classifier": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_model_artifacts": False,
            "new_lr_instances_fitted": True,
            "new_lr_saved": False,
            "metric_type_note": (
                "inner_cv_oof: genuine OOF over training pool using tuned C. "
                "apparent_train: in-sample (from prior audit). "
                "heldout_site: true OOF from saved test_predictions_logreg.csv."
            ),
        },
        "outputs": [
            "inner_cv_oof_metrics_by_site.csv",
            "inner_cv_oof_metrics_by_site.md",
            "three_way_comparison_apparent_vs_innercv_vs_heldout.csv",
            "three_way_comparison_apparent_vs_innercv_vs_heldout.md",
            "fold_level_inner_cv_predictions.csv",
            "final_interpretation.md",
            "command_log.json",
        ],
    }
    with open(OUT_DIR / "command_log.json", "w") as f:
        json.dump(cmd_log, f, indent=2)
    log.info("Done in %.1fs. Outputs in %s", elapsed, OUT_DIR)


def _write_md_table(df: pd.DataFrame, path: Path, title: str, notes: str = "") -> None:
    lines = [f"# {title}", f"Generated: {datetime.now().date()}", ""]
    if notes:
        lines += [notes, ""]
    lines += [df.to_markdown(index=False), ""]
    path.write_text("\n".join(lines))


def _write_three_way_md(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Three-Way Comparison: Apparent Train vs. Inner-CV OOF vs. Held-out Site AUC",
        f"Generated: {datetime.now().date()}",
        "",
        "| site | apparent_train_auc | inner_cv_oof_auc | heldout_site_auc | "
        "apparent_vs_innercv_gap | innercv_vs_heldout_gap |",
        "|------|:-----------------:|:---------------:|:---------------:|:----------------------:|:---------------------:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['site']} | {row['apparent_train_auc']:.4f} | "
            f"**{row['inner_cv_oof_auc']:.4f}** | {row['heldout_site_auc']:.4f} | "
            f"{row['apparent_vs_innercv_gap']:+.4f} | {row['innercv_vs_heldout_gap']:+.4f} |"
        )
    lines += [
        "",
        "**apparent_vs_innercv_gap**: how much of the apparent AUC was inflated "
        "by in-sample scoring (≥0.08–0.10 → overfitting concern).",
        "**innercv_vs_heldout_gap**: the remaining gap attributable to site-transfer "
        "after correcting for any in-sample inflation.",
        "",
    ]
    path.write_text("\n".join(lines))


def _write_interpretation(three_way_df: pd.DataFrame, inner_df: pd.DataFrame,
                           path: Path) -> None:
    lines = [
        "# Final Interpretation: LOSO Inner-CV Overfitting Check",
        f"Generated: {datetime.now().date()}",
        "",
        "**Question (Martín — '¿en validación qué te da?'):** Is the high apparent training-sites AUC",
        "(0.92–0.96) a trustworthy upper bound, or is it inflated by in-sample scoring?",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| site | apparent_train_auc | inner_cv_oof_auc | heldout_site_auc | "
        "apparent_vs_innercv_gap | verdict |",
        "|------|:-----------------:|:---------------:|:---------------:|:----------------------:|:-------:|",
    ]
    verdicts = {}
    for _, row in three_way_df.iterrows():
        gap = row["apparent_vs_innercv_gap"]
        if gap <= 0.05:
            verdict = "NOT OVERFITTING"
        elif gap <= 0.10:
            verdict = "MILD INFLATION"
        else:
            verdict = "OVERFITTING CONCERN"
        verdicts[row["site"]] = (verdict, gap)
        lines.append(
            f"| {row['site']} | {row['apparent_train_auc']:.4f} | "
            f"**{row['inner_cv_oof_auc']:.4f}** | {row['heldout_site_auc']:.4f} | "
            f"{gap:+.4f} | **{verdict}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation by site",
        "",
    ]

    for _, row in three_way_df.iterrows():
        site = row["site"]
        verdict, gap = verdicts[site]
        icv = row["inner_cv_oof_auc"]
        apt = row["apparent_train_auc"]
        held = row["heldout_site_auc"]
        shift_gap = row["innercv_vs_heldout_gap"]
        inner_row = inner_df[inner_df["site"] == site].iloc[0]
        sens = inner_row["inner_cv_oof_sensitivity"]
        spec = inner_row["inner_cv_oof_specificity"]

        lines += [f"### {site}"]
        if verdict == "NOT OVERFITTING":
            lines += [
                f"- Apparent train AUC={apt:.4f}, inner-CV OOF AUC=**{icv:.4f}** → gap={gap:+.4f}",
                f"- Inner-CV sensitivity={sens:.4f}, specificity={spec:.4f} at threshold 0.5",
                f"- Held-out site AUC={held:.4f} → inner-CV vs held-out gap={shift_gap:+.4f}",
                f"- **VERDICT: The model is NOT overfitting to the training pool.** The apparent AUC "
                f"was a fair estimate of within-distribution generalisation. The {shift_gap:+.4f} "
                f"gap to the held-out site is attributable to site-transfer / threshold-transfer, "
                f"not in-sample inflation. The original interpretation (score distribution shift as "
                f"the primary mechanism of near-zero held-out sensitivity) stands.",
            ]
        elif verdict == "MILD INFLATION":
            lines += [
                f"- Apparent train AUC={apt:.4f}, inner-CV OOF AUC=**{icv:.4f}** → gap={gap:+.4f}",
                f"- Inner-CV sensitivity={sens:.4f}, specificity={spec:.4f} at threshold 0.5",
                f"- Held-out site AUC={held:.4f} → inner-CV vs held-out gap={shift_gap:+.4f}",
                f"- **VERDICT: Mild in-sample inflation ({gap:+.4f}).** The apparent AUC slightly "
                f"overstated generalisation within the training pool, but the inner-CV AUC of "
                f"{icv:.4f} still shows strong training-distribution performance. The held-out gap "
                f"({shift_gap:+.4f} from inner-CV to held-out) remains primarily attributable to "
                f"site-transfer. The threshold-transfer interpretation stands but the ceiling "
                f"({apt:.4f}) should be replaced by the inner-CV figure ({icv:.4f}).",
            ]
        else:
            lines += [
                f"- Apparent train AUC={apt:.4f}, inner-CV OOF AUC=**{icv:.4f}** → gap={gap:+.4f}",
                f"- Inner-CV sensitivity={sens:.4f}, specificity={spec:.4f} at threshold 0.5",
                f"- Held-out site AUC={held:.4f} → inner-CV vs held-out gap={shift_gap:+.4f}",
                f"- **VERDICT: OVERFITTING CONCERN CONFIRMED.** The apparent AUC was substantially "
                f"inflated by in-sample scoring. The true within-distribution generalisation ceiling "
                f"is {icv:.4f}, not {apt:.4f}. The corrected gap to the held-out site is "
                f"{shift_gap:+.4f} (not {held - apt:+.4f}). While site-transfer is still a "
                f"contributing factor, the classifier is also overfitting to the training pool.",
            ]
        lines.append("")

    lines += [
        "---",
        "",
        "## Pooled verdict",
        "",
    ]

    gaps = [v[1] for v in verdicts.values()]
    mean_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    any_overfitting = any(v[0] == "OVERFITTING CONCERN" for v in verdicts.values())
    any_mild = any(v[0] == "MILD INFLATION" for v in verdicts.values())

    if not any_overfitting and not any_mild:
        pooled = (
            f"Across all four sites, the mean apparent-vs-inner-CV gap is {mean_gap:+.4f} "
            f"(max={max_gap:+.4f}). **The model is NOT overfitting to the training pool.** "
            "The apparent AUC figures (0.92–0.96) accurately reflect within-distribution "
            "generalisation under the 5-fold inner-CV. The near-zero held-out sensitivity "
            "is driven by site-transfer / threshold-transfer, not in-sample inflation. "
            "The original `final_interpretation.md` from the apparent-performance audit is confirmed."
        )
    elif not any_overfitting:
        pooled = (
            f"Across all four sites, the mean apparent-vs-inner-CV gap is {mean_gap:+.4f} "
            f"(max={max_gap:+.4f}). There is mild in-sample inflation but no site shows a "
            "critical overfitting concern (gap > 0.10). The inner-CV AUC figures provide the "
            "more conservative and correct ceiling for the manuscript. The threshold-transfer "
            "interpretation of near-zero held-out sensitivity stands."
        )
    else:
        pooled = (
            f"Across all four sites, the mean apparent-vs-inner-CV gap is {mean_gap:+.4f} "
            f"(max={max_gap:+.4f}). At least one site shows substantial in-sample inflation "
            "(gap > 0.10). **Overfitting concern is confirmed for at least one model.** "
            "The inner-CV AUC figures must replace the apparent AUC as the training-distribution "
            "ceiling in the manuscript."
        )

    lines += [pooled, "", "---", "", "## Guardrail status", ""]
    lines += [
        "- did_train_vae: False",
        "- did_retrain_classifier: False (original pipeline.joblib unchanged)",
        "- new_lr_instances_fitted: True (temporary, not saved)",
        "- did_modify_tensors: False",
        "- did_modify_metadata: False",
        "- Computation: frozen VAE encode + fresh LR per fold, tuned C reused from optuna",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
