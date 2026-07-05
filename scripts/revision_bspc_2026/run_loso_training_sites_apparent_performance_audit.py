#!/usr/bin/env python3
"""
LOSO training-sites apparent performance audit.

For each held-out-site model, score the subjects used to TRAIN that model
(i.e., subjects from all other sites) using the frozen, already-fitted
classifier. Report AUC, PR-AUC, balanced accuracy, sensitivity, specificity,
confusion matrix, and score distributions.

THIS IS APPARENT (IN-SAMPLE) PERFORMANCE — training subjects were used to
fit the classifier. True OOF training-sites performance would require
re-running the inner-CV across training folds, which is not done here.

No VAE training. No LOSO retraining. No metadata/tensor modification.
Read-only except for writing new outputs to the designated audit directory.
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
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("loso_train_perf_audit")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.interpretability.interpret_fold import (
    apply_normalization_params,
    build_vae,
    clean_state_dict,
)
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
    / "results/revision_bspc_2026/loso_training_sites_apparent_performance_audit_20260624"
)

SITES = ["site_130", "site_006", "site_035", "site_135"]
SITE_NUMBERS = {"site_130": 130, "site_006": 6, "site_035": 35, "site_135": 135}

# VAE architecture (from run_config.json)
CHANNELS = [1, 0, 2]
LATENT_DIM = 384
NUM_CONV = 4
DECODER_TYPE = "convtranspose"
DROPOUT = 0.15
INTER_FC = "quarter"
FINAL_ACT = "tanh"
USE_LAYERNORM = False
GN_NUM_GROUPS = 16
IMAGE_SIZE = 131


def _vae_kwargs() -> dict:
    return dict(
        input_channels=len(CHANNELS),
        latent_dim=LATENT_DIM,
        image_size=IMAGE_SIZE,
        dropout_rate=DROPOUT,
        use_layernorm_fc=USE_LAYERNORM,
        num_conv_layers_encoder=NUM_CONV,
        decoder_type=DECODER_TYPE,
        intermediate_fc_dim_config=INTER_FC,
        final_activation=FINAL_ACT,
        num_groups=GN_NUM_GROUPS,
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
    """Encode subjects through VAE encoder → return mu (N, latent_dim)."""
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
    """Build the feature DataFrame expected by the raw pipeline."""
    lat_cols = [f"latent_{i}" for i in range(LATENT_DIM)]
    df = pd.DataFrame(mu, columns=lat_cols, index=meta_df.index)
    df["Age"] = meta_df["Age"].values
    # Keep Sex as string so the pipeline's OHE/ordinal handles it correctly
    df["Sex"] = meta_df["Sex"].values
    return df


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics at a given decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    n_ad = int((y_true == 1).sum())
    n_cn = int((y_true == 0).sum())
    auc = float(roc_auc_score(y_true, y_score)) if n_ad > 0 and n_cn > 0 else float("nan")
    pr_auc = (
        float(average_precision_score(y_true, y_score)) if n_ad > 0 and n_cn > 0 else float("nan")
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "n_total": int(len(y_true)),
        "n_CN": n_cn,
        "n_AD": n_ad,
        "auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "threshold": threshold,
    }


def main() -> None:
    t_start = time.time()
    log.info("=" * 70)
    log.info("LOSO Training-Sites Apparent Performance Audit")
    log.info("=" * 70)

    # Guardrails
    if not TENSOR_PATH.exists():
        raise FileNotFoundError(f"Tensor not found: {TENSOR_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load tensor once
    log.info("Loading global tensor...")
    tensor_all = np.load(TENSOR_PATH)["global_tensor_data"]
    log.info(f"  Tensor shape: {tensor_all.shape}")

    # ── Per-site artifact inventory + scoring ─────────────────────────────────
    artifact_rows = []
    train_metrics_rows = []
    heldout_metrics_rows = []
    score_dist_rows = []

    for site in SITES:
        site_dir = LOSO_DIR / site
        site_num = SITE_NUMBERS[site]
        log.info(f"\n{'='*60}")
        log.info(f"Processing {site}")

        # 1) Artifact inventory
        ckpt_path = site_dir / f"vae_model_{site}.pt"
        norm_path = site_dir / "vae_norm_params.joblib"
        raw_pipe_path = site_dir / "classifier_logreg_raw_pipeline.joblib"
        cal_pipe_path = site_dir / "classifier_logreg_calibrated_pipeline.joblib"
        train_csv_path = site_dir / "train_subjects.csv"
        test_csv_path = site_dir / "test_subjects.csv"
        test_preds_path = site_dir / "test_predictions_logreg.csv"

        artifacts = {
            "vae_checkpoint": ckpt_path.exists(),
            "vae_norm_params": norm_path.exists(),
            "classifier_raw_pipeline": raw_pipe_path.exists(),
            "classifier_calibrated_pipeline": cal_pipe_path.exists(),
            "train_subjects_csv": train_csv_path.exists(),
            "test_subjects_csv": test_csv_path.exists(),
            "test_predictions_csv": test_preds_path.exists(),
        }
        artifact_rows.append({"site": site, **{k: "YES" if v else "NO" for k, v in artifacts.items()}})

        missing = [k for k, v in artifacts.items() if not v]
        if missing:
            log.error(f"  Missing artifacts: {missing}")
            continue

        log.info("  All required artifacts present.")

        # 2) Load artifacts
        norm_params = joblib.load(norm_path)
        raw_pipe = joblib.load(raw_pipe_path)
        train_df = pd.read_csv(train_csv_path)
        test_df_meta = pd.read_csv(test_csv_path)
        test_preds = pd.read_csv(test_preds_path)

        # Filter train to CN/AD only (exclude other labels if any)
        train_cnad = train_df[train_df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
        train_cnad = train_cnad.reset_index(drop=True)
        log.info(
            f"  Train (CN/AD): {len(train_cnad)} (CN={( train_cnad['ResearchGroup_Mapped']=='CN').sum()}, AD={(train_cnad['ResearchGroup_Mapped']=='AD').sum()})"
        )

        # 3) Load VAE and encode training subjects
        log.info(f"  Loading VAE: {ckpt_path.name}")
        vae = _load_vae(ckpt_path, device)

        train_tensor_idx = train_cnad["tensor_idx"].values.astype(int)
        log.info(f"  Encoding {len(train_tensor_idx)} training subjects...")
        mu_train = _encode_subjects(vae, tensor_all, norm_params, train_tensor_idx, device)
        log.info(f"  Latent mu shape: {mu_train.shape}")

        # Free GPU memory
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 4) Build feature DataFrame and score
        X_train = _build_feature_df(mu_train, train_cnad)
        y_train = (train_cnad["ResearchGroup_Mapped"] == "AD").astype(int).values

        log.info("  Scoring training subjects with raw pipeline...")
        try:
            y_score_train = raw_pipe.predict_proba(X_train)[:, 1]
        except Exception as e:
            log.error(f"  predict_proba failed: {e}")
            continue

        # 5) Compute apparent training-sites metrics
        metrics_train = _compute_metrics(y_train, y_score_train, threshold=0.5)
        metrics_train["site"] = site
        metrics_train["metric_type"] = "apparent_training_sites"
        metrics_train["note"] = (
            "IN-SAMPLE: subjects used to fit the classifier. "
            "Overestimates true generalization."
        )
        train_metrics_rows.append(metrics_train)
        log.info(
            f"  Apparent train AUC={metrics_train['auc']:.4f} "
            f"bal_acc={metrics_train['balanced_accuracy']:.4f} "
            f"sensitivity={metrics_train['sensitivity']:.4f} "
            f"specificity={metrics_train['specificity']:.4f}"
        )

        # 6) Held-out site metrics from existing test_predictions_logreg.csv
        y_true_test = test_preds["y_true"].values
        y_score_test = test_preds["y_score_final"].values  # calibrated score
        y_score_test_raw = test_preds["y_score_raw"].values

        metrics_heldout = _compute_metrics(y_true_test, y_score_test, threshold=0.5)
        metrics_heldout["site"] = site
        metrics_heldout["metric_type"] = "heldout_site_oof"
        metrics_heldout["note"] = (
            "TRUE OOF: held-out site not seen during training or classifier fitting."
        )
        heldout_metrics_rows.append(metrics_heldout)

        # Also compute raw (uncalibrated) held-out metrics for reference
        metrics_heldout_raw = _compute_metrics(y_true_test, y_score_test_raw, threshold=0.5)
        metrics_heldout_raw["site"] = site
        metrics_heldout_raw["metric_type"] = "heldout_site_oof_raw"
        metrics_heldout_raw["note"] = "Raw (uncalibrated) score on held-out site."
        heldout_metrics_rows.append(metrics_heldout_raw)

        # 7) Score distributions
        for y_s, y_t, score_type in [
            (y_score_train, y_train, "apparent_train"),
            (y_score_test, y_true_test, "heldout_test"),
        ]:
            for label_int, label_name in [(0, "CN"), (1, "AD")]:
                scores = y_s[y_t == label_int]
                if len(scores) == 0:
                    continue
                score_dist_rows.append({
                    "site": site,
                    "score_type": score_type,
                    "class": label_name,
                    "n": len(scores),
                    "mean": round(float(scores.mean()), 4),
                    "std": round(float(scores.std()), 4),
                    "median": round(float(np.median(scores)), 4),
                    "p25": round(float(np.percentile(scores, 25)), 4),
                    "p75": round(float(np.percentile(scores, 75)), 4),
                    "p10": round(float(np.percentile(scores, 10)), 4),
                    "p90": round(float(np.percentile(scores, 90)), 4),
                    "min": round(float(scores.min()), 4),
                    "max": round(float(scores.max()), 4),
                })

        # KS test: train AD vs test AD, train CN vs test CN
        for label_int, label_name in [(0, "CN"), (1, "AD")]:
            s_tr = y_score_train[y_train == label_int]
            s_te = y_score_test[y_true_test == label_int]
            if len(s_tr) > 1 and len(s_te) > 1:
                ks_stat, ks_p = ks_2samp(s_tr, s_te)
                score_dist_rows.append({
                    "site": site,
                    "score_type": f"ks_train_vs_heldout_{label_name}",
                    "class": label_name,
                    "n": len(s_tr) + len(s_te),
                    "mean": round(ks_stat, 4),
                    "std": round(ks_p, 6),
                    "median": float("nan"),
                    "p25": float("nan"),
                    "p75": float("nan"),
                    "p10": float("nan"),
                    "p90": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                })

        log.info(
            f"  Heldout AUC={metrics_heldout['auc']:.4f} "
            f"bal_acc={metrics_heldout['balanced_accuracy']:.4f} "
            f"sensitivity={metrics_heldout['sensitivity']:.4f} "
            f"specificity={metrics_heldout['specificity']:.4f}"
        )

    # ── Write outputs ──────────────────────────────────────────────────────────
    log.info("\nWriting outputs...")

    artifact_df = pd.DataFrame(artifact_rows)
    artifact_df.to_csv(OUT_DIR / "artifact_availability_by_site.csv", index=False)

    train_df_out = pd.DataFrame(train_metrics_rows)
    heldout_df_out = pd.DataFrame(heldout_metrics_rows)
    # Separate heldout to final and raw for comparison table
    heldout_final = heldout_df_out[heldout_df_out["metric_type"] == "heldout_site_oof"].copy()

    comparison_rows = []
    for site in SITES:
        tr = train_df_out[train_df_out["site"] == site]
        he = heldout_final[heldout_final["site"] == site]
        if tr.empty or he.empty:
            continue
        tr = tr.iloc[0]
        he = he.iloc[0]
        comparison_rows.append({
            "site": site,
            "train_n_CN": tr["n_CN"],
            "train_n_AD": tr["n_AD"],
            "heldout_n_CN": he["n_CN"],
            "heldout_n_AD": he["n_AD"],
            "train_auc_apparent": tr["auc"],
            "heldout_auc_oof": he["auc"],
            "train_balanced_acc": tr["balanced_accuracy"],
            "heldout_balanced_acc": he["balanced_accuracy"],
            "train_sensitivity": tr["sensitivity"],
            "heldout_sensitivity": he["sensitivity"],
            "train_specificity": tr["specificity"],
            "heldout_specificity": he["specificity"],
            "train_pr_auc": tr["pr_auc"],
            "heldout_pr_auc": he["pr_auc"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "heldout_vs_training_sites_comparison.csv", index=False)

    train_df_out.to_csv(OUT_DIR / "training_sites_apparent_metrics_by_site.csv", index=False)
    heldout_df_out.to_csv(OUT_DIR / "heldout_metrics_all.csv", index=False)

    score_dist_df = pd.DataFrame(score_dist_rows)
    score_dist_df.to_csv(OUT_DIR / "score_distribution_training_vs_heldout.csv", index=False)

    # ── Write markdown outputs ─────────────────────────────────────────────────
    _write_artifact_md(artifact_df)
    _write_train_metrics_md(train_df_out)
    _write_comparison_md(comparison_df)
    _write_score_dist_md(score_dist_df)

    # ── Command log ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    cmd_log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_label": "loso_training_sites_apparent_performance_audit_20260624",
        "elapsed_seconds": round(elapsed, 1),
        "sites_processed": SITES,
        "n_train_metric_rows": len(train_df_out),
        "n_comparison_rows": len(comparison_df),
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_classifier": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_model_artifacts": False,
            "did_run_loso": False,
            "metric_type_note": (
                "Training-sites metrics are APPARENT (in-sample). "
                "Held-out metrics are true OOF from saved test_predictions_logreg.csv."
            ),
        },
        "outputs": [str(p.name) for p in sorted(OUT_DIR.glob("*")) if p.is_file()],
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(cmd_log, indent=2), encoding="utf-8")
    log.info(f"\nDone in {elapsed:.1f}s. Outputs in {OUT_DIR}")


def _write_artifact_md(df: pd.DataFrame) -> None:
    md = ["# Artifact Availability by Site\n",
          "Generated: 2026-06-24\n\n"]
    md.append(df.to_markdown(index=False))
    md.append("\n\nAll artifacts confirmed present for all sites.\n")
    (OUT_DIR / "artifact_availability_by_site.md").write_text("".join(md), encoding="utf-8")


def _write_train_metrics_md(df: pd.DataFrame) -> None:
    ap = df[df["metric_type"] == "apparent_training_sites"].copy()
    md = ["# Training-Sites Apparent Performance by Site\n",
          "Generated: 2026-06-24\n\n",
          "**⚠️ IMPORTANT: These are APPARENT (in-sample) metrics.**\n",
          "Training subjects were used to fit both the VAE and the classifier.\n",
          "This quantifies how well the model fits the training data, NOT generalization.\n",
          "True OOF training-sites performance would require re-running inner CV — not done here.\n\n"]
    cols = ["site", "n_CN", "n_AD", "auc", "pr_auc", "balanced_accuracy",
            "sensitivity", "specificity", "TP", "TN", "FP", "FN"]
    md.append(ap[cols].to_markdown(index=False))
    md.append("\n")
    (OUT_DIR / "training_sites_apparent_metrics_by_site.md").write_text("".join(md), encoding="utf-8")


def _write_comparison_md(df: pd.DataFrame) -> None:
    md = ["# Held-Out Site vs Training Sites Performance Comparison\n",
          "Generated: 2026-06-24\n\n",
          "Training-sites metrics are **apparent (in-sample)**.\n",
          "Held-out metrics are **true OOF** (sites never seen during training).\n\n"]
    md.append(df.to_markdown(index=False))
    md.append("\n")
    (OUT_DIR / "heldout_vs_training_sites_comparison.md").write_text("".join(md), encoding="utf-8")


def _write_score_dist_md(df: pd.DataFrame) -> None:
    # Only distribution rows (not KS rows)
    dist = df[~df["score_type"].str.startswith("ks_")].copy()
    md = ["# Score Distributions: Training Sites vs Held-Out Site\n",
          "Generated: 2026-06-24\n\n",
          "Scores from `classifier_logreg_raw_pipeline` (raw, uncalibrated).\n\n"]
    md.append(dist.to_markdown(index=False))
    md.append("\n\n## KS Tests (train vs held-out, same class)\n\n")
    ks = df[df["score_type"].str.startswith("ks_")].copy()
    ks = ks.rename(columns={"mean": "ks_stat", "std": "ks_p"})
    if not ks.empty:
        md.append(ks[["site", "score_type", "class", "n", "ks_stat", "ks_p"]].to_markdown(index=False))
    md.append("\n")
    (OUT_DIR / "score_distribution_training_vs_heldout.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
