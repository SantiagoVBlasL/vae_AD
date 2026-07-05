#!/usr/bin/env python3
"""
Score OASIS pilot_parity runwise tensors using the ADNI horizon4480 Stage-B classifier ensemble.

Model  : adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5
         latent_dim=256, logreg_l2, z_plus_age_sex, 5 outer folds
Tensors: runwise_140TR_pilot_parity, runwise164_pilot_parity
         from oasis_next_60cn_60ad_tensor_build_pilot_parity_runwise_20260531/
Split  : oasis_60cn_60ad_calibration_test_protocol/split_calibration_test.csv
         calibration 30CN+30AD | locked_test 30CN+30AD

Thresholds reported
  adni_fixed           mean of per-fold inner_oof_target_sens_ge_0p70_max_spec thresholds
  oasis_calib_youden   Youden-J fitted on calibration set, applied to locked_test
  oasis_calib_target   target_sens>=0.70 max_spec fitted on calibration set, applied to locked_test

Constraints
  - No training
  - No OASIS model selection
  - Locked test is the only source of final metrics
  - No threshold fitting on locked test

Usage
  python scripts/revision_bspc_2026/score_oasis_pilot_parity_runwise_horizon4480_20260531.py \\
      [--confirm-score] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from betavae_xai.data.preprocessing import apply_normalization_params
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"

ADNI_RUN_DIR = (
    RESULTS_DIR
    / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_pilot_parity_runwise_20260531"
SPLIT_CSV = RESULTS_DIR / "oasis_60cn_60ad_calibration_test_protocol" / "split_calibration_test.csv"
OUTPUT_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_pilot_parity_runwise_horizon4480_20260531"

TENSOR_CANDIDATES: Dict[str, str] = {
    "runwise_140TR_pilot_parity": "tensor_runwise_140TR_pilot_parity.npz",
    "runwise164_pilot_parity": "tensor_runwise164_pilot_parity.npz",
}

SELECTED_CHANNEL_NAMES: List[str] = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]

N_FOLDS = 5
LATENT_DIM = 256
BATCH_SIZE = 64

# Per-fold ADNI thresholds from classifier_only_readout/classifier_sweep_thresholds_by_fold.csv
# strategy=inner_oof_target_sens_ge_0p70_max_spec
_ADNI_FOLD_THRESHOLDS: Dict[int, float] = {
    1: 0.487426400288,
    2: 0.480150502682,
    3: 0.482609561734,
    4: 0.481806162274,
    5: 0.483671386853,
}
_ADNI_FIXED_THRESHOLD = float(np.mean(list(_ADNI_FOLD_THRESHOLDS.values())))

EPS = 1e-7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_sex(value: Any) -> str:
    s = str(value).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} not found: {path}")


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ba = (sens + spec) / 2
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else float("nan")
    try:
        auc = float(roc_auc_score(y_true, prob))
    except Exception:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, prob))
    except Exception:
        pr_auc = float("nan")
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "ba": ba,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": len(y_true),
        "n_ad": int(y_true.sum()),
        "n_cn": int((y_true == 0).sum()),
    }


def select_threshold_youden(y_true: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def select_threshold_target_sens(
    y_true: np.ndarray, prob: np.ndarray, min_sens: float = 0.70
) -> float:
    candidates = np.linspace(0.0, 1.0, 1001)
    best_t = 0.5
    best_spec = -1.0
    for t in candidates:
        pred = (prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if sens >= min_sens and spec > best_spec:
            best_spec = spec
            best_t = t
    return float(best_t)


# ---------------------------------------------------------------------------
# VAE inference
# ---------------------------------------------------------------------------
def _load_vae_helpers():
    from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (
        encode_mu,
        make_model,
    )
    return encode_mu, make_model


def _load_run_cfg() -> Dict[str, Any]:
    cfg_raw = json.loads((ADNI_RUN_DIR / "run_config.json").read_text())
    return cfg_raw.get("args", {})


def encode_fold(
    fold: int,
    x_oasis: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return shape (N, LATENT_DIM) mu vectors for OASIS subjects using fold VAE."""
    encode_mu, make_model = _load_vae_helpers()
    fold_dir = ADNI_RUN_DIR / f"fold_{fold}"
    checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
    norm_path = fold_dir / f"vae_norm_params.joblib"
    require(checkpoint, f"fold {fold} VAE checkpoint")
    require(norm_path, f"fold {fold} norm params")

    norm_params = joblib.load(norm_path)
    x_norm = apply_normalization_params(x_oasis, norm_params)

    cfg = _load_run_cfg()
    model_cfg = {
        "latent_dim": LATENT_DIM,
        "dropout_rate_vae": float(cfg.get("dropout_rate_vae", 0.15)),
        "vae_final_activation": str(cfg.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_vae": cfg.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(cfg.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(cfg.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(cfg.get("decoder_type", "convtranspose")),
        "encoder_norm_mode": str(cfg.get("encoder_norm_mode", "groupnorm")),
        "vae_dropout_scope": str(cfg.get("vae_dropout_scope", "legacy_all")),
        "vae_block_order": str(cfg.get("vae_block_order", "legacy_act_norm")),
        "vae_conditioning_mode": "none",
        "vae_conditioning_vars": "none",
    }
    model = make_model(model_cfg, image_size=x_norm.shape[-1], n_channels=x_norm.shape[1], device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    mu = encode_mu(model, x_norm, batch_size=BATCH_SIZE, device=device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mu


def score_fold(
    fold: int,
    mu: np.ndarray,
    subjects_meta: pd.DataFrame,
) -> np.ndarray:
    """Apply saved logreg pipeline to latent mu → return raw probabilities (N,)."""
    fold_dir = ADNI_RUN_DIR / f"fold_{fold}"
    pipeline_path = fold_dir / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
    feature_cols_path = fold_dir / "feature_columns.json"
    require(pipeline_path, f"fold {fold} logreg pipeline")
    require(feature_cols_path, f"fold {fold} feature_columns.json")

    feature_cols = json.loads(feature_cols_path.read_text())["final_feature_columns"]
    latent_cols = [c for c in feature_cols if c.startswith("latent_")]
    assert len(latent_cols) == LATENT_DIM, (
        f"Latent dim mismatch: expected {LATENT_DIM}, got {len(latent_cols)}"
    )

    feat_df = pd.DataFrame(mu, columns=latent_cols)
    feat_df["Age"] = subjects_meta["Age"].values
    feat_df["Sex"] = subjects_meta["Sex"].values
    feat_df = feat_df[feature_cols]

    pipeline = joblib.load(pipeline_path)
    return pipeline.predict_proba(feat_df)[:, 1].astype(float)


# ---------------------------------------------------------------------------
# Data loading and alignment
# ---------------------------------------------------------------------------
def load_tensor(candidate: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    npz_path = TENSOR_DIR / TENSOR_CANDIDATES[candidate]
    require(npz_path, f"tensor {candidate}")
    data = np.load(npz_path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    subject_ids = data["subject_ids"].astype(str)
    channel_names = data["channel_names"].astype(str).tolist()
    return tensor, subject_ids, channel_names


def select_channels(tensor: np.ndarray, tensor_channels: List[str], target: List[str]) -> np.ndarray:
    idx_map = {n: i for i, n in enumerate(tensor_channels)}
    try:
        indices = [idx_map[n] for n in target]
    except KeyError as exc:
        raise KeyError(f"Channel {exc} not in tensor. Available: {tensor_channels}") from exc
    return tensor[:, indices, :, :]


def load_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    require(SPLIT_CSV, "split CSV")
    df = pd.read_csv(SPLIT_CSV)
    cal = df[df["protocol_subset"] == "calibration"].copy()
    test = df[df["protocol_subset"] == "locked_test"].copy()
    assert len(cal) > 0 and len(test) > 0
    return cal, test


def align_subjects(
    tensor_subject_ids: np.ndarray,
    split_subset: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return tensor row indices and subject metadata aligned to split_subset."""
    split_ids = split_subset["subject_id"].astype(str).tolist()
    id_to_row = {sid: i for i, sid in enumerate(tensor_subject_ids)}
    missing = [s for s in split_ids if s not in id_to_row]
    if missing:
        raise RuntimeError(f"{len(missing)} split subjects not in tensor: {missing[:10]}")
    indices = np.array([id_to_row[s] for s in split_ids], dtype=int)

    meta = split_subset[["subject_id", "diagnosis", "age_at_MR", "sex"]].copy()
    meta = meta.rename(columns={"subject_id": "SubjectID"}).reset_index(drop=True)
    meta["Age"] = pd.to_numeric(meta["age_at_MR"], errors="coerce")
    meta["Sex"] = meta["sex"].map(normalize_sex)
    meta["y"] = (meta["diagnosis"].isin(["AD", "AD_DEMENTIA"])).astype(int)
    return indices, meta


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------
def run_scoring(device: torch.device) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cal_split, test_split = load_split()
    print(f"Split loaded: cal={len(cal_split)} (CN={( cal_split['diagnosis']=='CN').sum()}, "
          f"AD={cal_split['diagnosis'].isin(['AD','AD_DEMENTIA']).sum()}), "
          f"test={len(test_split)} (CN={(test_split['diagnosis']=='CN').sum()}, "
          f"AD={test_split['diagnosis'].isin(['AD','AD_DEMENTIA']).sum()})")

    all_rows: List[Dict[str, Any]] = []
    score_dfs: List[pd.DataFrame] = []

    for candidate in TENSOR_CANDIDATES:
        print(f"\n=== Tensor candidate: {candidate} ===")
        tensor, subject_ids, channel_names = load_tensor(candidate)
        print(f"  Shape: {tensor.shape}, channels: {channel_names}")

        # Select channels by name in VAE training order
        x_all = select_channels(tensor, channel_names, SELECTED_CHANNEL_NAMES)
        print(f"  After channel selection: {x_all.shape}")

        # Align calibration and test subsets
        cal_idx, cal_meta = align_subjects(subject_ids, cal_split)
        test_idx, test_meta = align_subjects(subject_ids, test_split)
        print(f"  Cal subjects: {len(cal_idx)}, Test subjects: {len(test_idx)}")

        # Encode and score with each fold
        fold_probs_all: Dict[int, np.ndarray] = {}
        for fold in range(1, N_FOLDS + 1):
            print(f"  Fold {fold}: encoding {x_all.shape[0]} subjects ...", end=" ", flush=True)
            # Encode ALL 120 subjects with this fold's VAE
            mu_all = encode_fold(fold, x_all, device)
            # Build metadata for all subjects (needed for pipeline)
            # Subject order in x_all follows tensor/subject_ids order
            # We build full metadata from cal+test split
            full_split = pd.concat([cal_split, test_split], ignore_index=True)
            all_idx, all_meta = align_subjects(subject_ids, full_split)
            # Score in tensor order to preserve alignment
            mu_ordered = mu_all[all_idx]
            probs_ordered = score_fold(fold, mu_ordered, all_meta)
            # Store indexed by original tensor position
            prob_by_pos = np.full(len(subject_ids), np.nan)
            for i, tensor_pos in enumerate(all_idx):
                prob_by_pos[tensor_pos] = probs_ordered[i]

            fold_probs_all[fold] = prob_by_pos
            print(f"done (prob range [{probs_ordered.min():.3f}, {probs_ordered.max():.3f}])")

        # Ensemble: average across folds
        prob_matrix = np.stack([fold_probs_all[f] for f in range(1, N_FOLDS + 1)], axis=1)
        # prob_matrix shape: (n_tensor_subjects, 5)
        mean_prob_all = np.nanmean(prob_matrix, axis=1)

        # Extract calibration and test ensemble probs
        cal_prob = mean_prob_all[cal_idx]
        test_prob = mean_prob_all[test_idx]
        cal_y = cal_meta["y"].values
        test_y = test_meta["y"].values

        # --- Threshold strategies ---
        # 1. ADNI fixed (mean of per-fold thresholds)
        t_adni = _ADNI_FIXED_THRESHOLD

        # 2. OASIS calibration Youden-J (fitted on cal, applied to test)
        t_calib_youden = select_threshold_youden(cal_y, cal_prob)

        # 3. OASIS calibration target_sens>=0.70 (fitted on cal, applied to test)
        t_calib_target = select_threshold_target_sens(cal_y, cal_prob, min_sens=0.70)

        thresholds = {
            "adni_fixed": t_adni,
            "oasis_calib_youden": t_calib_youden,
            "oasis_calib_target_sens": t_calib_target,
        }

        print(f"\n  Thresholds: adni_fixed={t_adni:.4f}, "
              f"calib_youden={t_calib_youden:.4f}, calib_target_sens={t_calib_target:.4f}")

        print("\n  LOCKED TEST METRICS:")
        for strategy, threshold in thresholds.items():
            m = compute_metrics(test_y, test_prob, threshold)
            print(f"    [{strategy}] AUC={m['auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                  f"BA={m['ba']:.4f} Sens={m['sensitivity']:.4f} "
                  f"Spec={m['specificity']:.4f} F1={m['f1']:.4f} "
                  f"(thr={threshold:.4f})")
            row = {
                "tensor_candidate": candidate,
                "subset": "locked_test",
                "threshold_strategy": strategy,
                **m,
            }
            if strategy != "adni_fixed":
                row["threshold_fitted_on"] = "calibration"
                row["threshold_fitted_value"] = threshold
            all_rows.append(row)

        # Also record calibration metrics (informational)
        print("  CALIBRATION METRICS (informational):")
        for strategy, threshold in thresholds.items():
            m = compute_metrics(cal_y, cal_prob, threshold)
            print(f"    [{strategy}] AUC={m['auc']:.4f} BA={m['ba']:.4f} "
                  f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}")
            row = {
                "tensor_candidate": candidate,
                "subset": "calibration",
                "threshold_strategy": strategy,
                **m,
            }
            all_rows.append(row)

        # Save per-subject scores
        score_rows = []
        for subset_name, idx, meta_df, prob in [
            ("calibration", cal_idx, cal_meta, cal_prob),
            ("locked_test", test_idx, test_meta, test_prob),
        ]:
            for i in range(len(idx)):
                tensor_pos = idx[i]
                r = {
                    "candidate": candidate,
                    "protocol_subset": subset_name,
                    "SubjectID": meta_df.iloc[i]["SubjectID"],
                    "diagnosis": meta_df.iloc[i]["diagnosis"],
                    "y": int(meta_df.iloc[i]["y"]),
                    "Age": meta_df.iloc[i]["Age"],
                    "Sex": meta_df.iloc[i]["Sex"],
                    "prob_ensemble": float(prob[i]),
                }
                for fold in range(1, N_FOLDS + 1):
                    r[f"prob_fold{fold}"] = float(fold_probs_all[fold][tensor_pos])
                score_rows.append(r)
        score_dfs.append(pd.DataFrame(score_rows))

    # Write outputs
    summary_df = pd.DataFrame(all_rows)
    summary_path = OUTPUT_DIR / "scoring_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    if score_dfs:
        all_scores = pd.concat(score_dfs, ignore_index=True)
        scores_path = OUTPUT_DIR / "subject_scores.csv"
        all_scores.to_csv(scores_path, index=False)
        print(f"Subject scores saved: {scores_path}")

    # Print final locked_test comparison table
    print("\n" + "=" * 70)
    print("FINAL LOCKED TEST SUMMARY")
    print("=" * 70)
    lt = summary_df[summary_df["subset"] == "locked_test"]
    for _, row in lt.iterrows():
        print(f"  {row['tensor_candidate']:<35s} [{row['threshold_strategy']:<30s}] "
              f"AUC={row['auc']:.4f} PR={row['pr_auc']:.4f} BA={row['ba']:.4f} "
              f"Sens={row['sensitivity']:.4f} Spec={row['specificity']:.4f} "
              f"F1={row['f1']:.4f}")

    log = {
        "script": __file__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adni_run_dir": str(ADNI_RUN_DIR),
        "tensor_dir": str(TENSOR_DIR),
        "split_csv": str(SPLIT_CSV),
        "output_dir": str(OUTPUT_DIR),
        "tensor_candidates": list(TENSOR_CANDIDATES.keys()),
        "n_folds": N_FOLDS,
        "latent_dim": LATENT_DIM,
        "adni_fixed_threshold": _ADNI_FIXED_THRESHOLD,
        "adni_per_fold_thresholds": _ADNI_FOLD_THRESHOLDS,
        "no_training": True,
        "no_oasis_model_selection": True,
        "final_metrics_subset": "locked_test",
        "threshold_fitting_subset": "calibration",
    }
    log_path = OUTPUT_DIR / "command_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"Command log: {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-score",
        action="store_true",
        help="Required flag to actually run scoring (safety guard).",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or cuda:N).")
    args = parser.parse_args()

    if not args.confirm_score:
        print("Dry-run mode: pass --confirm-score to execute scoring.")
        print("Checking required paths...")
        ok = True
        for p, label in [
            (ADNI_RUN_DIR, "ADNI run dir"),
            (TENSOR_DIR, "tensor dir"),
            (SPLIT_CSV, "split CSV"),
        ]:
            status = "OK" if p.exists() else "MISSING"
            print(f"  [{status}] {label}: {p}")
            if not p.exists():
                ok = False
        for candidate, fname in TENSOR_CANDIDATES.items():
            p = TENSOR_DIR / fname
            status = "OK" if p.exists() else "MISSING"
            print(f"  [{status}] tensor {candidate}: {p}")
            if not p.exists():
                ok = False
        for fold in range(1, N_FOLDS + 1):
            fold_dir = ADNI_RUN_DIR / f"fold_{fold}"
            for fname in [
                f"vae_model_fold_{fold}.pt",
                f"vae_norm_params.joblib",
                f"classifier_logreg_raw_pipeline_fold_{fold}.joblib",
                "feature_columns.json",
            ]:
                p = fold_dir / fname
                status = "OK" if p.exists() else "MISSING"
                if not p.exists():
                    print(f"  [{status}] {p}")
                    ok = False
        print(f"\nPre-flight {'PASSED' if ok else 'FAILED'}.")
        print(f"ADNI fixed threshold (mean of fold thresholds): {_ADNI_FIXED_THRESHOLD:.6f}")
        if ok:
            print("Re-run with --confirm-score to execute.")
        sys.exit(0 if ok else 1)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")
    run_scoring(device)


if __name__ == "__main__":
    main()
