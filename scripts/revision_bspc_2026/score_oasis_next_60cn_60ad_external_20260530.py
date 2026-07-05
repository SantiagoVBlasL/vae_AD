#!/usr/bin/env python
"""Score OASIS 60CN/60AD external cohort with frozen ADNI models.

3 models × 3 tensor build candidates × 2 threshold strategies.
Uses the pre-locked OASIS 30CN+30AD calibration / 30CN+30AD test split.

Models
------
1. v5_1b_locked_raw   – adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate
                        saved logreg raw pipeline; ADNI threshold derived from
                        per-fold OOF raw scores (sens ≥ 0.70, max spec).
2. recover035_raw     – recover035_latent384_beta3p75_T80_h10000_p560_full5x5
                        saved logreg raw pipeline; ADNI threshold from
                        classifier_only_readout classifier_sweep_thresholds_by_fold.csv.
3. recover035_oof_logitz – same VAE + pipeline, but scores passed through
                        OOF-logitz calibration (logit-standardise-sigmoid);
                        calibration params from stageB_oof_score_calibration/calib_meta.csv.

Threshold strategies
--------------------
• adni_fixed  – ADNI inner-OOF threshold transferred directly (no OASIS data used).
• oasis_cal   – threshold fit on OASIS calibration set only (sens ≥ 0.70 max spec).
                Locked test labels NEVER used.

Outputs (locked test only)
--------------------------
metrics_locked_test.csv, predictions_locked_test.csv,
predictions_calibration.csv, score_distribution.csv,
calibration_curve_{model}_{candidate}.png, command_log.json, README.md.

Usage
-----
    python score_oasis_next_60cn_60ad_external_20260530.py            # dry-run
    python score_oasis_next_60cn_60ad_external_20260530.py --confirm-score
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
for _p in [str(SCRIPT_DIR), str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from betavae_xai.data.preprocessing import apply_normalization_params  # noqa: E402

# Lazy import – only needed for VAE encoding
def _import_vae_helpers():
    from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (
        encode_mu,
        make_model,
    )
    return encode_mu, make_model


RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"

ADNI_V5_1B_RUN = RESULTS_DIR / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
RECOVER035_RUN = RESULTS_DIR / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
RECOVER035_CALIB_DIR = RESULTS_DIR / "recover035_latent384_beta3p75_stageB_oof_score_calibration"

DEFAULT_TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_20260530"
DEFAULT_SPLIT_CSV = RESULTS_DIR / "oasis_60cn_60ad_calibration_test_protocol" / "split_calibration_test.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_20260530"

SELECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]

TENSOR_CANDIDATES = {
    "concatenated_timeseries": "tensor_concatenated_timeseries.npz",
    "runwise_140TR_connectome_average": "tensor_runwise_140TR_connectome_average.npz",
    "runwise164_connectome_average": "tensor_runwise164_connectome_average.npz",
}

N_OUTER_FOLDS = 5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
ADNI_THRESHOLD_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"
OOF_LOGITZ_EPS = 1e-7

# Per-fold ADNI thresholds hard-coded from audit (see script docstring for sources)
# v5.1b: derived from fold-OOF y_score_raw in all_folds_clf_predictions_MULTI_logreg_... .csv
# using sens >= 0.70, max spec.
_V5_1B_ADNI_THRESHOLDS: Dict[int, float] = {
    1: 0.4892,
    2: 0.5559,
    3: 0.5441,
    4: 0.5113,
    5: 0.5251,
}

# recover035 raw: from classifier_only_readout/classifier_sweep_thresholds_by_fold.csv
# strategy inner_oof_target_sens_ge_0p70_max_spec
_RECOVER035_RAW_ADNI_THRESHOLDS: Dict[int, float] = {
    1: 0.403303,
    2: 0.485141,
    3: 0.491351,
    4: 0.481261,
    5: 0.448284,
}

# recover035 oof_logitz: from stageB_oof_score_calibration/calib_foldwise_metrics.csv
# model=logreg_l2_original, feature_set=z_plus_age_sex, calib_method=oof_logitz,
# threshold_strategy=inner_oof_target_sens_ge_0p70_max_spec
_RECOVER035_LOGITZ_ADNI_THRESHOLDS: Dict[int, float] = {
    1: 0.614169,
    2: 0.569592,
    3: 0.606673,
    4: 0.575836,
    5: 0.478365,
}

# recover035 oof_logitz calibration params: from calib_meta.csv
# model=logreg_l2_original, feature_set=z_plus_age_sex, calib_method=oof_logitz
_RECOVER035_LOGITZ_PARAMS: Dict[int, Tuple[float, float]] = {
    # fold: (oof_logit_mean, oof_logit_std)
    1: (-1.859970, 3.205055),
    2: (-0.167127, 0.387162),
    3: (-0.196003, 0.397396),
    4: (-0.203518, 0.422902),
    5: (-0.175009, 0.348853),
}


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    label: str
    run_dir: Path
    calib_method: str           # "raw" | "oof_logitz"
    adni_thresholds: Dict[int, float]   # per-fold
    feature_set: str = "z_plus_age_sex"


MODEL_SPECS = [
    ModelSpec(
        label="v5_1b_locked_raw",
        run_dir=ADNI_V5_1B_RUN,
        calib_method="raw",
        adni_thresholds=_V5_1B_ADNI_THRESHOLDS,
    ),
    ModelSpec(
        label="recover035_raw",
        run_dir=RECOVER035_RUN,
        calib_method="raw",
        adni_thresholds=_RECOVER035_RAW_ADNI_THRESHOLDS,
    ),
    ModelSpec(
        label="recover035_oof_logitz",
        run_dir=RECOVER035_RUN,
        calib_method="oof_logitz",
        adni_thresholds=_RECOVER035_LOGITZ_ADNI_THRESHOLDS,
    ),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--confirm-score", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument(
        "--candidates",
        nargs="+",
        default=list(TENSOR_CANDIDATES.keys()),
        choices=list(TENSOR_CANDIDATES.keys()),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing [{label}]: {path}")


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_sex(value: Any) -> str:
    s = str(value).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split(split_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require(split_csv, "calibration-test split CSV")
    df = pd.read_csv(split_csv)
    cal = df[df["protocol_subset"] == "calibration"].copy()
    test = df[df["protocol_subset"] == "locked_test"].copy()
    assert len(cal) > 0, "No calibration subjects in split file"
    assert len(test) > 0, "No locked_test subjects in split file"
    print(f"Split: {len(cal)} calibration ({(cal['diagnosis']=='CN').sum()} CN / "
          f"{(cal['diagnosis'].isin(['AD','AD_DEMENTIA'])).sum()} AD), "
          f"{len(test)} locked test "
          f"({(test['diagnosis']=='CN').sum()} CN / "
          f"{(test['diagnosis'].isin(['AD','AD_DEMENTIA'])).sum()} AD)")
    return cal, test


def load_tensor_candidate(tensor_dir: Path, candidate: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Load NPZ and merge with subject_manifest. Returns (tensor, subjects_df, channel_names)."""
    npz_path = tensor_dir / TENSOR_CANDIDATES[candidate]
    require(npz_path, f"OASIS tensor ({candidate})")
    data = np.load(npz_path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    channel_names = data["channel_names"].astype(str).tolist()
    subjects = pd.DataFrame({
        "SubjectID": data["subject_ids"].astype(str),
        "session_id": data["session_ids"].astype(str),
        "experiment_id": data["experiment_ids"].astype(str),
        "diagnosis": data["diagnosis"].astype(str),
    })
    manifest_path = tensor_dir / "subject_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path).rename(columns={"subject_id": "SubjectID"})
        keep_cols = [c for c in ["SubjectID", "session_id", "age_at_MR", "sex"]
                     if c in manifest.columns]
        subjects = subjects.merge(
            manifest[keep_cols].drop_duplicates(["SubjectID", "session_id"]),
            on=["SubjectID", "session_id"], how="left",
        )
    subjects["y_true"] = subjects["diagnosis"].map(
        {"CN": 0, "AD_DEMENTIA": 1, "AD": 1}
    ).astype(int)
    subjects["Age"] = pd.to_numeric(subjects.get("age_at_MR"), errors="coerce")
    subjects["Sex"] = subjects.get("sex", pd.Series(["UNKNOWN"] * len(subjects))).map(normalize_sex)
    assert tensor.shape[0] == len(subjects), (
        f"Tensor/subjects mismatch: {tensor.shape[0]} vs {len(subjects)}"
    )
    print(f"  [{candidate}] Tensor: {tensor.shape}, channels: {channel_names}")
    return tensor, subjects, channel_names


def select_channels(
    tensor: np.ndarray,
    oasis_names: Sequence[str],
    target_names: Sequence[str],
) -> np.ndarray:
    idx_map = {name: i for i, name in enumerate(oasis_names)}
    try:
        indices = [idx_map[n] for n in target_names]
    except KeyError as exc:
        raise KeyError(f"Channel {exc} not found in OASIS tensor. Available: {oasis_names}") from exc
    return tensor[:, indices, :, :]


def align_subjects_to_split(
    subjects: pd.DataFrame,
    split_subset: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return tensor row indices and metadata for subjects matching the split subset."""
    if "subject_id" not in split_subset.columns:
        raise KeyError("split_subset must contain column 'subject_id'")
    missing_recommended = [c for c in ["age_at_MR", "sex"] if c not in split_subset.columns]
    if missing_recommended:
        raise KeyError(
            "split_subset must contain age_at_MR and sex for z_plus_age_sex scoring; "
            f"missing columns: {missing_recommended}"
        )

    split_ids = set(split_subset["subject_id"].astype(str))
    mask = subjects["SubjectID"].isin(split_ids)
    idx = np.where(mask)[0]
    sub_meta = subjects.iloc[idx].reset_index(drop=True)

    missing = sorted(split_ids - set(sub_meta["SubjectID"].astype(str)))
    if missing:
        raise RuntimeError(
            f"{len(missing)} split subjects were not found in tensor subjects: {missing[:20]}"
        )

    # Merge split metadata after removing stale or duplicate metadata columns.
    # This avoids age_at_MR_x / age_at_MR_y suffixes and guarantees Age/Sex are
    # derived from the locked calibration/test split file, not tensor-manifest
    # leftovers.
    drop_cols = [
        "age_at_MR",
        "sex",
        "Manufacturer",
        "ScannerModel",
        "Age",
        "Sex",
    ]
    sub_meta = sub_meta.drop(columns=[c for c in drop_cols if c in sub_meta.columns], errors="ignore")
    merge_from_split = split_subset[
        [c for c in ["subject_id", "protocol_subset", "age_at_MR", "sex", "Manufacturer", "ScannerModel"]
         if c in split_subset.columns]
    ].rename(columns={"subject_id": "SubjectID"})

    merge_from_split = merge_from_split.drop_duplicates("SubjectID")
    sub_meta = sub_meta.merge(merge_from_split, on="SubjectID", how="left", validate="one_to_one")

    if "age_at_MR" not in sub_meta.columns:
        raise KeyError("Merged metadata is missing 'age_at_MR'; cannot create Age.")
    if "sex" not in sub_meta.columns:
        raise KeyError("Merged metadata is missing 'sex'; cannot create Sex.")
    sub_meta["Age"] = pd.to_numeric(sub_meta["age_at_MR"], errors="coerce")
    sub_meta["Sex"] = sub_meta["sex"].map(normalize_sex)

    bad_age = sub_meta[sub_meta["Age"].isna()]["SubjectID"].astype(str).tolist()
    bad_sex = sub_meta[sub_meta["Sex"].isna() | sub_meta["Sex"].eq("UNKNOWN")]["SubjectID"].astype(str).tolist()
    if bad_age or bad_sex:
        pieces = []
        if bad_age:
            pieces.append(f"missing/non-numeric Age for subjects: {bad_age[:30]}")
        if bad_sex:
            pieces.append(f"missing/unknown Sex for subjects: {bad_sex[:30]}")
        raise RuntimeError("OASIS metadata alignment failed: " + " | ".join(pieces))
    return idx, sub_meta


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _load_run_config_latent_dim(run_dir: Path) -> int:
    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    args = cfg.get("args", {})
    return int(args.get("latent_dim", cfg.get("latent_dim", 256)))


def apply_oof_logitz(
    raw_scores: np.ndarray,
    oof_logit_mean: float,
    oof_logit_std: float,
) -> np.ndarray:
    """Logit-transform → OOF-standardise → sigmoid."""
    s = np.clip(raw_scores, OOF_LOGITZ_EPS, 1.0 - OOF_LOGITZ_EPS)
    logit = np.log(s / (1.0 - s))
    logit_std = max(abs(oof_logit_std), OOF_LOGITZ_EPS)
    standardised = (logit - oof_logit_mean) / logit_std
    return 1.0 / (1.0 + np.exp(-standardised))


def encode_and_score_fold(
    run_dir: Path,
    fold: int,
    x_subset: np.ndarray,
    subjects_meta: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    latent_dim: int,
) -> np.ndarray:
    """Encode subjects with fold VAE → apply saved logreg → return raw probabilities."""
    encode_mu, make_model = _import_vae_helpers()

    fold_dir = run_dir / f"fold_{fold}"
    checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
    norm_path = fold_dir / f"vae_norm_params.joblib"
    pipeline_path = fold_dir / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"
    feature_cols_path = fold_dir / "feature_columns.json"

    require(checkpoint, f"{run_dir.name} fold {fold} VAE checkpoint")
    require(norm_path, f"{run_dir.name} fold {fold} norm params")
    require(pipeline_path, f"{run_dir.name} fold {fold} raw pipeline")

    # Normalise OASIS tensor with ADNI fold norm params
    norm_params = joblib.load(norm_path)
    x_norm = apply_normalization_params(x_subset, norm_params)

    # Load config for architecture
    cfg_path = run_dir / "run_config.json"
    cfg_raw = json.loads(cfg_path.read_text())
    cfg_args = cfg_raw.get("args", {})

    # Build VAE and encode
    cfg_for_model = {
        "latent_dim": latent_dim,
        "dropout_rate_vae": float(cfg_args.get("dropout_rate_vae", 0.15)),
        "vae_final_activation": str(cfg_args.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_vae": cfg_args.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(cfg_args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(cfg_args.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(cfg_args.get("decoder_type", "convtranspose")),
        "encoder_norm_mode": str(cfg_args.get("encoder_norm_mode", "groupnorm")),
        "vae_dropout_scope": str(cfg_args.get("vae_dropout_scope", "legacy_all")),
        "vae_block_order": str(cfg_args.get("vae_block_order", "legacy_act_norm")),
        "vae_conditioning_mode": "none",
        "vae_conditioning_vars": "none",
    }
    model = make_model(cfg_for_model, image_size=x_norm.shape[-1], n_channels=x_norm.shape[1], device=device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    mu = encode_mu(model, x_norm, batch_size=batch_size, device=device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Build feature DataFrame
    feature_cols = json.loads(feature_cols_path.read_text())["final_feature_columns"]
    latent_cols = [c for c in feature_cols if c.startswith("latent_")]
    assert len(latent_cols) == latent_dim, (
        f"Latent dim mismatch: expected {latent_dim}, got {len(latent_cols)}"
    )
    feat_df = pd.DataFrame(mu, columns=latent_cols)
    feat_df["Age"] = subjects_meta["Age"].values
    feat_df["Sex"] = subjects_meta["Sex"].values
    # Ensure exact column order expected by pipeline
    feat_df = feat_df[feature_cols]

    # Apply saved raw logreg pipeline
    pipeline = joblib.load(pipeline_path)
    raw_scores = pipeline.predict_proba(feat_df)[:, 1].astype(float)
    return raw_scores


def score_subjects_all_folds(
    spec: ModelSpec,
    x_subset: np.ndarray,
    subjects_meta: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """Score OASIS subset with all 5 folds; return per-fold + ensemble rows."""
    latent_dim = _load_run_config_latent_dim(spec.run_dir)
    fold_scores: Dict[int, np.ndarray] = {}
    fold_cal_scores: Dict[int, np.ndarray] = {}

    for fold in range(1, N_OUTER_FOLDS + 1):
        raw = encode_and_score_fold(
            spec.run_dir, fold, x_subset, subjects_meta, batch_size, device, latent_dim
        )
        fold_scores[fold] = raw
        if spec.calib_method == "oof_logitz":
            mean, std = _RECOVER035_LOGITZ_PARAMS[fold]
            cal = apply_oof_logitz(raw, mean, std)
        else:
            cal = raw.copy()
        fold_cal_scores[fold] = cal

    n = len(subjects_meta)
    rows = []
    for fold in range(1, N_OUTER_FOLDS + 1):
        for i in range(n):
            rows.append({
                "SubjectID": subjects_meta.iloc[i]["SubjectID"],
                "session_id": subjects_meta.iloc[i].get("session_id", ""),
                "diagnosis": subjects_meta.iloc[i]["diagnosis"],
                "y_true": int(subjects_meta.iloc[i]["y_true"]),
                "Age": subjects_meta.iloc[i].get("Age", np.nan),
                "Sex": subjects_meta.iloc[i].get("Sex", "UNKNOWN"),
                "model_label": spec.label,
                "fold": fold,
                "y_score_raw": float(fold_scores[fold][i]),
                "y_score": float(fold_cal_scores[fold][i]),
            })

    fold_df = pd.DataFrame(rows)

    # Ensemble: mean calibrated score per subject
    ens_rows = []
    for subj_id, grp in fold_df.groupby("SubjectID", sort=False):
        base = grp.iloc[0].to_dict()
        base["fold"] = "ensemble"
        base["y_score_raw"] = float(grp["y_score_raw"].mean())
        base["y_score"] = float(grp["y_score"].mean())
        ens_rows.append(base)
    ens_df = pd.DataFrame(ens_rows)

    return pd.concat([fold_df, ens_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def _find_threshold_sens_ge_070_max_spec(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    thresholds = np.sort(np.unique(y_score))[::-1]
    best_t, best_spec = float(thresholds[-1]) - 1e-6, -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if sens >= 0.70 and spec > best_spec:
            best_spec = spec
            best_t = float(t)
    return best_t


def fit_oasis_calibration_threshold(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
) -> float:
    """Fit threshold on OASIS calibration set (sens >= 0.70, max spec)."""
    return _find_threshold_sens_ge_070_max_spec(cal_labels, cal_scores)


def ensemble_adni_threshold(spec: ModelSpec) -> float:
    """Mean per-fold ADNI threshold as the ensemble ADNI threshold."""
    return float(np.mean(list(spec.adni_thresholds.values())))


# ---------------------------------------------------------------------------
# Metrics + Bootstrap
# ---------------------------------------------------------------------------
def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan
    pr_auc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "n": n,
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
        "accuracy": float((tp + tn) / n) if n else np.nan,
        "auc": auc,
        "pr_auc": pr_auc,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boots.append(float(metric_fn(yt, ys)))
        except Exception:
            continue
    if not boots:
        return point, np.nan, np.nan
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Calibration curve plot
# ---------------------------------------------------------------------------
def plot_calibration_curve_figure(
    y_true_cal: np.ndarray,
    y_score_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_score_test: np.ndarray,
    model_label: str,
    build_candidate: str,
    adni_threshold: float,
    oasis_cal_threshold: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (yt, ys, subset_name) in zip(
        axes,
        [(y_true_cal, y_score_cal, "OASIS Calibration (30+30)"),
         (y_true_test, y_score_test, "OASIS Locked Test (30+30)")],
    ):
        try:
            frac_pos, mean_pred = calibration_curve(yt, ys, n_bins=8, strategy="uniform")
            ax.plot(mean_pred, frac_pos, "s-", label="Model")
        except Exception:
            pass
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(f"{subset_name}")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{model_label} | {build_candidate}\n"
        f"ADNI thr={adni_threshold:.3f}  OASIS-cal thr={oasis_cal_threshold:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------
def validate_artifacts() -> Tuple[List[Dict[str, str]], bool]:
    checks = []
    ok = True
    # Tensor candidates
    for cand, fname in TENSOR_CANDIDATES.items():
        p = DEFAULT_TENSOR_DIR / fname
        exists = p.exists()
        if not exists:
            ok = False
        checks.append({"artifact": f"tensor_{cand}", "path": str(p), "exists": str(exists)})
    # Split file
    p = DEFAULT_SPLIT_CSV
    exists = p.exists()
    if not exists:
        ok = False
    checks.append({"artifact": "split_csv", "path": str(p), "exists": str(exists)})
    # Model checkpoints
    for spec in MODEL_SPECS:
        for fold in range(1, N_OUTER_FOLDS + 1):
            fold_dir = spec.run_dir / f"fold_{fold}"
            for name, fname in [
                ("vae", f"vae_model_fold_{fold}.pt"),
                ("norm", f"vae_norm_params.joblib"),
                ("pipeline", f"classifier_logreg_raw_pipeline_fold_{fold}.joblib"),
                ("feature_cols", "feature_columns.json"),
            ]:
                p = fold_dir / fname
                exists = p.exists()
                if not exists:
                    ok = False
                checks.append({
                    "artifact": f"{spec.label}_fold{fold}_{name}",
                    "path": str(p),
                    "exists": str(exists),
                })
    # OOF-logitz calib meta
    p = RECOVER035_CALIB_DIR / "calib_meta.csv"
    exists = p.exists()
    if not exists:
        ok = False
    checks.append({"artifact": "oof_logitz_calib_meta", "path": str(p), "exists": str(exists)})
    return checks, ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"OASIS 60CN/60AD External Scoring | device={device} | confirm={args.confirm_score}")
    print(f"Tensor dir : {args.tensor_dir}")
    print(f"Split CSV  : {args.split_csv}")
    print(f"Output dir : {output_dir}")
    print(f"Candidates : {args.candidates}")

    # Validate artifacts
    checks, all_ok = validate_artifacts()
    checks_df = pd.DataFrame(checks)
    write_csv_md(
        checks_df,
        output_dir / "artifact_validation.csv",
        output_dir / "artifact_validation.md",
        "Artifact Validation",
    )
    missing = [c for c in checks if c["exists"] == "False"]
    if missing:
        print(f"\nMissing artifacts ({len(missing)}):")
        for m in missing:
            print(f"  MISSING: {m['path']}")
        if not args.confirm_score:
            print("\nDry-run: missing artifacts flagged. Add --confirm-score to abort on missing.")

    if not args.confirm_score:
        print("\nDry-run complete. No scoring performed. Add --confirm-score to execute.")
        write_json(output_dir / "command_log.json", {
            "script": str(Path(__file__).resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "dry_run",
            "confirm_score": False,
            "n_missing_artifacts": len(missing),
            "candidates": args.candidates,
            "models": [s.label for s in MODEL_SPECS],
        })
        return 0

    if missing:
        print(f"\nERROR: {len(missing)} artifacts missing. Cannot score.")
        return 1

    # Load split
    cal_split, test_split = load_split(args.split_csv)

    all_metrics: List[Dict[str, Any]] = []
    all_predictions_cal: List[pd.DataFrame] = []
    all_predictions_test: List[pd.DataFrame] = []
    all_score_dist: List[Dict[str, Any]] = []
    metadata_alignment_rows: List[pd.DataFrame] = []

    for candidate in args.candidates:
        print(f"\n=== Tensor candidate: {candidate} ===")
        tensor, subjects, channel_names = load_tensor_candidate(args.tensor_dir, candidate)

        # Select channel subset for all models (same 3 channels)
        x_all = select_channels(tensor, channel_names, SELECTED_CHANNEL_NAMES)

        # Align cal / test subjects
        cal_idx, cal_meta = align_subjects_to_split(subjects, cal_split)
        test_idx, test_meta = align_subjects_to_split(subjects, test_split)
        for subset_name, aligned in [("calibration", cal_meta), ("locked_test", test_meta)]:
            audit_cols = [
                "SubjectID",
                "protocol_subset",
                "diagnosis",
                "age_at_MR",
                "Age",
                "sex",
                "Sex",
                "Manufacturer",
                "ScannerModel",
            ]
            audit = aligned[[c for c in audit_cols if c in aligned.columns]].copy()
            if "protocol_subset" not in audit.columns:
                audit["protocol_subset"] = subset_name
            audit.insert(0, "build_candidate", candidate)
            metadata_alignment_rows.append(audit)
        print(f"  Calibration subjects aligned: {len(cal_meta)} "
              f"({(cal_meta['y_true']==0).sum()} CN / {(cal_meta['y_true']==1).sum()} AD)")
        print(f"  Locked test subjects aligned: {len(test_meta)} "
              f"({(test_meta['y_true']==0).sum()} CN / {(test_meta['y_true']==1).sum()} AD)")

        x_cal = x_all[cal_idx]
        x_test = x_all[test_idx]

        for spec in MODEL_SPECS:
            print(f"\n  --- Model: {spec.label} ---")

            # Score calibration set
            print("    Scoring calibration set...")
            preds_cal = score_subjects_all_folds(spec, x_cal, cal_meta, args.batch_size, device)
            preds_cal["build_candidate"] = candidate
            preds_cal["split_subset"] = "calibration"
            all_predictions_cal.append(preds_cal)

            # Score locked test set
            print("    Scoring locked test set...")
            preds_test = score_subjects_all_folds(spec, x_test, test_meta, args.batch_size, device)
            preds_test["build_candidate"] = candidate
            preds_test["split_subset"] = "locked_test"
            all_predictions_test.append(preds_test)

            # Ensemble predictions only
            ens_cal = preds_cal[preds_cal["fold"] == "ensemble"].reset_index(drop=True)
            ens_test = preds_test[preds_test["fold"] == "ensemble"].reset_index(drop=True)

            y_true_cal = ens_cal["y_true"].astype(int).values
            y_score_cal = ens_cal["y_score"].values
            y_true_test = ens_test["y_true"].astype(int).values
            y_score_test = ens_test["y_score"].values

            # ADNI fixed threshold (ensemble = mean of per-fold thresholds)
            adni_thr = ensemble_adni_threshold(spec)

            # OASIS calibration threshold
            oasis_cal_thr = fit_oasis_calibration_threshold(y_score_cal, y_true_cal)

            print(f"    ADNI fixed threshold  : {adni_thr:.4f}")
            print(f"    OASIS cal threshold   : {oasis_cal_thr:.4f}")

            # Calibration curve
            plot_calibration_curve_figure(
                y_true_cal, y_score_cal,
                y_true_test, y_score_test,
                spec.label, candidate,
                adni_thr, oasis_cal_thr,
                output_dir / f"calibration_curve_{spec.label}_{candidate}.png",
            )

            # Metrics on locked test with both threshold strategies
            for thr_strategy, thr in [("adni_fixed", adni_thr), ("oasis_calibration", oasis_cal_thr)]:
                m = binary_metrics(y_true_test, y_score_test, thr)
                # Bootstrap CI
                auc_pt, auc_lo, auc_hi = bootstrap_ci(
                    y_true_test, y_score_test, roc_auc_score, args.n_bootstrap, BOOTSTRAP_SEED
                )
                pr_pt, pr_lo, pr_hi = bootstrap_ci(
                    y_true_test, y_score_test, average_precision_score, args.n_bootstrap, BOOTSTRAP_SEED
                )
                m.update({
                    "model_label": spec.label,
                    "build_candidate": candidate,
                    "threshold_strategy": thr_strategy,
                    "adni_threshold_ensemble": adni_thr,
                    "oasis_cal_threshold": oasis_cal_thr,
                    "calib_method": spec.calib_method,
                    "auc_bootstrap_mean": auc_pt,
                    "auc_ci_lo_95": auc_lo,
                    "auc_ci_hi_95": auc_hi,
                    "pr_auc_bootstrap_mean": pr_pt,
                    "pr_auc_ci_lo_95": pr_lo,
                    "pr_auc_ci_hi_95": pr_hi,
                })
                all_metrics.append(m)
                print(
                    f"    [{thr_strategy}] AUC={auc_pt:.4f} [{auc_lo:.3f}–{auc_hi:.3f}]  "
                    f"PR-AUC={pr_pt:.4f} [{pr_lo:.3f}–{pr_hi:.3f}]  "
                    f"sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}"
                )

            # Score distribution (shift)
            for subset_name, ens_df_sub in [("calibration", ens_cal), ("locked_test", ens_test)]:
                for diag in ["CN", "AD_DEMENTIA"]:
                    mask = ens_df_sub["diagnosis"] == diag
                    scores = ens_df_sub.loc[mask, "y_score"].values
                    if len(scores) == 0:
                        continue
                    all_score_dist.append({
                        "model_label": spec.label,
                        "build_candidate": candidate,
                        "split_subset": subset_name,
                        "diagnosis": diag,
                        "n": len(scores),
                        "score_mean": float(np.mean(scores)),
                        "score_std": float(np.std(scores, ddof=1)),
                        "score_min": float(np.min(scores)),
                        "score_median": float(np.median(scores)),
                        "score_max": float(np.max(scores)),
                    })

    # Write outputs
    metadata_alignment = pd.concat(metadata_alignment_rows, ignore_index=True) if metadata_alignment_rows else pd.DataFrame()
    if not metadata_alignment.empty:
        metadata_alignment.to_csv(output_dir / "metadata_alignment_audit.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    col_order = [
        "model_label", "build_candidate", "threshold_strategy", "calib_method",
        "n", "n_cn", "n_ad", "threshold",
        "auc", "auc_ci_lo_95", "auc_ci_hi_95",
        "pr_auc", "pr_auc_ci_lo_95", "pr_auc_ci_hi_95",
        "sensitivity", "specificity", "balanced_accuracy", "f1",
        "tn", "fp", "fn", "tp",
        "adni_threshold_ensemble", "oasis_cal_threshold",
    ]
    col_order = [c for c in col_order if c in metrics_df.columns]
    metrics_df = metrics_df[col_order + [c for c in metrics_df.columns if c not in col_order]]
    write_csv_md(
        metrics_df,
        output_dir / "metrics_locked_test.csv",
        output_dir / "metrics_locked_test.md",
        "OASIS External Scoring — Locked Test Metrics",
    )

    preds_cal_df = pd.concat(all_predictions_cal, ignore_index=True)
    preds_test_df = pd.concat(all_predictions_test, ignore_index=True)
    preds_cal_df.to_csv(output_dir / "predictions_calibration.csv", index=False)
    preds_test_df.to_csv(output_dir / "predictions_locked_test.csv", index=False)

    score_dist_df = pd.DataFrame(all_score_dist)
    write_csv_md(
        score_dist_df,
        output_dir / "score_distribution.csv",
        output_dir / "score_distribution.md",
        "Score Distribution (Calibration vs Locked Test)",
    )

    # README
    readme_lines = [
        "# OASIS 60CN/60AD External Scoring Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Models scored",
        "",
        "| Model | Run | Calib method |",
        "|-------|-----|-------------|",
    ]
    for spec in MODEL_SPECS:
        readme_lines.append(f"| {spec.label} | {spec.run_dir.name} | {spec.calib_method} |")
    readme_lines += [
        "",
        "## Tensor build candidates",
        "",
        "| Candidate | File |",
        "|-----------|------|",
    ]
    for cand, fname in TENSOR_CANDIDATES.items():
        readme_lines.append(f"| {cand} | {fname} |")
    readme_lines += [
        "",
        "## Key metrics (locked test, AUC)",
        "",
        "See `metrics_locked_test.csv`.",
        "",
        "## Safety",
        "",
        "- OASIS subjects were never used for model training or architecture selection.",
        "- Only the calibration threshold strategy (`oasis_calibration`) uses OASIS data.",
        "- Locked test labels were never used for threshold fitting or model selection.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n")

    # Command log
    write_json(output_dir / "command_log.json", {
        "script": str(Path(__file__).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "scoring",
        "confirm_score": True,
        "device": str(device),
        "n_bootstrap": args.n_bootstrap,
        "candidates": args.candidates,
        "models": [s.label for s in MODEL_SPECS],
        "tensor_dir": str(args.tensor_dir),
        "split_csv": str(args.split_csv),
        "output_dir": str(output_dir),
        "safety": {
            "oasis_training": False,
            "locked_test_used_for_threshold": False,
            "oasis_merged_with_adni": False,
        },
    })

    print(f"\nDone. Results in: {output_dir}")
    print("\nKey results (locked test, AUC):")
    summary = metrics_df[metrics_df["threshold_strategy"] == "adni_fixed"][
        ["model_label", "build_candidate", "auc", "auc_ci_lo_95", "auc_ci_hi_95",
         "pr_auc", "sensitivity", "specificity"]
    ]
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
