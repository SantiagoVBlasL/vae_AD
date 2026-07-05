#!/usr/bin/env python
"""Read-only mega-OASIS 90CN/90AD external inference panel.

This audit scores the pooled OASIS pilot+new 90CN/90AD tensors with frozen
ADNI fold VAEs and the leakage-safe ADNI Stage B OOF-ECDF readout protocol.

Guardrails:
- no VAE training;
- no classifier fitting, threshold fitting, calibration fitting, or scaler
  fitting on OASIS;
- OASIS labels are used only for final metric evaluation;
- tensors, metadata, and model artifacts are not modified.

The Stage B OOF calibration package stores thresholds and prediction tables,
not serialized classifier estimators. To score external data, this script
deterministically reconstructs each fold's Stage B classifier and OOF-ECDF
mapping from ADNI train/dev latent caches only, then applies those frozen
ADNI-derived objects to OASIS.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (  # noqa: E402
    apply_conditioning_transformer,
    apply_normalization_params,
    encode_mu,
    load_config,
    make_model,
    score_1d,
)


MEGA_DIR = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"
DEFAULT_OUTPUT = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]
INNER_FOLDS = 5
FOLDS = [1, 2, 3, 4, 5]
SEED = 42

TENSORS = {
    "concatenated_timeseries": MEGA_DIR / "tensor_concatenated_timeseries.npz",
    "runwise_140TR_pilot_parity": MEGA_DIR / "tensor_runwise_140TR_pilot_parity.npz",
    "runwise164_pilot_parity": MEGA_DIR / "tensor_runwise164_pilot_parity.npz",
}


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    role: str
    run_dir: Path
    oof_dir: Path
    harmonization: str = "none"


CANDIDATES = [
    CandidateSpec(
        label="promoted_beta3p75_oof_ecdf",
        role="primary_model",
        run_dir=RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
    ),
    CandidateSpec(
        label="mfrBalancedVAE_beta3p75_oof_ecdf",
        role="external_sensitivity_only",
        run_dir=RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE",
        oof_dir=RESULTS / "recover035_latent384_beta3p75_mfrBalancedVAE_stageB_oof_score_calibration",
    ),
    CandidateSpec(
        label="latent512_beta3p75_oof_ecdf",
        role="external_sensitivity_only",
        run_dir=RESULTS / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
    ),
    CandidateSpec(
        label="promoted_beta3p75_residualized_mfr_oof_ecdf",
        role="posthoc_external_sensitivity_only",
        run_dir=RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602",
        harmonization="residualize_mfr_preserve_age_sex",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 100) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def normalize_dx(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"CN", "CONTROL", "NORMAL", "0"}:
        return "CN"
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA", "1"}:
        return "AD"
    return str(v).strip()


def normalize_sex(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return s if s else "UNKNOWN"


def normalize_mfr(v: Any) -> str:
    s = str(v).strip().replace("_", " ").replace("-", " ").upper()
    s = " ".join(s.split())
    mapping = {
        "GE": "GE",
        "G E": "GE",
        "GENERAL ELECTRIC": "GE",
        "GE MEDICAL SYSTEMS": "GE",
        "PHILIPS": "Philips",
        "PHILIPS MEDICAL SYSTEMS": "Philips",
        "PHILIPS HEALTHCARE": "Philips",
        "SIEMENS": "SIEMENS",
        "SIEMENS HEALTHCARE": "SIEMENS",
        "SIEMENS HEALTHINEERS": "SIEMENS",
        "SIEMENS MEDICAL SYSTEMS": "SIEMENS",
    }
    return mapping.get(s, str(v).strip())


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(pred.mean()) if len(pred) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str], include_age_sex: bool = True) -> ColumnTransformer:
    transformers: list = [("latent", Pipeline([("scaler", StandardScaler())]), mu_cols)]
    if include_age_sex:
        transformers.append(
            (
                "age",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ["Age"],
            )
        )
        transformers.append(
            (
                "sex",
                Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]),
                ["Sex"],
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def ensure_y(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ResearchGroup_Mapped" not in out.columns:
        raise ValueError("Latent frame missing ResearchGroup_Mapped")
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].map(normalize_dx)
    if "y" not in out.columns:
        out["y"] = out["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out[out["y"].isin([0, 1])].copy()
    out["y"] = out["y"].astype(int)
    out["Sex"] = out["Sex"].map(normalize_sex)
    out["Manufacturer"] = out["Manufacturer"].map(normalize_mfr)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    key_df = pd.DataFrame(
        {
            "dx": df["ResearchGroup_Mapped"].fillna("DX_UNKNOWN").astype(str),
            "mfr": df["Manufacturer"].fillna("MFR_UNKNOWN").astype(str),
        }
    )
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        label_key = df["y"].astype(int)
        return label_key, "label_only_fallback", int(label_key.value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def oof_ecdf_interp(oof_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_oof = np.sort(np.asarray(oof_scores, dtype=float))
    n = len(sorted_oof)
    pctiles = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(np.asarray(values, dtype=float), sorted_oof, pctiles, left=0.0, right=1.0)


def oof_ecdf_searchsorted(oof_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_oof = np.sort(np.asarray(oof_scores, dtype=float))
    return np.searchsorted(sorted_oof, np.asarray(values, dtype=float), side="right") / max(len(sorted_oof), 1)


def fit_adni_stageb_and_score_external(
    train_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    fold: int,
    n_jobs: int,
    ecdf_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train = ensure_y(train_df)
    ext = ext_df.copy()
    mu_cols = sorted([c for c in train.columns if c.startswith("mu_")], key=lambda c: int(c.split("_")[1]))
    if not mu_cols:
        raise ValueError("No mu_* latent columns found")
    missing_ext = [c for c in mu_cols if c not in ext.columns]
    if missing_ext:
        raise ValueError(f"External latent frame missing {len(missing_ext)} mu columns, first={missing_ext[:5]}")

    x_train = train[mu_cols + ["Age", "Sex"]].copy()
    y_train = train["y"].to_numpy(dtype=int)
    x_ext = ext[mu_cols + ["Age", "Sex"]].copy()

    pre = make_preprocessor(mu_cols, include_age_sex=True)
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED + fold,
                ),
            ),
        ]
    )
    key, context, min_count = inner_stratification_key(train, INNER_FOLDS)
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 30)
    splits = list(cv.split(x_train, key))
    search = GridSearchCV(
        pipe,
        {"model__C": ORIGINAL_C_GRID},
        scoring="roc_auc",
        cv=splits,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    oof_scores = cross_val_predict(
        clone(best), x_train, y_train, cv=splits, method="predict_proba", n_jobs=n_jobs
    )[:, 1]
    raw_ext = score_1d(best, x_ext)
    if ecdf_mode == "searchsorted":
        ecdf_ext = oof_ecdf_searchsorted(oof_scores, raw_ext)
    else:
        ecdf_ext = oof_ecdf_interp(oof_scores, raw_ext)
    meta = {
        "best_inner_auc": float(search.best_score_),
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "inner_cv_context": context,
        "minimum_inner_stratum_count": int(min_count),
        "ecdf_mode": ecdf_mode,
        "n_oof": int(len(oof_scores)),
        "oof_score_min": float(np.min(oof_scores)),
        "oof_score_max": float(np.max(oof_scores)),
    }
    return raw_ext, ecdf_ext, meta


def residualize_manufacturer_preserve_age_sex(
    train_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    mu_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train = train_df.copy()
    ext = ext_df.copy()
    train["Manufacturer"] = train["Manufacturer"].map(normalize_mfr)
    ext["Manufacturer"] = ext["Manufacturer"].map(normalize_mfr)

    age_train = pd.to_numeric(train["Age"], errors="coerce")
    age_mean = float(age_train.mean())
    age_std = float(age_train.std(ddof=0) or 1.0)
    train_mfr_levels = sorted(train["Manufacturer"].fillna("UNKNOWN").astype(str).unique().tolist())
    dummy_cols = train_mfr_levels[1:]

    def design(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        age = pd.to_numeric(df["Age"], errors="coerce").fillna(age_mean)
        age_z = ((age - age_mean) / age_std).to_numpy(dtype=float).reshape(-1, 1)
        sex = (
            df["Sex"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .map({"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0})
        )
        sex_fill = float(np.nanmean(sex.dropna())) if sex.notna().any() else 0.0
        sex_arr = sex.fillna(sex_fill).to_numpy(dtype=float).reshape(-1, 1)
        mfr = df["Manufacturer"].fillna("UNKNOWN").astype(str)
        dummies = np.zeros((len(df), len(dummy_cols)), dtype=float)
        for i, level in enumerate(dummy_cols):
            dummies[:, i] = (mfr == level).astype(float)
        return np.concatenate([np.ones((len(df), 1)), age_z, sex_arr, dummies], axis=1), dummies

    x_train, mfr_train = design(train)
    x_ext, mfr_ext = design(ext)
    reg = LinearRegression(fit_intercept=False)
    y_train = train[mu_cols].to_numpy(dtype=float)
    reg.fit(x_train, y_train)
    coef = np.asarray(reg.coef_, dtype=float)
    if dummy_cols:
        coef_mfr = coef[:, -len(dummy_cols) :].T
        train_adjust = mfr_train @ coef_mfr
        ext_adjust = mfr_ext @ coef_mfr
    else:
        train_adjust = 0.0
        ext_adjust = 0.0
    train.loc[:, mu_cols] = y_train - train_adjust
    ext.loc[:, mu_cols] = ext[mu_cols].to_numpy(dtype=float) - ext_adjust
    unknown_ext = sorted(set(ext["Manufacturer"].fillna("UNKNOWN").astype(str)) - set(train_mfr_levels))
    meta = {
        "train_manufacturer_levels": json.dumps(train_mfr_levels),
        "dummy_reference": train_mfr_levels[0] if train_mfr_levels else "",
        "dummy_cols": json.dumps(dummy_cols),
        "unknown_external_manufacturer_levels_treated_as_reference": json.dumps(unknown_ext),
    }
    return train, ext, meta


def load_threshold_from_oof(spec: CandidateSpec, fold: int) -> float:
    if spec.harmonization == "none":
        path = spec.oof_dir / "calib_foldwise_metrics.csv"
        df = pd.read_csv(path)
        mask = (
            (df["fold"] == fold)
            & (df["model_name"] == PRIMARY_MODEL)
            & (df["feature_set"] == PRIMARY_FEATURE_SET)
            & (df["calib_method"] == PRIMARY_CALIB)
            & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
        )
    else:
        path = spec.oof_dir / "thresholds_by_fold.csv"
        df = pd.read_csv(path)
        mask = (
            (df["fold"] == fold)
            & (df["harmonization_method"] == spec.harmonization)
            & (df["calib_method"] == PRIMARY_CALIB)
            & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
        )
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one threshold row for {spec.label} fold {fold}, found {len(rows)} in {path}")
    return float(rows.iloc[0]["threshold"])


def load_adni_reference_metrics(spec: CandidateSpec) -> Dict[str, Any]:
    if spec.harmonization == "none":
        path = spec.oof_dir / "calib_pooled_metrics.csv"
        df = pd.read_csv(path)
        rows = df[
            (df["model_name"] == PRIMARY_MODEL)
            & (df["feature_set"] == PRIMARY_FEATURE_SET)
            & (df["calib_method"] == PRIMARY_CALIB)
            & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
        ]
    else:
        path = spec.oof_dir / "pooled_metrics.csv"
        df = pd.read_csv(path)
        rows = df[
            (df["harmonization_method"] == spec.harmonization)
            & (df["model_name"] == PRIMARY_MODEL)
            & (df["feature_set"] == PRIMARY_FEATURE_SET)
            & (df["calib_method"] == PRIMARY_CALIB)
            & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
        ]
    if len(rows) != 1:
        return {}
    row = rows.iloc[0].to_dict()
    return {f"adni_{k}": v for k, v in row.items() if k in {"auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp"}}


def validate_artifacts() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in CANDIDATES:
        row: Dict[str, Any] = {
            "candidate": spec.label,
            "role": spec.role,
            "run_dir": str(spec.run_dir),
            "oof_dir": str(spec.oof_dir),
            "harmonization": spec.harmonization,
        }
        required = [spec.run_dir, spec.run_dir / "run_config.json", spec.oof_dir]
        if spec.harmonization == "none":
            required += [spec.oof_dir / "calib_foldwise_metrics.csv", spec.oof_dir / "calib_pooled_metrics.csv"]
        else:
            required += [spec.oof_dir / "thresholds_by_fold.csv", spec.oof_dir / "pooled_metrics.csv"]
        for fold in FOLDS:
            required += [
                spec.run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt",
                spec.run_dir / f"fold_{fold}" / "vae_norm_params.joblib",
                spec.run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
            ]
        missing = [str(p) for p in required if not p.exists()]
        row["status"] = "available" if not missing else "missing_required_artifact"
        row["missing_count"] = len(missing)
        row["missing_first"] = "; ".join(missing[:5])
        rows.append(row)
    return pd.DataFrame(rows)


def load_mega_tensor(path: Path) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    manifest = pd.read_csv(MEGA_DIR / "mega_manifest.csv")
    with np.load(path, allow_pickle=True) as zf:
        tensor = np.asarray(zf["global_tensor_data"], dtype=np.float32)
        subject_ids = np.asarray(zf["subject_ids"]).astype(str)
        session_ids = np.asarray(zf["session_ids"]).astype(str) if "session_ids" in zf.files else np.array([""] * len(subject_ids))
        experiment_ids = (
            np.asarray(zf["experiment_ids"]).astype(str) if "experiment_ids" in zf.files else np.array([""] * len(subject_ids))
        )
        diagnosis = np.asarray(zf["diagnosis"]).astype(str) if "diagnosis" in zf.files else np.array([""] * len(subject_ids))
        channel_names = np.asarray(zf["channel_names"]).astype(str).tolist()

    ids = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "session_id": session_ids,
            "experiment_id": experiment_ids,
            "diagnosis_tensor": diagnosis,
            "tensor_row": np.arange(len(subject_ids), dtype=int),
        }
    )
    meta = ids.merge(manifest, on=["subject_id", "session_id", "experiment_id"], how="left", suffixes=("", "_manifest"))
    if meta["diagnosis"].isna().any():
        bad = meta.loc[meta["diagnosis"].isna(), ["subject_id", "session_id", "experiment_id"]].head(10)
        raise ValueError(f"Mega manifest alignment failed for rows:\n{bad.to_string(index=False)}")
    meta["SubjectID"] = meta["subject_id"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["diagnosis"].map(normalize_dx)
    meta["y"] = meta["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    meta["Age"] = pd.to_numeric(meta["age_at_MR"], errors="coerce")
    meta["Sex"] = meta["sex_normalized"].where(meta["sex_normalized"].notna(), meta["sex"]).map(normalize_sex)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    if meta["Age"].isna().any() or meta["Sex"].isin(["", "UNKNOWN", "NAN"]).any():
        bad = meta.loc[meta["Age"].isna() | meta["Sex"].isin(["", "UNKNOWN", "NAN"]), ["SubjectID", "Age", "Sex"]]
        raise ValueError(f"OASIS Age/Sex missing after alignment:\n{bad.to_string(index=False)}")
    return tensor, meta, channel_names


def selected_channel_indices(model_channel_names: Sequence[str], oasis_channel_names: Sequence[str]) -> List[int]:
    if not model_channel_names:
        return list(range(len(oasis_channel_names)))
    idx: List[int] = []
    for name in model_channel_names:
        if name not in oasis_channel_names:
            raise ValueError(f"Model selected channel {name!r} not present in OASIS channels {list(oasis_channel_names)}")
        idx.append(list(oasis_channel_names).index(name))
    return idx


def encode_external_fold(
    spec: CandidateSpec,
    cfg: Dict[str, Any],
    tensor: np.ndarray,
    meta: pd.DataFrame,
    fold: int,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    fold_dir = spec.run_dir / f"fold_{fold}"
    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
    conditioning_path = fold_dir / "vae_conditioning_transformer.joblib"
    conditioning_transformer = (
        joblib.load(conditioning_path)
        if str(cfg.get("vae_conditioning_mode", "none")) != "none"
        else {"vars_mode": "none", "conditioning_dim": 0}
    )
    x_norm = apply_normalization_params(tensor, norm_params)
    cond = apply_conditioning_transformer(meta.rename(columns={"subject_id": "SubjectID"}), conditioning_transformer)
    model = make_model(
        cfg,
        image_size=tensor.shape[-1],
        n_channels=tensor.shape[1],
        device=device,
        conditioning_dim_override=int(conditioning_transformer.get("conditioning_dim", 0)),
    )
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    mu = encode_mu(model, x_norm, batch_size=batch_size, device=device, condition=cond)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frame = meta.copy()
    frame["fold"] = fold
    for j in range(mu.shape[1]):
        frame[f"mu_{j}"] = mu[:, j]
    return frame


def score_candidate_on_tensor(
    spec: CandidateSpec,
    build_name: str,
    tensor: np.ndarray,
    meta: pd.DataFrame,
    oasis_channel_names: List[str],
    device: torch.device,
    batch_size: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(spec.run_dir)
    channel_names = list(cfg.get("selected_channel_names") or [])
    idx = selected_channel_indices(channel_names, oasis_channel_names)
    x = tensor[:, idx, :, :].astype(np.float32)
    fold_pred_rows: List[pd.DataFrame] = []
    fold_meta_rows: List[Dict[str, Any]] = []

    for fold in FOLDS:
        ext = encode_external_fold(spec, cfg, x, meta, fold, device, batch_size)
        train = pd.read_csv(spec.run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv")
        train = ensure_y(train)
        ext["ResearchGroup_Mapped"] = ext["ResearchGroup_Mapped"].map(normalize_dx)
        ext["Sex"] = ext["Sex"].map(normalize_sex)
        ext["Manufacturer"] = ext["Manufacturer"].map(normalize_mfr)
        ext["Age"] = pd.to_numeric(ext["Age"], errors="coerce")
        mu_cols = sorted([c for c in train.columns if c.startswith("mu_")], key=lambda c: int(c.split("_")[1]))
        harmonization_meta: Dict[str, Any] = {}
        ecdf_mode = "interp"
        if spec.harmonization != "none":
            train, ext, harmonization_meta = residualize_manufacturer_preserve_age_sex(train, ext, mu_cols)
            ecdf_mode = "searchsorted"

        raw_score, ecdf_score, fit_meta = fit_adni_stageb_and_score_external(
            train, ext, fold=fold, n_jobs=n_jobs, ecdf_mode=ecdf_mode
        )
        threshold = load_threshold_from_oof(spec, fold)
        pred = (ecdf_score >= threshold).astype(int)
        keep_cols = [
            "SubjectID",
            "subject_id",
            "session_id",
            "experiment_id",
            "source_batch",
            "protocol_subset",
            "diagnosis",
            "ResearchGroup_Mapped",
            "Age",
            "Sex",
            "Manufacturer",
            "ScannerModel",
            "selected_qc_runs",
            "selected_run_ids",
            "mean_fd_subject",
            "max_fd_subject",
            "y",
        ]
        out = ext[[c for c in keep_cols if c in ext.columns]].copy()
        out["build_candidate"] = build_name
        out["candidate"] = spec.label
        out["role"] = spec.role
        out["harmonization"] = spec.harmonization
        out["model_name"] = PRIMARY_MODEL
        out["feature_set"] = PRIMARY_FEATURE_SET
        out["calib_method"] = PRIMARY_CALIB
        out["threshold_strategy"] = PRIMARY_THRESHOLD
        out["fold"] = fold
        out["prediction_level"] = "fold_model"
        out["y_score_raw"] = raw_score
        out["y_score"] = ecdf_score
        out["threshold"] = threshold
        out["y_pred"] = pred
        out["selected_channel_names"] = json.dumps(channel_names)
        fold_pred_rows.append(out)
        row = {
            "build_candidate": build_name,
            "candidate": spec.label,
            "fold": fold,
            "threshold": threshold,
            **fit_meta,
            **harmonization_meta,
        }
        fold_meta_rows.append(row)

    fold_preds = pd.concat(fold_pred_rows, ignore_index=True)
    ensemble_keys = [
        "build_candidate",
        "candidate",
        "role",
        "harmonization",
        "model_name",
        "feature_set",
        "calib_method",
        "threshold_strategy",
        "SubjectID",
        "subject_id",
        "session_id",
        "experiment_id",
        "source_batch",
        "protocol_subset",
        "diagnosis",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "ScannerModel",
        "selected_qc_runs",
        "selected_run_ids",
        "mean_fd_subject",
        "max_fd_subject",
        "y",
        "selected_channel_names",
    ]
    present_keys = [c for c in ensemble_keys if c in fold_preds.columns]
    ens = (
        fold_preds.groupby(present_keys, dropna=False)
        .agg(
            y_score_raw=("y_score_raw", "mean"),
            y_score=("y_score", "mean"),
            threshold=("threshold", "mean"),
            fold_score_std=("y_score", "std"),
            fold_score_min=("y_score", "min"),
            fold_score_max=("y_score", "max"),
            fold_positive_votes=("y_pred", "sum"),
        )
        .reset_index()
    )
    ens["fold"] = "ensemble_mean_score_majority_vote"
    ens["prediction_level"] = "ensemble_mean_score_majority_vote"
    ens["y_pred"] = (ens["fold_positive_votes"] >= 3).astype(int)
    all_preds = pd.concat([fold_preds, ens[fold_preds.columns.intersection(ens.columns).tolist()]], ignore_index=True)
    return all_preds, pd.DataFrame(fold_meta_rows)


def metrics_tables(preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    for keys, g in preds.groupby(["build_candidate", "candidate", "role", "prediction_level", "fold"], dropna=False):
        build, candidate, role, level, fold = keys
        m = binary_metrics(g["y"], g["y_score"], g["y_pred"])
        row = {
            "build_candidate": build,
            "candidate": candidate,
            "role": role,
            "prediction_level": level,
            "fold": fold,
            **m,
        }
        metric_rows.append(row)
        confusion_rows.append(
            {
                "build_candidate": build,
                "candidate": candidate,
                "prediction_level": level,
                "fold": fold,
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
            }
        )
    metrics = pd.DataFrame(metric_rows)
    foldwise = metrics[metrics["prediction_level"] == "fold_model"].copy()
    primary = metrics[metrics["prediction_level"] == "ensemble_mean_score_majority_vote"].copy()
    confusion = pd.DataFrame(confusion_rows)

    dist_rows: List[Dict[str, Any]] = []
    ens = preds[preds["prediction_level"] == "ensemble_mean_score_majority_vote"].copy()
    for keys, g in ens.groupby(["build_candidate", "candidate", "diagnosis"], dropna=False):
        build, cand, dx = keys
        q = g["y_score"].quantile([0.25, 0.5, 0.75])
        dist_rows.append(
            {
                "build_candidate": build,
                "candidate": cand,
                "diagnosis": dx,
                "n": int(len(g)),
                "mean_score": float(g["y_score"].mean()),
                "std_score": float(g["y_score"].std(ddof=1)),
                "median_score": float(q.loc[0.5]),
                "iqr_low": float(q.loc[0.25]),
                "iqr_high": float(q.loc[0.75]),
                "min_score": float(g["y_score"].min()),
                "max_score": float(g["y_score"].max()),
                "predicted_ad_rate": float(g["y_pred"].mean()),
            }
        )
    dist = pd.DataFrame(dist_rows)

    scanner_rows: List[Dict[str, Any]] = []
    group_cols = ["build_candidate", "candidate", "source_batch", "Manufacturer", "ScannerModel", "diagnosis"]
    for keys, g in ens.groupby([c for c in group_cols if c in ens.columns], dropna=False):
        values = list(keys) if isinstance(keys, tuple) else [keys]
        row = dict(zip([c for c in group_cols if c in ens.columns], values))
        q = g["y_score"].quantile([0.25, 0.5, 0.75])
        row.update(
            {
                "n": int(len(g)),
                "mean_score": float(g["y_score"].mean()),
                "median_score": float(q.loc[0.5]),
                "iqr_low": float(q.loc[0.25]),
                "iqr_high": float(q.loc[0.75]),
                "predicted_ad_rate": float(g["y_pred"].mean()),
                "cn_fpr_if_cn": float(g.loc[g["y"] == 0, "y_pred"].mean()) if (g["y"] == 0).any() else float("nan"),
                "ad_sensitivity_if_ad": float(g.loc[g["y"] == 1, "y_pred"].mean()) if (g["y"] == 1).any() else float("nan"),
            }
        )
        scanner_rows.append(row)
    scanner = pd.DataFrame(scanner_rows)
    return primary, foldwise, confusion, dist, scanner


def adni_vs_oasis(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    ref_rows: List[Dict[str, Any]] = []
    for spec in CANDIDATES:
        ref = load_adni_reference_metrics(spec)
        if not ref:
            continue
        sub = primary_metrics[primary_metrics["candidate"] == spec.label]
        for _, row in sub.iterrows():
            out = {
                "candidate": spec.label,
                "role": spec.role,
                "build_candidate": row["build_candidate"],
                **ref,
            }
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                out[f"oasis_{metric}"] = row.get(metric)
                out[f"generalization_drop_{metric}"] = row.get(metric) - ref.get(f"adni_{metric}", np.nan)
            ref_rows.append(out)
    return pd.DataFrame(ref_rows)


def interpretation_labels(primary_metrics: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    primary = primary_metrics[primary_metrics["candidate"] == "promoted_beta3p75_oof_ecdf"]
    primary_by_build = {r["build_candidate"]: r for _, r in primary.iterrows()}
    for _, row in primary_metrics.iterrows():
        label = "external sensitivity only"
        rationale = "secondary model or post-hoc sensitivity; not eligible for OASIS-only promotion"
        if row["candidate"] == "promoted_beta3p75_oof_ecdf":
            if float(row["auc"]) >= 0.60:
                label = "primary external support"
                rationale = "primary ADNI model shows above-chance external ranking on this build"
            else:
                label = "fails external generalization"
                rationale = "primary ADNI model has weak or below-threshold external ranking on this build"
        else:
            base = primary_by_build.get(row["build_candidate"])
            if base is not None and float(row["auc"]) > float(base["auc"]) and float(row["pr_auc"]) >= float(base["pr_auc"]):
                if "mfrBalanced" in str(row["candidate"]):
                    label = "deconfounding improves external but not internal"
                    rationale = "manufacturer-balanced sensitivity improves external ranking versus primary on this build but was not internally promoted"
                else:
                    label = "external sensitivity only"
                    rationale = "sensitivity candidate improves this external build post hoc; not a primary selection criterion"
        rows.append(
            {
                "build_candidate": row["build_candidate"],
                "candidate": row["candidate"],
                "role": row["role"],
                "interpretation_label": label,
                "rationale": rationale,
                "auc": row["auc"],
                "pr_auc": row["pr_auc"],
                "balanced_accuracy": row["balanced_accuracy"],
                "f1": row["f1"],
            }
        )
    return pd.DataFrame(rows)


def write_recommendation(outdir: Path, primary: pd.DataFrame, comp: pd.DataFrame, labels: pd.DataFrame, artifact_df: pd.DataFrame) -> None:
    primary_model = primary[primary["candidate"] == "promoted_beta3p75_oof_ecdf"].sort_values("auc", ascending=False)
    best_primary = primary_model.iloc[0] if not primary_model.empty else None
    best_all = primary.sort_values(["auc", "pr_auc"], ascending=False).iloc[0] if not primary.empty else None
    unavailable = artifact_df[artifact_df["status"] != "available"]
    lines = [
        "# Final Recommendation",
        "",
        "This is a read-only external inference audit. The ADNI fold VAEs and ADNI-only Stage B OOF-ECDF readout protocol were applied to mega-OASIS tensors. No OASIS scaler, classifier, threshold, calibration, residualization, ComBat transform, or model-selection step was fit.",
        "",
    ]
    if best_primary is not None:
        lines.append(
            f"Primary model best mega-OASIS build: `{best_primary['build_candidate']}` with "
            f"AUC={best_primary['auc']:.4f}, PR-AUC={best_primary['pr_auc']:.4f}, "
            f"BA={best_primary['balanced_accuracy']:.4f}, F1={best_primary['f1']:.4f}."
        )
    if best_all is not None:
        lines.append(
            f"Best panel entry by OASIS AUC was `{best_all['candidate']}` on `{best_all['build_candidate']}` "
            f"(AUC={best_all['auc']:.4f}, PR-AUC={best_all['pr_auc']:.4f}). This is not a promotion criterion because OASIS is external evaluation, not model selection."
        )
    lines.extend(
        [
            "",
            "Manuscript recommendation:",
            "- Report `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` as the primary ADNI model.",
            "- Report mfrBalancedVAE, latent512, and manufacturer-residualized readouts only as external sensitivity analyses if their artifact validation passed.",
            "- Interpret OASIS as an external stress test of transferability. It may support a ranking signal when AUC is above chance, but any sensitivity-model advantage is post-hoc and should not override the internally promoted ADNI model.",
        ]
    )
    if not unavailable.empty:
        lines.append("")
        lines.append("Unavailable sensitivity candidates were skipped rather than substituted:")
        for _, r in unavailable.iterrows():
            lines.append(f"- `{r['candidate']}`: {r['missing_count']} missing artifacts; first missing: {r['missing_first']}")
    lines.append("")
    lines.append("Primary/sensitivity labels were assigned in `interpretation_labels.csv`.")
    (outdir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = resolve(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    command_log = {
        "script": str(Path(__file__).resolve()),
        "timestamp_start": now_iso(),
        "args": vars(args),
        "guardrails": [
            "no VAE training",
            "no classifier fitting on OASIS",
            "no OASIS threshold fitting",
            "no OASIS calibration fitting",
            "no tensor/metadata/model artifact modification",
        ],
    }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    artifact_df = validate_artifacts()
    tensor_df = pd.DataFrame(
        [
            {"build_candidate": k, "tensor_path": str(v), "exists": v.exists()}
            for k, v in TENSORS.items()
        ]
    )
    write_table(outdir, "artifact_validation", artifact_df, max_rows=100)
    write_table(outdir, "tensor_artifact_validation", tensor_df, max_rows=100)
    if args.dry_run:
        command_log["timestamp_end"] = now_iso()
        command_log["dry_run"] = True
        write_json(outdir / "command_log.json", command_log)
        readme = [
            "# OASIS Mega 90/90 External Inference Model Panel",
            "",
            "Dry-run completed. Artifact validation tables were written; no scoring was performed.",
        ]
        (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        return

    available = [spec for spec in CANDIDATES if artifact_df.loc[artifact_df["candidate"] == spec.label, "status"].iloc[0] == "available"]
    if not available:
        raise RuntimeError("No candidate has complete artifacts.")

    all_predictions: List[pd.DataFrame] = []
    fold_fit_rows: List[pd.DataFrame] = []
    for build_name, tensor_path in TENSORS.items():
        if not tensor_path.exists():
            continue
        tensor, meta, oasis_channel_names = load_mega_tensor(tensor_path)
        for spec in available:
            preds, fit_meta = score_candidate_on_tensor(
                spec,
                build_name,
                tensor,
                meta,
                oasis_channel_names,
                device=device,
                batch_size=args.batch_size,
                n_jobs=args.n_jobs,
            )
            all_predictions.append(preds)
            fold_fit_rows.append(fit_meta)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_fit = pd.concat(fold_fit_rows, ignore_index=True) if fold_fit_rows else pd.DataFrame()
    predictions.to_csv(outdir / "predictions.csv", index=False)
    write_table(outdir, "fold_readout_reconstruction_audit", fold_fit, max_rows=200)

    primary, foldwise, confusion, dist, scanner = metrics_tables(predictions)
    write_table(outdir, "primary_metrics", primary.sort_values(["build_candidate", "candidate"]), max_rows=200)
    write_table(outdir, "foldwise_metrics", foldwise.sort_values(["build_candidate", "candidate", "fold"]), max_rows=500)
    write_table(outdir, "confusion_matrices", confusion.sort_values(["build_candidate", "candidate", "prediction_level", "fold"]), max_rows=500)
    write_table(outdir, "score_distribution_by_diagnosis", dist.sort_values(["build_candidate", "candidate", "diagnosis"]), max_rows=300)
    write_table(outdir, "score_distribution_by_scanner_site_manufacturer", scanner, max_rows=500)

    comparison = adni_vs_oasis(primary)
    write_table(outdir, "adni_vs_oasis_generalization", comparison.sort_values(["candidate", "build_candidate"]), max_rows=200)
    labels = interpretation_labels(primary, comparison)
    write_table(outdir, "interpretation_labels", labels.sort_values(["build_candidate", "candidate"]), max_rows=200)

    fold_contrib = (
        predictions[predictions["prediction_level"] == "fold_model"]
        .groupby(["build_candidate", "candidate", "fold", "diagnosis"], dropna=False)
        .agg(
            n=("y", "size"),
            mean_score=("y_score", "mean"),
            median_score=("y_score", "median"),
            std_score=("y_score", "std"),
            predicted_ad_rate=("y_pred", "mean"),
        )
        .reset_index()
    )
    write_table(outdir, "fold_score_contribution", fold_contrib, max_rows=500)

    write_recommendation(outdir, primary, comparison, labels, artifact_df)
    readme = [
        "# OASIS Mega 90/90 External Inference Model Panel",
        "",
        "Scored the mega-OASIS 90CN/90AD tensors with ADNI-trained fold VAEs and ADNI-only Stage B OOF-ECDF readout reconstruction.",
        "",
        "Primary convention: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "Outputs include predictions, primary/foldwise metrics, confusion matrices, score distributions, ADNI-vs-OASIS generalization deltas, interpretation labels, and final recommendation.",
        "",
        "Guardrails: no OASIS training, threshold fitting, calibration fitting, tensor modification, metadata modification, or model artifact modification.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log["timestamp_end"] = now_iso()
    command_log["dry_run"] = False
    command_log["n_prediction_rows"] = int(len(predictions))
    command_log["candidates_scored"] = [s.label for s in available]
    command_log["builds_scored"] = [k for k, p in TENSORS.items() if p.exists()]
    write_json(outdir / "command_log.json", command_log)


if __name__ == "__main__":
    main()
