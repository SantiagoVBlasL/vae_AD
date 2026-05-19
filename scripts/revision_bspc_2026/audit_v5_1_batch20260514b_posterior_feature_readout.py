#!/usr/bin/env python3
"""Posterior-feature classifier-only audit for ADNI v5.1 FULL [1,0,2].

This script is read-only with respect to the VAE, tensors, metadata, ledger,
and existing run outputs. It loads saved fold VAE checkpoints, extracts
posterior features (mu/logvar/std/per-dim KLD), and evaluates classifier-only
readouts using the existing outer folds and true inner-CV OOF threshold
selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.data.preprocessing import apply_normalization_params


BASE_SWEEP_PATH = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
spec = importlib.util.spec_from_file_location("mfrsplit_classifier_sweep", BASE_SWEEP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import base sweep helpers from {BASE_SWEEP_PATH}")
base_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_sweep)


DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_posterior_feature_readout_audit"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
TARGET_SENSITIVITY = 0.70
FEATURE_SETS = {
    "A_mu": ["mu"],
    "B_mu_logvar": ["mu", "logvar"],
    "C_mu_std": ["mu", "std"],
    "D_mu_per_dim_kld": ["mu", "per_dim_kld"],
    "E_mu_logvar_per_dim_kld": ["mu", "logvar", "per_dim_kld"],
}
ALL_MODELS = ["logreg_l2", "logreg_elasticnet", "svm_rbf_narrow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=[PRIMARY_MODEL])
    parser.add_argument("--include-secondary", action="store_true", help="Also run logreg_elasticnet and svm_rbf_narrow.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-posterior-cache", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool, reuse_cache: bool) -> Path:
    path = resolve(path)
    generated = [
        "README.md",
        "posterior_feature_main_comparison.csv",
        "posterior_feature_main_comparison.md",
        "posterior_feature_all_model_comparison.csv",
        "posterior_feature_all_model_comparison.md",
        "foldwise_comparison.csv",
        "foldwise_comparison.md",
        "subgroup_by_manufacturer.csv",
        "subgroup_by_manufacturer.md",
        "subgroup_by_sex.csv",
        "subgroup_by_sex.md",
        "fold4_posterior_feature_deep_dive.csv",
        "fold4_posterior_feature_deep_dive.md",
        "fold4_error_summary.csv",
        "posterior_feature_thresholds_by_fold.csv",
        "posterior_feature_predictions.csv",
        "posterior_feature_manifest.json",
        "command_log.json",
    ]
    if path.exists() and any((path / name).exists() for name in generated):
        if not overwrite:
            raise FileExistsError(f"{path} already contains audit outputs; pass --overwrite")
        for name in generated:
            p = path / name
            if p.exists():
                p.unlink()
        if not reuse_cache and (path / "posterior_feature_cache").exists():
            shutil.rmtree(path / "posterior_feature_cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    posterior = Pipeline([("scaler", StandardScaler())])
    age = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    sex = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    return ColumnTransformer(
        [
            ("posterior", posterior, feature_cols),
            ("age", age, ["Age"]),
            ("sex", sex, ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def classifier_specs(seed: int) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    return {
        "logreg_l2": (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l2",
                            solver="lbfgs",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.001, 0.01, 0.1, 1.0]},
        ),
        "logreg_elasticnet": (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.001, 0.01, 0.1], "model__l1_ratio": [0.05, 0.1, 0.25, 0.5]},
        ),
        "svm_rbf_narrow": (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
                ]
            ),
            {"model__C": [0.3, 1.0, 3.0], "model__gamma": ["scale", 0.001, 0.003]},
        ),
    }


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError(f"Estimator has no predict_proba/decision_function: {type(estimator)}")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(pred.mean()) if len(pred) else float("nan"),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
    }


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


def threshold_table(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    for thr in threshold_candidates(score):
        pred = (score >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(binary_metrics(y, score, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    selections: List[Dict[str, Any]] = []
    youden = tbl.sort_values(["youden_j", "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
    selections.append(
        {
            "threshold_strategy": "inner_oof_youden_j",
            "threshold": float(youden["threshold"]),
            "selection_metric": "youden_j",
            "inner_oof_sensitivity": float(youden["sensitivity"]),
            "inner_oof_specificity": float(youden["specificity"]),
            "inner_oof_balanced_accuracy": float(youden["balanced_accuracy"]),
        }
    )
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        target = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        selection_metric = "target_not_reached_inner_oof"
    else:
        target = eligible.sort_values(
            ["specificity", "sensitivity", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        selection_metric = "selected_inner_oof"
    selections.append(
        {
            "threshold_strategy": PRIMARY_THRESHOLD,
            "threshold": float(target["threshold"]),
            "selection_metric": selection_metric,
            "inner_oof_sensitivity": float(target["sensitivity"]),
            "inner_oof_specificity": float(target["specificity"]),
            "inner_oof_balanced_accuracy": float(target["balanced_accuracy"]),
        }
    )
    selections.append(
        {
            "threshold_strategy": "fixed_0p5",
            "threshold": 0.5,
            "selection_metric": "fixed_no_selection",
            "inner_oof_sensitivity": np.nan,
            "inner_oof_specificity": np.nan,
            "inner_oof_balanced_accuracy": np.nan,
        }
    )
    return selections


def encode_posterior(model: Any, tensor: np.ndarray, batch_size: int, device: torch.device) -> Dict[str, np.ndarray]:
    mu_chunks: List[np.ndarray] = []
    logvar_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            x = torch.from_numpy(tensor[start : start + batch_size]).float().to(device)
            mu, logvar = model.encode(x)
            mu_chunks.append(mu.detach().cpu().numpy())
            logvar_chunks.append(logvar.detach().cpu().numpy())
    mu = np.concatenate(mu_chunks, axis=0).astype(np.float32)
    logvar = np.concatenate(logvar_chunks, axis=0).astype(np.float32)
    std = np.exp(0.5 * logvar).astype(np.float32)
    per_dim_kld = (0.5 * (np.square(mu) + np.exp(logvar) - logvar - 1.0)).astype(np.float32)
    return {"mu": mu, "logvar": logvar, "std": std, "per_dim_kld": per_dim_kld}


def merge_subject_metadata(subjects: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    base = subjects.copy()
    base["SubjectID"] = base["SubjectID"].astype(str)
    meta_cols = [
        c
        for c in [
            "SubjectID",
            "Age",
            "Sex",
            "Manufacturer",
            "source_batch",
            "source_label",
            "tensor_source",
            "ResearchGroup_Mapped",
        ]
        if c in metadata.columns
    ]
    merged = base.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    for col in ["Age", "Sex", "Manufacturer", "source_batch", "source_label", "tensor_source", "ResearchGroup_Mapped"]:
        meta_col = f"{col}_meta"
        if meta_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
            merged = merged.drop(columns=[meta_col])
    return merged


def posterior_frame(subjects: pd.DataFrame, metadata: pd.DataFrame, posterior: Dict[str, np.ndarray], fold: int, split: str) -> pd.DataFrame:
    base = merge_subject_metadata(subjects, metadata)
    y = base["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    if y.isna().any():
        bad = base.loc[y.isna(), ["SubjectID", "ResearchGroup_Mapped"]].head(5).to_dict("records")
        raise ValueError(f"Fold {fold} {split} contains non-CN/AD labels: {bad}")
    base["y"] = y.astype(int)
    base["fold"] = fold
    base["split"] = split
    parts = [base.reset_index(drop=True)]
    for key, arr in posterior.items():
        parts.append(pd.DataFrame(arr, columns=[f"{key}_{i}" for i in range(arr.shape[1])]))
    return pd.concat(parts, axis=1)


def build_or_load_posterior_cache(
    run_dir: Path,
    outdir: Path,
    cfg: Dict[str, Any],
    metadata: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    reuse: bool,
) -> Dict[str, Any]:
    cache_dir = outdir / "posterior_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outer_folds = int(cfg["outer_folds"])
    expected = [cache_dir / f"fold_{fold}_{split}_posterior_features.csv" for fold in range(1, outer_folds + 1) for split in ["trainDev", "test"]]
    if reuse and all(p.exists() for p in expected):
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "posterior_source": "reused_existing_posterior_feature_cache",
            "posterior_cache_dir": str(cache_dir),
            "vae_retrained": False,
            "outer_folds": outer_folds,
            "inner_folds": int(cfg["inner_folds"]),
            "feature_blocks": list(FEATURE_SETS),
        }

    tensor_info = base_sweep.load_selected_tensor(resolve(cfg["global_tensor_path"]), cfg["channels_to_use"])
    if tensor_info["python_bandpass_applied"] is not False:
        raise RuntimeError(f"Expected python_bandpass_applied=False, got {tensor_info['python_bandpass_applied']}")
    tensor = tensor_info["tensor"]
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "posterior_source": "generated_by_inference_from_saved_vae_checkpoint",
        "run_dir": str(run_dir),
        "tensor_path": str(resolve(cfg["global_tensor_path"])),
        "channels_to_use": cfg["channels_to_use"],
        "selected_channel_names": cfg["selected_channel_names"],
        "latent_dim": int(cfg["latent_dim"]),
        "outer_folds": outer_folds,
        "inner_folds": int(cfg["inner_folds"]),
        "device": str(device),
        "vae_retrained": False,
        "feature_blocks": list(FEATURE_SETS),
        "folds": [],
    }

    for fold in range(1, outer_folds + 1):
        fold_dir = run_dir / f"fold_{fold}"
        train_subjects_path = fold_dir / "train_dev_subjects_fold.csv"
        test_subjects_path = fold_dir / "test_subjects_fold.csv"
        norm_path = fold_dir / "vae_norm_params.joblib"
        checkpoint_path = fold_dir / f"vae_model_fold_{fold}.pt"
        require_files([train_subjects_path, test_subjects_path, norm_path, checkpoint_path])
        train_subjects = pd.read_csv(train_subjects_path)
        test_subjects = pd.read_csv(test_subjects_path)
        norm_params = joblib.load(norm_path)
        model = base_sweep.make_model(cfg, image_size=tensor.shape[-1], n_channels=tensor.shape[1], device=device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        fold_info: Dict[str, Any] = {
            "fold": fold,
            "checkpoint": str(checkpoint_path),
            "normalization_params": str(norm_path),
        }
        for split, subjects in [("trainDev", train_subjects), ("test", test_subjects)]:
            idx = subjects["tensor_idx"].astype(int).to_numpy()
            x_norm = apply_normalization_params(tensor[idx], norm_params)
            posterior = encode_posterior(model, x_norm, batch_size=batch_size, device=device)
            frame = posterior_frame(subjects, metadata, posterior, fold, split)
            path = cache_dir / f"fold_{fold}_{split}_posterior_features.csv"
            frame.to_csv(path, index=False)
            fold_info[f"n_{split}"] = int(len(frame))
            fold_info[f"{split}_path"] = str(path)
        manifest["folds"].append(fold_info)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return manifest


def feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    prefixes = FEATURE_SETS[feature_set]
    cols: List[str] = []
    for prefix in prefixes:
        cols.extend([c for c in df.columns if c.startswith(f"{prefix}_")])
    return cols


def inner_stratification_key(df: pd.DataFrame, n_splits: int) -> Tuple[pd.Series, str, int]:
    cols = ["ResearchGroup_Mapped", "Manufacturer"]
    key_df = df[cols].copy()
    for col in cols:
        key_df[col] = key_df[col].fillna(f"{col}_UNKNOWN").astype(str)
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    if min_count < n_splits:
        key = df["y"].astype(int)
        return key, "label_only_fallback", int(pd.Series(key).value_counts().min())
    return key, "ResearchGroup_Mapped+Manufacturer", min_count


def load_fold_pair(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_posterior_features.csv")
    test = pd.read_csv(cache_dir / f"fold_{fold}_test_posterior_features.csv")
    return train, test


def run_readouts(outdir: Path, cfg: Dict[str, Any], models: Sequence[str], n_jobs: int) -> Dict[str, pd.DataFrame]:
    cache_dir = outdir / "posterior_feature_cache"
    fold_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    outer_folds = int(cfg["outer_folds"])
    inner_folds = int(cfg["inner_folds"])

    for fold in range(1, outer_folds + 1):
        train_df, test_df = load_fold_pair(cache_dir, fold)
        y_train = train_df["y"].astype(int).to_numpy()
        y_test = test_df["y"].astype(int).to_numpy()
        inner_key, inner_context, min_inner_cell = inner_stratification_key(train_df, n_splits=inner_folds)
        inner_cv = list(
            StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=int(cfg["seed"]) + fold + 30).split(
                np.zeros(len(train_df)),
                inner_key,
            )
        )
        specs = classifier_specs(seed=int(cfg["seed"]) + fold)
        for feature_set in FEATURE_SETS:
            fcols = feature_columns(train_df, feature_set)
            if not fcols:
                raise RuntimeError(f"No posterior feature columns found for {feature_set}")
            x_train = train_df[fcols + ["Age", "Sex"]].copy()
            x_test = test_df[fcols + ["Age", "Sex"]].copy()
            pre = make_preprocessor(fcols)
            for model_name in models:
                base_pipe, grid = specs[model_name]
                pipe = clone(base_pipe)
                pipe.steps[0] = ("pre", pre)
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring="roc_auc",
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    refit=True,
                    error_score=np.nan,
                )
                search.fit(x_train, y_train)
                best = search.best_estimator_
                oof_score = cross_val_predict(clone(best), x_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
                test_score = score_1d(best, x_test)
                status_rows.append(
                    {
                        "fold": fold,
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "status": "fit_ok",
                        "n_features_posterior": len(fcols),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        "best_inner_auc": float(search.best_score_),
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_inner_cell),
                    }
                )
                for sel in select_thresholds(y_train, oof_score):
                    thr = float(sel["threshold"])
                    y_pred = (test_score >= thr).astype(int)
                    row: Dict[str, Any] = {
                        "fold": fold,
                        "feature_set": feature_set,
                        "feature_blocks": "+".join(FEATURE_SETS[feature_set]),
                        "model_name": model_name,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "threshold_selection_context": "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection",
                        "inner_cv_context": inner_context,
                        "minimum_inner_stratum_count": int(min_inner_cell),
                        "best_inner_auc": float(search.best_score_),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        "n_features_posterior": len(fcols),
                        **sel,
                    }
                    row.update(binary_metrics(y_test, test_score, y_pred))
                    fold_rows.append(row)
                    pred = test_df[
                        [
                            "SubjectID",
                            "tensor_idx",
                            "ResearchGroup_Mapped",
                            "Manufacturer",
                            "Age",
                            "Sex",
                            "source_batch",
                            "source_label",
                            "tensor_source",
                        ]
                    ].copy()
                    pred["fold"] = fold
                    pred["feature_set"] = feature_set
                    pred["feature_blocks"] = "+".join(FEATURE_SETS[feature_set])
                    pred["model_name"] = model_name
                    pred["threshold_strategy"] = sel["threshold_strategy"]
                    pred["threshold"] = thr
                    pred["y_true"] = y_test
                    pred["y_score"] = test_score
                    pred["y_pred"] = y_pred
                    pred_rows.append(pred)
                    threshold_rows.append({k: row[k] for k in [
                        "fold",
                        "feature_set",
                        "feature_blocks",
                        "model_name",
                        "threshold_strategy",
                        "threshold",
                        "threshold_selection_context",
                        "inner_cv_context",
                        "minimum_inner_stratum_count",
                        "best_inner_auc",
                        "selection_metric",
                        "inner_oof_sensitivity",
                        "inner_oof_specificity",
                        "inner_oof_balanced_accuracy",
                    ]})
    return {
        "foldwise": pd.DataFrame(fold_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True, sort=False),
        "thresholds": pd.DataFrame(threshold_rows),
        "model_status": pd.DataFrame(status_rows),
    }


def pooled_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (feature_set, model_name, strategy), sub in pred.groupby(["feature_set", "model_name", "threshold_strategy"], dropna=False):
        row = {
            "feature_set": feature_set,
            "feature_blocks": str(sub["feature_blocks"].iloc[0]),
            "model_name": model_name,
            "threshold_strategy": strategy,
            "threshold": "fold_specific" if strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_name", "threshold_strategy", "feature_set"]).reset_index(drop=True)


def metric_from_predictions(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def fold4_deep_dive(pred: pd.DataFrame) -> pd.DataFrame:
    fold4 = pred[pred["fold"].eq(4)].copy()
    fold4["diagnosis"] = np.where(fold4["y_true"].eq(1), "AD", "CN")
    fold4["prediction"] = np.where(fold4["y_pred"].eq(1), "AD_like", "CN_like")
    fold4["error_type"] = np.select(
        [
            fold4["y_true"].eq(1) & fold4["y_pred"].eq(0),
            fold4["y_true"].eq(0) & fold4["y_pred"].eq(1),
        ],
        ["false_negative_AD", "false_positive_CN"],
        default="correct",
    )
    fold4["margin_to_threshold"] = fold4["y_score"].astype(float) - fold4["threshold"].astype(float)
    keep = [
        "feature_set",
        "feature_blocks",
        "model_name",
        "SubjectID",
        "diagnosis",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "tensor_source",
        "fold",
        "threshold_strategy",
        "threshold",
        "y_score",
        "margin_to_threshold",
        "prediction",
        "error_type",
        "y_true",
        "y_pred",
    ]
    return fold4[[c for c in keep if c in fold4.columns]].sort_values(
        ["model_name", "feature_set", "error_type", "Manufacturer", "SubjectID"]
    ).reset_index(drop=True)


def write_readme(outdir: Path, primary: pd.DataFrame, all_models: pd.DataFrame, manifest: Dict[str, Any], models: Sequence[str]) -> None:
    best = primary.sort_values("auc", ascending=False).iloc[0]
    baseline = primary[primary["feature_set"].eq("A_mu")]
    lines = [
        "# Posterior Feature Readout Audit",
        "",
        "Scope: classifier-only readout on saved VAE posterior features from the current FULL `[1,0,2]` run.",
        "",
        "- VAE retraining: `False`.",
        "- Tensor/metadata/ledger modification: `False`.",
        f"- Posterior source: `{manifest.get('posterior_source')}`.",
        f"- Models run: `{', '.join(models)}`.",
        f"- Primary decision model: `{PRIMARY_MODEL}`.",
        f"- Primary threshold: `{PRIMARY_THRESHOLD}` using true inner-CV OOF predictions.",
        "",
        "## Primary LogReg L2 Result",
        "",
        f"- Best primary feature set by AUC: `{best['feature_set']}` ({best['feature_blocks']}) with AUC={best['auc']:.4f}, PR-AUC={best['pr_auc']:.4f}, BA={best['balanced_accuracy']:.4f}, sensitivity={best['sensitivity']:.4f}, specificity={best['specificity']:.4f}.",
    ]
    if not baseline.empty:
        b = baseline.iloc[0]
        lines.append(
            f"- Baseline `mu + Age + Sex`: AUC={b['auc']:.4f}, PR-AUC={b['pr_auc']:.4f}, BA={b['balanced_accuracy']:.4f}, sensitivity={b['sensitivity']:.4f}, specificity={b['specificity']:.4f}."
        )
        if best["feature_set"] != "A_mu":
            lines.append(
                f"- Delta best vs `mu`: AUC={best['auc'] - b['auc']:+.4f}, PR-AUC={best['pr_auc'] - b['pr_auc']:+.4f}, BA={best['balanced_accuracy'] - b['balanced_accuracy']:+.4f}."
            )
    lines.extend(
        [
            "",
            "## Primary Table",
            "",
            md_table(primary),
            "",
            "## Notes",
            "",
            "- Additional posterior blocks increase feature dimensionality and are therefore audited as readout-only changes, not as VAE improvements.",
            "- Any non-0.5 threshold in these outputs is selected from train/dev inner-CV OOF predictions only.",
            "- Optional secondary models, if run, are exploratory and separated from the primary decision.",
        ]
    )
    if len(set(all_models["model_name"])) > 1:
        lines.extend(["", "## All Models", "", md_table(all_models)])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, overwrite=args.overwrite, reuse_cache=args.reuse_posterior_cache)
    run_dir = resolve(args.run_dir)
    cfg = base_sweep.load_config(run_dir)
    metadata = base_sweep.normalize_metadata(pd.read_csv(resolve(cfg["metadata_path"])))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    models = list(dict.fromkeys(args.models + (["logreg_elasticnet", "svm_rbf_narrow"] if args.include_secondary else [])))
    manifest = build_or_load_posterior_cache(
        run_dir=run_dir,
        outdir=outdir,
        cfg=cfg,
        metadata=metadata,
        batch_size=int(args.batch_size),
        device=device,
        reuse=bool(args.reuse_posterior_cache),
    )
    write_json(outdir / "posterior_feature_manifest.json", manifest)
    sweep = run_readouts(outdir=outdir, cfg=cfg, models=models, n_jobs=int(args.n_jobs))
    foldwise = sweep["foldwise"].sort_values(["model_name", "threshold_strategy", "feature_set", "fold"]).reset_index(drop=True)
    predictions = sweep["predictions"]
    pooled = pooled_from_predictions(predictions)
    primary = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))].sort_values(
        "auc", ascending=False
    ).reset_index(drop=True)
    all_primary_strategy = pooled[pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD)].sort_values(
        ["model_name", "auc"], ascending=[True, False]
    ).reset_index(drop=True)
    primary_predictions = predictions[(predictions["model_name"].eq(PRIMARY_MODEL)) & (predictions["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    manufacturer = metric_from_predictions(primary_predictions, ["feature_set", "feature_blocks", "model_name", "Manufacturer"]).sort_values(
        ["Manufacturer", "auc", "feature_set"], ascending=[True, False, True]
    ).reset_index(drop=True)
    sex = metric_from_predictions(primary_predictions, ["feature_set", "feature_blocks", "model_name", "Sex"]).sort_values(
        ["Sex", "auc", "feature_set"], ascending=[True, False, True]
    ).reset_index(drop=True)
    fold4 = fold4_deep_dive(primary_predictions)
    fold4_errors = fold4[fold4["error_type"].ne("correct")]
    fold4_summary = (
        fold4_errors.groupby(["feature_set", "feature_blocks", "model_name", "error_type", "Manufacturer"], dropna=False)
        .size()
        .reset_index(name="n_errors")
        .sort_values(["feature_set", "error_type", "Manufacturer"])
        .reset_index(drop=True)
    )

    primary.to_csv(outdir / "posterior_feature_main_comparison.csv", index=False)
    (outdir / "posterior_feature_main_comparison.md").write_text(md_table(primary), encoding="utf-8")
    all_primary_strategy.to_csv(outdir / "posterior_feature_all_model_comparison.csv", index=False)
    (outdir / "posterior_feature_all_model_comparison.md").write_text(md_table(all_primary_strategy), encoding="utf-8")
    foldwise.to_csv(outdir / "foldwise_comparison.csv", index=False)
    (outdir / "foldwise_comparison.md").write_text(md_table(foldwise), encoding="utf-8")
    manufacturer.to_csv(outdir / "subgroup_by_manufacturer.csv", index=False)
    (outdir / "subgroup_by_manufacturer.md").write_text(md_table(manufacturer), encoding="utf-8")
    sex.to_csv(outdir / "subgroup_by_sex.csv", index=False)
    (outdir / "subgroup_by_sex.md").write_text(md_table(sex), encoding="utf-8")
    fold4.to_csv(outdir / "fold4_posterior_feature_deep_dive.csv", index=False)
    (outdir / "fold4_posterior_feature_deep_dive.md").write_text(md_table(fold4), encoding="utf-8")
    fold4_summary.to_csv(outdir / "fold4_error_summary.csv", index=False)
    sweep["thresholds"].to_csv(outdir / "posterior_feature_thresholds_by_fold.csv", index=False)
    predictions.to_csv(outdir / "posterior_feature_predictions.csv", index=False)
    sweep["model_status"].to_csv(outdir / "posterior_feature_model_status.csv", index=False)
    write_readme(outdir, primary, all_primary_strategy, manifest, models)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "output_dir": str(outdir),
        "models_requested": models,
        "feature_sets": FEATURE_SETS,
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof_for_selected_hyperparameters",
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "device": str(device),
        "posterior_source": manifest.get("posterior_source"),
    }
    write_json(outdir / "command_log.json", command_log)
    print(f"output_dir={outdir}")
    print("vae_retrained=False")
    print("tensor_modified=False")
    print("metadata_modified=False")
    print("ledger_modified=False")
    print(primary[["feature_set", "feature_blocks", "model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
