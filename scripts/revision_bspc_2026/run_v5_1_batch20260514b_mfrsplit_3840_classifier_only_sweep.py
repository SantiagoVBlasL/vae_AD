#!/usr/bin/env python3
"""Classifier-only sweep on ADNI v5.1 batch20260514b mfrsplit 3840 fold latents.

This script does not train or modify any VAE. If full fold-level latent mu
feature CSVs are not already present, it caches them by inference through the
saved fold VAE checkpoints and saved fold splits. Classifier thresholds are
selected from inner-CV out-of-fold train/dev probabilities only; outer test
labels are never used for threshold selection.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from betavae_xai.models import ConvolutionalVAE


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
)
CLASSIFIERS = [
    "logreg_elasticnet",
    "logreg_l2",
    "svm_rbf",
    "random_forest",
    "gradient_boosting",
    "lightgbm",
    "xgboost",
]
TARGET_SENSITIVITY = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--outer-folds", type=int, default=None, help="Override outer fold count; defaults to run_config.")
    parser.add_argument("--inner-folds", type=int, default=None, help="Override inner CV fold count; defaults to run_config.")
    parser.add_argument(
        "--folds-to-run",
        type=int,
        nargs="*",
        default=None,
        help="Optional 1-based outer folds to read/evaluate. Default evaluates all folds.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CLASSIFIERS,
        default=CLASSIFIERS,
        help="Classifier-only readout models to run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-latent-cache", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool, reuse_latent_cache: bool) -> Path:
    path = resolve(path)
    generated = [
        "README.md",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_confusion_by_fold.csv",
        "classifier_sweep_pooled_confusion.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        "classifier_sweep_predictions.csv",
        "comparison_vs_original_logreg_svm.csv",
        "latent_feature_manifest.json",
        "command_log.json",
    ]
    if path.exists() and any((path / name).exists() for name in generated):
        if not overwrite:
            raise FileExistsError(f"{path} already contains sweep outputs; pass --overwrite")
        for name in generated:
            p = path / name
            if p.exists():
                p.unlink()
        if not reuse_latent_cache and (path / "latent_cache").exists():
            shutil.rmtree(path / "latent_cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {path}")
    cfg = read_json(path)
    args = cfg.get("args", {})
    channels_to_use = list(cfg.get("channels_to_use_indices") or args.get("channels_to_use"))
    return {
        "source": str(path),
        "raw": cfg,
        "global_tensor_path": Path(cfg.get("global_tensor_path") or args.get("global_tensor_path")),
        "metadata_path": Path(cfg.get("metadata_path") or args.get("metadata_path")),
        "channels_to_use": channels_to_use,
        "input_channels": int(args.get("input_channels", len(channels_to_use))),
        "selected_channel_names": list(cfg.get("channel_names_selected") or args.get("selected_channel_names") or []),
        "latent_dim": int(args.get("latent_dim", 256)),
        "dropout_rate_vae": float(args.get("dropout_rate_vae", 0.15)),
        "vae_final_activation": str(args.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_vae": args.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(args.get("decoder_type", "convtranspose")),
        "encoder_norm_mode": str(args.get("encoder_norm_mode", "groupnorm")),
        "vae_dropout_scope": str(args.get("vae_dropout_scope", "legacy_all")),
        "vae_block_order": str(args.get("vae_block_order", "legacy_act_norm")),
        "metadata_features": list(args.get("metadata_features") or ["Age", "Sex"]),
        "classifier_stratify_cols": list(args.get("classifier_stratify_cols") or ["Manufacturer"]),
        "outer_folds": int(args.get("outer_folds", 5)),
        "inner_folds": int(args.get("inner_folds", 5)),
        "seed": int(args.get("seed", 42)),
        "python_bandpass_applied": False,
    }


def load_selected_tensor(tensor_path: Path, channels: Sequence[int]) -> Dict[str, Any]:
    with np.load(tensor_path, allow_pickle=False) as zf:
        tensor = np.asarray(zf["global_tensor_data"][:, list(channels), :, :], dtype=np.float32)
        subject_ids = np.asarray(zf["subject_ids"]).astype(str) if "subject_ids" in zf.files else None
        channel_names = np.asarray(zf["channel_names"]).astype(str).tolist() if "channel_names" in zf.files else []
        python_bandpass_applied = bool(zf["python_bandpass_applied"]) if "python_bandpass_applied" in zf.files else None
    return {
        "tensor": tensor,
        "subject_ids": subject_ids,
        "channel_names": channel_names,
        "python_bandpass_applied": python_bandpass_applied,
    }


def normalize_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if "tensor_idx" not in out.columns and "tensor_index" in out.columns:
        out = out.rename(columns={"tensor_index": "tensor_idx"})
    out["SubjectID"] = out["SubjectID"].astype(str)
    for col in ["ResearchGroup_Mapped", "Diagnosis", "Manufacturer", "Sex", "source_batch", "source_label", "tensor_source"]:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce") if "Age" in out.columns else np.nan
    return out


def make_model(cfg: Dict[str, Any], image_size: int, n_channels: int, device: torch.device) -> ConvolutionalVAE:
    model = ConvolutionalVAE(
        input_channels=n_channels,
        latent_dim=int(cfg["latent_dim"]),
        image_size=image_size,
        final_activation=str(cfg["vae_final_activation"]),
        intermediate_fc_dim_config=cfg["intermediate_fc_dim_vae"],
        dropout_rate=float(cfg["dropout_rate_vae"]),
        use_layernorm_fc=bool(cfg["use_layernorm_vae_fc"]),
        num_conv_layers_encoder=int(cfg["num_conv_layers_encoder"]),
        decoder_type=str(cfg["decoder_type"]),
        encoder_norm_mode=str(cfg.get("encoder_norm_mode", "groupnorm")),
        dropout_scope=str(cfg.get("vae_dropout_scope", "legacy_all")),
        block_order=str(cfg.get("vae_block_order", "legacy_act_norm")),
    )
    model.to(device)
    model.eval()
    return model


def encode_mu(model: ConvolutionalVAE, tensor: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            x = torch.from_numpy(tensor[start : start + batch_size]).float().to(device)
            mu, _logvar = model.encode(x)
            chunks.append(mu.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def merge_subject_metadata(subjects: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    base = subjects.copy()
    base["SubjectID"] = base["SubjectID"].astype(str)
    meta_cols = [
        col
        for col in [
            "SubjectID",
            "Age",
            "Sex",
            "Manufacturer",
            "source_batch",
            "source_label",
            "tensor_source",
            "ResearchGroup_Mapped",
        ]
        if col in metadata.columns
    ]
    merged = base.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    if "ResearchGroup_Mapped_meta" in merged.columns:
        merged["ResearchGroup_Mapped"] = merged["ResearchGroup_Mapped"].where(
            merged["ResearchGroup_Mapped"].notna(), merged["ResearchGroup_Mapped_meta"]
        )
        merged = merged.drop(columns=["ResearchGroup_Mapped_meta"])
    return merged


def subject_latent_frame(subjects: pd.DataFrame, metadata: pd.DataFrame, mu: np.ndarray, fold: int, split: str) -> pd.DataFrame:
    base = merge_subject_metadata(subjects, metadata)
    y = base["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    if y.isna().any():
        bad = base.loc[y.isna(), ["SubjectID", "ResearchGroup_Mapped"]].head(5).to_dict("records")
        raise ValueError(f"Fold {fold} {split} contains non CN/AD labels: {bad}")
    base["y"] = y.astype(int)
    base["fold"] = fold
    base["split"] = split
    mu_df = pd.DataFrame(mu, columns=[f"mu_{i}" for i in range(mu.shape[1])])
    return pd.concat([base.reset_index(drop=True), mu_df], axis=1)


def build_or_load_latent_cache(
    run_dir: Path,
    outdir: Path,
    cfg: Dict[str, Any],
    metadata: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    reuse: bool,
) -> Dict[str, Any]:
    cache_dir = outdir / "latent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outer_folds = int(cfg.get("outer_folds", 5))
    folds_to_run = [int(x) for x in cfg.get("folds_to_run", list(range(1, outer_folds + 1)))]
    expected = [cache_dir / f"fold_{fold}_{split}_latent_mu.csv" for fold in folds_to_run for split in ["trainDev", "test"]]
    if reuse and all(p.exists() for p in expected):
        return {
            "latent_source": "reused_existing_latent_cache",
            "latent_cache_dir": str(cache_dir),
            "vae_retrained": False,
            "outer_folds": outer_folds,
            "inner_folds": int(cfg.get("inner_folds", 5)),
            "folds_to_run": folds_to_run,
            "folds": [{"fold": f, "status": "reused"} for f in folds_to_run],
        }

    tensor_path = resolve(cfg["global_tensor_path"])
    tensor_info = load_selected_tensor(tensor_path, cfg["channels_to_use"])
    if tensor_info["python_bandpass_applied"] is not False:
        raise RuntimeError(f"Expected python_bandpass_applied=False, got {tensor_info['python_bandpass_applied']}")
    tensor = tensor_info["tensor"]
    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "latent_source": "generated_by_inference_from_saved_vae_checkpoint",
        "run_dir": str(run_dir),
        "tensor_path": str(tensor_path),
        "channels_to_use": cfg["channels_to_use"],
        "selected_channel_names": cfg["selected_channel_names"],
        "latent_dim": int(cfg["latent_dim"]),
        "vae_final_activation": str(cfg["vae_final_activation"]),
        "intermediate_fc_dim_vae": cfg["intermediate_fc_dim_vae"],
        "decoder_type": str(cfg["decoder_type"]),
        "num_conv_layers_encoder": int(cfg["num_conv_layers_encoder"]),
        "encoder_norm_mode": str(cfg.get("encoder_norm_mode", "groupnorm")),
        "vae_dropout_scope": str(cfg.get("vae_dropout_scope", "legacy_all")),
        "vae_block_order": str(cfg.get("vae_block_order", "legacy_act_norm")),
        "outer_folds": outer_folds,
        "inner_folds": int(cfg.get("inner_folds", 5)),
        "folds_to_run": folds_to_run,
        "device": str(device),
        "vae_retrained": False,
        "folds": [],
    }

    for fold in folds_to_run:
        fold_dir = run_dir / f"fold_{fold}"
        train_subjects_path = fold_dir / "train_dev_subjects_fold.csv"
        test_subjects_path = fold_dir / "test_subjects_fold.csv"
        norm_path = fold_dir / "vae_norm_params.joblib"
        checkpoint_path = fold_dir / f"vae_model_fold_{fold}.pt"
        require_files([train_subjects_path, test_subjects_path, norm_path, checkpoint_path])
        train_subjects = pd.read_csv(train_subjects_path)
        test_subjects = pd.read_csv(test_subjects_path)
        norm_params = joblib.load(norm_path)

        model = make_model(cfg, image_size=tensor.shape[-1], n_channels=tensor.shape[1], device=device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_info = {"fold": fold, "checkpoint": str(checkpoint_path), "normalization_params": str(norm_path)}
        for split, subjects in [("trainDev", train_subjects), ("test", test_subjects)]:
            idx = subjects["tensor_idx"].astype(int).to_numpy()
            x_norm = apply_normalization_params(tensor[idx], norm_params)
            mu = encode_mu(model, x_norm, batch_size=batch_size, device=device)
            frame = subject_latent_frame(subjects, metadata, mu, fold, split)
            path = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            frame.to_csv(path, index=False)
            fold_info[f"n_{split}"] = int(len(frame))
            fold_info[f"{split}_path"] = str(path)
        manifest["folds"].append(fold_info)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return manifest


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    numeric_latent = Pipeline([("scaler", StandardScaler())])
    numeric_age = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())])
    return ColumnTransformer(
        [
            ("latent", numeric_latent, mu_cols),
            ("age", numeric_age, ["Age"]),
            ("sex", categorical, ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def classifier_specs(seed: int, y_train: np.ndarray) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]], str]]:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos else 1.0
    specs: Dict[str, Tuple[Pipeline, Dict[str, List[Any]], str]] = {
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
            {"model__C": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.7]},
            "available",
        ),
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
            "available",
        ),
        "svm_rbf": (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.001]},
            "available",
        ),
        "random_forest": (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {"model__max_depth": [4, None], "model__min_samples_leaf": [2, 8]},
            "available",
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    ("model", GradientBoostingClassifier(random_state=seed)),
                ]
            ),
            {"model__n_estimators": [100, 250], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2]},
            "available",
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        LGBMClassifier(
                            objective="binary",
                            class_weight="balanced",
                            random_state=seed,
                            n_jobs=1,
                            verbosity=-1,
                            force_col_wise=True,
                        ),
                    ),
                ]
            ),
            {"model__n_estimators": [100, 250], "model__learning_rate": [0.03, 0.1], "model__num_leaves": [7]},
            "available",
        )
    except Exception as exc:
        specs["lightgbm"] = (Pipeline([("pre", "passthrough"), ("model", LogisticRegression())]), {}, f"unavailable: {exc}")

    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = (
            Pipeline(
                [
                    ("pre", "passthrough"),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=seed,
                            n_jobs=1,
                            tree_method="hist",
                            scale_pos_weight=scale_pos_weight,
                        ),
                    ),
                ]
            ),
            {"model__n_estimators": [100, 250], "model__learning_rate": [0.03, 0.1], "model__max_depth": [2]},
            "available",
        )
    except Exception as exc:
        specs["xgboost"] = (Pipeline([("pre", "passthrough"), ("model", LogisticRegression())]), {}, f"unavailable: {exc}")

    return specs


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


def score_1d(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        return np.asarray(proba[:, 1], dtype=float)
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
    out: Dict[str, float] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
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


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


def threshold_table(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    for thr in threshold_candidates(s):
        pred = (s >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(binary_metrics(y, s, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    selections: List[Dict[str, Any]] = []
    for criterion, metric in [("inner_oof_youden_j", "youden_j"), ("inner_oof_balanced_accuracy", "balanced_accuracy")]:
        r = tbl.sort_values([metric, "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
        selections.append(
            {
                "threshold_strategy": criterion,
                "threshold": float(r["threshold"]),
                "selection_metric": metric,
                "inner_oof_sensitivity": float(r["sensitivity"]),
                "inner_oof_specificity": float(r["specificity"]),
                "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
            }
        )
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        status = "target_not_reached_inner_oof"
    else:
        r = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
        status = "selected_inner_oof"
    selections.append(
        {
            "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
            "threshold": float(r["threshold"]),
            "selection_metric": status,
            "inner_oof_sensitivity": float(r["sensitivity"]),
            "inner_oof_specificity": float(r["specificity"]),
            "inner_oof_balanced_accuracy": float(r["balanced_accuracy"]),
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


def load_latent_pair(cache_dir: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv")
    return train, test


def original_predictions(run_dir: Path) -> pd.DataFrame:
    path = sorted(run_dir.glob("all_folds_clf_predictions_MULTI*.csv"))[0]
    df = pd.read_csv(path)
    if "classifier" not in df.columns:
        df["classifier"] = df["classifier_type"]
    df["threshold_strategy"] = "original_fixed_0p5"
    df["model_name"] = "original_" + df["classifier"].astype(str)
    df["y_pred_eval"] = (df["y_score_final"].astype(float) >= 0.5).astype(int)
    return df


def run_sweep(run_dir: Path, outdir: Path, cfg: Dict[str, Any], n_jobs: int, models: Sequence[str]) -> Dict[str, pd.DataFrame]:
    cache_dir = outdir / "latent_cache"
    fold_metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []
    subgroup_rows: List[Dict[str, Any]] = []
    model_status_rows: List[Dict[str, Any]] = []

    outer_folds = int(cfg.get("outer_folds", 5))
    inner_folds = int(cfg.get("inner_folds", 5))
    folds_to_run = [int(x) for x in cfg.get("folds_to_run", list(range(1, outer_folds + 1)))]
    for fold in folds_to_run:
        train_df, test_df = load_latent_pair(cache_dir, fold)
        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        feature_cols = mu_cols + ["Age", "Sex"]
        x_train = train_df[feature_cols].copy()
        y_train = train_df["y"].astype(int).to_numpy()
        x_test = test_df[feature_cols].copy()
        y_test = test_df["y"].astype(int).to_numpy()
        inner_key, inner_context, min_inner_cell = inner_stratification_key(train_df, n_splits=inner_folds)
        inner_cv = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=cfg["seed"] + fold + 30).split(np.zeros(len(train_df)), inner_key))
        pre = make_preprocessor(mu_cols)
        specs = classifier_specs(seed=cfg["seed"] + fold, y_train=y_train)

        for model_name in models:
            base_pipe, grid, status = specs[model_name]
            if status != "available":
                model_status_rows.append({"fold": fold, "model_name": model_name, "status": status})
                continue
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
            thresholds = select_thresholds(y_train, oof_score)
            model_status_rows.append(
                {
                    "fold": fold,
                    "model_name": model_name,
                    "status": "fit_ok",
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "best_inner_auc": float(search.best_score_),
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_inner_cell),
                }
            )
            for sel in thresholds:
                thr = float(sel["threshold"])
                y_pred = (test_score >= thr).astype(int)
                row = {
                    "fold": fold,
                    "model_name": model_name,
                    "threshold_strategy": sel["threshold_strategy"],
                    "threshold": thr,
                    "threshold_selection_context": "true_inner_cv_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed_no_selection",
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_inner_cell),
                    "best_inner_auc": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    **sel,
                }
                row.update(binary_metrics(y_test, test_score, y_pred))
                fold_metric_rows.append(row)
                confusion_rows.append({k: row[k] for k in ["fold", "model_name", "threshold_strategy", "threshold", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1"]})
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
                pred["model_name"] = model_name
                pred["threshold_strategy"] = sel["threshold_strategy"]
                pred["threshold"] = thr
                pred["y_true"] = y_test
                pred["y_score"] = test_score
                pred["y_pred"] = y_pred
                pred_rows.append(pred)

                for manufacturer, sub_idx in pred.groupby("Manufacturer", dropna=False).groups.items():
                    sub_pred = pred.loc[list(sub_idx)]
                    sub_row = {
                        "fold": fold,
                        "model_name": model_name,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "Manufacturer": manufacturer,
                    }
                    sub_row.update(binary_metrics(sub_pred["y_true"], sub_pred["y_score"], sub_pred["y_pred"]))
                    subgroup_rows.append(sub_row)

    return {
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True, sort=False),
        "thresholds": pd.DataFrame(threshold_rows) if threshold_rows else pd.DataFrame(fold_metric_rows)[
            [
                "fold",
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
            ]
        ].drop_duplicates(),
        "confusion_by_fold": pd.DataFrame(confusion_rows),
        "subgroup_by_manufacturer": pd.DataFrame(subgroup_rows),
        "model_status": pd.DataFrame(model_status_rows),
    }


def pooled_from_predictions(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    conf_rows: List[Dict[str, Any]] = []
    for (model_name, strategy), sub in pred.groupby(["model_name", "threshold_strategy"], dropna=False):
        row = {
            "model_name": model_name,
            "threshold_strategy": strategy,
            "threshold": "fold_specific" if strategy != "fixed_0p5" else 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
        conf_rows.append({k: row[k] for k in ["model_name", "threshold_strategy", "threshold", "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1"]})
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "model_name"]), pd.DataFrame(conf_rows).sort_values(["threshold_strategy", "model_name"])


def compare_against_original(run_dir: Path, sweep_pooled: pd.DataFrame) -> pd.DataFrame:
    orig = original_predictions(run_dir)
    rows: List[Dict[str, Any]] = []
    for model_name, sub in orig.groupby("model_name"):
        row = {"model_name": model_name, "threshold_strategy": "original_fixed_0p5"}
        row.update(binary_metrics(sub["y_true"], sub["y_score_final"], sub["y_pred_eval"]))
        rows.append(row)
    orig_metrics = pd.DataFrame(rows)
    combined = pd.concat([orig_metrics, sweep_pooled], ignore_index=True, sort=False)
    ref_map = {r["model_name"]: r for _, r in orig_metrics.iterrows()}
    delta_rows: List[Dict[str, Any]] = []
    for _, row in sweep_pooled.iterrows():
        for ref_name in ["original_logreg", "original_svm"]:
            ref = ref_map.get(ref_name)
            if ref is None:
                continue
            delta = {
                "model_name": row["model_name"],
                "threshold_strategy": row["threshold_strategy"],
                "reference_model": ref_name,
            }
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                delta[f"{metric}_delta"] = float(row[metric] - ref[metric])
            delta_rows.append(delta)
    return pd.concat([combined.assign(row_type="absolute"), pd.DataFrame(delta_rows).assign(row_type="delta")], ignore_index=True, sort=False)


def make_readme(outdir: Path, pooled: pd.DataFrame, comparison: pd.DataFrame, manifest: Dict[str, Any], model_status: pd.DataFrame) -> None:
    fixed = pooled[pooled["threshold_strategy"].eq("fixed_0p5")].sort_values("auc", ascending=False)
    inner = pooled[pooled["threshold_strategy"].eq("inner_oof_youden_j")].sort_values("balanced_accuracy", ascending=False)
    unavailable = model_status[model_status["status"].astype(str).str.startswith("unavailable")]

    def table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> List[str]:
        if df.empty:
            return ["No rows."]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.head(n).iterrows():
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.3f}" if np.isfinite(v) else "NA")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# ADNI v5.1 batch20260514b mfrsplit 3840 Classifier-Only Sweep",
        "",
        "## Scope",
        "",
        "- VAE retraining: `False`.",
        "- Tensor modification: `False`.",
        "- Metadata/ledger modification: `False`.",
        f"- Latent source: `{manifest.get('latent_source')}`.",
        "- Threshold selection for non-0.5 strategies: `true_inner_cv_oof` on train/dev only.",
        f"- Outer folds read: `{manifest.get('outer_folds')}`.",
        f"- Inner CV folds: `{manifest.get('inner_folds')}`.",
        "",
        "## Best Fixed Threshold 0.5 Models by AUC",
        "",
        *table(fixed, ["model_name", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]),
        "",
        "## Best Inner-OOF Youden Threshold Models by Balanced Accuracy",
        "",
        *table(inner, ["model_name", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]),
        "",
        "## Optional Classifier Availability",
        "",
    ]
    if unavailable.empty:
        lines.append("All optional classifiers requested were available.")
    else:
        lines.extend(table(unavailable, ["fold", "model_name", "status"], n=20))
    lines += [
        "",
        "## Outputs",
        "",
        "- `classifier_sweep_foldwise_metrics.csv`",
        "- `classifier_sweep_pooled_metrics.csv`",
        "- `classifier_sweep_confusion_by_fold.csv`",
        "- `classifier_sweep_pooled_confusion.csv`",
        "- `classifier_sweep_thresholds_by_fold.csv`",
        "- `classifier_sweep_subgroup_metrics_by_manufacturer.csv`",
        "- `classifier_sweep_predictions.csv`",
        "- `comparison_vs_original_logreg_svm.csv`",
        "- `latent_feature_manifest.json`",
        "- `command_log.json`",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite, args.reuse_latent_cache)
    cfg = load_config(run_dir)
    if args.outer_folds is not None:
        cfg["outer_folds"] = int(args.outer_folds)
    if args.inner_folds is not None:
        cfg["inner_folds"] = int(args.inner_folds)
    if args.folds_to_run:
        outer_folds = int(cfg.get("outer_folds", 5))
        folds_to_run = sorted({int(x) for x in args.folds_to_run})
        invalid = [x for x in folds_to_run if x < 1 or x > outer_folds]
        if invalid:
            raise RuntimeError(f"Invalid --folds-to-run values {invalid}; allowed range is 1..{outer_folds}")
        cfg["folds_to_run"] = folds_to_run
    metadata = normalize_metadata(pd.read_csv(resolve(cfg["metadata_path"])))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    latent_manifest = build_or_load_latent_cache(
        run_dir=run_dir,
        outdir=outdir,
        cfg=cfg,
        metadata=metadata,
        batch_size=int(args.batch_size),
        device=device,
        reuse=bool(args.reuse_latent_cache),
    )
    write_json(outdir / "latent_feature_manifest.json", latent_manifest)

    requested_models = list(dict.fromkeys(args.models))
    sweep = run_sweep(run_dir=run_dir, outdir=outdir, cfg=cfg, n_jobs=int(args.n_jobs), models=requested_models)
    foldwise = sweep["foldwise_metrics"].sort_values(["model_name", "threshold_strategy", "fold"])
    predictions = sweep["predictions"]
    pooled, pooled_conf = pooled_from_predictions(predictions)
    comparison = compare_against_original(run_dir, pooled)

    foldwise.to_csv(outdir / "classifier_sweep_foldwise_metrics.csv", index=False)
    pooled.to_csv(outdir / "classifier_sweep_pooled_metrics.csv", index=False)
    sweep["confusion_by_fold"].to_csv(outdir / "classifier_sweep_confusion_by_fold.csv", index=False)
    pooled_conf.to_csv(outdir / "classifier_sweep_pooled_confusion.csv", index=False)
    sweep["thresholds"].to_csv(outdir / "classifier_sweep_thresholds_by_fold.csv", index=False)
    sweep["subgroup_by_manufacturer"].to_csv(outdir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv", index=False)
    predictions.to_csv(outdir / "classifier_sweep_predictions.csv", index=False)
    comparison.to_csv(outdir / "comparison_vs_original_logreg_svm.csv", index=False)
    sweep["model_status"].to_csv(outdir / "classifier_sweep_model_status.csv", index=False)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "output_dir": str(outdir),
        "classifiers_requested": requested_models,
        "outer_folds": int(cfg.get("outer_folds", 5)),
        "inner_folds": int(cfg.get("inner_folds", 5)),
        "folds_to_run": cfg.get("folds_to_run", list(range(1, int(cfg.get("outer_folds", 5)) + 1))),
        "vae_architecture": {
            "channels_to_use": cfg.get("channels_to_use"),
            "input_channels": int(cfg.get("input_channels", len(cfg.get("channels_to_use", [])))),
            "latent_dim": int(cfg.get("latent_dim", 256)),
            "vae_final_activation": str(cfg.get("vae_final_activation", "tanh")),
            "intermediate_fc_dim_vae": cfg.get("intermediate_fc_dim_vae", "quarter"),
            "decoder_type": str(cfg.get("decoder_type", "convtranspose")),
            "num_conv_layers_encoder": int(cfg.get("num_conv_layers_encoder", 4)),
            "encoder_norm_mode": str(cfg.get("encoder_norm_mode", "groupnorm")),
            "vae_dropout_scope": str(cfg.get("vae_dropout_scope", "legacy_all")),
            "vae_block_order": str(cfg.get("vae_block_order", "legacy_act_norm")),
        },
        "threshold_selection": "true_inner_cv_oof_for_selected_hyperparameters",
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "latent_source": latent_manifest.get("latent_source"),
        "device": str(device),
    }
    write_json(outdir / "command_log.json", command_log)
    make_readme(outdir, pooled, comparison, latent_manifest, sweep["model_status"])

    print(f"output_dir={outdir}")
    print("vae_retrained=False")
    print("tensor_modified=False")
    print("threshold_selection=true_inner_cv_oof")
    print(pooled[["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
