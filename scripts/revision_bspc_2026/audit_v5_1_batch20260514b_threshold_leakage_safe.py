#!/usr/bin/env python3
"""Leakage-safe threshold audit for ADNI v5.1 batch20260514b baseline.

This script is deliberately read-only with respect to tensors, trained models,
and ledgers. It does not train a VAE or classifier. When no train/inner
prediction tables are present, it reconstructs train/dev probabilities by
running inference through the saved fold VAE checkpoints and saved final
classifier pipelines. Thresholds are selected only on train/dev scores and then
applied to the already-held-out outer test predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.data.preprocessing import apply_normalization_params
from betavae_xai.models import ConvolutionalVAE


DEFAULT_RUN_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_threshold_leakage_safe_audit"
)
EXPECTED_CLASSIFIERS = ["logreg", "svm"]
TARGET_SENSITIVITY = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "README.md",
        "threshold_by_fold.csv",
        "metrics_threshold_0p5_vs_selected.csv",
        "confusion_by_fold_selected_threshold.csv",
        "subgroup_metrics_by_manufacturer_selected_threshold.csv",
        "command_log.json",
        "prediction_artifact_inventory.csv",
        "train_dev_predictions_for_threshold_selection.csv",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for p in existing:
            p.unlink()
    return path


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {path}")
    cfg = read_json(path)
    args = cfg.get("args", {})
    return {
        "raw": cfg,
        "source": str(path),
        "global_tensor_path": Path(cfg.get("global_tensor_path") or args.get("global_tensor_path")),
        "metadata_path": Path(cfg.get("metadata_path") or args.get("metadata_path")),
        "channels_to_use": list(cfg.get("channels_to_use_indices") or args.get("channels_to_use")),
        "selected_channel_names": list(cfg.get("channel_names_selected") or args.get("selected_channel_names") or []),
        "classifier_types": list(args.get("classifier_types") or EXPECTED_CLASSIFIERS),
        "metadata_features": list(args.get("metadata_features") or []),
        "latent_dim": int(args.get("latent_dim", 256)),
        "latent_features_type": str(args.get("latent_features_type", "mu")),
        "dropout_rate_vae": float(args.get("dropout_rate_vae", 0.15)),
        "vae_final_activation": str(args.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_vae": args.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(args.get("decoder_type", "convtranspose")),
        "python_bandpass_applied_expected": False,
        "classifier_calibrate": bool(args.get("classifier_calibrate", False)),
    }


def load_selected_tensor(tensor_path: Path, channels: Sequence[int]) -> Dict[str, Any]:
    with np.load(tensor_path, allow_pickle=False) as zf:
        if "global_tensor_data" not in zf.files:
            raise KeyError(f"{tensor_path} does not contain global_tensor_data")
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
    )
    model.to(device)
    model.eval()
    return model


def encode_features(
    model: ConvolutionalVAE,
    tensor: np.ndarray,
    latent_features_type: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            x = torch.from_numpy(tensor[start : start + batch_size]).float().to(device)
            if latent_features_type == "mu":
                mu, _logvar = model.encode(x)
                out = mu
            elif latent_features_type == "z":
                _recon, _mu, _logvar, z = model(x)
                out = z
            else:
                raise ValueError(f"Unsupported latent_features_type={latent_features_type!r}")
            chunks.append(out.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def get_score_1d(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).ravel()
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError(f"Model has neither predict_proba nor decision_function: {type(model)}")


def prediction_artifact_inventory(run_dir: Path, classifiers: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        for clf in classifiers:
            train_pred_patterns = [
                f"train_dev_predictions_{clf}.csv",
                f"train_predictions_{clf}.csv",
                f"inner_oof_predictions_{clf}.csv",
                f"oof_predictions_{clf}.csv",
            ]
            train_pred_files = [fold_dir / pat for pat in train_pred_patterns]
            rows.append(
                {
                    "fold": fold,
                    "classifier": clf,
                    "test_predictions_csv": str(fold_dir / f"test_predictions_{clf}.csv"),
                    "test_predictions_available": (fold_dir / f"test_predictions_{clf}.csv").exists(),
                    "train_dev_subjects_available": (fold_dir / "train_dev_subjects_fold.csv").exists(),
                    "inner_or_train_prediction_files_found": ";".join(
                        str(p) for p in train_pred_files if p.exists()
                    ),
                    "saved_final_classifier_available": (
                        fold_dir / f"classifier_{clf}_final_pipeline_fold_{fold}.joblib"
                    ).exists(),
                    "saved_vae_checkpoint_available": (fold_dir / f"vae_model_fold_{fold}.pt").exists(),
                    "vae_norm_params_available": (fold_dir / "vae_norm_params.joblib").exists(),
                }
            )
    return pd.DataFrame(rows)


def normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    out["SubjectID"] = out["SubjectID"].astype(str)
    for col in ["ResearchGroup_Mapped", "Diagnosis", "Manufacturer", "Sex"]:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    if "Age" not in out.columns:
        out["Age"] = np.nan
    return out


def feature_frame(
    latent: np.ndarray,
    subjects: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: Sequence[str],
    metadata_features: Sequence[str],
) -> pd.DataFrame:
    latent_cols = [f"latent_{i}" for i in range(latent.shape[1])]
    x = pd.DataFrame(latent, columns=latent_cols)
    if metadata_features:
        sub = subjects[["SubjectID"]].copy()
        sub["SubjectID"] = sub["SubjectID"].astype(str)
        meta_cols = ["SubjectID", *metadata_features]
        merged = sub.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left")
        meta_x = merged[list(metadata_features)].reset_index(drop=True)
        x = pd.concat([x.reset_index(drop=True), meta_x], axis=1)
    missing = [col for col in feature_columns if col not in x.columns]
    if missing:
        raise RuntimeError(f"Generated feature frame is missing columns: {missing[:10]}")
    return x.loc[:, list(feature_columns)]


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    def div(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    out: Dict[str, float] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": div(tp + tn, len(y)),
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "balanced_accuracy": div(div(tp, tp + fn) + div(tn, tn + fp), 2.0),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(p.mean()) if len(p) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def threshold_grid(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite scores for threshold selection")
    values = np.unique(np.concatenate(([0.0, 0.5, 1.0], s)))
    values = np.clip(values, 0.0, 1.0)
    return np.unique(np.round(values, 12))


def threshold_table(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    for thr in threshold_grid(s):
        pred = (s >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(binary_metrics(y, s, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_thresholds(y_true: Sequence[int], y_score: Sequence[float]) -> List[Dict[str, Any]]:
    tbl = threshold_table(y_true, y_score)
    selections: List[Dict[str, Any]] = []

    selectors = {
        "youden_j": ["youden_j", "sensitivity", "specificity", "threshold"],
        "balanced_accuracy": ["balanced_accuracy", "sensitivity", "specificity", "threshold"],
    }
    for criterion, sort_cols in selectors.items():
        chosen = tbl.sort_values(sort_cols, ascending=[False, False, False, False]).iloc[0]
        selections.append(
            {
                "criterion": criterion,
                "selection_status": "selected_on_train_dev_scores",
                "selected_threshold": float(chosen["threshold"]),
                "train_selection_metric": float(chosen[criterion]),
                "train_sensitivity": float(chosen["sensitivity"]),
                "train_specificity": float(chosen["specificity"]),
                "train_balanced_accuracy": float(chosen["balanced_accuracy"]),
                "train_f1": float(chosen["f1"]),
            }
        )

    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY].copy()
    if eligible.empty:
        chosen = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        status = f"target_sensitivity_{TARGET_SENSITIVITY:.2f}_not_reached_on_train_dev"
    else:
        chosen = eligible.sort_values(
            ["specificity", "sensitivity", "balanced_accuracy", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
        status = "selected_on_train_dev_scores"
    selections.append(
        {
            "criterion": "target_sensitivity_ge_0p70_max_specificity",
            "selection_status": status,
            "selected_threshold": float(chosen["threshold"]),
            "train_selection_metric": float(chosen["specificity"]),
            "train_sensitivity": float(chosen["sensitivity"]),
            "train_specificity": float(chosen["specificity"]),
            "train_balanced_accuracy": float(chosen["balanced_accuracy"]),
            "train_f1": float(chosen["f1"]),
        }
    )
    return selections


def load_test_predictions(run_dir: Path, metadata: pd.DataFrame, classifiers: Sequence[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    meta_cols = [
        c
        for c in [
            "SubjectID",
            "ResearchGroup_Mapped",
            "Diagnosis",
            "Age",
            "Sex",
            "Manufacturer",
            "source_batch",
            "source_label",
            "tensor_source",
        ]
        if c in metadata.columns
    ]
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        for clf in classifiers:
            path = fold_dir / f"test_predictions_{clf}.csv"
            df = pd.read_csv(path)
            df["fold"] = fold
            df["classifier"] = clf
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["SubjectID"] = out["SubjectID"].astype(str)
    out = out.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    out["Manufacturer"] = out["Manufacturer"].fillna("UNKNOWN").astype(str)
    out["Sex"] = out["Sex"].fillna("UNKNOWN").astype(str)
    return out


def generate_train_dev_predictions(
    run_dir: Path,
    cfg: Dict[str, Any],
    tensor: np.ndarray,
    metadata: pd.DataFrame,
    classifiers: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        subjects_path = fold_dir / "train_dev_subjects_fold.csv"
        feature_columns_path = fold_dir / "feature_columns.json"
        norm_path = fold_dir / "vae_norm_params.joblib"
        checkpoint_path = fold_dir / f"vae_model_fold_{fold}.pt"
        require_files([subjects_path, feature_columns_path, norm_path, checkpoint_path])

        subjects = pd.read_csv(subjects_path)
        subjects["SubjectID"] = subjects["SubjectID"].astype(str)
        idx = subjects["tensor_idx"].astype(int).to_numpy()
        norm_params = joblib.load(norm_path)
        tensor_norm = apply_normalization_params(tensor[idx], norm_params)

        model = make_model(cfg, image_size=tensor.shape[-1], n_channels=tensor.shape[1], device=device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        latent = encode_features(model, tensor_norm, cfg["latent_features_type"], batch_size, device)

        feature_cfg = read_json(feature_columns_path)
        x = feature_frame(
            latent=latent,
            subjects=subjects,
            metadata=metadata,
            feature_columns=feature_cfg["final_feature_columns"],
            metadata_features=cfg["metadata_features"],
        )
        y_true = subjects["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
        if y_true.isna().any():
            bad = subjects.loc[y_true.isna(), ["SubjectID", "ResearchGroup_Mapped"]].head(5).to_dict("records")
            raise RuntimeError(f"Fold {fold} train/dev includes non CN/AD labels: {bad}")
        for clf in classifiers:
            model_path = fold_dir / f"classifier_{clf}_final_pipeline_fold_{fold}.joblib"
            require_files([model_path])
            pipe = joblib.load(model_path)
            score = get_score_1d(pipe, x)
            out = pd.DataFrame(
                {
                    "fold": fold,
                    "classifier": clf,
                    "SubjectID": subjects["SubjectID"].values,
                    "tensor_idx": subjects["tensor_idx"].astype(int).values,
                    "y_true": y_true.astype(int).values,
                    "y_score_final": score,
                    "prediction_source": "generated_train_dev_from_saved_vae_and_final_classifier",
                    "threshold_selection_context": "train_dev_resubstitution_no_outer_test",
                }
            )
            rows.append(out)

        del model, tensor_norm, latent, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred = pd.concat(rows, ignore_index=True)
    meta_cols = [c for c in ["SubjectID", "Age", "Sex", "Manufacturer", "ResearchGroup_Mapped"] if c in metadata.columns]
    pred = pred.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    pred["Manufacturer"] = pred["Manufacturer"].fillna("UNKNOWN").astype(str)
    pred["Sex"] = pred["Sex"].fillna("UNKNOWN").astype(str)
    return pred


def make_threshold_rows(train_pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (fold, clf), sub in train_pred.groupby(["fold", "classifier"], dropna=False):
        for selected in select_thresholds(sub["y_true"], sub["y_score_final"]):
            row = {
                "fold": int(fold),
                "classifier": clf,
                "selection_data": "train_dev",
                "selection_context": "train_dev_resubstitution_no_outer_test",
                "inner_cv_oof_predictions_available": False,
                "outer_test_used_for_selection": False,
            }
            row.update(selected)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["classifier", "fold", "criterion"])


def evaluate_thresholds(test_pred: pd.DataFrame, threshold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    conf_rows: List[Dict[str, Any]] = []
    selected_pred_chunks: List[pd.DataFrame] = []

    for (fold, clf), sub in test_pred.groupby(["fold", "classifier"], dropna=False):
        sub = sub.copy()
        y = sub["y_true"].to_numpy(int)
        score = sub["y_score_final"].to_numpy(float)

        pred_05 = (score >= 0.5).astype(int)
        row = {
            "scope": "outer_test_fold",
            "fold": int(fold),
            "classifier": clf,
            "criterion": "threshold_0p5",
            "threshold_strategy": "fixed_0p5",
            "threshold": 0.5,
            "selection_context": "predefined_no_test_selection",
        }
        row.update(binary_metrics(y, score, pred_05))
        metric_rows.append(row)

        fold_thresholds = threshold_df[
            threshold_df["fold"].eq(fold) & threshold_df["classifier"].eq(clf)
        ]
        for _, thr_row in fold_thresholds.iterrows():
            thr = float(thr_row["selected_threshold"])
            pred = (score >= thr).astype(int)
            mrow = {
                "scope": "outer_test_fold",
                "fold": int(fold),
                "classifier": clf,
                "criterion": thr_row["criterion"],
                "threshold_strategy": "selected_train_dev",
                "threshold": thr,
                "selection_context": thr_row["selection_context"],
            }
            mrow.update(binary_metrics(y, score, pred))
            metric_rows.append(mrow)

            conf = {
                "fold": int(fold),
                "classifier": clf,
                "criterion": thr_row["criterion"],
                "threshold": thr,
            }
            conf.update({k: mrow[k] for k in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy", "f1"]})
            conf_rows.append(conf)

            pred_chunk = sub.copy()
            pred_chunk["criterion"] = thr_row["criterion"]
            pred_chunk["selected_threshold"] = thr
            pred_chunk["y_pred_selected"] = pred
            selected_pred_chunks.append(pred_chunk)

    selected_pred = pd.concat(selected_pred_chunks, ignore_index=True) if selected_pred_chunks else pd.DataFrame()

    metrics = pd.DataFrame(metric_rows)
    pooled_rows: List[Dict[str, Any]] = []
    for (clf, criterion, strategy), sub in metrics.groupby(["classifier", "criterion", "threshold_strategy"], dropna=False):
        if strategy == "fixed_0p5":
            source = test_pred[test_pred["classifier"].eq(clf)].copy()
            pred = (source["y_score_final"].to_numpy(float) >= 0.5).astype(int)
            threshold_desc = "0.5"
        else:
            source = selected_pred[
                selected_pred["classifier"].eq(clf) & selected_pred["criterion"].eq(criterion)
            ].copy()
            pred = source["y_pred_selected"].to_numpy(int)
            threshold_desc = "fold_specific"
        if source.empty:
            continue
        row = {
            "scope": "outer_test_pooled",
            "fold": "pooled",
            "classifier": clf,
            "criterion": criterion,
            "threshold_strategy": strategy,
            "threshold": threshold_desc,
            "selection_context": (
                "predefined_no_test_selection" if strategy == "fixed_0p5" else "train_dev_resubstitution_no_outer_test"
            ),
        }
        row.update(binary_metrics(source["y_true"], source["y_score_final"], pred))
        pooled_rows.append(row)

    if pooled_rows:
        metrics = pd.concat([metrics, pd.DataFrame(pooled_rows)], ignore_index=True, sort=False)

    return (
        metrics.sort_values(["classifier", "criterion", "scope", "fold"]).reset_index(drop=True),
        pd.DataFrame(conf_rows).sort_values(["classifier", "criterion", "fold"]).reset_index(drop=True),
        selected_pred,
    )


def subgroup_metrics_by_manufacturer(selected_pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if selected_pred.empty:
        return pd.DataFrame()
    for (clf, criterion, manufacturer), sub in selected_pred.groupby(["classifier", "criterion", "Manufacturer"], dropna=False):
        y = sub["y_true"].to_numpy(int)
        score = sub["y_score_final"].to_numpy(float)
        pred = sub["y_pred_selected"].to_numpy(int)
        row = {
            "classifier": clf,
            "criterion": criterion,
            "grouping": "manufacturer_all",
            "Manufacturer": manufacturer,
        }
        row.update(binary_metrics(y, score, pred))
        rows.append(row)

        ad = sub[sub["y_true"].eq(1)]
        if not ad.empty:
            rows.append(
                {
                    "classifier": clf,
                    "criterion": criterion,
                    "grouping": "ad_sensitivity_by_manufacturer",
                    "Manufacturer": manufacturer,
                    "n": int(len(ad)),
                    "n_ad": int(len(ad)),
                    "n_cn": 0,
                    "tp": int(ad["y_pred_selected"].eq(1).sum()),
                    "fn": int(ad["y_pred_selected"].eq(0).sum()),
                    "sensitivity": float(ad["y_pred_selected"].eq(1).mean()),
                    "specificity": np.nan,
                    "balanced_accuracy": np.nan,
                    "score_mean": float(ad["y_score_final"].mean()),
                    "score_median": float(ad["y_score_final"].median()),
                }
            )
        cn = sub[sub["y_true"].eq(0)]
        if not cn.empty:
            rows.append(
                {
                    "classifier": clf,
                    "criterion": criterion,
                    "grouping": "cn_specificity_by_manufacturer",
                    "Manufacturer": manufacturer,
                    "n": int(len(cn)),
                    "n_ad": 0,
                    "n_cn": int(len(cn)),
                    "tn": int(cn["y_pred_selected"].eq(0).sum()),
                    "fp": int(cn["y_pred_selected"].eq(1).sum()),
                    "sensitivity": np.nan,
                    "specificity": float(cn["y_pred_selected"].eq(0).mean()),
                    "balanced_accuracy": np.nan,
                    "score_mean": float(cn["y_score_final"].mean()),
                    "score_median": float(cn["y_score_final"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["classifier", "criterion", "grouping", "Manufacturer"])


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
        if math.isnan(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


def make_readme(
    outdir: Path,
    run_dir: Path,
    cfg: Dict[str, Any],
    inventory: pd.DataFrame,
    thresholds: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    pooled = metrics[metrics["scope"].eq("outer_test_pooled")].copy()
    lines = [
        "# ADNI v5.1 batch20260514b Threshold Leakage-Safe Audit",
        "",
        "## Scope",
        "",
        "- Baseline: `adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline`.",
        "- Dataset/tensors/ledger were not modified.",
        "- VAE and classifiers were not retrained.",
        "- Python bandpass remains OFF in the source dataset.",
        "",
        "## Prediction Artifacts",
        "",
        f"- Outer test prediction CSVs found: `{int(inventory['test_predictions_available'].sum())}/{len(inventory)}`.",
        "- Saved train/inner prediction CSVs found: `0`.",
        "- Train/dev threshold-selection scores were generated by inference only, using saved fold VAE checkpoints and saved final classifier pipelines.",
        "- Selection context: `train_dev_resubstitution_no_outer_test`. This is leakage-safe relative to the outer test folds, but less conservative than inner-CV out-of-fold threshold selection.",
        "",
        "## Threshold Selection",
        "",
        "- Criteria per outer fold and classifier:",
        "  - `youden_j`: maximize sensitivity + specificity - 1 on train/dev.",
        "  - `balanced_accuracy`: maximize balanced accuracy on train/dev.",
        "  - `target_sensitivity_ge_0p70_max_specificity`: among train/dev thresholds with sensitivity >= 0.70, maximize specificity.",
        "- Outer test scores were used only after thresholds were fixed.",
        "",
        "## Pooled Outer-Test Results",
        "",
    ]
    if pooled.empty:
        lines.append("No pooled metrics were generated.")
    else:
        keep = [
            "classifier",
            "criterion",
            "threshold_strategy",
            "roc_auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "f1",
            "tp",
            "fn",
            "tn",
            "fp",
        ]
        lines.append(pooled[keep].to_markdown(index=False, floatfmt=".3f"))

    lines.extend(
        [
            "",
            "## Clinical Interpretation",
            "",
            "- This audit implements threshold selection without looking at outer test labels for threshold choice.",
            "- The selected thresholds can improve AD sensitivity by accepting lower CN specificity; the exact trade-off is criterion-dependent.",
            "- For a final report, prefer an inner-CV out-of-fold threshold rerun over train/dev resubstitution if time allows.",
            "",
            "## Recommended Lightweight Rerun",
            "",
            "If we want the strictest clinical threshold selection, rerun only the classifier stage on saved/exported fold latents:",
            "",
            "1. Export or cache fold train/dev latents from the saved VAE checkpoints.",
            "2. For each outer fold, run inner-CV classifier fits and save out-of-fold train/dev probabilities.",
            "3. Select thresholds on those inner out-of-fold probabilities.",
            "4. Apply the fixed threshold to the untouched outer test fold.",
            "",
            "This still does not retrain the VAE and does not modify tensors.",
            "",
            "## Outputs",
            "",
            "- `prediction_artifact_inventory.csv`",
            "- `train_dev_predictions_for_threshold_selection.csv`",
            "- `threshold_by_fold.csv`",
            "- `metrics_threshold_0p5_vs_selected.csv`",
            "- `confusion_by_fold_selected_threshold.csv`",
            "- `subgroup_metrics_by_manufacturer_selected_threshold.csv`",
            "- `command_log.json`",
            "",
            "## Provenance",
            "",
            f"- Run dir: `{run_dir}`",
            f"- Tensor: `{cfg['global_tensor_path']}`",
            f"- Metadata: `{cfg['metadata_path']}`",
            f"- Channels: `{cfg['channels_to_use']}`",
            f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    cfg = load_run_config(run_dir)
    classifiers = cfg["classifier_types"]
    if sorted(classifiers) != sorted(EXPECTED_CLASSIFIERS):
        raise RuntimeError(f"Unexpected classifiers in run_config: {classifiers}")

    tensor_path = resolve(cfg["global_tensor_path"])
    metadata_path = resolve(cfg["metadata_path"])
    require_files([tensor_path, metadata_path])

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    inventory = prediction_artifact_inventory(run_dir, classifiers)
    inventory.to_csv(outdir / "prediction_artifact_inventory.csv", index=False)

    metadata = normalize_metadata(pd.read_csv(metadata_path))
    tensor_info = load_selected_tensor(tensor_path, cfg["channels_to_use"])
    if tensor_info["python_bandpass_applied"] is not False:
        raise RuntimeError(f"Expected python_bandpass_applied=False, got {tensor_info['python_bandpass_applied']}")
    tensor = tensor_info["tensor"]

    train_pred = generate_train_dev_predictions(
        run_dir=run_dir,
        cfg=cfg,
        tensor=tensor,
        metadata=metadata,
        classifiers=classifiers,
        batch_size=int(args.batch_size),
        device=device,
    )
    train_pred.to_csv(outdir / "train_dev_predictions_for_threshold_selection.csv", index=False)

    thresholds = make_threshold_rows(train_pred)
    thresholds.to_csv(outdir / "threshold_by_fold.csv", index=False)

    test_pred = load_test_predictions(run_dir, metadata, classifiers)
    metrics, confusion, selected_pred = evaluate_thresholds(test_pred, thresholds)
    metrics.to_csv(outdir / "metrics_threshold_0p5_vs_selected.csv", index=False)
    confusion.to_csv(outdir / "confusion_by_fold_selected_threshold.csv", index=False)
    subgroup = subgroup_metrics_by_manufacturer(selected_pred)
    subgroup.to_csv(outdir / "subgroup_metrics_by_manufacturer_selected_threshold.csv", index=False)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "output_dir": str(outdir),
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "channels_to_use": cfg["channels_to_use"],
        "selected_channel_names": cfg["selected_channel_names"],
        "classifiers": classifiers,
        "device": str(device),
        "n_train_dev_prediction_rows": int(len(train_pred)),
        "n_outer_test_prediction_rows": int(len(test_pred)),
        "inner_or_train_prediction_csvs_found": int(
            inventory["inner_or_train_prediction_files_found"].astype(str).str.len().gt(0).sum()
        ),
        "train_dev_scores_generated_from_saved_models": True,
        "threshold_selection_used_outer_test": False,
        "vae_retrained": False,
        "classifier_retrained": False,
        "tensor_modified": False,
        "ledger_modified": False,
        "python_bandpass_applied": False,
        "selection_context": "train_dev_resubstitution_no_outer_test",
        "recommended_stricter_followup": "inner_cv_oof_threshold_rerun_on_saved_or_exported_latents_no_vae_retraining",
    }
    write_json(outdir / "command_log.json", command_log)
    make_readme(outdir, run_dir, cfg, inventory, thresholds, metrics)

    pooled = metrics[metrics["scope"].eq("outer_test_pooled")].copy()
    print(f"output_dir={outdir}")
    print("train_dev_scores_generated_from_saved_models=True")
    print("threshold_selection_used_outer_test=False")
    print("vae_retrained=False")
    print("classifier_retrained=False")
    if not pooled.empty:
        cols = ["classifier", "criterion", "threshold_strategy", "sensitivity", "specificity", "balanced_accuracy", "f1"]
        print(pooled[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
