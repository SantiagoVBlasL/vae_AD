#!/usr/bin/env python3
"""Evaluate leakage-safe downstream-guided VAE checkpoint selection.

For each outer fold, this script selects among saved VAE checkpoints using only
train/dev subjects and inner-CV downstream ROC-AUC. After one checkpoint and
readout are selected, the held-out outer test fold is encoded once and evaluated.

No VAE training is performed here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"Missing src directory: {SRC_DIR}")

from betavae_xai.data.preprocessing import apply_normalization_params, normalize_inter_channel_fold
from betavae_xai.models import ConvolutionalVAE


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_ckptselect"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/downstream_checkpoint_selection"
)
DEFAULT_CONFIG_FALLBACK = PROJECT_ROOT / "configs/runs/adni_expanded_v4_beta25_ch4_1_0_ckptselect.json"

GENERATED_FILES = [
    "checkpoint_selection_by_fold.csv",
    "test_metrics_by_fold.csv",
    "pooled_predictions.csv",
    "README.md",
    "downstream_checkpoint_selection_manifest.json",
]

EPOCH_PATTERNS = [
    re.compile(r"(?:epoch|ep|e)[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"_(\d{3,5})(?:\.pt|\.pth)$", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-fallback", type=Path, default=DEFAULT_CONFIG_FALLBACK)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-rbf-svm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def load_config(run_dir: Path, fallback: Path) -> Dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_cfg = read_json(run_config_path)
        args = run_cfg.get("args", {})
        return {
            "source": str(run_config_path),
            "global_tensor_path": run_cfg.get("global_tensor_path") or args.get("global_tensor_path"),
            "metadata_path": run_cfg.get("metadata_path") or args.get("metadata_path"),
            "channels_to_use": run_cfg.get("channels_to_use_indices") or args.get("channels_to_use"),
            "selected_channel_names": run_cfg.get("channel_names_selected"),
            "latent_dim": args.get("latent_dim", 256),
            "dropout_rate_vae": args.get("dropout_rate_vae", 0.15),
            "vae_final_activation": args.get("vae_final_activation", "tanh"),
            "intermediate_fc_dim_vae": args.get("intermediate_fc_dim_vae", "quarter"),
            "use_layernorm_vae_fc": bool(args.get("use_layernorm_vae_fc", False)),
            "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
            "decoder_type": args.get("decoder_type", "convtranspose"),
            "norm_mode": args.get("norm_mode", "zscore_offdiag"),
            "beta_vae": args.get("beta_vae", 2.5),
            "metadata_features": args.get("metadata_features", ["Age", "Sex"]),
        }

    fallback = resolve(fallback)
    cfg = read_json(fallback)
    params = cfg.get("parameters", {})
    paths = cfg.get("paths", {})
    return {
        "source": str(fallback),
        "global_tensor_path": paths.get("global_tensor_path"),
        "metadata_path": paths.get("metadata_path"),
        "channels_to_use": params.get("channels_to_use"),
        "selected_channel_names": cfg.get("selected_channel_names"),
        "latent_dim": params.get("latent_dim", 256),
        "dropout_rate_vae": params.get("dropout_rate_vae", 0.15),
        "vae_final_activation": params.get("vae_final_activation", "tanh"),
        "intermediate_fc_dim_vae": params.get("intermediate_fc_dim_vae", "quarter"),
        "use_layernorm_vae_fc": bool(params.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(params.get("num_conv_layers_encoder", 4)),
        "decoder_type": params.get("decoder_type", "convtranspose"),
        "norm_mode": params.get("norm_mode", "zscore_offdiag"),
        "beta_vae": params.get("beta_vae", 2.5),
        "metadata_features": params.get("metadata_features", ["Age", "Sex"]),
    }


def validate_scope(cfg: Dict[str, Any]) -> None:
    channels = [int(c) for c in cfg["channels_to_use"]]
    if channels != [4, 1, 0]:
        raise ValueError(f"This evaluator is scoped to channels [4,1,0]; config requested {channels}")
    expected = {
        "beta_vae": 2.5,
        "latent_dim": 256,
        "dropout_rate_vae": 0.15,
        "vae_final_activation": "tanh",
        "use_layernorm_vae_fc": False,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Scope guard failed for {key}: got {cfg.get(key)!r}, expected {value!r}")
    metadata_features = list(cfg.get("metadata_features", []))
    if metadata_features != ["Age", "Sex"]:
        raise ValueError(f"Metadata features must be exactly ['Age', 'Sex']; got {metadata_features!r}")


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def infer_epoch(path: Path) -> Optional[int]:
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(path.name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def checkpoint_role(path: Path, fold: int) -> str:
    if path.name == f"vae_model_fold_{fold}.pt":
        return "final_best_checkpoint"
    if "checkpoint" in path.name.lower():
        return "periodic_checkpoint"
    return "unknown_checkpoint"


def discover_checkpoints(fold_dir: Path, fold: int) -> List[Path]:
    patterns = [
        f"vae_model_fold_{fold}.pt",
        "vae_checkpoint*.pt",
        "vae_checkpoint*.pth",
        "checkpoint*.pt",
        "checkpoint*.pth",
        "vae_checkpoints/*.pt",
        "vae_checkpoints/*.pth",
    ]
    found: Dict[Path, None] = {}
    for pattern in patterns:
        for path in fold_dir.glob(pattern):
            if path.is_file():
                found[path] = None
    return sorted(found.keys(), key=lambda p: (infer_epoch(p) is None, infer_epoch(p) or 10**9, str(p)))


def load_selected_tensor(tensor_path: Path, channels: List[int]) -> Dict[str, Any]:
    with np.load(tensor_path, allow_pickle=True) as npz:
        if "global_tensor_data" not in npz.files:
            raise KeyError(f"{tensor_path} does not contain global_tensor_data")
        tensor = np.asarray(npz["global_tensor_data"][:, channels, :, :], dtype=np.float32)
        channel_names = np.asarray(npz["channel_names"]).astype(str).tolist() if "channel_names" in npz.files else None
    return {"tensor": tensor, "channel_names": channel_names}


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


def state_dict_from_checkpoint(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ["model_state_dict", "state_dict", "vae_state_dict"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise ValueError("Unsupported checkpoint format; expected raw state_dict or dict with model_state_dict")


def encode_mu(model: ConvolutionalVAE, tensor: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    mus: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = torch.from_numpy(tensor[start : start + batch_size]).float().to(device)
            mu, _logvar = model.encode(batch)
            mus.append(mu.detach().cpu().numpy())
    return np.concatenate(mus, axis=0)


def encode_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    train_norm: np.ndarray,
    test_norm: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model = make_model(cfg, image_size=train_norm.shape[-1], n_channels=train_norm.shape[1], device=device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict_from_checkpoint(payload))
    model.eval()
    mu_train = encode_mu(model, train_norm, batch_size, device)
    mu_test = encode_mu(model, test_norm, batch_size, device) if test_norm is not None else None
    del model, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mu_train, mu_test


def subject_base(subjects: pd.DataFrame, metadata: pd.DataFrame, fold: int, split: str) -> pd.DataFrame:
    base = subjects.copy()
    base["SubjectID"] = base["SubjectID"].astype(str)
    meta_cols = [c for c in ["SubjectID", "Age", "Sex"] if c in metadata.columns]
    if "SubjectID" not in meta_cols:
        raise ValueError("Metadata must contain SubjectID")
    base = base.merge(metadata[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    if "ResearchGroup_Mapped" not in base.columns:
        raise ValueError("Fold subject file must contain ResearchGroup_Mapped")
    base["y_true"] = base["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    if base["y_true"].isna().any():
        raise ValueError(f"Fold {fold} {split} contains non-CN/AD labels")
    base["fold"] = fold
    base["split"] = split
    return base


def feature_matrix(base: pd.DataFrame, mu: np.ndarray) -> pd.DataFrame:
    if len(base) != mu.shape[0]:
        raise ValueError(f"Subject/mu length mismatch: {len(base)} vs {mu.shape[0]}")
    latent = pd.DataFrame(mu, columns=[f"mu_{i:03d}" for i in range(mu.shape[1])])
    features = pd.concat([latent, base[["Age", "Sex"]].reset_index(drop=True)], axis=1)
    features["Age"] = pd.to_numeric(features["Age"], errors="coerce")
    features["Sex"] = features["Sex"].astype(str).replace({"nan": np.nan, "None": np.nan})
    features = pd.get_dummies(features, columns=["Sex"], drop_first=False)
    forbidden = [c for c in features.columns if c.lower().startswith(("manufacturer", "site"))]
    if forbidden:
        raise ValueError(f"Forbidden scanner/site features present: {forbidden}")
    return features.astype(float)


def align_test_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> pd.DataFrame:
    return test_x.reindex(columns=train_x.columns, fill_value=0).astype(float)


def classifier_grid(seed: int, include_rbf_svm: bool) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    grids: Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]] = {
        "lda_shrinkage": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", LinearDiscriminantAnalysis(solver="lsqr")),
                ]
            ),
            {"clf__shrinkage": ["auto", 0.5]},
        ),
        "logreg_l2": (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            solver="liblinear",
                            penalty="l2",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
    }
    if include_rbf_svm:
        grids["rbf_svm"] = (
            Pipeline(
                [
                    ("impute", SimpleImputer()),
                    ("scale", StandardScaler()),
                    ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
                ]
            ),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.01]},
        )
    return grids


def score_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float)
        score = 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
    else:
        score = np.asarray(model.predict(x), dtype=float)
    return np.clip(score, 1e-6, 1.0 - 1e-6)


def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (y_score >= low) & (y_score <= high) if i == n_bins - 1 else (y_score >= low) & (y_score < high)
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_score[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def metrics_from_scores(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_CN": int((y == 0).sum()),
        "n_AD": int((y == 1).sum()),
        "roc_auc": roc_auc_score(y, score) if len(set(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(set(y)) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity_AD": tp / (tp + fn) if tp + fn else np.nan,
        "specificity_CN": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, score),
        "ece": ece_score(y, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_readout_for_checkpoint(
    train_x: pd.DataFrame,
    y_train: np.ndarray,
    fold: int,
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
    checkpoint_role_name: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.random_seed)
    for classifier, (pipe, grid) in classifier_grid(args.random_seed, include_rbf_svm=not args.skip_rbf_svm).items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1, error_score="raise", refit=True)
        search.fit(train_x, y_train)
        rows.append(
            {
                "fold": fold,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_role": checkpoint_role_name,
                "epoch_inferable": checkpoint_epoch,
                "classifier": classifier,
                "inner_cv_auc_trainDev": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "selected": False,
            }
        )
    return rows


def fit_selected_readout(
    classifier: str,
    best_params_json: str,
    train_x: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    pipe, _grid = classifier_grid(seed, include_rbf_svm=True)[classifier]
    best_params = json.loads(best_params_json) if best_params_json else {}
    pipe.set_params(**best_params)
    pipe.fit(train_x, y_train)
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    calibrated.fit(train_x, y_train)
    return calibrated


def write_readme(outdir: Path, selection: pd.DataFrame, test_metrics: pd.DataFrame, pooled: Dict[str, Any]) -> None:
    if test_metrics.empty:
        body = "No folds were evaluated."
    else:
        body = (
            f"Mean-fold ROC-AUC: {test_metrics['roc_auc'].mean():.4f} +/- "
            f"{test_metrics['roc_auc'].std(ddof=1):.4f}; pooled ROC-AUC: {pooled['roc_auc']:.4f}."
        )

    selected_table = (
        selection[selection["selected"]][
            ["fold", "checkpoint_name", "epoch_inferable", "classifier", "inner_cv_auc_trainDev"]
        ].to_markdown(index=False, floatfmt=".4f")
        if not selection.empty
        else ""
    )

    lines = [
        "# Downstream-Guided Checkpoint Selection",
        "",
        "This evaluation does not change the CVAE architecture and does not retrain the VAE. It selects among already saved VAE checkpoints.",
        "",
        "- Fixed architecture: channels `[4,1,0]`, beta `2.5`, latent dim `256`, dropout `0.15`, tanh decoder, no LayerNorm.",
        "- Predictive features: checkpoint-specific `mu` plus Age and Sex.",
        "- Manufacturer/Site features are not used.",
        "- Checkpoint and readout selection use only each outer fold train/dev split.",
        "- Held-out test folds are encoded and evaluated once after checkpoint selection.",
        "",
        "## Result",
        "",
        body,
        "",
        "## Selected Checkpoints",
        "",
        selected_table,
        "",
        "## Comparison Targets",
        "",
        "- Baseline tanh [4,1,0]: original reconstruction-selected checkpoint and original downstream classifiers.",
        "- Frozen-latent LDA result: all `mu` + LDA shrinkage, approximately 0.771 mean-fold ROC-AUC in the latent selection sweep.",
        "",
        "## Outputs",
        "",
        "- `checkpoint_selection_by_fold.csv`: all train/dev checkpoint/readout inner-CV scores and selected rows.",
        "- `test_metrics_by_fold.csv`: held-out test metrics after selection.",
        "- `pooled_predictions.csv`: held-out test predictions pooled across folds.",
        "- `downstream_checkpoint_selection_manifest.json`: run manifest.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    cfg = load_config(run_dir, args.config_fallback)
    validate_scope(cfg)

    tensor_path = resolve(Path(str(cfg["global_tensor_path"])))
    metadata_path = resolve(Path(str(cfg["metadata_path"])))
    channels = [int(c) for c in cfg["channels_to_use"]]
    require_files([tensor_path, metadata_path])

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    tensor_info = load_selected_tensor(tensor_path, channels)
    tensor = tensor_info["tensor"]
    metadata = pd.read_csv(metadata_path)
    selected_channel_names = cfg.get("selected_channel_names") or [
        tensor_info["channel_names"][i] for i in channels
    ]

    selection_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []

    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        checkpoints = discover_checkpoints(fold_dir, fold)
        if not checkpoints:
            raise FileNotFoundError(f"No VAE checkpoints found for fold {fold}: {fold_dir}")

        train_subjects_path = fold_dir / "train_dev_subjects_fold.csv"
        test_subjects_path = fold_dir / "test_subjects_fold.csv"
        vae_pool_idx_path = fold_dir / "vae_training_pool_tensor_idx.npy"
        vae_train_local_idx_path = fold_dir / "vae_actual_train_idx_local_to_pool.npy"
        require_files([train_subjects_path, test_subjects_path, vae_pool_idx_path, vae_train_local_idx_path])

        train_subjects = pd.read_csv(train_subjects_path)
        test_subjects = pd.read_csv(test_subjects_path)
        train_base = subject_base(train_subjects, metadata, fold, "trainDev")
        test_base = subject_base(test_subjects, metadata, fold, "test")
        y_train = train_base["y_true"].astype(int).to_numpy()
        y_test = test_base["y_true"].astype(int).to_numpy()

        vae_pool_idx = np.load(vae_pool_idx_path).astype(int)
        vae_train_local_idx = np.load(vae_train_local_idx_path).astype(int)
        vae_pool_tensor = tensor[vae_pool_idx]
        _pool_norm, norm_params = normalize_inter_channel_fold(
            vae_pool_tensor,
            vae_train_local_idx,
            mode=str(cfg["norm_mode"]),
            selected_channel_original_names=list(selected_channel_names),
        )
        train_norm = apply_normalization_params(tensor[train_subjects["tensor_idx"].astype(int).to_numpy()], norm_params)
        test_norm = apply_normalization_params(tensor[test_subjects["tensor_idx"].astype(int).to_numpy()], norm_params)

        fold_candidate_rows: List[Dict[str, Any]] = []
        for checkpoint_path in checkpoints:
            mu_train, _ = encode_checkpoint(cfg, checkpoint_path, train_norm, None, args.batch_size, device)
            train_x = feature_matrix(train_base, mu_train)
            fold_candidate_rows.extend(
                select_readout_for_checkpoint(
                    train_x,
                    y_train,
                    fold,
                    checkpoint_path,
                    infer_epoch(checkpoint_path),
                    checkpoint_role(checkpoint_path, fold),
                    args,
                )
            )

        if not fold_candidate_rows:
            raise RuntimeError(f"No checkpoint/readout candidates completed for fold {fold}")

        best_row = max(fold_candidate_rows, key=lambda row: row["inner_cv_auc_trainDev"])
        for row in fold_candidate_rows:
            row["selected"] = (
                row["checkpoint_path"] == best_row["checkpoint_path"]
                and row["classifier"] == best_row["classifier"]
            )
        selection_rows.extend(fold_candidate_rows)

        selected_checkpoint = Path(best_row["checkpoint_path"])
        mu_train, mu_test = encode_checkpoint(cfg, selected_checkpoint, train_norm, test_norm, args.batch_size, device)
        if mu_test is None:
            raise RuntimeError("Selected checkpoint test encoding unexpectedly missing")
        train_x = feature_matrix(train_base, mu_train)
        test_x = align_test_features(train_x, feature_matrix(test_base, mu_test))
        final_model = fit_selected_readout(
            best_row["classifier"],
            best_row["best_params"],
            train_x,
            y_train,
            args.random_seed,
        )
        score = score_model(final_model, test_x)
        metrics = metrics_from_scores(y_test, score)
        metric_rows.append(
            {
                "fold": fold,
                "selected_checkpoint_path": str(selected_checkpoint),
                "selected_checkpoint_name": selected_checkpoint.name,
                "selected_checkpoint_role": checkpoint_role(selected_checkpoint, fold),
                "selected_epoch_inferable": infer_epoch(selected_checkpoint),
                "selected_classifier": best_row["classifier"],
                "selected_inner_cv_auc_trainDev": best_row["inner_cv_auc_trainDev"],
                "selected_best_params": best_row["best_params"],
                **metrics,
            }
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "SubjectID": test_base["SubjectID"].values,
                    "fold": fold,
                    "selected_checkpoint_name": selected_checkpoint.name,
                    "selected_classifier": best_row["classifier"],
                    "y_true": y_test,
                    "y_score": score,
                    "y_pred": (score >= 0.5).astype(int),
                }
            )
        )

    selection_df = pd.DataFrame(selection_rows)
    metrics_df = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    pooled_metrics = metrics_from_scores(
        preds["y_true"].astype(int).to_numpy(),
        preds["y_score"].astype(float).to_numpy(),
    )

    selection_df.to_csv(outdir / "checkpoint_selection_by_fold.csv", index=False)
    metrics_df.to_csv(outdir / "test_metrics_by_fold.csv", index=False)
    preds.to_csv(outdir / "pooled_predictions.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()) if run_dir.exists() else None,
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "config_source": cfg["source"],
        "channels_to_use": channels,
        "selected_channel_names": selected_channel_names,
        "vae_retrained_by_this_script": False,
        "architecture_modified": False,
        "metadata_features": ["Age", "Sex"],
        "manufacturer_site_features_used": False,
        "inner_folds": args.inner_folds,
        "readouts": list(classifier_grid(args.random_seed, include_rbf_svm=not args.skip_rbf_svm).keys()),
        "pooled_metrics": pooled_metrics,
    }
    (outdir / "downstream_checkpoint_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(outdir, selection_df, metrics_df, pooled_metrics)

    print(metrics_df.to_string(index=False))
    print(json.dumps({"pooled_metrics": pooled_metrics, "output_dir": str(outdir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
