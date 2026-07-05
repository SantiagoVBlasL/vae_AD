#!/usr/bin/env python3
"""
Post-hoc metadata and latent baselines using the same outer fold splits from a VAE run.

Uses fold_N/train_dev_subjects_fold.csv and fold_N/test_subjects_fold.csv
(saved by run_vae_clf_ad_inference.py with save_fold_artefacts=True) to replicate
the exact outer-fold assignments without retraining the VAE.

Baselines:
  age_sex        — LogReg on [Age, Sex_binary]
  scanner        — LogReg on OHE(Manufacturer, Site3)   ← leakage check, not a model
  latent         — LogReg on VAE mu (requires saved model or pre-extracted CSV)
  latent_age_sex — LogReg on [VAE mu, Age, Sex_binary]

Classifier: LogisticRegression(C=1.0, class_weight='balanced') + StandardScaler.
            Fixed C — intentionally simpler than the gridsearch used for the main model.

Latent extraction mode (--extract-latent):
  Loads fold_N/vae_model_fold_N.pt + fold_N/vae_norm_params.joblib,
  selects channels (from --config or --channels-to-use),
  normalises, encodes through VAE, saves fold_N/latent_representations_all.csv.

Outputs (in --output-dir or results_dir/baselines/):
  baselines_by_fold.csv
  baselines_pooled.csv
  baselines_vs_vae_comparison.csv   (if VAE test_predictions_*.csv are found)
  README_baselines.md
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRIC_COLS = [
    "baseline", "fold",
    "N", "N_AD", "N_CN",
    "roc_auc", "pr_auc", "balanced_accuracy",
    "accuracy", "sensitivity_AD", "specificity_CN", "f1",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc metadata and latent baselines for an ADNI expanded VAE run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Run results dir containing fold_1/, fold_2/, …")
    p.add_argument("--metadata-path", type=Path, required=True,
                   help="Full subject_metadata CSV (all subjects).")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON run config (to read channels_to_use, latent_dim, seed).")
    p.add_argument("--global-tensor-path", type=Path, default=None,
                   help="Global tensor NPZ (needed for --extract-latent).")
    p.add_argument("--manifest-path", type=Path, default=None,
                   help="Per-subject manifest CSV (for extra SourceCohort column).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir. Defaults to results_dir/baselines/.")
    p.add_argument("--positive-label", type=str, default="AD",
                   help="ResearchGroup_Mapped label for the positive class.")
    p.add_argument(
        "--baselines", nargs="+",
        choices=["age_sex", "scanner", "latent", "latent_age_sex", "all"],
        default=["age_sex", "scanner"],
        help="Baseline types to run. 'all' expands to all four types.",
    )
    p.add_argument(
        "--extract-latent", action="store_true",
        help="Extract and save latent representations from each fold's saved VAE model.",
    )
    p.add_argument("--latent-csv", type=Path, default=None,
                   help="Pre-extracted latent representations CSV (SubjectID + latent_0…).")
    p.add_argument("--channels-to-use", type=int, nargs="+", default=None,
                   help="Channel indices to select from the global tensor (overrides config).")
    p.add_argument("--latent-dim", type=int, default=None,
                   help="VAE latent dimension (overrides config).")
    p.add_argument("--classifier-C", type=float, default=1.0,
                   help="LogReg regularisation strength C.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for VAE inference ('cpu' or 'cuda').")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Fold discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_folds(results_dir: Path) -> List[Dict]:
    """Return list of fold dicts with paths; folds are 1-indexed (fold_1, fold_2…)."""
    folds = []
    for fold_dir in sorted(results_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold_n = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        train_csv = fold_dir / "train_dev_subjects_fold.csv"
        test_csv = fold_dir / "test_subjects_fold.csv"
        if not train_csv.exists() or not test_csv.exists():
            continue
        vae_model = fold_dir / f"vae_model_fold_{fold_n}.pt"
        norm_params = fold_dir / "vae_norm_params.joblib"
        folds.append(dict(
            n=fold_n,
            dir=fold_dir,
            train_csv=train_csv,
            test_csv=test_csv,
            vae_model=vae_model if vae_model.exists() else None,
            norm_params=norm_params if norm_params.exists() else None,
            latent_csv=fold_dir / "latent_representations_all.csv",
        ))
    return folds


def load_fold_subjects(fold: Dict, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test subject frames and join with full metadata."""
    train_raw = pd.read_csv(fold["train_csv"])
    test_raw = pd.read_csv(fold["test_csv"])

    train_raw["SubjectID"] = train_raw["SubjectID"].astype(str).str.strip()
    test_raw["SubjectID"] = test_raw["SubjectID"].astype(str).str.strip()

    meta_idx = metadata.set_index("SubjectID")

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        base_cols = [c for c in df.columns]
        extra_cols = [c for c in ["Age", "Sex", "Manufacturer", "Site3", "SourceCohort"]
                      if c in meta_idx.columns and c not in base_cols]
        if extra_cols:
            extra = meta_idx.loc[df["SubjectID"].values, extra_cols].reset_index(drop=True)
            df = pd.concat([df.reset_index(drop=True), extra], axis=1)
        return df

    return enrich(train_raw), enrich(test_raw)


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prep_age_sex(df: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(df["Age"], errors="coerce")
    age = age.fillna(float(age.median()))
    sex = df["Sex"].astype(str).str.strip().str.upper().str[:1].map({"M": 1.0, "F": 0.0}).fillna(0.5)
    return np.column_stack([age.values, sex.values]).astype(np.float32)


def prep_scanner_ohe(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot encode Manufacturer + Site3; fit on train, align test columns."""
    cols = [c for c in ["Manufacturer", "Site3"] if c in train_df.columns]
    if not cols:
        return np.zeros((len(train_df), 0)), np.zeros((len(test_df), 0))

    # Pass a subset DataFrame; pd.get_dummies uses column names as prefix automatically.
    train_sub = train_df[cols].astype(str)
    test_sub = test_df[[c for c in cols if c in test_df.columns]].astype(str)

    train_ohe = pd.get_dummies(train_sub, dtype=float)
    test_ohe = pd.get_dummies(test_sub, dtype=float)
    # Align test to train columns (unseen categories → 0)
    test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0.0)
    return train_ohe.values.astype(np.float32), test_ohe.values.astype(np.float32)


def get_labels(df: pd.DataFrame, positive_label: str) -> np.ndarray:
    col = "ResearchGroup_Mapped" if "ResearchGroup_Mapped" in df.columns else "y_true"
    return (df[col].astype(str) == positive_label).astype(int).values


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

def make_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced",
            random_state=42, solver="lbfgs",
        )),
    ])


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, baseline: str, fold_n: int,
) -> Dict:
    y_pred = (y_score >= 0.5).astype(int)
    n = len(y_true)
    n_ad = int(y_true.sum())
    n_cn = n - n_ad

    row: Dict = {
        "baseline": baseline,
        "fold": fold_n,
        "N": n, "N_AD": n_ad, "N_CN": n_cn,
        "roc_auc": None, "pr_auc": None,
        "balanced_accuracy": None, "accuracy": None,
        "sensitivity_AD": None, "specificity_CN": None, "f1": None,
    }
    if n_ad == 0 or n_cn == 0:
        return row

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            row["roc_auc"] = float(roc_auc_score(y_true, y_score))
            row["pr_auc"] = float(average_precision_score(y_true, y_score))
            row["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
            row["accuracy"] = float(accuracy_score(y_true, y_pred))
            row["sensitivity_AD"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
            row["specificity_CN"] = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
            row["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        except Exception:
            pass
    return row


def run_one_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    name: str, fold_n: int, C: float,
) -> Dict:
    if X_train.shape[1] == 0:
        return {"baseline": name, "fold": fold_n, "error": "no features"}
    pipe = make_pipeline(C)
    try:
        pipe.fit(X_train, y_train)
        y_score = pipe.predict_proba(X_test)[:, 1]
    except Exception as exc:
        return {"baseline": name, "fold": fold_n, "error": str(exc)}
    return compute_metrics(y_test, y_score, name, fold_n)


# ─────────────────────────────────────────────────────────────────────────────
# Latent representations
# ─────────────────────────────────────────────────────────────────────────────

def load_latent_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load pre-extracted latent CSV with SubjectID index."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "SubjectID" not in df.columns:
        return None
    df["SubjectID"] = df["SubjectID"].astype(str).str.strip()
    return df.set_index("SubjectID")


def extract_latent_from_vae(
    fold: Dict,
    global_tensor: np.ndarray,
    tensor_subject_ids: np.ndarray,
    channels_to_use: List[int],
    latent_dim: int,
    device: str = "cpu",
) -> Optional[pd.DataFrame]:
    """
    Load saved VAE model + norm params, encode all subjects, return DataFrame.
    Saved to fold_dir/latent_representations_all.csv for future reuse.
    """
    try:
        import torch
        import joblib
        from betavae_xai.models import ConvolutionalVAE
        from betavae_xai.data.preprocessing import apply_normalization_params
    except ImportError as e:
        print(f"  [latent] Import error: {e}. Install torch + betavae_xai to use --extract-latent.")
        return None

    vae_path = fold["vae_model"]
    norm_path = fold["norm_params"]
    if vae_path is None or norm_path is None:
        print(f"  [latent] fold_{fold['n']}: VAE model or norm_params not found. Skipping.")
        return None

    print(f"  [latent] fold_{fold['n']}: loading VAE from {vae_path}")
    try:
        norm_params = joblib.load(str(norm_path))

        # Select channels and normalise
        tensor_sel = global_tensor[:, channels_to_use, :, :].astype(np.float32)
        tensor_normed = apply_normalization_params(tensor_sel, norm_params)

        image_size = global_tensor.shape[-1]
        vae = ConvolutionalVAE(
            input_channels=len(channels_to_use),
            latent_dim=latent_dim,
            image_size=image_size,
        )
        state = torch.load(str(vae_path), map_location=device)
        vae.load_state_dict(state)
        vae.eval()
        vae.to(device)

        batch_size = 32
        mus: List[np.ndarray] = []
        n_total = tensor_normed.shape[0]
        with torch.no_grad():
            for start in range(0, n_total, batch_size):
                batch = torch.FloatTensor(tensor_normed[start:start + batch_size]).to(device)
                _, mu, _, _ = vae(batch)
                mus.append(mu.cpu().numpy())

        mu_np = np.concatenate(mus, axis=0)
        cols = [f"latent_{i}" for i in range(latent_dim)]
        df = pd.DataFrame(mu_np, columns=cols)
        df.insert(0, "SubjectID", tensor_subject_ids.astype(str))
        df = df.set_index("SubjectID")

        # Save for future reuse
        save_path = fold["dir"] / "latent_representations_all.csv"
        df.reset_index().to_csv(save_path, index=False)
        print(f"  [latent] Saved: {save_path}")
        return df

    except Exception as exc:
        print(f"  [latent] fold_{fold['n']}: extraction failed: {exc}")
        return None


def get_latent_for_fold(
    fold: Dict,
    global_tensor: Optional[np.ndarray],
    tensor_subject_ids: Optional[np.ndarray],
    channels_to_use: List[int],
    latent_dim: int,
    do_extract: bool,
    shared_latent: Optional[pd.DataFrame],
    device: str,
) -> Optional[pd.DataFrame]:
    """Return DataFrame (SubjectID index, latent_0…latent_D columns) or None."""
    # 1. Use shared pre-extracted CSV if provided
    if shared_latent is not None:
        return shared_latent
    # 2. Look for fold-local latent CSV
    latent_df = load_latent_csv(fold["latent_csv"])
    if latent_df is not None:
        return latent_df
    # 3. Extract on-the-fly if requested
    if do_extract and global_tensor is not None and tensor_subject_ids is not None:
        return extract_latent_from_vae(
            fold, global_tensor, tensor_subject_ids, channels_to_use, latent_dim, device
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VAE main-model metrics (from saved test_predictions CSVs)
# ─────────────────────────────────────────────────────────────────────────────

def load_vae_metrics(results_dir: Path, positive_label: str) -> List[Dict]:
    """Pool test_predictions_*.csv from all folds and compute per-fold metrics."""
    rows: List[Dict] = []
    for fold_dir in sorted(results_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold_n = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for pred_csv in sorted(fold_dir.glob("test_predictions_*.csv")):
            clf = pred_csv.stem.replace("test_predictions_", "")
            df = pd.read_csv(pred_csv)
            score_col = next(
                (c for c in ["y_score_final", "y_score_cal", "y_score_raw", "y_score"] if c in df.columns),
                None,
            )
            true_col = next(
                (c for c in ["y_true", "ResearchGroup_Mapped", "label"] if c in df.columns),
                None,
            )
            if score_col is None or true_col is None:
                continue
            y_true = (df[true_col].astype(str) == positive_label).astype(int).values
            y_score = pd.to_numeric(df[score_col], errors="coerce").values
            valid = ~np.isnan(y_score)
            m = compute_metrics(y_true[valid], y_score[valid], f"vae_{clf}", fold_n)
            rows.append(m)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────

def pool_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled (macro-average) statistics across folds per baseline."""
    if df.empty:
        return df
    numeric = [c for c in METRIC_COLS if c not in ("baseline", "fold")]
    rows = []
    for name, sub in df.groupby("baseline"):
        row: Dict = {"baseline": name, "fold": "pooled"}
        for col in numeric:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                row[col] = float(vals.mean()) if len(vals) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.results_dir / "baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Expand "all" shorthand ────────────────────────────────────────────────
    baselines_to_run: List[str] = []
    for b in args.baselines:
        if b == "all":
            baselines_to_run = ["age_sex", "scanner", "latent", "latent_age_sex"]
            break
        baselines_to_run.append(b)
    need_latent = any(b in ("latent", "latent_age_sex") for b in baselines_to_run)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg: Dict = {}
    if args.config and args.config.exists():
        with args.config.open() as f:
            cfg = json.load(f)

    channels_to_use: List[int] = (
        args.channels_to_use
        or cfg.get("parameters", {}).get("channels_to_use")
        or [1, 0, 2]
    )
    latent_dim: int = (
        args.latent_dim
        or cfg.get("parameters", {}).get("latent_dim")
        or 256
    )

    # ── Load metadata ─────────────────────────────────────────────────────────
    if not args.metadata_path.exists():
        print(f"ERROR: metadata not found: {args.metadata_path}")
        return 1
    metadata = pd.read_csv(args.metadata_path)
    metadata["SubjectID"] = metadata["SubjectID"].astype(str).str.strip()

    # Optionally enrich with manifest (adds SourceCohort if missing)
    if args.manifest_path and args.manifest_path.exists():
        mfst = pd.read_csv(args.manifest_path)
        mfst["SubjectID"] = mfst["SubjectID"].astype(str).str.strip()
        for col in ["SourceCohort", "IsMartin59", "IsSantiProcessed"]:
            if col in mfst.columns and col not in metadata.columns:
                metadata = metadata.merge(
                    mfst[["SubjectID", col]], on="SubjectID", how="left"
                )

    # ── Load global tensor (for latent extraction) ────────────────────────────
    global_tensor: Optional[np.ndarray] = None
    tensor_subject_ids: Optional[np.ndarray] = None
    if need_latent and (args.extract_latent or args.global_tensor_path):
        tensor_path = args.global_tensor_path or (
            PROJECT_ROOT / cfg.get("paths", {}).get("global_tensor_path", "")
            if cfg else None
        )
        if tensor_path and tensor_path.exists():
            print(f"Loading global tensor from {tensor_path} …")
            with np.load(str(tensor_path), allow_pickle=True) as npz:
                global_tensor = np.asarray(npz["global_tensor_data"]).astype(np.float32)
                tensor_subject_ids = np.asarray(npz["subject_ids"]).astype(str)
            print(f"Tensor shape: {global_tensor.shape}")

    # ── Pre-extracted latent CSV ──────────────────────────────────────────────
    shared_latent: Optional[pd.DataFrame] = None
    if args.latent_csv:
        shared_latent = load_latent_csv(args.latent_csv)
        if shared_latent is not None:
            print(f"Loaded shared latent CSV: {args.latent_csv} ({len(shared_latent)} rows)")

    # ── Discover folds ────────────────────────────────────────────────────────
    folds = discover_folds(args.results_dir)
    if not folds:
        print(
            f"No completed fold directories found in {args.results_dir}.\n"
            f"Expected fold_N/train_dev_subjects_fold.csv + test_subjects_fold.csv.\n"
            f"Run training first (or check --results-dir)."
        )
        _write_empty_outputs(output_dir)
        return 0

    print(f"Found {len(folds)} fold(s): {[f['n'] for f in folds]}")
    print(f"Baselines: {baselines_to_run}")
    print(f"Channels: {channels_to_use}, latent_dim: {latent_dim}")

    # ── Run baselines ─────────────────────────────────────────────────────────
    all_rows: List[Dict] = []
    latent_skipped_folds: List[int] = []

    for fold in folds:
        fold_n = fold["n"]
        print(f"\nFold {fold_n}:")

        train_df, test_df = load_fold_subjects(fold, metadata)
        y_train = get_labels(train_df, args.positive_label)
        y_test = get_labels(test_df, args.positive_label)

        n_tr_ad = int(y_train.sum())
        n_te_ad = int(y_test.sum())
        print(f"  train: N={len(y_train)}, AD={n_tr_ad}, CN={len(y_train)-n_tr_ad}")
        print(f"  test:  N={len(y_test)},  AD={n_te_ad}, CN={len(y_test)-n_te_ad}")

        # ── age_sex ────────────────────────────────────────────────────────────
        if "age_sex" in baselines_to_run:
            if all(c in train_df.columns for c in ["Age", "Sex"]):
                X_tr = prep_age_sex(train_df)
                X_te = prep_age_sex(test_df)
                row = run_one_baseline(X_tr, y_train, X_te, y_test, "age_sex", fold_n, args.classifier_C)
                all_rows.append(row)
                print(f"  age_sex     ROC-AUC={row.get('roc_auc', 'err'):.4f}" if row.get("roc_auc") else f"  age_sex     error")
            else:
                print(f"  age_sex: Age or Sex missing from metadata. Skipping.")

        # ── scanner ────────────────────────────────────────────────────────────
        if "scanner" in baselines_to_run:
            if any(c in train_df.columns for c in ["Manufacturer", "Site3"]):
                X_tr, X_te = prep_scanner_ohe(train_df, test_df)
                if X_tr.shape[1] > 0:
                    row = run_one_baseline(X_tr, y_train, X_te, y_test,
                                          "scanner_leakage_check", fold_n, args.classifier_C)
                    all_rows.append(row)
                    print(f"  scanner     ROC-AUC={row.get('roc_auc', 'err'):.4f}" if row.get("roc_auc") else f"  scanner     error")
            else:
                print(f"  scanner: Manufacturer/Site3 missing from metadata. Skipping.")

        # ── latent ─────────────────────────────────────────────────────────────
        if need_latent:
            latent_df = get_latent_for_fold(
                fold, global_tensor, tensor_subject_ids,
                channels_to_use, latent_dim,
                args.extract_latent, shared_latent, args.device,
            )
            if latent_df is None:
                latent_skipped_folds.append(fold_n)
            else:
                # Align latent rows to train/test subject order
                def get_latent_matrix(sids: pd.Series) -> Optional[np.ndarray]:
                    sids = sids.astype(str).str.strip()
                    missing = [s for s in sids if s not in latent_df.index]
                    if missing:
                        print(f"    [latent] {len(missing)} subjects missing from latent CSV.")
                        return None
                    return latent_df.loc[sids].values.astype(np.float32)

                Ltr = get_latent_matrix(train_df["SubjectID"])
                Lte = get_latent_matrix(test_df["SubjectID"])

                if Ltr is not None and Lte is not None:
                    if "latent" in baselines_to_run:
                        row = run_one_baseline(Ltr, y_train, Lte, y_test,
                                               "latent", fold_n, args.classifier_C)
                        all_rows.append(row)
                        print(f"  latent      ROC-AUC={row.get('roc_auc', 'err'):.4f}" if row.get("roc_auc") else f"  latent      error")

                    if "latent_age_sex" in baselines_to_run and all(
                        c in train_df.columns for c in ["Age", "Sex"]
                    ):
                        Atr = prep_age_sex(train_df)
                        Ate = prep_age_sex(test_df)
                        X_tr_comb = np.concatenate([Ltr, Atr], axis=1)
                        X_te_comb = np.concatenate([Lte, Ate], axis=1)
                        row = run_one_baseline(X_tr_comb, y_train, X_te_comb, y_test,
                                               "latent_age_sex", fold_n, args.classifier_C)
                        all_rows.append(row)
                        print(f"  lat+age_sex ROC-AUC={row.get('roc_auc', 'err'):.4f}" if row.get("roc_auc") else f"  lat+age_sex error")

    if latent_skipped_folds:
        print(
            f"\nNOTE: Latent baselines skipped for fold(s) {latent_skipped_folds}.\n"
            f"  To enable: pass --extract-latent --global-tensor-path <NPZ> "
            f"(requires torch + betavae_xai),\n"
            f"  OR pre-extract and pass --latent-csv <CSV with SubjectID + latent_0…>."
        )

    # ── Save per-fold table ───────────────────────────────────────────────────
    fold_df = pd.DataFrame(all_rows).reindex(columns=METRIC_COLS)
    fold_df.to_csv(output_dir / "baselines_by_fold.csv", index=False)

    # ── Pooled table ──────────────────────────────────────────────────────────
    pooled_df = pool_metrics(fold_df)
    pooled_df.to_csv(output_dir / "baselines_pooled.csv", index=False)

    # ── Load VAE main-model metrics and compare ───────────────────────────────
    vae_rows = load_vae_metrics(args.results_dir, args.positive_label)
    if vae_rows:
        vae_df = pd.DataFrame(vae_rows).reindex(columns=METRIC_COLS)
        vae_pooled = pool_metrics(vae_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            comparison_df = pd.concat(
                [pooled_df, vae_pooled], ignore_index=True
            ).sort_values("baseline")
        comparison_df.to_csv(output_dir / "baselines_vs_vae_comparison.csv", index=False)
    else:
        comparison_df = pooled_df
        comparison_df.to_csv(output_dir / "baselines_vs_vae_comparison.csv", index=False)

    # ── README ────────────────────────────────────────────────────────────────
    _write_readme(output_dir, pooled_df, comparison_df, baselines_to_run,
                  latent_skipped_folds, len(folds), args)

    print(f"\nBaseline results written to: {output_dir}")
    _print_summary(pooled_df)
    return 0


def _print_summary(pooled: pd.DataFrame) -> None:
    if pooled.empty:
        return
    print("\nPooled baseline summary:")
    for _, row in pooled.iterrows():
        parts = [f"{row['baseline']:<20}"]
        for m in ("roc_auc", "pr_auc", "balanced_accuracy"):
            v = row.get(m)
            parts.append(f"{m}={v:.4f}" if isinstance(v, float) else f"{m}=N/A")
        print("  " + "  ".join(parts))


def _write_empty_outputs(output_dir: Path) -> None:
    pd.DataFrame(columns=METRIC_COLS).to_csv(output_dir / "baselines_by_fold.csv", index=False)
    pd.DataFrame(columns=METRIC_COLS).to_csv(output_dir / "baselines_pooled.csv", index=False)
    pd.DataFrame(columns=METRIC_COLS).to_csv(output_dir / "baselines_vs_vae_comparison.csv", index=False)
    readme = (
        "# Metadata Baselines\n\n"
        "No fold results found. Run training first.\n"
    )
    (output_dir / "README_baselines.md").write_text(readme, encoding="utf-8")


def _write_readme(
    output_dir: Path,
    pooled: pd.DataFrame,
    comparison: pd.DataFrame,
    baselines_requested: List[str],
    latent_skipped: List[int],
    n_folds: int,
    args: argparse.Namespace,
) -> None:
    def fmt_row(row: "pd.Series") -> str:
        roc = f"{row['roc_auc']:.4f}" if isinstance(row.get("roc_auc"), float) else "N/A"
        pr = f"{row['pr_auc']:.4f}" if isinstance(row.get("pr_auc"), float) else "N/A"
        ba = f"{row['balanced_accuracy']:.4f}" if isinstance(row.get("balanced_accuracy"), float) else "N/A"
        return f"| {row['baseline']} | {roc} | {pr} | {ba} |"

    table_header = "| Baseline | ROC-AUC | PR-AUC | Bal.Acc |"
    table_sep = "|---|---|---|---|"

    lines = [
        "# Metadata and Latent Baselines",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Results dir: {args.results_dir}",
        f"Outer folds: {n_folds}",
        f"Positive label: {args.positive_label} (CN = negative)",
        f"Classifier: LogisticRegression(C={args.classifier_C}, class_weight='balanced')",
        "",
        "## Baseline descriptions",
        "",
        "| Baseline | Description | Interpretation |",
        "|---|---|---|",
        "| age_sex | LogReg on [Age, Sex_binary] | Demographic confound check |",
        "| scanner_leakage_check | LogReg on OHE(Manufacturer, Site3) | Scanner confound check — **not a model** |",
        "| latent | LogReg on VAE mu (256-dim) | Upper bound: VAE features only |",
        "| latent_age_sex | LogReg on [VAE mu, Age, Sex_binary] | VAE + demographics |",
        "",
        "**Note**: scanner_leakage_check is intentionally labelled as a check, not a model.",
        "A high ROC-AUC here indicates scanner/site confound in the data.",
        "The gridsearch classifiers in the main run already include a leakage QC step",
        "(qc_check_scanner_leakage=true).",
        "",
        "## Pooled results (macro-average across folds)",
        "",
        table_header,
        table_sep,
    ]
    for _, row in comparison.iterrows():
        lines.append(fmt_row(row))

    lines += [
        "",
        "## Files",
        "",
        "- baselines_by_fold.csv — per-fold metrics",
        "- baselines_pooled.csv — macro-averaged metrics",
        "- baselines_vs_vae_comparison.csv — baselines + VAE main-model side by side",
        "",
    ]
    if latent_skipped:
        lines += [
            "## Latent baseline notes",
            "",
            f"Latent reps not available for fold(s): {latent_skipped}.",
            "To enable latent baselines, run with:",
            "  --extract-latent --global-tensor-path <NPZ>  (requires torch + betavae_xai)",
            "  OR pre-extract and pass --latent-csv <CSV>",
            "",
        ]

    (output_dir / "README_baselines.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
