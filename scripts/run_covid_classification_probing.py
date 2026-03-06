#!/usr/bin/env python3
"""
run_covid_classification_probing.py
===================================
Frozen-encoder probing: COVID vs CONTROL using ADNI-trained β-VAE latents.

Scientific question:
   Can representations learned from ADNI AD-vs-CN connectomes transfer to
   discriminate COVID vs CONTROL in an independent cohort?

Design:
   - 5 frozen ADNI VAE encoders (one per ADNI outer fold) → 5 latent spaces
   - Supervised CV *within* the COVID cohort (StratifiedKFold)
   - 3 feature families: latent_only | metadata_only | latent_plus_metadata
   - Late fusion: average predicted scores across ADNI encoders
     (predict_proba for logreg; decision_function for SVM)
   - metadata_only is encoder-independent → trained once, not fused
   - CRITICAL: never average latent vectors across ADNI folds (rotational
     indeterminacy of β-VAEs)

Usage:
   python scripts/run_covid_classification_probing.py --smoke_test
   python scripts/run_covid_classification_probing.py
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════════════════════════
# Imports from the betavae_xai package
# ═══════════════════════════════════════════════════════════════════════
from betavae_xai.data.preprocessing import apply_normalization_params  # noqa: E402
from betavae_xai.models import ConvolutionalVAE  # noqa: E402
from betavae_xai.models.classifiers import get_classifier_and_grid  # noqa: E402
from betavae_xai.utils.run_io import _safe_json_dump  # noqa: E402

# Optuna import (for OptunaSearchCV used in tuning)
import optuna  # noqa: E402
try:
    from optuna.integration import OptunaSearchCV
except ImportError:
    from optuna_integration import OptunaSearchCV


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════
def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "N/A"


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()[:4096]).hexdigest()[:16]


def _get_score_1d(estimator, X):
    """
    Score helper consistent with run_vae_clf_ad_inference.py:
      - predict_proba[:,1] if available
      - decision_function (raw) otherwise
      - fallback: predict() as float
    """
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel()
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    return np.asarray(estimator.predict(X)).astype(float).ravel()


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                     y_pred: np.ndarray | None = None,
                     threshold: float = 0.5) -> Dict[str, float]:
    """Compute classification metrics.
    
    y_score may come from decision_function (unbounded) or predict_proba.
    y_pred is the model's native prediction; if not provided, threshold y_score.
    """
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Data loading & alignment
# ═══════════════════════════════════════════════════════════════════════
def load_covid_cohort(
    tensor_path: Path,
    metadata_path: Path,
    channels_to_use: List[int],
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """Load COVID tensor & metadata, align by ID, select channels."""
    npz = np.load(str(tensor_path), allow_pickle=True)
    tensor_full = npz["global_tensor_data"]          # (N, 7, 131, 131)
    tensor_ids = npz["subject_ids"].astype(str)
    roi_names = npz["roi_names_in_order"].astype(str).tolist()
    channel_names = npz["channel_names"].astype(str).tolist()

    meta = pd.read_csv(metadata_path)
    assert "ID" in meta.columns, f"Expected 'ID' column in {metadata_path}"
    assert "Grupo" in meta.columns, f"Expected 'Grupo' column in {metadata_path}"
    meta["ID"] = meta["ID"].astype(str).str.strip()

    # Build tensor-index lookup
    tdf = pd.DataFrame({"ID": tensor_ids, "tensor_idx": np.arange(len(tensor_ids))})
    merged = pd.merge(tdf, meta, on="ID", how="inner")
    assert len(merged) > 0, "No subjects matched between tensor and metadata"

    # Filter to COVID + CONTROL only
    merged = merged[merged["Grupo"].isin(["COVID", "CONTROL"])].reset_index(drop=True)
    log.info(f"Aligned cohort: {len(merged)} subjects "
             f"(COVID={sum(merged['Grupo']=='COVID')}, "
             f"CONTROL={sum(merged['Grupo']=='CONTROL')})")

    # Subject alignment audit
    tensor_set = set(tensor_ids)
    meta_set = set(meta["ID"].astype(str))
    log.info(f"  Tensor IDs: {len(tensor_set)} | Meta IDs: {len(meta_set)} | "
             f"Intersection: {len(tensor_set & meta_set)} | "
             f"Meta-only: {len(meta_set - tensor_set)} | "
             f"Tensor-only: {len(tensor_set - meta_set)}")

    # Select channels and reindex tensor
    idx = merged["tensor_idx"].values
    tensor_sel = tensor_full[idx][:, channels_to_use, :, :]
    selected_ch_names = [channel_names[i] for i in channels_to_use]

    # Labels
    merged["y"] = (merged["Grupo"] == "COVID").astype(int)

    return tensor_sel, merged, roi_names, selected_ch_names, channel_names


# ═══════════════════════════════════════════════════════════════════════
#  Compatibility checks
# ═══════════════════════════════════════════════════════════════════════
def validate_compatibility(
    run_config: dict,
    covid_roi_names: list,
    covid_channel_names: list,
    channels_to_use: List[int],
) -> Tuple[str, str]:
    """Fail-fast checks and return (roi_hash, channel_hash)."""
    adni_rois = [str(r) for r in run_config.get("roi_names_in_order", [])]
    adni_chs = [str(c) for c in run_config.get("channel_names_master_in_tensor_order", [])]

    if adni_rois and covid_roi_names:
        assert adni_rois == covid_roi_names, (
            f"ROI mismatch: ADNI has {len(adni_rois)} ROIs, COVID has {len(covid_roi_names)}")
    if adni_chs and covid_channel_names:
        assert adni_chs == covid_channel_names, (
            f"Channel mismatch: ADNI={adni_chs}, COVID={covid_channel_names}")

    for ci in channels_to_use:
        assert ci < len(covid_channel_names), f"Channel index {ci} out of range"

    roi_hash = hashlib.sha256(",".join(covid_roi_names).encode()).hexdigest()[:16]
    ch_hash = hashlib.sha256(",".join(covid_channel_names).encode()).hexdigest()[:16]
    log.info(f"Compatibility OK — ROI hash: {roi_hash}, Channel hash: {ch_hash}")
    return roi_hash, ch_hash


# ═══════════════════════════════════════════════════════════════════════
#  Latent extraction (frozen encoder)
# ═══════════════════════════════════════════════════════════════════════
def extract_latents(
    vae: ConvolutionalVAE,
    tensor_norm: np.ndarray,
    device: torch.device,
    latent_type: str = "mu",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract latent μ or z from a frozen VAE encoder."""
    vae.eval()
    latents = []
    with torch.no_grad():
        for s in range(0, len(tensor_norm), batch_size):
            batch = torch.from_numpy(tensor_norm[s:s + batch_size]).float().to(device)
            _, mu, _, z = vae(batch)
            lat = mu if latent_type == "mu" else z
            latents.append(lat.cpu().numpy())
    return np.concatenate(latents, axis=0)


def load_vae_for_fold(
    fold_dir: Path, fold_id: int, run_args: dict, device: torch.device,
) -> ConvolutionalVAE:
    """Instantiate and load a frozen VAE from a fold directory."""
    n_ch = len(run_args["channels_to_use"])
    vae = ConvolutionalVAE(
        input_channels=n_ch,
        latent_dim=run_args["latent_dim"],
        image_size=131,  # AAL3 ROIs
        final_activation=run_args.get("vae_final_activation", "tanh"),
        intermediate_fc_dim_config=run_args.get("intermediate_fc_dim_vae", "quarter"),
        dropout_rate=float(run_args.get("dropout_rate_vae", 0.15)),
        use_layernorm_fc=bool(run_args.get("use_layernorm_vae_fc", False)),
        num_conv_layers_encoder=int(run_args.get("num_conv_layers_encoder", 4)),
        decoder_type=run_args.get("decoder_type", "convtranspose"),
    ).to(device)
    ckpt = fold_dir / f"vae_model_fold_{fold_id}.pt"
    assert ckpt.exists(), f"VAE checkpoint not found: {ckpt}"
    vae.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


# ═══════════════════════════════════════════════════════════════════════
#  COVID CV splitter
# ═══════════════════════════════════════════════════════════════════════
def build_covid_cv_splits(
    cohort: pd.DataFrame,
    n_splits: int,
    seed: int,
    artifacts_dir: Path,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build stratified CV splits inside the COVID cohort."""
    y = cohort["y"].values
    sex = cohort["Sex"].values.astype(str)

    # Try Grupo×Sex stratification; fallback to Grupo-only
    strat_key = np.array([f"{yi}_{si}" for yi, si in zip(y, sex)])
    _, counts = np.unique(strat_key, return_counts=True)
    if counts.min() >= n_splits:
        strat_col = strat_key
        strat_used = "Grupo_x_Sex"
    else:
        strat_col = y
        strat_used = "Grupo_only"
    log.info(f"COVID CV stratification: {strat_used}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    split_dir = artifacts_dir / "covid_cv_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), strat_col)):
        splits.append((train_idx, test_idx))
        np.save(split_dir / f"fold_{i}_train_idx.npy", train_idx)
        np.save(split_dir / f"fold_{i}_test_idx.npy", test_idx)

    return splits


# ═══════════════════════════════════════════════════════════════════════
#  Build feature matrices
# ═══════════════════════════════════════════════════════════════════════
def build_features(
    latent: np.ndarray | None,
    cohort: pd.DataFrame,
    idx: np.ndarray,
    family: str,
    latent_dim: int,
    metadata_cols: List[str] = ("Age", "Sex"),
) -> pd.DataFrame:
    """
    Build feature DataFrame for a given subset (train or test) and feature family.

    Returns a **pd.DataFrame** so that _AutoPreprocessor in the classifier
    pipeline can detect the 'Sex' column and route it through the proper
    sex-encoder → imputer → scaler sub-pipeline.  This matches the design
    of run_vae_clf_ad_inference.py where metadata is passed RAW.

    For 'metadata_only', *latent* may be None (no encoder dependency).
    """
    sub = cohort.iloc[idx].reset_index(drop=True)

    if family == "metadata_only":
        return sub[list(metadata_cols)].reset_index(drop=True)

    # Latent-dependent families require a valid latent array
    assert latent is not None, f"latent must not be None for family={family}"
    lat_names = [f"latent_{j}" for j in range(latent_dim)]
    lat_df = pd.DataFrame(latent[idx], columns=lat_names)

    if family == "latent_only":
        return lat_df
    elif family == "latent_plus_metadata":
        meta_df = sub[list(metadata_cols)].reset_index(drop=True)
        return pd.concat([lat_df, meta_df], axis=1)
    else:
        raise ValueError(f"Unknown feature family: {family}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Frozen-encoder probing: COVID vs CONTROL")
    parser.add_argument("--results_dir", type=str,
                        default="results/vae_3channels_beta65_pro")
    parser.add_argument("--covid_tensor_path", type=str, default=None,
                        help="Path to COVID NPZ tensor (auto-detected if omitted)")
    parser.add_argument("--covid_metadata_path", type=str,
                        default="data/SubjectsData_AAL3_COVID.csv")
    parser.add_argument("--classifier_types", nargs="+", default=["logreg", "svm"])
    parser.add_argument("--feature_families", nargs="+",
                        default=["latent_only", "metadata_only", "latent_plus_metadata"])
    parser.add_argument("--encoder_folds", nargs="+", type=int, default=None,
                        help="ADNI fold IDs to use (default: all available)")
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--latent_features_type", type=str, default="mu",
                        choices=["mu", "z"])
    parser.add_argument("--metadata_features", nargs="+", default=["Age", "Sex"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_optuna_trials", type=int, default=30)
    args = parser.parse_args()

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Paths ─────────────────────────────────────────────────────
    results_dir = PROJECT_ROOT / args.results_dir
    config_path = results_dir / "run_config.json"
    assert config_path.exists(), f"run_config.json not found: {config_path}"

    with open(config_path) as f:
        run_config = json.load(f)
    run_args = run_config["args"]

    covid_meta_path = PROJECT_ROOT / args.covid_metadata_path

    # Auto-detect COVID tensor
    if args.covid_tensor_path:
        covid_tensor_path = PROJECT_ROOT / args.covid_tensor_path
    else:
        candidates = sorted(PROJECT_ROOT.glob(
            "data/COVID_AAL3_Tensor_v1_*/GLOBAL_TENSOR_from_COVID_*.npz"))
        assert len(candidates) >= 1, "COVID tensor not found — provide --covid_tensor_path"
        covid_tensor_path = candidates[0]
    log.info(f"COVID tensor: {covid_tensor_path.relative_to(PROJECT_ROOT)}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        results_dir / "covid_classification_probing")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    tables_dir = output_dir / "Tables"
    figures_dir = output_dir / "Figures"
    logs_dir = output_dir / "Logs"
    artifacts_dir = output_dir / "Artifacts"
    for d in [tables_dir, figures_dir, logs_dir, artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    # ── Discover ADNI folds ───────────────────────────────────────
    channels_to_use = run_args["channels_to_use"]
    latent_dim = run_args["latent_dim"]

    if args.encoder_folds is not None:
        adni_folds = args.encoder_folds
    else:
        adni_folds = sorted([
            int(d.name.split("_")[1])
            for d in results_dir.iterdir()
            if d.is_dir() and d.name.startswith("fold_")
            and (d / f"vae_model_fold_{d.name.split('_')[1]}.pt").exists()
        ])
    log.info(f"ADNI encoder folds: {adni_folds}")

    if args.smoke_test:
        adni_folds = adni_folds[:1]
        log.info(f"  → smoke_test: using only fold {adni_folds}")

    # ── Load COVID cohort ─────────────────────────────────────────
    tensor_sel, cohort, roi_names, sel_ch_names, all_ch_names = load_covid_cohort(
        covid_tensor_path, covid_meta_path, channels_to_use)

    roi_hash, ch_hash = validate_compatibility(
        run_config, roi_names, all_ch_names, channels_to_use)

    # Symmetry and diagonal sanity on a sample
    sample = tensor_sel[0, 0, :, :]
    assert np.allclose(sample, sample.T, atol=1e-5), "Connectome not symmetric"
    log.info("Symmetry and shape checks passed.")

    # ── COVID CV splits ───────────────────────────────────────────
    n_outer = args.outer_folds
    if args.smoke_test:
        n_outer = min(2, n_outer)
    splits = build_covid_cv_splits(cohort, n_outer, SEED, artifacts_dir)
    log.info(f"COVID CV: {len(splits)} outer folds")

    # ── Extract latents per ADNI encoder ──────────────────────────
    latent_dir = artifacts_dir / "latents"
    latent_dir.mkdir(exist_ok=True)
    latents_by_adni: Dict[int, np.ndarray] = {}

    for adni_k in adni_folds:
        fold_dir = results_dir / f"fold_{adni_k}"
        norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
        tensor_norm = apply_normalization_params(tensor_sel.copy(), norm_params)

        vae = load_vae_for_fold(fold_dir, adni_k, run_args, device)
        latent = extract_latents(vae, tensor_norm, device,
                                 latent_type=args.latent_features_type)
        latents_by_adni[adni_k] = latent

        save_path = latent_dir / f"latent_{args.latent_features_type}_adniFold{adni_k}.npy"
        np.save(save_path, latent)
        log.info(f"Latent extracted: ADNI fold {adni_k} → shape {latent.shape}")

        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Classifier training loop ──────────────────────────────────
    n_optuna = args.n_optuna_trials
    if args.smoke_test:
        n_optuna = 2

    y_all = cohort["y"].values
    all_rows = []       # per-fold metrics
    all_oof = []        # OOF predictions
    coef_rows = []      # logreg coefficients

    for adni_k in adni_folds:
        latent = latents_by_adni[adni_k]

        for covid_fi, (train_idx, test_idx) in enumerate(splits):
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            test_ids = cohort.iloc[test_idx]["ID"].values

            for clf_type in args.classifier_types:
                for family in args.feature_families:
                    # metadata_only is encoder-independent:
                    # train once (first encoder fold), skip the rest.
                    if family == "metadata_only" and adni_k != adni_folds[0]:
                        continue
                    encoder_fold_label = 0 if family == "metadata_only" else adni_k

                    X_train = build_features(
                        latent if family != "metadata_only" else None,
                        cohort, train_idx, family, latent_dim)
                    X_test = build_features(
                        latent if family != "metadata_only" else None,
                        cohort, test_idx, family, latent_dim)

                    # Get classifier pipeline + Optuna search space
                    # Matches AD pipeline: balance=True, no SMOTE, no calibration
                    pipeline, param_dist, n_iter_suggested = get_classifier_and_grid(
                        clf_type, seed=SEED, balance=True)

                    # Inner CV: vary seed by outer fold + ADNI encoder.
                    # metadata_only uses encoder-independent seed.
                    if family == "metadata_only":
                        inner_seed = SEED + covid_fi + 30
                    else:
                        inner_seed = SEED + covid_fi + adni_k * 100 + 30
                    effective_inner_folds = min(
                        args.inner_folds,
                        max(2, min(int(sum(y_train == 0)),
                                   int(sum(y_train == 1)))))
                    inner_cv = StratifiedKFold(
                        n_splits=effective_inner_folds,
                        shuffle=True, random_state=inner_seed)

                    # Explicit Optuna study with seeded sampler
                    # (matches AD pipeline pattern)
                    sampler = optuna.samplers.TPESampler(seed=inner_seed)
                    study = optuna.create_study(
                        direction="maximize", sampler=sampler)
                    effective_trials = min(n_optuna, n_iter_suggested)

                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    search = OptunaSearchCV(
                        pipeline, param_dist,
                        study=study,
                        cv=inner_cv, scoring="roc_auc",
                        n_trials=effective_trials,
                        random_state=inner_seed,
                        n_jobs=1, refit=True,
                        timeout=300 if not args.smoke_test else 30,
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        search.fit(X_train, y_train)

                    best_pipe = search.best_estimator_

                    # Score using _get_score_1d (consistent with AD pipeline)
                    y_score = _get_score_1d(best_pipe, X_test)
                    y_pred = best_pipe.predict(X_test)
                    metrics = _compute_metrics(y_test, y_score, y_pred=y_pred)

                    row = {
                        "covid_outer_fold": covid_fi,
                        "adni_encoder_fold": encoder_fold_label,
                        "model_type": clf_type,
                        "feature_family": family,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        **metrics,
                    }
                    all_rows.append(row)

                    for i, sid in enumerate(test_ids):
                        all_oof.append({
                            "SubjectID": sid,
                            "y_true": int(y_test[i]),
                            "covid_outer_fold": covid_fi,
                            "adni_encoder_fold": encoder_fold_label,
                            "model_type": clf_type,
                            "feature_family": family,
                            "y_score": float(y_score[i]),
                            "y_pred": int(y_pred[i]),
                        })

                    # Extract logreg coefficients if accessible
                    if clf_type == "logreg":
                        try:
                            model_step = best_pipe.named_steps["model"]
                            coefs = model_step.coef_[0]
                            for ci, cv in enumerate(coefs):
                                coef_rows.append({
                                    "covid_outer_fold": covid_fi,
                                    "adni_encoder_fold": encoder_fold_label,
                                    "feature_family": family,
                                    "feature_idx": ci,
                                    "coefficient": float(cv),
                                })
                        except Exception:
                            pass

            log.info(f"  ADNI={adni_k} COVID_fold={covid_fi} done "
                     f"({len(args.classifier_types)}×{len(args.feature_families)} combos)")

    # ── Save per-fold metrics ─────────────────────────────────────
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(tables_dir / "metrics_by_fold_encoder_model.csv", index=False)
    log.info(f"Saved: metrics_by_fold_encoder_model.csv ({len(metrics_df)} rows)")

    # ── Save OOF predictions ──────────────────────────────────────
    oof_df = pd.DataFrame(all_oof)
    oof_df.to_csv(tables_dir / "oof_predictions_by_encoder_model.csv", index=False)

    # ── Late fusion: average scores across ADNI encoders ─────────
    # NOTE: y_score is predict_proba[:,1] for logreg (bounded [0,1])
    # and decision_function for SVM (unbounded margin).
    # AUC is rank-based so both work per-encoder.
    # For fusion we average scores; for y_pred_fused we use majority-vote.
    # metadata_only rows have n_encoders_fused=1 (no actual fusion).
    fusion_rows = []
    if len(adni_folds) > 1:
        for (sid, cof, mtype, fam), grp in oof_df.groupby(
                ["SubjectID", "covid_outer_fold", "model_type", "feature_family"]):
            fused_score = float(grp["y_score"].mean())
            # Majority vote on per-encoder predictions (robust to score scale)
            fused_pred = int(grp["y_pred"].sum() > len(grp) / 2)
            fusion_rows.append({
                "SubjectID": sid,
                "y_true": int(grp["y_true"].iloc[0]),
                "covid_outer_fold": cof,
                "model_type": mtype,
                "feature_family": fam,
                "y_score_fused": fused_score,
                "y_pred_fused": fused_pred,
                "n_encoders_fused": len(grp),
            })
    else:
        # Single encoder → fusion = identity
        for _, r in oof_df.iterrows():
            fusion_rows.append({
                "SubjectID": r["SubjectID"],
                "y_true": int(r["y_true"]),
                "covid_outer_fold": r["covid_outer_fold"],
                "model_type": r["model_type"],
                "feature_family": r["feature_family"],
                "y_score_fused": float(r["y_score"]),
                "y_pred_fused": int(r["y_pred"]),
                "n_encoders_fused": 1,
            })

    fusion_df = pd.DataFrame(fusion_rows)
    fusion_df.to_csv(tables_dir / "oof_predictions_late_fusion.csv", index=False)

    # ── Summary tables ────────────────────────────────────────────
    # Per encoder/model summary (across COVID folds)
    summary_enc = (
        metrics_df
        .groupby(["adni_encoder_fold", "model_type", "feature_family"])
        .agg(
            auc_mean=("auc", "mean"), auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_mean=("f1", "mean"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            n_folds=("auc", "count"),
        ).reset_index()
    )
    summary_enc.to_csv(tables_dir / "summary_by_encoder_model.csv", index=False)

    # Late fusion summary
    fusion_summary_rows = []
    for (mtype, fam), grp in fusion_df.groupby(["model_type", "feature_family"]):
        # Compute per-fold metrics, then average
        fold_metrics = []
        for cof, fgrp in grp.groupby("covid_outer_fold"):
            m = _compute_metrics(
                fgrp["y_true"].values,
                fgrp["y_score_fused"].values,
                y_pred=fgrp["y_pred_fused"].values,
            )
            fold_metrics.append(m)
        fm_df = pd.DataFrame(fold_metrics)
        row = {"model_type": mtype, "feature_family": fam}
        for col in fm_df.columns:
            row[f"{col}_mean"] = float(fm_df[col].mean())
            row[f"{col}_std"] = float(fm_df[col].std())
        row["n_folds"] = len(fold_metrics)
        fusion_summary_rows.append(row)

    fusion_summary = pd.DataFrame(fusion_summary_rows)
    fusion_summary.to_csv(tables_dir / "summary_late_fusion.csv", index=False)

    # ── Feature schema summary ────────────────────────────────────
    schema_rows = []
    for fam in args.feature_families:
        if fam == "latent_only":
            n_feat = latent_dim
        elif fam == "metadata_only":
            n_feat = 2
        else:
            n_feat = latent_dim + 2
        schema_rows.append({
            "feature_family": fam,
            "latent_dim": latent_dim if "latent" in fam else 0,
            "metadata_columns": ", ".join(args.metadata_features) if "metadata" in fam else "",
            "final_feature_count": n_feat,
            "latent_type": args.latent_features_type if "latent" in fam else "N/A",
            "selected_channels": ", ".join(sel_ch_names) if "latent" in fam else "N/A",
        })
    pd.DataFrame(schema_rows).to_csv(tables_dir / "feature_schema_summary.csv", index=False)

    # ── Logreg coefficients ───────────────────────────────────────
    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(
            tables_dir / "logreg_coefficients.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 1: AUC by encoder and model
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot = metrics_df.pivot_table(
            index=["adni_encoder_fold", "feature_family"],
            columns="model_type", values="auc", aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_ylabel("Mean AUC (across COVID folds)")
        ax.set_title("AUC by ADNI Encoder Fold, Feature Family & Model")
        ax.legend(title="Model", fontsize=8)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        fig.savefig(figures_dir / "auc_by_encoder_and_model.png", dpi=150)
        plt.close(fig)

        # Fig 2: Late fusion AUC by feature family
        if len(fusion_summary):
            fig, ax = plt.subplots(figsize=(8, 5))
            for mtype in fusion_summary["model_type"].unique():
                sub = fusion_summary[fusion_summary["model_type"] == mtype]
                ax.barh(
                    [f"{r['feature_family']}\n({mtype})" for _, r in sub.iterrows()],
                    sub["auc_mean"], xerr=sub["auc_std"],
                    alpha=0.75, label=mtype, edgecolor="white")
            ax.set_xlabel("AUC (late fusion)")
            ax.set_title("Late Fusion AUC by Feature Family")
            ax.legend()
            ax.set_xlim(0, 1)
            plt.tight_layout()
            fig.savefig(figures_dir / "late_fusion_auc_by_feature_family.png", dpi=150)
            plt.close(fig)

        # Fig 3: Prediction distribution
        fig, axes = plt.subplots(1, len(args.feature_families),
                                  figsize=(5 * len(args.feature_families), 4))
        if len(args.feature_families) == 1:
            axes = [axes]
        for ax_i, fam in enumerate(args.feature_families):
            sub = fusion_df[fusion_df["feature_family"] == fam]
            for label, color in [(0, "#3498DB"), (1, "#E74C3C")]:
                vals = sub[sub["y_true"] == label]["y_score_fused"]
                axes[ax_i].hist(vals, bins=20, alpha=0.6, color=color,
                                label=f"{'CONTROL' if label == 0 else 'COVID'}",
                                edgecolor="white", density=True)
            axes[ax_i].set_title(fam, fontsize=9)
            axes[ax_i].set_xlabel("Fused Score")
            axes[ax_i].legend(fontsize=8)
        plt.suptitle("Late Fusion Prediction Distribution")
        plt.tight_layout()
        fig.savefig(figures_dir / "prediction_distribution_late_fusion.png", dpi=150)
        plt.close(fig)

        log.info("Figures saved.")
    except Exception as e:
        log.warning(f"Figure generation failed: {e}")

    # ── Manifest ──────────────────────────────────────────────────
    output_files = sorted([
        str(p.relative_to(output_dir))
        for p in output_dir.rglob("*") if p.is_file()
    ])

    manifest = {
        "script": "scripts/run_covid_classification_probing.py",
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "git_hash": _git_hash(),
        "seeds": {"global": SEED, "sampling": SEED},
        "device": str(device),
        "smoke_test": args.smoke_test,
        "input_artifacts": {
            "run_config": str(config_path.relative_to(PROJECT_ROOT)),
            "covid_tensor": str(covid_tensor_path.relative_to(PROJECT_ROOT)),
            "covid_metadata": str(covid_meta_path.relative_to(PROJECT_ROOT)),
        },
        "encoder_folds_used": adni_folds,
        "latent_type": args.latent_features_type,
        "latent_dim": latent_dim,
        "metadata_features": args.metadata_features,
        "channels_to_use_indices": channels_to_use,
        "channel_names_selected": sel_ch_names,
        "roi_hash": roi_hash,
        "channel_hash": ch_hash,
        "n_subjects": len(cohort),
        "n_covid": int(sum(cohort["y"] == 1)),
        "n_control": int(sum(cohort["y"] == 0)),
        "covid_outer_folds": len(splits),
        "classifier_types": args.classifier_types,
        "feature_families": args.feature_families,
        "n_optuna_trials": n_optuna,
        "output_files": output_files,
    }
    manifest_path = output_dir / "run_manifest.json"
    _safe_json_dump(manifest, manifest_path)

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  COVID Classification Probing — Summary")
    print("=" * 72)
    print(f"  Cohort: {len(cohort)} subjects "
          f"(COVID={sum(cohort['y']==1)}, CONTROL={sum(cohort['y']==0)})")
    print(f"  ADNI encoders: {adni_folds}")
    print(f"  COVID CV folds: {len(splits)}")
    print(f"  Classifiers: {args.classifier_types}")
    print(f"  Feature families: {args.feature_families}")
    print(f"  Latent type: {args.latent_features_type}, dim={latent_dim}")

    if len(fusion_summary):
        print(f"\n  Late Fusion AUC Summary:")
        for _, r in fusion_summary.iterrows():
            print(f"    {r['model_type']:8s} | {r['feature_family']:25s} | "
                  f"AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}")

    print(f"\n  Output: {output_dir.relative_to(PROJECT_ROOT)}/")
    for f_name in output_files[:15]:
        print(f"    {f_name}")
    if len(output_files) > 15:
        print(f"    ... ({len(output_files)} files total)")
    print(f"\n  Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
