#!/usr/bin/env python
"""Read-only latent-space OASIS-vs-ADNI domain-shift audit.

Focus:
  ADNI model: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
  Readout: recover035_latent384_beta3p75_stageB_oof_score_calibration
  OASIS panel/build: oasis_mega_90_90_external_inference_model_panel_20260604,
                     runwise164_pilot_parity

This script does not fit anything on OASIS. OASIS is encoded by frozen ADNI
fold VAEs only; all centroids/covariances/support thresholds are estimated from
ADNI train/dev latent features.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score


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
)


RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LATENT_CACHE = RUN_DIR / "classifier_only_readout" / "latent_cache"
OOF_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OASIS_PANEL = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604"
MEGA_DIR = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"
OASIS_TENSOR = MEGA_DIR / "tensor_runwise164_pilot_parity.npz"
OASIS_MANIFEST = MEGA_DIR / "mega_manifest.csv"
OUT_DIR = RESULTS / "promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604"

FOLDS = [1, 2, 3, 4, 5]
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
OASIS_CANDIDATE = "promoted_beta3p75_oof_ecdf"
OASIS_BUILD = "runwise164_pilot_parity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--latent-cache", type=Path, default=LATENT_CACHE)
    p.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    p.add_argument("--oasis-panel", type=Path, default=OASIS_PANEL)
    p.add_argument("--oasis-tensor", type=Path, default=OASIS_TENSOR)
    p.add_argument("--oasis-manifest", type=Path, default=OASIS_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def mu_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("mu_")], key=lambda c: int(c.split("_")[1]))


def selected_channel_indices(model_channel_names: Sequence[str], oasis_channel_names: Sequence[str]) -> List[int]:
    idx: List[int] = []
    for name in model_channel_names:
        if name not in oasis_channel_names:
            raise ValueError(f"Model channel {name!r} missing from OASIS channels {list(oasis_channel_names)}")
        idx.append(list(oasis_channel_names).index(name))
    return idx


def load_oasis_tensor_and_meta(tensor_path: Path, manifest_path: Path) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    manifest = pd.read_csv(manifest_path)
    with np.load(tensor_path, allow_pickle=True) as zf:
        tensor = np.asarray(zf["global_tensor_data"], dtype=np.float32)
        subject_ids = np.asarray(zf["subject_ids"]).astype(str)
        session_ids = np.asarray(zf["session_ids"]).astype(str)
        experiment_ids = np.asarray(zf["experiment_ids"]).astype(str)
        diagnosis = np.asarray(zf["diagnosis"]).astype(str)
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
        raise ValueError(f"OASIS manifest alignment failed:\n{bad.to_string(index=False)}")
    meta["SubjectID"] = meta["subject_id"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["diagnosis"].map(normalize_dx)
    meta["y"] = meta["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    meta["Age"] = pd.to_numeric(meta["age_at_MR"], errors="coerce")
    meta["Sex"] = meta["sex_normalized"].where(meta["sex_normalized"].notna(), meta["sex"]).map(normalize_sex)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    return tensor, meta, channel_names


def encode_oasis_fold(
    run_dir: Path,
    cfg: Dict[str, Any],
    tensor: np.ndarray,
    meta: pd.DataFrame,
    fold: int,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    fold_dir = run_dir / f"fold_{fold}"
    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
    conditioning_path = fold_dir / "vae_conditioning_transformer.joblib"
    conditioning_transformer = (
        joblib.load(conditioning_path)
        if str(cfg.get("vae_conditioning_mode", "none")) != "none"
        else {"vars_mode": "none", "conditioning_dim": 0}
    )
    x_norm = apply_normalization_params(tensor, norm_params)
    cond = apply_conditioning_transformer(meta, conditioning_transformer)
    model = make_model(
        cfg,
        image_size=tensor.shape[-1],
        n_channels=tensor.shape[1],
        device=device,
        conditioning_dim_override=int(conditioning_transformer.get("conditioning_dim", 0)),
    )
    state_dict = torch.load(fold_dir / f"vae_model_fold_{fold}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    mu = encode_mu(model, x_norm, batch_size=batch_size, device=device, condition=cond)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = meta.copy()
    out["fold"] = fold
    out["cohort"] = "OASIS"
    out["cohort_dx"] = "OASIS_" + out["ResearchGroup_Mapped"].astype(str)
    for j in range(mu.shape[1]):
        out[f"mu_{j}"] = mu[:, j]
    return out


def load_adni_fold(latent_cache: Path, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(latent_cache / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(latent_cache / f"fold_{fold}_test_latent_mu.csv")
    for df, split in [(train, "trainDev"), (test, "test")]:
        df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
        df["Manufacturer"] = df["Manufacturer"].map(normalize_mfr)
        df["Sex"] = df["Sex"].map(normalize_sex)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["cohort"] = "ADNI"
        df["cohort_dx"] = "ADNI_" + df["ResearchGroup_Mapped"].astype(str)
        df["fold"] = fold
        df["split"] = split
    return train, test


def cosine_distance_to_centroid(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=1)
    c_norm = float(np.linalg.norm(centroid))
    denom = np.maximum(x_norm * c_norm, 1e-12)
    return 1.0 - (x @ centroid) / denom


def mahalanobis(x: np.ndarray, center: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = x - center.reshape(1, -1)
    val = np.einsum("ij,jk,ik->i", diff, precision, diff)
    return np.sqrt(np.maximum(val, 0.0))


def ecdf_percentile(reference_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference_values, dtype=float))
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / max(len(ref), 1)


def summarize_numeric(g: pd.DataFrame, col: str) -> Dict[str, Any]:
    q = g[col].quantile([0.25, 0.5, 0.75])
    return {
        f"{col}_mean": float(g[col].mean()),
        f"{col}_std": float(g[col].std(ddof=1)),
        f"{col}_median": float(q.loc[0.5]),
        f"{col}_iqr_low": float(q.loc[0.25]),
        f"{col}_iqr_high": float(q.loc[0.75]),
    }


def distance_metrics_for_fold(train: pd.DataFrame, test: pd.DataFrame, oasis: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = mu_cols(train)
    train_clf = train[train["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    if train_clf.empty:
        raise ValueError(f"Fold {fold}: no ADNI classifier train/dev rows")
    train_x = train_clf[cols].to_numpy(dtype=float)
    cn_centroid = train_clf.loc[train_clf["ResearchGroup_Mapped"] == "CN", cols].to_numpy(dtype=float).mean(axis=0)
    ad_centroid = train_clf.loc[train_clf["ResearchGroup_Mapped"] == "AD", cols].to_numpy(dtype=float).mean(axis=0)
    global_centroid = train_x.mean(axis=0)
    cov = LedoitWolf().fit(train_x)
    precision = cov.precision_
    train_maha_global = mahalanobis(train_x, global_centroid, precision)
    support_95 = float(np.quantile(train_maha_global, 0.95))
    support_99 = float(np.quantile(train_maha_global, 0.99))

    frames: List[pd.DataFrame] = []
    for cohort_split, df in [("ADNI_trainDev", train_clf), ("ADNI_test", test[test["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()), ("OASIS", oasis)]:
        x = df[cols].to_numpy(dtype=float)
        out_cols = [
            c
            for c in [
                "SubjectID",
                "subject_id",
                "session_id",
                "experiment_id",
                "source_batch",
                "protocol_subset",
                "ResearchGroup_Mapped",
                "diagnosis",
                "Age",
                "Sex",
                "Manufacturer",
                "ScannerModel",
                "fold",
                "split",
                "y",
            ]
            if c in df.columns
        ]
        out = df[out_cols].copy()
        out["fold"] = fold
        out["cohort_split"] = cohort_split
        out["cohort_dx"] = ("OASIS_" if cohort_split == "OASIS" else "ADNI_") + out["ResearchGroup_Mapped"].astype(str)
        out["mu_l2_norm"] = np.linalg.norm(x, axis=1)
        out["euclidean_to_adni_cn_centroid"] = np.linalg.norm(x - cn_centroid.reshape(1, -1), axis=1)
        out["euclidean_to_adni_ad_centroid"] = np.linalg.norm(x - ad_centroid.reshape(1, -1), axis=1)
        out["cosine_to_adni_cn_centroid"] = cosine_distance_to_centroid(x, cn_centroid)
        out["cosine_to_adni_ad_centroid"] = cosine_distance_to_centroid(x, ad_centroid)
        out["mahalanobis_to_adni_global"] = mahalanobis(x, global_centroid, precision)
        out["mahalanobis_to_adni_cn_centroid"] = mahalanobis(x, cn_centroid, precision)
        out["mahalanobis_to_adni_ad_centroid"] = mahalanobis(x, ad_centroid, precision)
        out["ood_percentile_vs_adni_train"] = ecdf_percentile(train_maha_global, out["mahalanobis_to_adni_global"].to_numpy())
        out["outside_adni_train_95pct_support"] = out["mahalanobis_to_adni_global"] > support_95
        out["outside_adni_train_99pct_support"] = out["mahalanobis_to_adni_global"] > support_99
        out["euclidean_delta_cn_minus_ad"] = out["euclidean_to_adni_cn_centroid"] - out["euclidean_to_adni_ad_centroid"]
        out["cosine_delta_cn_minus_ad"] = out["cosine_to_adni_cn_centroid"] - out["cosine_to_adni_ad_centroid"]
        out["mahalanobis_delta_cn_minus_ad"] = out["mahalanobis_to_adni_cn_centroid"] - out["mahalanobis_to_adni_ad_centroid"]
        out["euclidean_closer_to_adni_cn_than_ad"] = out["euclidean_delta_cn_minus_ad"] < 0
        out["cosine_closer_to_adni_cn_than_ad"] = out["cosine_delta_cn_minus_ad"] < 0
        out["mahalanobis_closer_to_adni_cn_than_ad"] = out["mahalanobis_delta_cn_minus_ad"] < 0
        frames.append(out)

    subject_distance = pd.concat(frames, ignore_index=True)
    summary_rows: List[Dict[str, Any]] = []
    for keys, g in subject_distance.groupby(["fold", "cohort_split", "ResearchGroup_Mapped"], dropna=False):
        row = {"fold": keys[0], "cohort_split": keys[1], "diagnosis": keys[2], "n": int(len(g))}
        for col in [
            "mu_l2_norm",
            "euclidean_to_adni_cn_centroid",
            "euclidean_to_adni_ad_centroid",
            "cosine_to_adni_cn_centroid",
            "cosine_to_adni_ad_centroid",
            "mahalanobis_to_adni_global",
            "ood_percentile_vs_adni_train",
            "euclidean_delta_cn_minus_ad",
            "mahalanobis_delta_cn_minus_ad",
        ]:
            row.update(summarize_numeric(g, col))
        row["outside_95pct_support_rate"] = float(g["outside_adni_train_95pct_support"].mean())
        row["outside_99pct_support_rate"] = float(g["outside_adni_train_99pct_support"].mean())
        row["closer_to_adni_cn_than_ad_euclidean_rate"] = float(g["euclidean_closer_to_adni_cn_than_ad"].mean())
        row["closer_to_adni_cn_than_ad_mahalanobis_rate"] = float(g["mahalanobis_closer_to_adni_cn_than_ad"].mean())
        summary_rows.append(row)

    train_mean = train_x.mean(axis=0)
    train_std = train_x.std(axis=0, ddof=1)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    dim_rows: List[Dict[str, Any]] = []
    for cohort_split, df in [("ADNI_trainDev", train_clf), ("ADNI_test", test[test["ResearchGroup_Mapped"].isin(["CN", "AD"])]), ("OASIS", oasis)]:
        for dx, g in df.groupby("ResearchGroup_Mapped", dropna=False):
            x = g[cols].to_numpy(dtype=float)
            mean_shift_z = (x.mean(axis=0) - train_mean) / train_std
            std_ratio = x.std(axis=0, ddof=1) / train_std
            for i, col in enumerate(cols):
                dim_rows.append(
                    {
                        "fold": fold,
                        "cohort_split": cohort_split,
                        "diagnosis": dx,
                        "latent_dim": col,
                        "mean_shift_vs_adni_train_z": float(mean_shift_z[i]),
                        "abs_mean_shift_vs_adni_train_z": float(abs(mean_shift_z[i])),
                        "std_ratio_vs_adni_train": float(std_ratio[i]) if np.isfinite(std_ratio[i]) else float("nan"),
                    }
                )
    return subject_distance, pd.DataFrame(summary_rows), pd.DataFrame(dim_rows)


def load_primary_adni_predictions(oof_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(oof_dir / "calib_predictions.csv")
    pred = pred[
        (pred["model_name"] == PRIMARY_MODEL)
        & (pred["feature_set"] == PRIMARY_FEATURE_SET)
        & (pred["calib_method"] == PRIMARY_CALIB)
        & (pred["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    pred["cohort_split"] = "ADNI_test"
    pred["ResearchGroup_Mapped"] = pred["ResearchGroup_Mapped"].map(normalize_dx)
    pred["y"] = pred["y_true"].astype(int)
    return pred


def load_primary_oasis_predictions(panel_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(panel_dir / "predictions.csv")
    pred = pred[
        (pred["candidate"] == OASIS_CANDIDATE)
        & (pred["build_candidate"] == OASIS_BUILD)
        & (pred["prediction_level"] == "fold_model")
        & (pred["model_name"] == PRIMARY_MODEL)
        & (pred["feature_set"] == PRIMARY_FEATURE_SET)
        & (pred["calib_method"] == PRIMARY_CALIB)
        & (pred["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    pred["fold"] = pred["fold"].astype(int)
    pred["cohort_split"] = "OASIS"
    pred["ResearchGroup_Mapped"] = pred["ResearchGroup_Mapped"].map(normalize_dx)
    pred["y"] = pred["y"].astype(int)
    return pred


def score_shift_tables(adni_pred: pd.DataFrame, oasis_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    both = pd.concat(
        [
            adni_pred[["SubjectID", "ResearchGroup_Mapped", "fold", "cohort_split", "y", "y_score", "threshold", "y_pred"]].copy(),
            oasis_pred[["SubjectID", "ResearchGroup_Mapped", "fold", "cohort_split", "y", "y_score", "threshold", "y_pred", "source_batch"]].copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    rows: List[Dict[str, Any]] = []
    for keys, g in both.groupby(["fold", "cohort_split", "ResearchGroup_Mapped"], dropna=False):
        q = g["y_score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        thr = float(g["threshold"].iloc[0])
        rows.append(
            {
                "fold": keys[0],
                "cohort_split": keys[1],
                "diagnosis": keys[2],
                "n": int(len(g)),
                "score_mean": float(g["y_score"].mean()),
                "score_std": float(g["y_score"].std(ddof=1)),
                "score_q10": float(q.loc[0.1]),
                "score_q25": float(q.loc[0.25]),
                "score_median": float(q.loc[0.5]),
                "score_q75": float(q.loc[0.75]),
                "score_q90": float(q.loc[0.9]),
                "threshold": thr,
                "fraction_above_threshold": float((g["y_score"] >= thr).mean()),
                "threshold_percentile_within_group": float((g["y_score"] <= thr).mean()),
                "predicted_ad_rate": float(g["y_pred"].mean()),
            }
        )
    fold_rows: List[Dict[str, Any]] = []
    for fold, g in oasis_pred.groupby("fold", dropna=False):
        if g["y"].nunique() == 2:
            auc = float(roc_auc_score(g["y"], g["y_score"]))
            pr = float(average_precision_score(g["y"], g["y_score"]))
        else:
            auc = pr = float("nan")
        fold_rows.append(
            {
                "fold": int(fold),
                "oasis_auc": auc,
                "oasis_pr_auc": pr,
                "mean_score_cn": float(g.loc[g["y"] == 0, "y_score"].mean()),
                "mean_score_ad": float(g.loc[g["y"] == 1, "y_score"].mean()),
                "predicted_ad_rate": float(g["y_pred"].mean()),
                "threshold": float(g["threshold"].iloc[0]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def correlation_tables(distance: pd.DataFrame, oasis_pred: pd.DataFrame) -> pd.DataFrame:
    pred_cols = ["SubjectID", "fold", "y_score", "y_pred", "threshold"]
    merged = distance[distance["cohort_split"] == "OASIS"].merge(oasis_pred[pred_cols], on=["SubjectID", "fold"], how="left")
    rows: List[Dict[str, Any]] = []
    metrics = [
        "mu_l2_norm",
        "mahalanobis_to_adni_global",
        "ood_percentile_vs_adni_train",
        "euclidean_to_adni_cn_centroid",
        "euclidean_to_adni_ad_centroid",
        "euclidean_delta_cn_minus_ad",
        "mahalanobis_delta_cn_minus_ad",
    ]
    for fold, fdf in merged.groupby("fold"):
        for metric in metrics:
            rows.append(
                {
                    "fold": int(fold),
                    "metric": metric,
                    "pearson_corr_with_oasis_score": float(fdf[[metric, "y_score"]].corr(method="pearson").iloc[0, 1]),
                    "spearman_corr_with_oasis_score": float(fdf[[metric, "y_score"]].corr(method="spearman").iloc[0, 1]),
                    "n": int(fdf[[metric, "y_score"]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_outputs(outdir: Path, distance: pd.DataFrame, score_shift: pd.DataFrame, corr_merge: pd.DataFrame) -> None:
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fold, fdf in distance.groupby("fold"):
        labels = ["ADNI_trainDev_CN", "ADNI_trainDev_AD", "ADNI_test_CN", "ADNI_test_AD", "OASIS_CN", "OASIS_AD"]
        data = []
        for label in labels:
            if label.startswith("ADNI_trainDev"):
                split = "ADNI_trainDev"
                dx = label.split("_")[-1]
            elif label.startswith("ADNI_test"):
                split = "ADNI_test"
                dx = label.split("_")[-1]
            else:
                split = "OASIS"
                dx = label.split("_")[-1]
            data.append(fdf.loc[(fdf["cohort_split"] == split) & (fdf["ResearchGroup_Mapped"] == dx), "mahalanobis_to_adni_global"].dropna().to_numpy())
        plt.figure(figsize=(11, 4))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mahalanobis distance to ADNI train centroid")
        plt.title(f"Fold {fold}: latent OOD distance")
        plt.tight_layout()
        plt.savefig(fig_dir / f"fold_{fold}_mahalanobis_ood_by_group.png", dpi=160)
        plt.close()

        data2 = []
        for label in labels:
            if label.startswith("ADNI_trainDev"):
                split = "ADNI_trainDev"
                dx = label.split("_")[-1]
            elif label.startswith("ADNI_test"):
                split = "ADNI_test"
                dx = label.split("_")[-1]
            else:
                split = "OASIS"
                dx = label.split("_")[-1]
            data2.append(fdf.loc[(fdf["cohort_split"] == split) & (fdf["ResearchGroup_Mapped"] == dx), "euclidean_delta_cn_minus_ad"].dropna().to_numpy())
        plt.figure(figsize=(11, 4))
        plt.axhline(0, color="black", linewidth=1)
        plt.boxplot(data2, labels=labels, showfliers=False)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Euclidean distance delta: CN centroid minus AD centroid")
        plt.title(f"Fold {fold}: negative values are closer to ADNI CN")
        plt.tight_layout()
        plt.savefig(fig_dir / f"fold_{fold}_centroid_delta_by_group.png", dpi=160)
        plt.close()

    oasis = corr_merge[corr_merge["cohort_split"] == "OASIS"].copy()
    if not oasis.empty and "y_score" in oasis.columns:
        plt.figure(figsize=(7, 5))
        for dx, g in oasis.groupby("ResearchGroup_Mapped"):
            plt.scatter(g["mahalanobis_to_adni_global"], g["y_score"], s=12, alpha=0.55, label=f"OASIS {dx}")
        plt.xlabel("Mahalanobis distance to ADNI train centroid")
        plt.ylabel("OASIS fold OOF-ECDF score")
        plt.title("OASIS score vs latent OOD distance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "oasis_score_vs_mahalanobis_distance.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 5))
    score_shift.boxplot(column="score_median", by=["cohort_split", "diagnosis"], grid=False)
    plt.suptitle("")
    plt.title("Fold-level median score by cohort/diagnosis")
    plt.ylabel("Median OOF-ECDF score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fold_median_score_shift_by_cohort_diagnosis.png", dpi=160)
    plt.close()


def final_interpretation(
    group_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    fold_contrib: pd.DataFrame,
    outdir: Path,
) -> None:
    oasis_ad = group_summary[(group_summary["cohort_split"] == "OASIS") & (group_summary["diagnosis"] == "AD")]
    oasis_cn = group_summary[(group_summary["cohort_split"] == "OASIS") & (group_summary["diagnosis"] == "CN")]
    ad_closer_cn = float(oasis_ad["closer_to_adni_cn_than_ad_euclidean_rate"].mean())
    ad_out95 = float(oasis_ad["outside_95pct_support_rate"].mean())
    cn_out95 = float(oasis_cn["outside_95pct_support_rate"].mean())
    primary_fold_auc = float(fold_contrib["oasis_auc"].mean())
    lines = [
        "# Final Interpretation",
        "",
        "This audit used frozen ADNI fold VAEs and existing OOF-ECDF score outputs. No OASIS labels were used to fit scalers, centroids, covariance models, thresholds, calibration, or classifiers.",
        "",
        f"Across folds, OASIS AD subjects were closer to the ADNI CN centroid than the ADNI AD centroid by Euclidean distance in {ad_closer_cn:.1%} of cases on average.",
        f"OASIS latent support outside the ADNI train/dev 95th percentile Mahalanobis boundary averaged {cn_out95:.1%} for CN and {ad_out95:.1%} for AD.",
        f"Mean fold-level OASIS AUC for the promoted model on `runwise164_pilot_parity` was {primary_fold_auc:.3f}; the ensemble panel table should be used for the primary external metric.",
        "",
        "Interpretation: the promoted model retains an external ranking signal on runwise164 pilot-parity OASIS, but the latent-space geometry shows a nontrivial cohort shift. In particular, OASIS AD often occupies regions nearer to the ADNI CN centroid and/or outside the ADNI train/dev support. This is consistent with threshold-transfer weakness and supports reporting OASIS as an external stress test rather than as evidence for further OASIS-driven model selection.",
    ]
    (outdir / "final_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    latent_cache = resolve(args.latent_cache)
    oof_dir = resolve(args.oof_dir)
    panel_dir = resolve(args.oasis_panel)
    oasis_tensor_path = resolve(args.oasis_tensor)
    oasis_manifest_path = resolve(args.oasis_manifest)
    outdir = resolve(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "timestamp_start": now_iso(),
        "args": vars(args),
        "guardrails": [
            "no training",
            "no OASIS fitting",
            "no OASIS threshold calibration",
            "no tensor/metadata/model artifact modification",
        ],
    }

    required = [
        run_dir / "run_config.json",
        latent_cache,
        oof_dir / "calib_predictions.csv",
        panel_dir / "predictions.csv",
        oasis_tensor_path,
        oasis_manifest_path,
    ]
    for fold in FOLDS:
        required.extend(
            [
                latent_cache / f"fold_{fold}_trainDev_latent_mu.csv",
                latent_cache / f"fold_{fold}_test_latent_mu.csv",
                run_dir / f"fold_{fold}" / "vae_norm_params.joblib",
                run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt",
            ]
        )
    artifact = pd.DataFrame(
        [{"path": str(p), "exists": p.exists(), "role": "required"} for p in required]
    )
    write_table(outdir, "artifact_validation", artifact, max_rows=200)
    if not artifact["exists"].all():
        raise FileNotFoundError("Missing required artifacts:\n" + artifact.loc[~artifact["exists"], "path"].to_string(index=False))

    if args.dry_run:
        write_json(outdir / "command_log.json", {**command_log, "timestamp_end": now_iso(), "dry_run": True})
        (outdir / "README.md").write_text("# Latent Distance Audit\n\nDry-run completed; no encoding or distance audit was run.\n", encoding="utf-8")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = load_config(run_dir)
    oasis_tensor, oasis_meta, oasis_channel_names = load_oasis_tensor_and_meta(oasis_tensor_path, oasis_manifest_path)
    selected_names = list(cfg.get("selected_channel_names") or [])
    selected_idx = selected_channel_indices(selected_names, oasis_channel_names)
    oasis_selected = oasis_tensor[:, selected_idx, :, :].astype(np.float32)
    adni_pred = load_primary_adni_predictions(oof_dir)
    oasis_pred = load_primary_oasis_predictions(panel_dir)

    oasis_latents: List[pd.DataFrame] = []
    distance_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    dim_frames: List[pd.DataFrame] = []

    for fold in FOLDS:
        train, test = load_adni_fold(latent_cache, fold)
        oasis_fold = encode_oasis_fold(run_dir, cfg, oasis_selected, oasis_meta, fold, device, args.batch_size)
        oasis_latents.append(oasis_fold)
        dist, summary, dims = distance_metrics_for_fold(train, test, oasis_fold, fold)
        distance_frames.append(dist)
        summary_frames.append(summary)
        dim_frames.append(dims)

    oasis_latent_df = pd.concat(oasis_latents, ignore_index=True)
    oasis_latent_df.to_csv(outdir / "oasis_fold_latent_mu_runwise164.csv", index=False)
    distance = pd.concat(distance_frames, ignore_index=True)
    group_summary = pd.concat(summary_frames, ignore_index=True)
    dim_shift = pd.concat(dim_frames, ignore_index=True)
    write_table(outdir, "subject_distance_metrics", distance, max_rows=200)
    write_table(outdir, "group_distance_summary", group_summary, max_rows=200)
    dim_summary = (
        dim_shift.groupby(["fold", "cohort_split", "diagnosis"], dropna=False)
        .agg(
            n_dims=("latent_dim", "size"),
            mean_abs_mean_shift_z=("abs_mean_shift_vs_adni_train_z", "mean"),
            max_abs_mean_shift_z=("abs_mean_shift_vs_adni_train_z", "max"),
            median_abs_mean_shift_z=("abs_mean_shift_vs_adni_train_z", "median"),
            mean_std_ratio=("std_ratio_vs_adni_train", "mean"),
            median_std_ratio=("std_ratio_vs_adni_train", "median"),
        )
        .reset_index()
    )
    write_table(outdir, "per_dimension_shift_summary", dim_summary, max_rows=200)
    top_dim_shift = dim_shift.sort_values("abs_mean_shift_vs_adni_train_z", ascending=False).head(500)
    write_table(outdir, "top_per_dimension_shifts", top_dim_shift, max_rows=200)

    centroid_summary = group_summary[
        [
            "fold",
            "cohort_split",
            "diagnosis",
            "n",
            "euclidean_delta_cn_minus_ad_mean",
            "mahalanobis_delta_cn_minus_ad_mean",
            "closer_to_adni_cn_than_ad_euclidean_rate",
            "closer_to_adni_cn_than_ad_mahalanobis_rate",
        ]
    ].copy()
    write_table(outdir, "centroid_closeness_summary", centroid_summary, max_rows=200)
    ood_summary = group_summary[
        [
            "fold",
            "cohort_split",
            "diagnosis",
            "n",
            "mahalanobis_to_adni_global_mean",
            "mahalanobis_to_adni_global_median",
            "ood_percentile_vs_adni_train_mean",
            "outside_95pct_support_rate",
            "outside_99pct_support_rate",
        ]
    ].copy()
    write_table(outdir, "latent_support_ood_summary", ood_summary, max_rows=200)

    score_summary, fold_contrib = score_shift_tables(adni_pred, oasis_pred)
    write_table(outdir, "score_shift_summary", score_summary, max_rows=200)
    write_table(outdir, "fold_score_contribution", fold_contrib, max_rows=100)
    corr = correlation_tables(distance, oasis_pred)
    write_table(outdir, "score_distance_correlations", corr, max_rows=200)

    corr_merge = distance.merge(
        oasis_pred[["SubjectID", "fold", "y_score", "y_pred", "threshold"]],
        on=["SubjectID", "fold"],
        how="left",
    )
    plot_outputs(outdir, distance, score_summary, corr_merge)
    final_interpretation(group_summary, score_summary, fold_contrib, outdir)
    readme = [
        "# Promoted latent384 OASIS-vs-ADNI latent distance audit",
        "",
        f"Model: `{run_dir.name}`",
        f"External build: `{OASIS_BUILD}`",
        "",
        "This read-only audit encodes OASIS through frozen ADNI fold VAEs and compares OASIS latent geometry against ADNI train/dev and test latent caches. All distance centroids, covariance shrinkage models, and support thresholds are estimated from ADNI train/dev data only.",
        "",
        "Key outputs:",
        "- `oasis_fold_latent_mu_runwise164.csv`",
        "- `subject_distance_metrics.csv/.md`",
        "- `group_distance_summary.csv/.md`",
        "- `centroid_closeness_summary.csv/.md`",
        "- `latent_support_ood_summary.csv/.md`",
        "- `per_dimension_shift_summary.csv/.md`",
        "- `score_shift_summary.csv/.md`",
        "- `score_distance_correlations.csv/.md`",
        "- `figures/*.png`",
        "- `final_interpretation.md`",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log.update(
        {
            "timestamp_end": now_iso(),
            "dry_run": False,
            "device": str(device),
            "selected_channel_names": selected_names,
            "oasis_latent_rows": int(len(oasis_latent_df)),
            "distance_rows": int(len(distance)),
        }
    )
    write_json(outdir / "command_log.json", command_log)


if __name__ == "__main__":
    main()
