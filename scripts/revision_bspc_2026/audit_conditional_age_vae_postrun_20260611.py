#!/usr/bin/env python3
"""Read-only post-run audit for the conditional-age VAE experiment.

This audit never retrains a VAE and never modifies tensors or metadata. It may
create a classifier-only latent cache by running inference through saved fold
checkpoints, then trains predefined Stage B readouts on existing fold latents.
All thresholds are selected from inner-CV train/dev OOF scores only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep as sweep


DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_age_vae_latent384_beta3p75_T80_p560_full5x5_20260610"
DEFAULT_PROMOTED_DIR = PROJECT_ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
DEFAULT_OUTDIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_age_vae_postrun_audit_20260611"

READOUTS = {
    "mu_only": "z_only",
    "mu_plus_sex": "z_plus_sex",
    "mu_plus_age_sex": "z_plus_age_sex",
}
MODELS = ["logreg_l2", "svm_rbf"]
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIBRATION = "oof_ecdf"

PROMOTED_REFERENCE = {
    "model": "promoted_baseline_ch102_latent384_beta3p75",
    "classifier": "logreg_l2",
    "readout": "mu_plus_age_sex",
    "calibration": "oof_ecdf",
    "threshold_strategy": PRIMARY_THRESHOLD,
    "auc": 0.795155,
    "pr_auc": 0.573934,
    "balanced_accuracy": 0.725979,
    "sensitivity": 0.731959,
    "specificity": 0.720000,
    "f1": 0.563492,
    "philips_cn_fpr": 0.454545,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--promoted-dir", type=Path, default=DEFAULT_PROMOTED_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--reuse-latent-cache", action="store_true", default=True)
    return ap.parse_args()


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else PROJECT_ROOT / p


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_df(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    csv_path = outdir / f"{stem}.csv"
    md_path = outdir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    if df.empty:
        md_path.write_text("_No rows._\n", encoding="utf-8")
    else:
        md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, p, zero_division=0)),
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))) if len(np.unique(y)) == 2 else np.nan,
        "predicted_ad_rate": float(np.mean(p)) if len(p) else np.nan,
    }
    return out


def ecdf_train_test(oof_score: np.ndarray, test_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    oof_score = np.asarray(oof_score, dtype=float)
    test_score = np.asarray(test_score, dtype=float)
    order = np.argsort(oof_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(oof_score) + 1)
    oof_ecdf = (ranks - 0.5) / max(len(oof_score), 1)
    sorted_oof = np.sort(oof_score)
    test_ecdf = np.searchsorted(sorted_oof, test_score, side="right") / max(len(sorted_oof), 1)
    return oof_ecdf.astype(float), test_ecdf.astype(float)


def derive_raw_tp_group(df: pd.DataFrame) -> pd.Series:
    if "raw_tp_group" in df.columns:
        return df["raw_tp_group"].astype(str)
    if "n_timepoints_raw" in df.columns:
        n = pd.to_numeric(df["n_timepoints_raw"], errors="coerce")
        values = pd.Series("otherTP", index=df.index, dtype=object)
        values.loc[n <= 160] = "140TP"
        values.loc[n >= 180] = "197TP"
        values.loc[n.isna()] = "unknown"
        return values
    return pd.Series(["unknown"] * len(df), index=df.index)


def infer_label_from_metadata(df: pd.DataFrame) -> pd.Series:
    rg = df["ResearchGroup_Mapped"].astype(str)
    return rg.map({"CN": 0, "AD": 1}).astype("Int64")


def load_metadata(cfg: Dict[str, Any]) -> pd.DataFrame:
    meta_path = resolve(Path(cfg["metadata_path"]))
    meta = sweep.normalize_metadata(pd.read_csv(meta_path))
    meta["raw_tp_group"] = derive_raw_tp_group(meta)
    return meta


def completion_integrity(run_dir: Path, outdir: Path, metadata: pd.DataFrame, outer_folds: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    removed_subjects_all: List[str] = []
    for fold in range(1, outer_folds + 1):
        fd = run_dir / f"fold_{fold}"
        train_subjects = pd.read_csv(fd / "train_dev_subjects_fold.csv") if (fd / "train_dev_subjects_fold.csv").exists() else pd.DataFrame()
        test_subjects = pd.read_csv(fd / "test_subjects_fold.csv") if (fd / "test_subjects_fold.csv").exists() else pd.DataFrame()
        removed_path = fd / f"vae_pool_required_metadata_removed_fold_{fold}.csv"
        removed = pd.read_csv(removed_path) if removed_path.exists() else pd.DataFrame()
        if "SubjectID" in removed.columns:
            removed_subjects_all.extend(removed["SubjectID"].astype(str).tolist())
        val_idx_path = fd / "vae_internal_val_idx_local_to_pool.npy"
        val_n = int(len(np.load(val_idx_path))) if val_idx_path.exists() else -1
        required = {
            "vae_model": fd / f"vae_model_fold_{fold}.pt",
            "train_subjects": fd / "train_dev_subjects_fold.csv",
            "test_subjects": fd / "test_subjects_fold.csv",
            "train_indices": fd / "train_dev_indices.npy",
            "test_indices": fd / "test_indices.npy",
            "conditioning_transformer": fd / "vae_conditioning_transformer.joblib",
            "conditioning_summary": fd / "vae_conditioning_summary.csv",
            "logreg_predictions": fd / "test_predictions_logreg.csv",
            "svm_predictions": fd / "test_predictions_svm.csv",
            "history": fd / f"vae_train_history_fold_{fold}.joblib",
            "rate_distortion": fd / f"fold_{fold}_rate_distortion.csv",
        }
        pred_nan = False
        for pred_name in ["test_predictions_logreg.csv", "test_predictions_svm.csv"]:
            pp = fd / pred_name
            if pp.exists():
                p = pd.read_csv(pp)
                score_cols = [c for c in p.columns if "score" in c.lower()]
                pred_nan = pred_nan or p[score_cols].isna().any().any()
        classifier_subjects = set(train_subjects.get("SubjectID", pd.Series(dtype=str)).astype(str)) | set(
            test_subjects.get("SubjectID", pd.Series(dtype=str)).astype(str)
        )
        rows.append(
            {
                "fold": fold,
                **{f"has_{name}": path.exists() for name, path in required.items()},
                "vae_internal_val_n": val_n,
                "vae_internal_val_nonzero": val_n > 0,
                "n_train_dev_subjects": int(len(train_subjects)),
                "n_test_subjects": int(len(test_subjects)),
                "removed_required_metadata_subjects": ",".join(sorted(set(removed.get("SubjectID", pd.Series(dtype=str)).astype(str)))) if not removed.empty and "SubjectID" in removed else "",
                "removed_only_128_S_2002": set(removed.get("SubjectID", pd.Series(dtype=str)).astype(str)).issubset({"128_S_2002"}) if not removed.empty else True,
                "128_S_2002_in_classifier_pool": "128_S_2002" in classifier_subjects,
                "prediction_scores_have_nan": bool(pred_nan),
                "status": "PASS" if all(path.exists() for path in required.values()) and val_n > 0 and "128_S_2002" not in classifier_subjects and not pred_nan else "CHECK",
            }
        )
    return pd.DataFrame(rows)


def add_latent_nan_integrity(integrity: pd.DataFrame, cache_dir: Path, outer_folds: int) -> pd.DataFrame:
    out = integrity.copy()
    for fold in range(1, outer_folds + 1):
        fold_mask = out["fold"].eq(fold)
        for split in ["trainDev", "test"]:
            p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            if not p.exists():
                out.loc[fold_mask, f"{split}_latent_cache_exists"] = False
                out.loc[fold_mask, f"{split}_latent_has_nan"] = True
                continue
            df = pd.read_csv(p)
            mu_cols = [c for c in df.columns if c.startswith("mu_")]
            out.loc[fold_mask, f"{split}_latent_cache_exists"] = True
            out.loc[fold_mask, f"{split}_latent_has_nan"] = bool(df[mu_cols].isna().any().any())
        bad_latent = bool(out.loc[fold_mask, ["trainDev_latent_has_nan", "test_latent_has_nan"]].astype(bool).any(axis=None))
        if bad_latent:
            out.loc[fold_mask, "status"] = "CHECK"
    return out


def fold_identity(run_dir: Path, promoted_dir: Path, metadata: pd.DataFrame, outer_folds: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, outer_folds + 1):
        cand = pd.read_csv(run_dir / f"fold_{fold}" / "test_subjects_fold.csv")
        ref = pd.read_csv(promoted_dir / f"fold_{fold}" / "test_subjects_fold.csv")
        cand_ids = set(cand["SubjectID"].astype(str))
        ref_ids = set(ref["SubjectID"].astype(str))
        cand_meta = cand.merge(metadata[["SubjectID", "Manufacturer"]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
        ref_meta = ref.merge(metadata[["SubjectID", "Manufacturer"]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
        cand_counts = cand_meta["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
        ref_counts = ref_meta["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
        cand_mfr = cand_meta["Manufacturer"].value_counts(dropna=False).to_dict()
        ref_mfr = ref_meta["Manufacturer"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "fold": fold,
                "same_test_subject_ids": cand_ids == ref_ids,
                "missing_from_candidate": ",".join(sorted(ref_ids - cand_ids)),
                "extra_in_candidate": ",".join(sorted(cand_ids - ref_ids)),
                "candidate_cn": int(cand_counts.get("CN", 0)),
                "candidate_ad": int(cand_counts.get("AD", 0)),
                "reference_cn": int(ref_counts.get("CN", 0)),
                "reference_ad": int(ref_counts.get("AD", 0)),
                "same_cn_ad_counts": {k: int(cand_counts.get(k, 0)) for k in ["CN", "AD"]} == {k: int(ref_counts.get(k, 0)) for k in ["CN", "AD"]},
                "candidate_manufacturer_counts": json.dumps(cand_mfr, sort_keys=True),
                "reference_manufacturer_counts": json.dumps(ref_mfr, sort_keys=True),
                "same_manufacturer_counts": cand_mfr == ref_mfr,
                "status": "PASS" if cand_ids == ref_ids and cand_counts == ref_counts and cand_mfr == ref_mfr else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def train_stageb_ecdf(cache_dir: Path, cfg: Dict[str, Any], inner_folds: int, n_jobs: int, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    hyper_rows: List[Dict[str, Any]] = []
    for fold in range(1, int(cfg["outer_folds"]) + 1):
        train_df, test_df = sweep.load_latent_pair(cache_dir, fold)
        for extra in ["Site3", "n_timepoints_raw", "raw_tp_group"]:
            if extra not in test_df.columns and extra in metadata.columns:
                test_df = test_df.merge(metadata[["SubjectID", extra]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
            if extra not in train_df.columns and extra in metadata.columns:
                train_df = train_df.merge(metadata[["SubjectID", extra]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
        if "raw_tp_group" not in test_df.columns:
            test_df["raw_tp_group"] = derive_raw_tp_group(test_df)
        if "raw_tp_group" not in train_df.columns:
            train_df["raw_tp_group"] = derive_raw_tp_group(train_df)

        mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
        y_train = train_df["y"].astype(int).to_numpy()
        y_test = test_df["y"].astype(int).to_numpy()
        inner_key, inner_context, min_inner_cell = sweep.inner_stratification_key(train_df, n_splits=inner_folds)
        inner_cv = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=cfg["seed"] + fold + 30).split(np.zeros(len(train_df)), inner_key))
        specs = sweep.classifier_specs(seed=cfg["seed"] + fold, y_train=y_train)
        for readout_label, feature_set in READOUTS.items():
            if feature_set == "z_only":
                feature_cols = mu_cols
            elif feature_set == "z_plus_sex":
                feature_cols = mu_cols + ["Sex"]
            else:
                feature_cols = mu_cols + ["Age", "Sex"]
            x_train = train_df[feature_cols].copy()
            x_test = test_df[feature_cols].copy()
            pre = sweep.make_preprocessor(mu_cols, readout_feature_set=feature_set)
            for model_name in MODELS:
                base_pipe, grid, status = specs[model_name]
                if status != "available":
                    continue
                pipe = clone(base_pipe)
                pipe.steps[0] = ("pre", pre)
                search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=n_jobs, refit=True, error_score=np.nan)
                search.fit(x_train, y_train)
                best = search.best_estimator_
                oof_raw = cross_val_predict(clone(best), x_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
                test_raw = sweep.score_1d(best, x_test)
                oof_ecdf, test_ecdf = ecdf_train_test(oof_raw, test_raw)
                thresholds = sweep.select_thresholds(y_train, oof_ecdf)
                selected = [t for t in thresholds if t["threshold_strategy"] == PRIMARY_THRESHOLD][0]
                thr = float(selected["threshold"])
                y_pred = (test_ecdf >= thr).astype(int)
                row = {
                    "fold": fold,
                    "classifier": model_name,
                    "readout": readout_label,
                    "feature_set": feature_set,
                    "calibration": PRIMARY_CALIBRATION,
                    "threshold_strategy": PRIMARY_THRESHOLD,
                    "threshold": thr,
                    "best_inner_auc": float(search.best_score_),
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "inner_cv_context": inner_context,
                    "minimum_inner_stratum_count": int(min_inner_cell),
                    "inner_oof_sensitivity": float(selected["inner_oof_sensitivity"]),
                    "inner_oof_specificity": float(selected["inner_oof_specificity"]),
                    "inner_oof_balanced_accuracy": float(selected["inner_oof_balanced_accuracy"]),
                }
                row.update(metrics(y_test, test_ecdf, y_pred))
                fold_rows.append(row)
                hyper_rows.append({k: row[k] for k in ["fold", "classifier", "readout", "best_params", "best_inner_auc", "threshold", "inner_cv_context", "minimum_inner_stratum_count"]})
                pred = test_df[
                    [
                        "SubjectID",
                        "tensor_idx",
                        "ResearchGroup_Mapped",
                        "Manufacturer",
                        "Age",
                        "Sex",
                        "Site3",
                        "raw_tp_group",
                    ]
                ].copy()
                pred["fold"] = fold
                pred["classifier"] = model_name
                pred["readout"] = readout_label
                pred["calibration"] = PRIMARY_CALIBRATION
                pred["threshold_strategy"] = PRIMARY_THRESHOLD
                pred["threshold"] = thr
                pred["y_true"] = y_test
                pred["y_score"] = test_ecdf
                pred["y_score_raw"] = test_raw
                pred["y_pred"] = y_pred
                pred_frames.append(pred)
    pred_all = pd.concat(pred_frames, ignore_index=True)
    return pd.DataFrame(fold_rows), pred_all, pd.DataFrame(hyper_rows)


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (clf, readout), sub in pred.groupby(["classifier", "readout"], dropna=False):
        row = {
            "model": f"conditional_age_{readout}_{clf}",
            "classifier": clf,
            "readout": readout,
            "calibration": PRIMARY_CALIBRATION,
            "threshold_strategy": PRIMARY_THRESHOLD,
        }
        row.update(metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["classifier", "readout"]).reset_index(drop=True)


def manufacturer_errors(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    philips_rows: List[Dict[str, Any]] = []
    for (clf, readout), top in pred.groupby(["classifier", "readout"], dropna=False):
        for mfr, sub in top.groupby("Manufacturer", dropna=False):
            cn = sub[sub["y_true"].eq(0)]
            ad = sub[sub["y_true"].eq(1)]
            rows.append(
                {
                    "classifier": clf,
                    "readout": readout,
                    "Manufacturer": mfr,
                    "cn_n": int(len(cn)),
                    "cn_fp": int((cn["y_pred"] == 1).sum()),
                    "cn_fpr": safe_div(int((cn["y_pred"] == 1).sum()), len(cn)),
                    "ad_n": int(len(ad)),
                    "ad_fn": int((ad["y_pred"] == 0).sum()),
                    "ad_fnr": safe_div(int((ad["y_pred"] == 0).sum()), len(ad)),
                    "score_median_cn": float(cn["y_score"].median()) if len(cn) else np.nan,
                    "score_median_ad": float(ad["y_score"].median()) if len(ad) else np.nan,
                }
            )
        philips_cn = top[top["y_true"].eq(0) & top["Manufacturer"].astype(str).str.upper().str.contains("PHILIPS", na=False)]
        for group_col in ["raw_tp_group", "Site3"]:
            if group_col in philips_cn.columns:
                for g, sub in philips_cn.groupby(group_col, dropna=False):
                    philips_rows.append(
                        {
                            "classifier": clf,
                            "readout": readout,
                            "grouping": group_col,
                            "group": g,
                            "cn_n": int(len(sub)),
                            "cn_fp": int((sub["y_pred"] == 1).sum()),
                            "cn_fpr": safe_div(int((sub["y_pred"] == 1).sum()), len(sub)),
                            "high_confidence_fp_score_gt_0p75": int(((sub["y_pred"] == 1) & (sub["y_score"] > 0.75)).sum()),
                        }
                    )
        philips_rows.append(
            {
                "classifier": clf,
                "readout": readout,
                "grouping": "overall",
                "group": "PHILIPS",
                "cn_n": int(len(philips_cn)),
                "cn_fp": int((philips_cn["y_pred"] == 1).sum()),
                "cn_fpr": safe_div(int((philips_cn["y_pred"] == 1).sum()), len(philips_cn)),
                "high_confidence_fp_score_gt_0p75": int(((philips_cn["y_pred"] == 1) & (philips_cn["y_score"] > 0.75)).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(philips_rows)


def latent_association(cache_dir: Path, promoted_cache_dir: Path | None, metadata: pd.DataFrame, pred: pd.DataFrame, outer_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    geom_rows: List[Dict[str, Any]] = []

    def collect(cache: Path, label: str, primary_pred: pd.DataFrame | None = None) -> None:
        frames = []
        for fold in range(1, outer_folds + 1):
            test = pd.read_csv(cache / f"fold_{fold}_test_latent_mu.csv")
            train = pd.read_csv(cache / f"fold_{fold}_trainDev_latent_mu.csv")
            for df, split in [(test, "test"), (train, "trainDev")]:
                for extra in ["Site3", "n_timepoints_raw", "raw_tp_group"]:
                    if extra not in df.columns and extra in metadata.columns:
                        df = df.merge(metadata[["SubjectID", extra]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
                df["raw_tp_group"] = derive_raw_tp_group(df)
                df["fold"] = fold
                df["split"] = split
                frames.append(df)
            # Geometry uses train centroids and primary predictions on test.
            if primary_pred is not None and label == "conditional_age":
                mu_cols = [c for c in train.columns if c.startswith("mu_")]
                cn_cent = train.loc[train["y"].eq(0), mu_cols].mean().to_numpy()
                ad_cent = train.loc[train["y"].eq(1), mu_cols].mean().to_numpy()
                pp = primary_pred[primary_pred["fold"].eq(fold) & primary_pred["classifier"].eq("logreg_l2") & primary_pred["readout"].eq("mu_plus_age_sex")]
                merged = test.merge(pp[["SubjectID", "y_pred", "y_score"]], on="SubjectID", how="left")
                philips_cn = merged[merged["y"].eq(0) & merged["Manufacturer"].astype(str).str.upper().str.contains("PHILIPS", na=False)]
                for status, sub in [
                    ("Philips_CN_FP", philips_cn[philips_cn["y_pred"].eq(1)]),
                    ("Philips_CN_TN", philips_cn[philips_cn["y_pred"].eq(0)]),
                ]:
                    if sub.empty:
                        geom_rows.append({"source": label, "fold": fold, "group": status, "n": 0})
                    else:
                        x = sub[mu_cols].to_numpy(dtype=float)
                        geom_rows.append(
                            {
                                "source": label,
                                "fold": fold,
                                "group": status,
                                "n": int(len(sub)),
                                "latent_norm_mean": float(np.linalg.norm(x, axis=1).mean()),
                                "distance_to_cn_centroid_mean": float(np.linalg.norm(x - cn_cent, axis=1).mean()),
                                "distance_to_ad_centroid_mean": float(np.linalg.norm(x - ad_cent, axis=1).mean()),
                                "score_mean": float(sub["y_score"].mean()),
                            }
                        )
        all_test = pd.concat([f for f in frames if f["split"].iloc[0] == "test"], ignore_index=True)
        mu_cols = [c for c in all_test.columns if c.startswith("mu_")]
        age = pd.to_numeric(all_test["Age"], errors="coerce")
        corr = [
            abs(np.corrcoef(all_test[c].astype(float), age.astype(float))[0, 1])
            for c in mu_cols
            if age.notna().all() and np.std(all_test[c].astype(float)) > 0
        ]
        raw_group = all_test["raw_tp_group"].astype(str)
        raw_binary = raw_group.eq("140TP").astype(int)
        raw_corr = [
            abs(np.corrcoef(all_test[c].astype(float), raw_binary)[0, 1])
            for c in mu_cols
            if raw_binary.nunique() == 2 and np.std(all_test[c].astype(float)) > 0
        ]
        mfr = all_test["Manufacturer"].astype(str)
        # Lightweight latent age prediction from OOF test latents only.
        age_r2 = np.nan
        age_mae = np.nan
        if age.notna().all():
            x = all_test[mu_cols].to_numpy(dtype=float)
            cv = KFold(n_splits=5, shuffle=True, random_state=123)
            yhat = cross_val_predict(RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]), x, age.to_numpy(dtype=float), cv=cv)
            ss_res = float(np.sum((age.to_numpy(dtype=float) - yhat) ** 2))
            ss_tot = float(np.sum((age.to_numpy(dtype=float) - age.mean()) ** 2))
            age_r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
            age_mae = float(np.mean(np.abs(age.to_numpy(dtype=float) - yhat)))
        rows.append(
            {
                "source": label,
                "n_test": int(len(all_test)),
                "latent_dim": int(len(mu_cols)),
                "latent_age_abs_corr_mean": float(np.nanmean(corr)) if corr else np.nan,
                "latent_age_abs_corr_max": float(np.nanmax(corr)) if corr else np.nan,
                "latent_raw_tp_140_abs_corr_mean": float(np.nanmean(raw_corr)) if raw_corr else np.nan,
                "latent_raw_tp_140_abs_corr_max": float(np.nanmax(raw_corr)) if raw_corr else np.nan,
                "age_from_mu_cv_r2": age_r2,
                "age_from_mu_cv_mae": age_mae,
                "manufacturer_levels": ",".join(sorted(mfr.dropna().unique())),
            }
        )

    collect(cache_dir, "conditional_age", primary_pred=pred)
    if promoted_cache_dir is not None and promoted_cache_dir.exists():
        collect(promoted_cache_dir, "promoted_baseline", primary_pred=None)
    return pd.DataFrame(rows), pd.DataFrame(geom_rows)


def training_curve_summary(run_dir: Path, outer_folds: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, outer_folds + 1):
        fd = run_dir / f"fold_{fold}"
        hist_path = fd / f"vae_train_history_fold_{fold}.joblib"
        rd_path = fd / f"fold_{fold}_rate_distortion.csv"
        info_path = fd / f"fold_{fold}_trainDev_latent_info_summary.csv"
        hist = joblib.load(hist_path)
        val_modelsel = np.asarray(hist.get("val_loss_modelsel", hist.get("val_loss", [])), dtype=float)
        best_i = int(np.nanargmin(val_modelsel)) if len(val_modelsel) else -1
        rd = pd.read_csv(rd_path) if rd_path.exists() else pd.DataFrame()
        final_rd = rd.iloc[-1].to_dict() if not rd.empty else {}
        best_rd = rd.iloc[best_i].to_dict() if not rd.empty and 0 <= best_i < len(rd) else {}
        info = pd.read_csv(info_path) if info_path.exists() else pd.DataFrame()
        y_info = info[info["variable"].eq("Y_target")].iloc[0].to_dict() if not info.empty and info["variable"].eq("Y_target").any() else {}
        m_info = info[info["variable"].eq("Manufacturer")].iloc[0].to_dict() if not info.empty and info["variable"].eq("Manufacturer").any() else {}
        rows.append(
            {
                "fold": fold,
                "final_epoch": int(len(val_modelsel)),
                "best_epoch": int(best_i + 1) if best_i >= 0 else np.nan,
                "epochs_after_best": int(len(val_modelsel) - best_i - 1) if best_i >= 0 else np.nan,
                "best_val_loss_beta_max": float(val_modelsel[best_i]) if best_i >= 0 else np.nan,
                "final_train_loss": float(hist.get("train_loss", [np.nan])[-1]),
                "final_val_loss": float(hist.get("val_loss", [np.nan])[-1]),
                "final_D_val": final_rd.get("D_val", np.nan),
                "final_R_val_bits": final_rd.get("R_val_bits", np.nan),
                "final_beta_kld_over_recon": float(hist.get("val_beta_kld_over_recon", [np.nan])[-1]),
                "best_D_val": best_rd.get("D_val", np.nan),
                "best_R_val_bits": best_rd.get("R_val_bits", np.nan),
                "n_active_trainDev": y_info.get("n_active", np.nan),
                "total_correlation_trainDev_nats": y_info.get("total_correlation_nats", np.nan),
                "mi_y_sum_nats": y_info.get("mi_sum_nats", np.nan),
                "mi_manufacturer_sum_nats": m_info.get("mi_sum_nats", np.nan),
                "fold_instability_flag": "long_after_best" if (best_i >= 0 and len(val_modelsel) - best_i - 1 > 560) else "none",
            }
        )
    return pd.DataFrame(rows)


def final_decision(
    pooled: pd.DataFrame,
    philips: pd.DataFrame,
    latent_assoc: pd.DataFrame,
    mfr_errors: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    rows = [PROMOTED_REFERENCE.copy()]
    cond_assoc = latent_assoc[latent_assoc["source"].eq("conditional_age")]
    prom_assoc = latent_assoc[latent_assoc["source"].eq("promoted_baseline")]
    cond_age_assoc = float(cond_assoc["latent_age_abs_corr_mean"].iloc[0]) if not cond_assoc.empty else np.nan
    prom_age_assoc = float(prom_assoc["latent_age_abs_corr_mean"].iloc[0]) if not prom_assoc.empty else np.nan
    for _, r in pooled.iterrows():
        p = philips[(philips["classifier"].eq(r["classifier"])) & (philips["readout"].eq(r["readout"])) & philips["grouping"].eq("overall")]
        philips_fpr = float(p["cn_fpr"].iloc[0]) if not p.empty else np.nan
        p140 = philips[(philips["classifier"].eq(r["classifier"])) & (philips["readout"].eq(r["readout"])) & philips["grouping"].eq("raw_tp_group") & philips["group"].eq("140TP")]
        p140_fpr = float(p140["cn_fpr"].iloc[0]) if not p140.empty else np.nan
        ge = mfr_errors[
            (mfr_errors["classifier"].eq(r["classifier"]))
            & (mfr_errors["readout"].eq(r["readout"]))
            & (mfr_errors["Manufacturer"].astype(str).str.upper().eq("GE"))
        ]
        ge_ad_fnr = float(ge["ad_fnr"].iloc[0]) if not ge.empty else np.nan
        decision = "reject"
        if (
            r["auc"] >= PROMOTED_REFERENCE["auc"] - 0.005
            and r["pr_auc"] >= PROMOTED_REFERENCE["pr_auc"] - 0.005
            and r["f1"] >= PROMOTED_REFERENCE["f1"] - 0.02
            and philips_fpr < PROMOTED_REFERENCE["philips_cn_fpr"]
            and (np.isnan(cond_age_assoc) or np.isnan(prom_age_assoc) or cond_age_assoc < prom_age_assoc)
        ):
            decision = "sensitivity_only"
            if philips_fpr < 0.40 and r["auc"] >= PROMOTED_REFERENCE["auc"] and r["pr_auc"] >= PROMOTED_REFERENCE["pr_auc"]:
                decision = "promotion_discussion_candidate"
        rows.append(
            {
                "model": f"conditional_age_{r['readout']}_{r['classifier']}",
                "classifier": r["classifier"],
                "readout": r["readout"],
                "calibration": r["calibration"],
                "threshold_strategy": r["threshold_strategy"],
                "auc": r["auc"],
                "pr_auc": r["pr_auc"],
                "balanced_accuracy": r["balanced_accuracy"],
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "f1": r["f1"],
                "philips_cn_fpr": philips_fpr,
                "philips_140tp_cn_fpr": p140_fpr,
                "ge_ad_fnr": ge_ad_fnr,
                "latent_age_association": cond_age_assoc,
                "promoted_latent_age_association": prom_age_assoc,
                "decision": decision,
            }
        )
    decision_df = pd.DataFrame(rows)
    best = decision_df[decision_df["model"].astype(str).str.startswith("conditional_age")].sort_values(["decision", "auc"], ascending=[True, False])
    any_candidate = decision_df["decision"].eq("promotion_discussion_candidate").any()
    if any_candidate:
        text = "promotion_discussion_candidate: at least one conditional-age readout met the numerical gate; inspect manufacturer/geometric guardrails before any promotion."
    elif decision_df["decision"].eq("sensitivity_only").any():
        text = "sensitivity_only: conditional-age VAE produced at least one non-inferior profile but did not satisfy the full promotion gate."
    else:
        text = "reject: conditional-age VAE did not satisfy the joint AUC/PR-AUC, Philips FPR, and latent-age association promotion gate."
    return decision_df, text


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    promoted_dir = resolve(args.promoted_dir)
    outdir = resolve(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = sweep.load_config(run_dir)
    cfg["outer_folds"] = int(args.outer_folds)
    cfg["inner_folds"] = int(args.inner_folds)
    cfg["folds_to_run"] = list(range(1, int(args.outer_folds) + 1))
    metadata = load_metadata(cfg)
    device = sweep.torch.device("cuda" if args.device == "cuda" else "cpu")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "run_dir": str(run_dir),
        "promoted_dir": str(promoted_dir),
        "output_dir": str(outdir),
        "guardrails": {
            "vae_retraining": False,
            "tensor_modification": False,
            "metadata_modification": False,
            "oasis_scoring": False,
            "threshold_fitting_scope": "inner_cv_train_dev_only",
        },
    }

    integrity = completion_integrity(run_dir, outdir, metadata, args.outer_folds)
    identity = fold_identity(run_dir, promoted_dir, metadata, args.outer_folds)

    latent_manifest = sweep.build_or_load_latent_cache(
        run_dir=run_dir,
        outdir=outdir,
        cfg=cfg,
        metadata=metadata,
        batch_size=args.batch_size,
        device=device,
        reuse=args.reuse_latent_cache,
    )
    write_json(outdir / "latent_cache_manifest.json", latent_manifest)

    cache_dir = outdir / "latent_cache"
    integrity = add_latent_nan_integrity(integrity, cache_dir, args.outer_folds)
    stageb_foldwise, predictions, hyper = train_stageb_ecdf(cache_dir, cfg, args.inner_folds, args.n_jobs, metadata)
    pooled = pooled_metrics(predictions)
    mfr_errors, philips = manufacturer_errors(predictions)
    promoted_cache = promoted_dir / "classifier_only_readout" / "latent_cache"
    latent_assoc, latent_geom = latent_association(cache_dir, promoted_cache if promoted_cache.exists() else None, metadata, predictions, args.outer_folds)
    training = training_curve_summary(run_dir, args.outer_folds)
    decision_df, decision_text = final_decision(pooled, philips, latent_assoc, mfr_errors)

    write_df(outdir, "completion_integrity_audit", integrity)
    write_df(outdir, "fold_identity_vs_promoted", identity)
    write_df(outdir, "training_curve_summary", training)
    write_df(outdir, "stageB_three_readout_foldwise_metrics", stageb_foldwise)
    write_df(outdir, "stageB_three_readout_pooled_metrics", pooled)
    write_df(outdir, "manufacturer_error_by_readout", mfr_errors)
    write_df(outdir, "philips_cn_fpr_by_readout", philips)
    write_df(outdir, "latent_age_protocol_association", latent_assoc)
    write_df(outdir, "latent_geometry_philips_fp_tn", latent_geom)
    write_df(outdir, "stageB_three_readout_selected_hyperparameters", hyper)
    predictions.to_csv(outdir / "stageB_three_readout_oof_predictions.csv", index=False)
    write_df(outdir, "final_conditional_age_vae_decision_table", decision_df)

    summary_lines = [
        "# Conditional-Age VAE Post-Run Audit",
        "",
        "## Executive Summary",
        "",
        f"- Run: `{run_dir}`",
        f"- Promoted baseline comparator: `{promoted_dir}`",
        "- VAE retraining: `False`.",
        "- Tensor/metadata modification: `False`.",
        "- OASIS scoring: `False`.",
        "- Stage B readouts: `mu_only`, `mu_plus_sex`, `mu_plus_age_sex`.",
        "- Classifiers: `logreg_l2`, `svm_rbf`.",
        "- Primary calibration: `OOF-ECDF`; threshold selected inside train/dev inner CV only.",
        f"- Completion status: `{('PASS' if integrity['status'].eq('PASS').all() else 'CHECK')}`.",
        f"- Fold identity vs promoted: `{('PASS' if identity['status'].eq('PASS').all() else 'FAIL')}`.",
        f"- Decision: `{decision_text}`",
        "",
        "## Primary Pooled Metrics",
        "",
        pooled.to_markdown(index=False),
        "",
        "## Decision Table",
        "",
        decision_df.to_markdown(index=False),
        "",
    ]
    (outdir / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    decision_md = [
        "# Final Conditional-Age VAE Decision",
        "",
        decision_text,
        "",
        "## Gate Logic",
        "",
        "- Do not promote based only on AUC.",
        "- Promotion requires non-worse AUC/PR-AUC, non-worse BA/F1, meaningful Philips CN FPR reduction, Philips 140TP FPR reduction where available, no GE AD FNR worsening, and lower latent-age association.",
        "- This audit did not use OASIS labels, did not fit OASIS thresholds, and did not retrain the VAE.",
        "",
        decision_df.to_markdown(index=False),
        "",
    ]
    (outdir / "final_conditional_age_vae_decision.md").write_text("\n".join(decision_md), encoding="utf-8")

    write_json(outdir / "command_log.json", command_log)


if __name__ == "__main__":
    main()
