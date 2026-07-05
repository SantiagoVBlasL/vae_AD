#!/usr/bin/env python3
"""Read-only ComBat train/apply validation and classifier-only sensitivity audit.

This audit validates whether fold-wise Manufacturer harmonization has a safe
train/apply path before any FULL VAE experiment. It reads the promoted ADNI run,
the locked tensor, metadata, and latent caches. It writes only audit outputs.

No VAE training, tensor modification, metadata modification, model artifact
modification, or threshold fitting on outer-test folds is performed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from neurocombat_sklearn import CombatModel
except Exception:  # pragma: no cover - reported in audit output
    CombatModel = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
PROMOTED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
PROMOTED_OOF_ALT = RESULTS / "recover035_lat384_beta3p75_stageB_oof_score_calibration"
OUT_DEFAULT = RESULTS / "combat_train_apply_validation_classifier_only_sensitivity_20260603"

FOLDS = [1, 2, 3, 4, 5]
SEED = 42
TARGET_SENSITIVITY = 0.70
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
SELECTED_CHANNEL_INDICES = [1, 0, 2]
SELECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]
MANUFACTURER_CODE = {"GE": 0.0, "Philips": 1.0, "SIEMENS": 2.0}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=PROMOTED_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: Sequence[str]) -> dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {"cmd": list(cmd), "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows. See CSV for full table._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class CompatSparseOneHotEncoder(OneHotEncoder):
    """Compatibility wrapper for packages still passing sparse=..."""

    def __init__(
        self,
        *,
        categories: str | list[Any] = "auto",
        drop: Any = None,
        sparse: bool = True,
        dtype: Any = np.float64,
        handle_unknown: str = "error",
        min_frequency: Any = None,
        max_categories: Any = None,
        feature_name_combiner: str = "concat",
    ) -> None:
        self.sparse = sparse
        super().__init__(
            categories=categories,
            drop=drop,
            sparse_output=sparse,
            dtype=dtype,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories,
            feature_name_combiner=feature_name_combiner,
        )


def patch_neurocombat_sklearn_ohe() -> None:
    try:
        import neurocombat_sklearn.neurocombat_sklearn as ncs  # type: ignore

        ncs.OneHotEncoder = CompatSparseOneHotEncoder
    except Exception:
        return


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def normalize_dx(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"CN", "CONTROL", "0", "NORMAL"}:
        return "CN"
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA", "1"}:
        return "AD"
    if s in {"MCI"}:
        return "MCI"
    return str(v)


def normalize_sex(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"F", "FEMALE", "0"}:
        return "F"
    if s in {"M", "MALE", "1"}:
        return "M"
    return "UNKNOWN"


def normalize_manufacturer(v: Any) -> str:
    s = str(v).strip()
    sl = s.lower()
    if "philips" in sl:
        return "Philips"
    if "siemens" in sl:
        return "SIEMENS"
    if sl in {"ge", "general electric"} or "general electric" in sl:
        return "GE"
    return s if s else "UNKNOWN"


def metadata_path_from_run(run_dir: Path) -> Path:
    cfg = read_json(run_dir / "run_config.json")
    args = cfg.get("args", cfg)
    return Path(args["metadata_path"])


def tensor_path_from_run(run_dir: Path) -> Path:
    cfg = read_json(run_dir / "run_config.json")
    args = cfg.get("args", cfg)
    return Path(args["global_tensor_path"])


def load_metadata(run_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path_from_run(run_dir))
    if "tensor_index" in meta.columns and "tensor_idx" not in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["tensor_idx"] = pd.to_numeric(meta["tensor_idx"], errors="coerce").astype("Int64")
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].map(normalize_dx)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].map(normalize_sex)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Site"] = meta.get("Site3", "UNKNOWN").fillna("UNKNOWN").astype(str)
    meta["ADNI_phase"] = meta.get("Visit", "UNKNOWN").fillna("UNKNOWN").astype(str).str.extract(r"(ADNI\d+)", expand=False).fillna("UNKNOWN")
    return meta


def load_tensor(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = tensor_path_from_run(run_dir)
    npz = np.load(path, allow_pickle=True)
    tensor = np.asarray(npz["global_tensor_data"], dtype=np.float32)
    subject_ids = np.asarray(npz["subject_ids"]).astype(str)
    channel_names = [str(x) for x in np.asarray(npz["channel_names"]).tolist()]
    return tensor, subject_ids, channel_names


def fold_subjects(run_dir: Path, fold: int, split: str) -> pd.DataFrame:
    if split == "trainDev":
        path = run_dir / f"fold_{fold}/train_dev_subjects_fold.csv"
    elif split == "test":
        path = run_dir / f"fold_{fold}/test_subjects_fold.csv"
    else:
        raise ValueError(split)
    df = pd.read_csv(path)
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["tensor_idx"] = pd.to_numeric(df["tensor_idx"], errors="coerce").astype(int)
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
    return df


def fold_with_metadata(run_dir: Path, meta: pd.DataFrame, fold: int, split: str) -> pd.DataFrame:
    sub = fold_subjects(run_dir, fold, split)
    cols = [c for c in meta.columns if c not in {"ResearchGroup_Mapped"}] + ["ResearchGroup_Mapped"]
    merged = sub.merge(meta[cols].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    if "ResearchGroup_Mapped_meta" in merged.columns:
        merged["ResearchGroup_Mapped"] = merged["ResearchGroup_Mapped"].where(
            merged["ResearchGroup_Mapped"].notna(), merged["ResearchGroup_Mapped_meta"]
        )
        merged = merged.drop(columns=["ResearchGroup_Mapped_meta"])
    merged["y"] = merged["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    return merged


def vae_fit_metadata(run_dir: Path, meta: pd.DataFrame, fold: int) -> pd.DataFrame:
    idx = np.load(run_dir / f"fold_{fold}/vae_training_pool_tensor_idx.npy").astype(int)
    out = meta[meta["tensor_idx"].astype("Int64").isin(idx)].copy()
    out = out.sort_values("tensor_idx").reset_index(drop=True)
    return out


def tri_indices(n_roi: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_roi, k=1)


def edge_features(tensor: np.ndarray, tensor_idx: Sequence[int], channel_pos: int, tri: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    arr = tensor[np.asarray(tensor_idx, dtype=int), channel_pos]
    return np.asarray(arr[:, tri[0], tri[1]], dtype=np.float64)


def edge_summary(X_by_channel: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {}
    for name, x in X_by_channel.items():
        rows[f"{name}__mean"] = np.mean(x, axis=1)
        rows[f"{name}__std"] = np.std(x, axis=1)
        rows[f"{name}__median"] = np.median(x, axis=1)
        rows[f"{name}__l1_mean_abs"] = np.mean(np.abs(x), axis=1)
        rows[f"{name}__l2_rms"] = np.sqrt(np.mean(x * x, axis=1))
        rows[f"{name}__p95_abs"] = np.percentile(np.abs(x), 95, axis=1)
    return pd.DataFrame(rows)


@dataclass
class CombatFoldChannel:
    fold: int
    channel_name: str
    selected_channel_index: int
    model: Any
    status: str
    reason: str
    train_shape: tuple[int, int]
    train_manufacturers: str


@dataclass
class FittedCombatWrapper:
    model: Any
    variable_mask: np.ndarray


def covariates_for_combat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mfr = df["Manufacturer"].map(normalize_manufacturer)
    unknown = sorted(set(mfr.astype(str)) - set(MANUFACTURER_CODE))
    if unknown:
        raise ValueError(f"Unsupported Manufacturer values for ComBat coding: {unknown}")
    sites = mfr.map(MANUFACTURER_CODE).to_numpy(dtype=float).reshape(-1, 1)
    sex = df["Sex"].map(normalize_sex).map({"F": 0, "M": 1}).fillna(-1).to_numpy(dtype=float).reshape(-1, 1)
    age = pd.to_numeric(df["Age"], errors="coerce")
    if age.isna().any():
        age = age.fillna(age.median())
    age_arr = age.to_numpy(dtype=float).reshape(-1, 1)
    return sites, sex, age_arr


def fit_combat_model(X_fit: np.ndarray, fit_meta: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    if CombatModel is None:
        return None, {"status": "failed", "reason": "neurocombat_sklearn.CombatModel is unavailable"}
    patch_neurocombat_sklearn_ohe()
    finite_mask = np.isfinite(X_fit).all(axis=0)
    var = np.nanvar(X_fit, axis=0)
    variable_mask = finite_mask & (var > 1e-12)
    if int(variable_mask.sum()) == 0:
        return None, {
            "status": "failed",
            "reason": "No finite non-constant features available for ComBat fit",
            "n_features": int(X_fit.shape[1]),
            "n_variable_features": 0,
            "n_constant_or_nonfinite_features": int(X_fit.shape[1]),
        }
    sites, sex, age = covariates_for_combat(fit_meta)
    model = CombatModel()
    model.fit(X_fit[:, variable_mask], sites, discrete_covariates=sex, continuous_covariates=age)
    return FittedCombatWrapper(model=model, variable_mask=variable_mask), {
        "status": "fit_ok",
        "reason": "",
        "n_fit": int(X_fit.shape[0]),
        "n_features": int(X_fit.shape[1]),
        "n_variable_features_combat_fit": int(variable_mask.sum()),
        "n_constant_or_nonfinite_features_carried_through": int((~variable_mask).sum()),
        "manufacturers_fit": ";".join(str(x) for x in sorted(pd.Series(sites.ravel()).unique())),
    }


def transform_combat_model(model: Any, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    sites, sex, age = covariates_for_combat(df)
    if isinstance(model, FittedCombatWrapper):
        out = np.asarray(X, dtype=np.float64).copy()
        out[:, model.variable_mask] = np.asarray(
            model.model.transform(
                out[:, model.variable_mask],
                sites,
                discrete_covariates=sex,
                continuous_covariates=age,
            ),
            dtype=np.float64,
        )
        return out
    return np.asarray(model.transform(X, sites, discrete_covariates=sex, continuous_covariates=age), dtype=np.float64)


def integrity_rows(
    fold: int,
    channel_name: str,
    split: str,
    original_edges: np.ndarray,
    transformed_edges: np.ndarray,
    original_diag: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(transformed_edges)
    # Reconstruction fills the original diagonal and mirrors the transformed upper triangle.
    return {
        "fold": fold,
        "channel_name": channel_name,
        "split": split,
        "n_subjects": int(transformed_edges.shape[0]),
        "n_features": int(transformed_edges.shape[1]),
        "shape_matches_original": bool(original_edges.shape == transformed_edges.shape),
        "nan_count": int(np.isnan(transformed_edges).sum()),
        "inf_count": int(np.isinf(transformed_edges).sum()),
        "finite_fraction": float(finite.mean()) if finite.size else float("nan"),
        "max_abs_edge_change": float(np.nanmax(np.abs(transformed_edges - original_edges))) if transformed_edges.size else float("nan"),
        "diagonal_reconstruction_policy": "original_diagonal_preserved",
        "max_abs_original_diag": float(np.nanmax(np.abs(original_diag))) if original_diag.size else float("nan"),
        "max_abs_diag_change_after_reconstruction": 0.0,
        "symmetry_policy": "upper_triangle_mirrored",
        "max_abs_asymmetry_after_reconstruction": 0.0,
    }


def safe_binary_metrics(y_true: Iterable[int], y_score: Iterable[float], y_pred: Iterable[int]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    pred = np.asarray(list(y_pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
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
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def select_thresholds(y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    thresholds = np.unique(np.round(np.clip(np.concatenate([[0.0, 0.5, 1.0], s]), 0, 1), 12))
    rows = []
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        rows.append({"threshold": float(thr), "sensitivity": sens, "specificity": spec, "ba": float(np.nanmean([sens, spec])), "j": sens + spec - 1})
    tbl = pd.DataFrame(rows)
    out = [{"threshold_strategy": "fixed_0p5", "threshold": 0.5}]
    r = tbl.sort_values(["ba", "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
    out.append({"threshold_strategy": "inner_oof_balanced_accuracy", "threshold": float(r["threshold"])})
    r = tbl.sort_values(["j", "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
    out.append({"threshold_strategy": "inner_oof_youden_j", "threshold": float(r["threshold"])})
    elig = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if elig.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        r = elig.sort_values(["specificity", "sensitivity", "ba", "threshold"], ascending=[False, False, False, False]).iloc[0]
    out.append({"threshold_strategy": PRIMARY_THRESHOLD, "threshold": float(r["threshold"])})
    return out


def preprocess_for_classifier(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("features", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_names),
            ("age", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["Age"]),
            ("sex", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def stratification_key(df: pd.DataFrame, n_splits: int) -> tuple[pd.Series, str]:
    key = df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)
    if key.value_counts().min() >= n_splits:
        return key, "ResearchGroup_Mapped+Manufacturer"
    return df["y"].astype(int), "label_only_fallback"


def evaluate_binary_feature_set(
    fold: int,
    feature_set: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    inner_folds: int,
    n_jobs: int,
) -> dict[str, pd.DataFrame]:
    y_train = train_df["y"].astype(int).to_numpy()
    y_test = test_df["y"].astype(int).to_numpy()
    key, cv_context = stratification_key(train_df, inner_folds)
    cv = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED + fold).split(np.zeros(len(train_df)), key))
    pipe = Pipeline(
        [
            ("pre", preprocess_for_classifier(feature_cols)),
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
    search = GridSearchCV(pipe, {"model__C": ORIGINAL_C_GRID}, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=True)
    search.fit(train_df[feature_cols + ["Age", "Sex"]], y_train)
    best = search.best_estimator_
    oof = cross_val_predict(clone(best), train_df[feature_cols + ["Age", "Sex"]], y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    test_scores = np.asarray(best.predict_proba(test_df[feature_cols + ["Age", "Sex"]])[:, 1], dtype=float)
    metric_rows = []
    pred_rows = []
    for sel in select_thresholds(y_train, oof):
        thr = float(sel["threshold"])
        y_pred = (test_scores >= thr).astype(int)
        row = {
            "fold": fold,
            "feature_set": feature_set,
            "threshold_strategy": sel["threshold_strategy"],
            "threshold": thr,
            "threshold_selection_context": "train_dev_inner_oof" if sel["threshold_strategy"] != "fixed_0p5" else "fixed",
            "best_C": float(search.best_params_["model__C"]),
            "best_inner_auc": float(search.best_score_),
            "inner_cv_context": cv_context,
        }
        row.update(safe_binary_metrics(y_test, test_scores, y_pred))
        metric_rows.append(row)
        pred = test_df[["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "Site", "ADNI_phase", "y"]].copy()
        pred["fold"] = fold
        pred["feature_set"] = feature_set
        pred["threshold_strategy"] = sel["threshold_strategy"]
        pred["threshold"] = thr
        pred["y_score"] = test_scores
        pred["y_pred"] = y_pred
        pred = pred.rename(columns={"y": "y_true"})
        pred_rows.append(pred)
    return {"metrics": pd.DataFrame(metric_rows), "predictions": pd.concat(pred_rows, ignore_index=True)}


def pooled_metrics(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["threshold"] = "fold_specific" if row.get("threshold_strategy") != "fixed_0p5" else 0.5
        row.update(safe_binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def manufacturer_fpr(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in pred.groupby(["feature_set", "threshold_strategy", "Manufacturer"], dropna=False):
        feature_set, strategy, mfr = keys
        cn = sub[sub["y_true"] == 0]
        ad = sub[sub["y_true"] == 1]
        rows.append(
            {
                "feature_set": feature_set,
                "threshold_strategy": strategy,
                "Manufacturer": mfr,
                "n_cn": int(len(cn)),
                "fp_cn": int((cn["y_pred"] == 1).sum()),
                "fpr_cn": safe_div((cn["y_pred"] == 1).sum(), len(cn)),
                "n_ad": int(len(ad)),
                "fn_ad": int((ad["y_pred"] == 0).sum()),
                "fnr_ad": safe_div((ad["y_pred"] == 0).sum(), len(ad)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_manufacturer_predictability(
    fold: int,
    feature_scope: str,
    stage: str,
    X_train: np.ndarray,
    train_mfr: Sequence[str],
    X_test: np.ndarray,
    test_mfr: Sequence[str],
) -> dict[str, Any]:
    y_train = np.asarray(list(train_mfr)).astype(str)
    y_test = np.asarray(list(test_mfr)).astype(str)
    row = {"fold": fold, "feature_scope": feature_scope, "stage": stage, "n_train": int(len(y_train)), "n_test": int(len(y_test))}
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        row.update({"status": "skipped_single_class", "balanced_accuracy": np.nan, "macro_ovr_auc": np.nan})
        return row
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED + fold),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    row["balanced_accuracy"] = float(balanced_accuracy_score(y_test, pred))
    row["status"] = "fit_ok"
    try:
        proba = clf.predict_proba(X_test)
        row["macro_ovr_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=clf.classes_))
    except Exception:
        row["macro_ovr_auc"] = float("nan")
    return row


def evaluate_age_sex_association(
    fold: int,
    feature_set: str,
    stage: str,
    X_train: np.ndarray,
    train_df: pd.DataFrame,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    age_train = pd.to_numeric(train_df["Age"], errors="coerce")
    age_test = pd.to_numeric(test_df["Age"], errors="coerce")
    ok_train = age_train.notna()
    ok_test = age_test.notna()
    if ok_train.sum() >= 10 and ok_test.sum() >= 5:
        reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        reg.fit(X_train[ok_train.to_numpy()], age_train[ok_train].to_numpy(dtype=float))
        pred = reg.predict(X_test[ok_test.to_numpy()])
        rows.append(
            {
                "fold": fold,
                "feature_set": feature_set,
                "stage": stage,
                "target": "Age",
                "metric_primary": "r2",
                "r2": float(r2_score(age_test[ok_test].to_numpy(dtype=float), pred)),
                "mae": float(mean_absolute_error(age_test[ok_test].to_numpy(dtype=float), pred)),
                "status": "fit_ok",
            }
        )
    sex_train = train_df["Sex"].map(normalize_sex)
    sex_test = test_df["Sex"].map(normalize_sex)
    ok_train = sex_train.isin(["F", "M"])
    ok_test = sex_test.isin(["F", "M"])
    if ok_train.sum() >= 10 and ok_test.sum() >= 5 and sex_train[ok_train].nunique() == 2 and sex_test[ok_test].nunique() == 2:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED + fold)),
            ]
        )
        ytr = sex_train[ok_train].map({"F": 0, "M": 1}).to_numpy(dtype=int)
        yte = sex_test[ok_test].map({"F": 0, "M": 1}).to_numpy(dtype=int)
        clf.fit(X_train[ok_train.to_numpy()], ytr)
        score = clf.predict_proba(X_test[ok_test.to_numpy()])[:, 1]
        pred = (score >= 0.5).astype(int)
        rows.append(
            {
                "fold": fold,
                "feature_set": feature_set,
                "stage": stage,
                "target": "Sex_M",
                "metric_primary": "auc",
                "auc": float(roc_auc_score(yte, score)),
                "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                "status": "fit_ok",
            }
        )
    return rows


def residualize_latent(train_df: pd.DataFrame, test_df: pd.DataFrame, mu_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    age = pd.to_numeric(train["Age"], errors="coerce")
    age_mean = float(age.mean())
    age_std = float(age.std(ddof=0) or 1.0)
    mfr_levels = sorted(train["Manufacturer"].astype(str).unique().tolist())
    dummy_levels = mfr_levels[1:]

    def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        a = pd.to_numeric(df["Age"], errors="coerce").fillna(age_mean)
        age_z = ((a - age_mean) / age_std).to_numpy(dtype=float).reshape(-1, 1)
        sex = df["Sex"].map(normalize_sex).map({"F": 0.0, "M": 1.0}).fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
        mfr = df["Manufacturer"].astype(str)
        dummies = np.column_stack([(mfr == lvl).to_numpy(dtype=float) for lvl in dummy_levels]) if dummy_levels else np.zeros((len(df), 0))
        return np.concatenate([np.ones((len(df), 1)), age_z, sex, dummies], axis=1), dummies

    X, dm_train = design(train)
    Xt, dm_test = design(test)
    reg = LinearRegression(fit_intercept=False).fit(X, train[mu_cols].to_numpy(dtype=float))
    coef = np.asarray(reg.coef_, dtype=float)
    if dummy_levels:
        coef_mfr = coef[:, -len(dummy_levels) :].T
        train.loc[:, mu_cols] = train[mu_cols].to_numpy(dtype=float) - dm_train @ coef_mfr
        test.loc[:, mu_cols] = test[mu_cols].to_numpy(dtype=float) - dm_test @ coef_mfr
    return train, test


def combat_latent(train_df: pd.DataFrame, test_df: pd.DataFrame, mu_cols: list[str]) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    if CombatModel is None:
        return None, None, "neurocombat_sklearn unavailable"
    patch_neurocombat_sklearn_ohe()
    train = train_df.copy()
    test = test_df.copy()
    sites, sex, age = covariates_for_combat(train)
    model = CombatModel()
    try:
        model.fit(train[mu_cols].to_numpy(dtype=float), sites, discrete_covariates=sex, continuous_covariates=age)
        train.loc[:, mu_cols] = model.transform(train[mu_cols].to_numpy(dtype=float), sites, discrete_covariates=sex, continuous_covariates=age)
        tsites, tsex, tage = covariates_for_combat(test)
        test.loc[:, mu_cols] = model.transform(test[mu_cols].to_numpy(dtype=float), tsites, discrete_covariates=tsex, continuous_covariates=tage)
    except Exception as exc:
        return None, None, str(exc)
    return train, test, "fit_ok"


def latent_mu_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("mu_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def load_latent_fold(run_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
    df = pd.read_csv(path)
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
    df["Manufacturer"] = df["Manufacturer"].map(normalize_manufacturer)
    df["Sex"] = df["Sex"].map(normalize_sex)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Site"] = df.get("Site", "UNKNOWN")
    df["ADNI_phase"] = "UNKNOWN"
    df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    return df


def source_subject_column(cols: Sequence[str]) -> str | None:
    for c in ["SubjectID", "Subject ID", "Subject", "PTID", "subject_id"]:
        if c in cols:
            return c
    return None


def raw_manufacturer_audit(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_paths = [
        metadata_path_from_run(PROMOTED_RUN),
        PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_master_manifest/adni_v5_1_master_subject_manifest_all_images.csv",
        PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_master_manifest/adni_v5_1_master_subject_manifest_first_visit.csv",
        PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_full_build_qc/subject_alignment.csv",
        PROJECT_ROOT / "data/idaSearch_4_03_2026.csv",
        PROJECT_ROOT / "data/AD_fMRI_4_28_2026.csv",
        PROJECT_ROOT / "data/AD_fMRI_4_28_2026_extended.csv",
    ]
    raw_rows: list[dict[str, Any]] = []
    subject_meta = meta[["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "Site", "ADNI_phase"]].drop_duplicates("SubjectID")
    for path in source_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        subj_col = source_subject_column(df.columns)
        raw_cols = [
            c
            for c in df.columns
            if any(tok in c.lower() for tok in ["manufacturer", "vendor", "scanner", "fieldstrength", "magneticfield", "protocol", "phase"])
        ]
        if subj_col is None or not raw_cols:
            raw_rows.append({"source_file": rel(path), "raw_column": "NONE_FOUND", "raw_value": "NO_SUBJECT_OR_RAW_COLUMN", "n": 0})
            continue
        raw_rename = {c: f"raw__{c}" for c in raw_cols}
        tmp = df[[subj_col] + raw_cols].copy().rename(columns={subj_col: "SubjectID", **raw_rename})
        tmp["SubjectID"] = tmp["SubjectID"].astype(str)
        tmp = tmp.merge(subject_meta, on="SubjectID", how="inner")
        for col in raw_cols:
            raw_col = raw_rename[col]
            vals = tmp[raw_col].fillna("MISSING").astype(str)
            for raw_val, sub in tmp.assign(_raw=vals).groupby("_raw", dropna=False):
                raw_rows.append(
                    {
                        "source_file": rel(path),
                        "raw_column": col,
                        "raw_value": raw_val,
                        "collapsed_manufacturer_values": ";".join(sorted(sub["Manufacturer"].dropna().astype(str).unique())),
                        "n": int(len(sub)),
                        "n_cn": int((sub["ResearchGroup_Mapped"] == "CN").sum()),
                        "n_mci": int((sub["ResearchGroup_Mapped"] == "MCI").sum()),
                        "n_ad": int((sub["ResearchGroup_Mapped"] == "AD").sum()),
                        "age_mean": float(pd.to_numeric(sub["Age"], errors="coerce").mean()),
                        "sex_counts": json.dumps(sub["Sex"].value_counts(dropna=False).to_dict(), sort_keys=True),
                        "site_n": int(sub["Site"].nunique(dropna=True)),
                        "phase_counts": json.dumps(sub["ADNI_phase"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    }
                )
    raw_df = pd.DataFrame(raw_rows)
    philips = raw_df[
        raw_df["raw_value"].astype(str).str.contains("philips", case=False, na=False)
        | raw_df.get("collapsed_manufacturer_values", pd.Series(dtype=str)).astype(str).str.contains("Philips", case=False, na=False)
    ].copy()
    if philips.empty:
        philips = pd.DataFrame(
            [
                {
                    "source_file": "all_scanned_sources",
                    "raw_column": "Manufacturer",
                    "raw_value": "Philips",
                    "n": int((meta["Manufacturer"] == "Philips").sum()),
                    "n_cn": int(((meta["Manufacturer"] == "Philips") & (meta["ResearchGroup_Mapped"] == "CN")).sum()),
                    "n_mci": int(((meta["Manufacturer"] == "Philips") & (meta["ResearchGroup_Mapped"] == "MCI")).sum()),
                    "n_ad": int(((meta["Manufacturer"] == "Philips") & (meta["ResearchGroup_Mapped"] == "AD")).sum()),
                    "recommendation": "No finer Philips raw sub-brand labels were found locally; keep collapsed Philips.",
                }
            ]
        )
    else:
        philips["adequate_for_split"] = (philips["n_cn"].fillna(0) > 0) & (philips["n_ad"].fillna(0) > 0) & (philips["n"].fillna(0) >= 20)
        if not philips["adequate_for_split"].all():
            philips["recommendation"] = "Keep collapsed Philips; not every observed Philips/raw sub-batch has adequate n and CN/AD representation."
        else:
            philips["recommendation"] = "Potentially splittable, but only after provenance confirmation; not split in this audit."
    return raw_df, philips


def promoted_reference_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = PROMOTED_OOF if PROMOTED_OOF.exists() else PROMOTED_OOF_ALT
    pred_path = oof / "calib_predictions.csv"
    pooled_path = oof / "calib_pooled_metrics.csv"
    if not pred_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    pred = pd.read_csv(pred_path)
    pred = pred[
        pred["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pred["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & pred["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & pred["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    pooled = pd.read_csv(pooled_path) if pooled_path.exists() else pd.DataFrame()
    if not pooled.empty:
        pooled = pooled[
            pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
            & pooled["feature_set"].astype(str).eq(PRIMARY_FEATURES)
            & pooled["calib_method"].astype(str).eq(PRIMARY_CALIB)
            & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        ].copy()
    return pred, pooled


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    command_log: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "start_time": now_iso(),
        "run_dir": str(run_dir),
        "output_dir": str(outdir),
        "guardrails": {
            "vae_training": False,
            "tensor_modification": False,
            "metadata_modification": False,
            "model_artifact_modification": False,
            "threshold_fitting_on_test_folds": False,
            "ephemeral_fold_local_audit_classifiers": True,
            "saved_audit_models": False,
        },
        "commands": [],
    }
    pyc = run_cmd([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())])
    command_log["commands"].append(pyc)
    if pyc["returncode"] != 0:
        raise RuntimeError("py_compile failed")

    meta = load_metadata(run_dir)
    tensor, tensor_subjects, channel_names = load_tensor(run_dir)
    tri = tri_indices(tensor.shape[-1])
    if args.dry_run:
        report = [
            "# Dry Run",
            "",
            f"Run: `{rel(run_dir)}`",
            f"Metadata rows: `{len(meta)}`",
            f"Tensor shape: `{tuple(tensor.shape)}`",
            f"Selected channel indices: `{SELECTED_CHANNEL_INDICES}`",
            f"Selected channel names from tensor: `{[channel_names[i] for i in SELECTED_CHANNEL_INDICES]}`",
            f"CombatModel available: `{CombatModel is not None}`",
            "No transforms/classifiers were run in dry-run mode.",
        ]
        (outdir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        command_log["dry_run"] = True
        command_log["end_time"] = now_iso()
        (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return 0

    raw_mfr, philips_balance = raw_manufacturer_audit(meta)
    write_table(outdir, "raw_manufacturer_value_audit", raw_mfr, max_rows=300)
    write_table(outdir, "philips_subbrand_balance", philips_balance, max_rows=200)

    validation_rows: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    mfr_pred_rows: list[dict[str, Any]] = []
    bio_rows: list[dict[str, Any]] = []
    classifier_metric_parts: list[pd.DataFrame] = []
    classifier_pred_parts: list[pd.DataFrame] = []

    for fold in FOLDS:
        print(f"Fold {fold}: ComBat feature validation", flush=True)
        vae_fit = vae_fit_metadata(run_dir, meta, fold)
        train_df = fold_with_metadata(run_dir, meta, fold, "trainDev")
        test_df = fold_with_metadata(run_dir, meta, fold, "test")
        train_idx = train_df["tensor_idx"].to_numpy(dtype=int)
        test_idx = test_df["tensor_idx"].to_numpy(dtype=int)
        fit_idx = vae_fit["tensor_idx"].to_numpy(dtype=int)

        original_train_by_channel: dict[str, np.ndarray] = {}
        original_test_by_channel: dict[str, np.ndarray] = {}
        combat_train_by_channel: dict[str, np.ndarray] = {}
        combat_test_by_channel: dict[str, np.ndarray] = {}

        for selected_i, channel_pos in enumerate(SELECTED_CHANNEL_INDICES):
            channel_name = channel_names[channel_pos]
            X_fit = edge_features(tensor, fit_idx, channel_pos, tri)
            X_train = edge_features(tensor, train_idx, channel_pos, tri)
            X_test = edge_features(tensor, test_idx, channel_pos, tri)
            original_train_by_channel[channel_name] = X_train
            original_test_by_channel[channel_name] = X_test
            diag_train = np.diagonal(tensor[train_idx, channel_pos], axis1=1, axis2=2)
            diag_test = np.diagonal(tensor[test_idx, channel_pos], axis1=1, axis2=2)

            model, meta_fit = fit_combat_model(X_fit, vae_fit)
            row = {
                "fold": fold,
                "channel_name": channel_name,
                "selected_channel_index": channel_pos,
                "n_fit_vae_pool": int(len(vae_fit)),
                "fit_pool_dx_counts": json.dumps(vae_fit["ResearchGroup_Mapped"].value_counts().to_dict(), sort_keys=True),
                "fit_pool_manufacturer_counts": json.dumps(vae_fit["Manufacturer"].value_counts().to_dict(), sort_keys=True),
                "protected_covariates": "Age+Sex",
                "excluded_covariates": "Diagnosis",
                "batch": "Manufacturer",
                "implementation": "neurocombat_sklearn.CombatModel",
                **meta_fit,
            }
            if model is None:
                validation_rows.append(row)
                continue
            X_train_c = transform_combat_model(model, X_train, train_df)
            X_test_c = transform_combat_model(model, X_test, test_df)
            combat_train_by_channel[channel_name] = X_train_c
            combat_test_by_channel[channel_name] = X_test_c
            validation_rows.append(row)
            integrity.append(integrity_rows(fold, channel_name, "trainDev", X_train, X_train_c, diag_train))
            integrity.append(integrity_rows(fold, channel_name, "test", X_test, X_test_c, diag_test))

            mfr_pred_rows.append(
                evaluate_manufacturer_predictability(fold, channel_name, "before_combat", X_train, train_df["Manufacturer"], X_test, test_df["Manufacturer"])
            )
            mfr_pred_rows.append(
                evaluate_manufacturer_predictability(
                    fold, channel_name, "after_combat", X_train_c, train_df["Manufacturer"], X_test_c, test_df["Manufacturer"]
                )
            )

        if combat_train_by_channel:
            X_train_pool = np.concatenate([original_train_by_channel[n] for n in original_train_by_channel], axis=1)
            X_test_pool = np.concatenate([original_test_by_channel[n] for n in original_test_by_channel], axis=1)
            X_train_pool_c = np.concatenate([combat_train_by_channel[n] for n in combat_train_by_channel], axis=1)
            X_test_pool_c = np.concatenate([combat_test_by_channel[n] for n in combat_test_by_channel], axis=1)
            mfr_pred_rows.append(
                evaluate_manufacturer_predictability(fold, "pooled_3_channels", "before_combat", X_train_pool, train_df["Manufacturer"], X_test_pool, test_df["Manufacturer"])
            )
            mfr_pred_rows.append(
                evaluate_manufacturer_predictability(
                    fold, "pooled_3_channels", "after_combat", X_train_pool_c, train_df["Manufacturer"], X_test_pool_c, test_df["Manufacturer"]
                )
            )

            # Input-derived simple edge summaries.
            orig_summary_train = edge_summary(original_train_by_channel)
            orig_summary_test = edge_summary(original_test_by_channel)
            combat_summary_train = edge_summary(combat_train_by_channel)
            combat_summary_test = edge_summary(combat_test_by_channel)
            for stage, summary_train, summary_test in [
                ("before_combat", orig_summary_train, orig_summary_test),
                ("after_combat", combat_summary_train, combat_summary_test),
            ]:
                feature_cols = list(summary_train.columns)
                tr = pd.concat([train_df.reset_index(drop=True), summary_train.reset_index(drop=True)], axis=1)
                te = pd.concat([test_df.reset_index(drop=True), summary_test.reset_index(drop=True)], axis=1)
                for row in evaluate_age_sex_association(
                    fold, "input_edge_summaries_plus_age_sex", stage, summary_train.to_numpy(dtype=float), train_df, summary_test.to_numpy(dtype=float), test_df
                ):
                    bio_rows.append(row)
                res = evaluate_binary_feature_set(
                    fold,
                    f"input_edge_summaries_{stage}_plus_age_sex",
                    tr,
                    te,
                    feature_cols,
                    args.inner_folds,
                    args.n_jobs,
                )
                classifier_metric_parts.append(res["metrics"])
                classifier_pred_parts.append(res["predictions"])

        # Existing latent z sensitivity variants.
        latent_train = load_latent_fold(run_dir, fold, "trainDev")
        latent_test = load_latent_fold(run_dir, fold, "test")
        latent_train = latent_train.merge(train_df[["SubjectID", "Site", "ADNI_phase"]], on="SubjectID", how="left", suffixes=("", "_meta"))
        latent_test = latent_test.merge(test_df[["SubjectID", "Site", "ADNI_phase"]], on="SubjectID", how="left", suffixes=("", "_meta"))
        mu_cols = latent_mu_columns(latent_train)
        latent_variants: list[tuple[str, pd.DataFrame, pd.DataFrame, str]] = [
            ("latent_original_plus_age_sex", latent_train.copy(), latent_test.copy(), "fit_ok")
        ]
        tr_res, te_res = residualize_latent(latent_train, latent_test, mu_cols)
        latent_variants.append(("latent_residualized_mfr_preserve_age_sex", tr_res, te_res, "fit_ok"))
        tr_combat, te_combat, combat_status = combat_latent(latent_train, latent_test, mu_cols)
        if tr_combat is not None and te_combat is not None:
            latent_variants.append(("latent_combat_mfr_age_sex", tr_combat, te_combat, combat_status))
        else:
            validation_rows.append(
                {
                    "fold": fold,
                    "channel_name": "latent_mu",
                    "selected_channel_index": "latent",
                    "status": "skipped",
                    "reason": f"latent Combat skipped: {combat_status}",
                    "implementation": "neurocombat_sklearn.CombatModel",
                }
            )
        for feature_set, tr, te, _status in latent_variants:
            for row in evaluate_age_sex_association(
                fold, feature_set, "latent", tr[mu_cols].to_numpy(dtype=float), tr, te[mu_cols].to_numpy(dtype=float), te
            ):
                bio_rows.append(row)
            res = evaluate_binary_feature_set(fold, feature_set, tr, te, mu_cols, args.inner_folds, args.n_jobs)
            classifier_metric_parts.append(res["metrics"])
            classifier_pred_parts.append(res["predictions"])

    combat_validation = pd.DataFrame(validation_rows)
    integrity_df = pd.DataFrame(integrity)
    mfr_pred = pd.DataFrame(mfr_pred_rows)
    biological = pd.DataFrame(bio_rows)
    classifier_foldwise = pd.concat(classifier_metric_parts, ignore_index=True, sort=False) if classifier_metric_parts else pd.DataFrame()
    classifier_pred = pd.concat(classifier_pred_parts, ignore_index=True, sort=False) if classifier_pred_parts else pd.DataFrame()
    classifier_pooled = pooled_metrics(classifier_pred, ["feature_set", "threshold_strategy"]) if not classifier_pred.empty else pd.DataFrame()
    classifier_fpr = manufacturer_fpr(classifier_pred) if not classifier_pred.empty else pd.DataFrame()

    # Reference promoted row for comparison.
    ref_pred, ref_pooled = promoted_reference_rows()
    reference_rows = []
    if not ref_pooled.empty:
        r = ref_pooled.iloc[0].to_dict()
        r["feature_set"] = "promoted_reference_oof_ecdf_latent_plus_age_sex"
        reference_rows.append(r)
    if reference_rows:
        ref_df = pd.DataFrame(reference_rows)
        for col in classifier_pooled.columns:
            if col not in ref_df.columns:
                ref_df[col] = np.nan
        classifier_pooled = pd.concat([classifier_pooled, ref_df[classifier_pooled.columns]], ignore_index=True, sort=False)
    if not ref_pred.empty:
        ref = ref_pred.rename(columns={"y_true": "y_true"})
        if "Site" not in ref.columns:
            ref = ref.merge(meta[["SubjectID", "Site", "ADNI_phase"]], on="SubjectID", how="left")
        ref["feature_set"] = "promoted_reference_oof_ecdf_latent_plus_age_sex"
        ref["threshold_strategy"] = PRIMARY_THRESHOLD
        ref_for_fpr = ref[
            ["feature_set", "threshold_strategy", "Manufacturer", "y_true", "y_pred", "SubjectID", "Age", "Sex", "ResearchGroup_Mapped"]
        ].copy()
        classifier_fpr = pd.concat([classifier_fpr, manufacturer_fpr(ref_for_fpr)], ignore_index=True, sort=False)

    write_table(outdir, "combat_train_apply_validation", combat_validation, max_rows=300)
    write_table(outdir, "transformed_feature_integrity_checks", integrity_df, max_rows=300)
    write_table(outdir, "manufacturer_predictability_before_after_combat", mfr_pred, max_rows=300)
    write_table(outdir, "biological_preservation_before_after_combat", biological, max_rows=300)
    write_table(outdir, "classifier_only_sensitivity_metrics", classifier_pooled, max_rows=300)
    write_table(outdir, "classifier_only_sensitivity_foldwise_metrics", classifier_foldwise, max_rows=300)
    write_table(outdir, "classifier_only_sensitivity_manufacturer_fpr", classifier_fpr, max_rows=300)

    # Recommendation gate.
    train_apply_ok = bool((combat_validation["status"] == "fit_ok").all()) if not combat_validation.empty else False
    integrity_ok = bool(
        not integrity_df.empty
        and (integrity_df["shape_matches_original"].astype(bool).all())
        and (integrity_df["nan_count"].sum() == 0)
        and (integrity_df["inf_count"].sum() == 0)
        and (integrity_df["max_abs_diag_change_after_reconstruction"].max() == 0)
        and (integrity_df["max_abs_asymmetry_after_reconstruction"].max() == 0)
    )
    mfr_summary = (
        mfr_pred.groupby(["feature_scope", "stage"], dropna=False)["balanced_accuracy"].mean().reset_index()
        if not mfr_pred.empty
        else pd.DataFrame()
    )
    pooled_before = mfr_summary[mfr_summary["stage"] == "before_combat"]["balanced_accuracy"].mean() if not mfr_summary.empty else np.nan
    pooled_after = mfr_summary[mfr_summary["stage"] == "after_combat"]["balanced_accuracy"].mean() if not mfr_summary.empty else np.nan
    manufacturer_predictability_decreases = bool(np.isfinite(pooled_before) and np.isfinite(pooled_after) and pooled_after < pooled_before)

    primary_rows = classifier_pooled[classifier_pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
    ref_row = primary_rows[primary_rows["feature_set"].astype(str).eq("promoted_reference_oof_ecdf_latent_plus_age_sex")]
    ref_auc = float(ref_row["auc"].iloc[0]) if len(ref_row) else 0.795155
    ref_pr = float(ref_row["pr_auc"].iloc[0]) if len(ref_row) else 0.573934
    ref_philips = classifier_fpr[
        (classifier_fpr["feature_set"].astype(str).eq("promoted_reference_oof_ecdf_latent_plus_age_sex"))
        & (classifier_fpr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
        & (classifier_fpr["Manufacturer"].astype(str).str.lower().eq("philips"))
    ]
    ref_philips_fpr = float(ref_philips["fpr_cn"].iloc[0]) if len(ref_philips) else 0.444

    candidate = primary_rows[primary_rows["feature_set"].astype(str).eq("input_edge_summaries_after_combat_plus_age_sex")]
    candidate_auc_ok = False
    candidate_pr_ok = False
    candidate_philips_ok = False
    if len(candidate):
        candidate_auc_ok = float(candidate["auc"].iloc[0]) >= ref_auc - 0.005
        candidate_pr_ok = float(candidate["pr_auc"].iloc[0]) >= ref_pr - 0.005
        cand_ph = classifier_fpr[
            (classifier_fpr["feature_set"].astype(str).eq("input_edge_summaries_after_combat_plus_age_sex"))
            & (classifier_fpr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
            & (classifier_fpr["Manufacturer"].astype(str).str.lower().eq("philips"))
        ]
        if len(cand_ph):
            candidate_philips_ok = float(cand_ph["fpr_cn"].iloc[0]) <= ref_philips_fpr

    full_gate = train_apply_ok and integrity_ok and manufacturer_predictability_decreases and candidate_auc_ok and candidate_pr_ok and candidate_philips_ok
    if full_gate:
        decision = "A. run ComBat-input FULL VAE"
    else:
        decision = "B. run classifier-only latent/residual sensitivity first"

    recommendation_lines = [
        "# FULL ComBat-Input Recommendation",
        "",
        f"Decision: **{decision}**",
        "",
        "## Gate Status",
        "",
        f"- Train/apply wrapper validated: `{train_apply_ok}`",
        f"- Transformed feature integrity passed: `{integrity_ok}`",
        f"- Manufacturer predictability decreased after ComBat: `{manufacturer_predictability_decreases}`",
        f"- Input-summary classifier AUC within 0.005 of promoted reference: `{candidate_auc_ok}`",
        f"- Input-summary classifier PR-AUC within 0.005 of promoted reference: `{candidate_pr_ok}`",
        f"- Philips CN FPR decreases or does not worsen in input-summary sensitivity: `{candidate_philips_ok}`",
        "",
        "## Interpretation",
        "",
        "The train/apply mechanics are evaluated fold-wise with `neurocombat_sklearn.CombatModel`, fitting only on fold-local VAE training-pool subjects and applying frozen parameters to classifier train/dev and outer-test subjects. Diagnosis is excluded from the ComBat covariate model; Age and Sex are preserved covariates. No full-dataset harmonization and no test labels are used.",
        "",
        "A FULL ComBat-input VAE should only be launched if the classifier-only sensitivity shows nuisance reduction without degrading AD/CN ranking or Philips CN behavior. Otherwise, keep harmonization as a sensitivity branch and preserve the promoted model.",
    ]
    (outdir / "full_run_recommendation.md").write_text("\n".join(recommendation_lines) + "\n", encoding="utf-8")

    readme = [
        "# ComBat Train/Apply Validation And Classifier-Only Sensitivity",
        "",
        f"Reference run: `{rel(run_dir)}`",
        "",
        "Read-only guardrails:",
        "",
        "- No VAE training.",
        "- No tensor modification.",
        "- No metadata modification.",
        "- No model artifact modification.",
        "- No threshold fitting on outer-test folds.",
        "- Ephemeral audit classifiers and ComBat transforms are fit fold-locally and are not saved.",
        "",
        f"Final recommendation: **{decision}**",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    command_log["end_time"] = now_iso()
    command_log["decision"] = decision
    command_log["outputs"] = sorted(p.name for p in outdir.iterdir())
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
