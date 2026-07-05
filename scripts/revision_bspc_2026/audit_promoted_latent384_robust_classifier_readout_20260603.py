#!/usr/bin/env python3
"""Read-only robust classifier-only readout audit on promoted latent features.

The script uses saved fold-wise latent mu caches from the promoted VAE only.
It fits ephemeral classifier-only readouts inside the existing outer
train/dev folds, uses inner-CV OOF scores for hyperparameter and threshold
selection, and writes audit tables only.

No VAE training, tensor modification, metadata modification, model artifact
modification, or test-fold threshold fitting is performed.
"""

from __future__ import annotations

import argparse
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LATENT_CACHE = RUN_DIR / "classifier_only_readout/latent_cache"
PROMOTED_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OUT_DEFAULT = RESULTS / "promoted_latent384_robust_classifier_readout_audit_20260603"

FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = 5
SEED = 42
TARGET_SENSITIVITY = 0.70
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"

REF_AUC = 0.795155
REF_PR_AUC = 0.573934
REF_BA = 0.725979
REF_SENS = 0.731959
REF_SPEC = 0.720000
REF_F1 = 0.563492
REF_PHILIPS_FPR = 45 / 99


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
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


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows; full table is in CSV._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def logit(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def mu_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("mu_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def normalize_dx(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"CN", "CONTROL", "0"}:
        return "CN"
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA", "1"}:
        return "AD"
    return str(v)


def load_fold(run_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = run_dir / "classifier_only_readout/latent_cache"
    train = pd.read_csv(cache / f"fold_{fold}_trainDev_latent_mu.csv")
    test = pd.read_csv(cache / f"fold_{fold}_test_latent_mu.csv")
    for df in (train, test):
        df["SubjectID"] = df["SubjectID"].astype(str)
        df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
        df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
        df["Manufacturer"] = df["Manufacturer"].fillna("UNKNOWN").astype(str)
        df["Sex"] = df["Sex"].fillna("UNKNOWN").astype(str)
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    return train, test


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], y_pred: Iterable[int]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    p = np.asarray(list(y_pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
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
        "f1": float(f1_score(y, p, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate([[0.0, 0.5, 1.0], s]), 0, 1), 12))


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "youden_j": sens + spec - 1,
    }


def select_thresholds(y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for thr in threshold_candidates(scores):
        row = {"threshold": float(thr)}
        row.update(threshold_metrics(y_true, scores, float(thr)))
        rows.append(row)
    tbl = pd.DataFrame(rows)
    out = [{"threshold_strategy": "fixed_0p5", "threshold": 0.5}]
    for strategy, metric in [("inner_oof_balanced_accuracy", "balanced_accuracy"), ("inner_oof_youden_j", "youden_j")]:
        r = tbl.sort_values([metric, "sensitivity", "specificity", "threshold"], ascending=[False, False, False, False]).iloc[0]
        out.append({"threshold_strategy": strategy, "threshold": float(r["threshold"])})
    eligible = tbl[tbl["sensitivity"] >= TARGET_SENSITIVITY]
    if eligible.empty:
        r = tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        r = eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
    out.append({"threshold_strategy": PRIMARY_THRESHOLD, "threshold": float(r["threshold"])})
    return out


def calibrate_scores(method: str, y_oof: np.ndarray, oof_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if method == "raw":
        return oof_scores.copy(), test_scores.copy(), "raw estimator scores"
    if method == "oof_logitz":
        x_oof = logit(oof_scores)
        x_test = logit(test_scores)
        mu = float(np.mean(x_oof))
        sd = max(float(np.std(x_oof, ddof=1)), 1e-8)
        return sigmoid((x_oof - mu) / sd), sigmoid((x_test - mu) / sd), "train/dev OOF logit z-score"
    if method == "oof_ecdf":
        sorted_oof = np.sort(np.asarray(oof_scores, dtype=float))

        def ecdf(vals: np.ndarray) -> np.ndarray:
            return np.searchsorted(sorted_oof, vals, side="right") / max(len(sorted_oof), 1)

        ranks = pd.Series(oof_scores).rank(method="average").to_numpy()
        oof_ecdf = (ranks - 0.5) / len(oof_scores)
        return oof_ecdf.astype(float), ecdf(test_scores).astype(float), "train/dev OOF ECDF mapping"
    raise ValueError(method)


def preprocessing(mu_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("latent", Pipeline([("scaler", StandardScaler())]), mu_cols),
            ("age", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["Age"]),
            ("sex", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), ["Sex"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def strat_key(df: pd.DataFrame, n_splits: int) -> tuple[pd.Series, str]:
    key = df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)
    if key.value_counts().min() >= n_splits:
        return key, "ResearchGroup_Mapped+Manufacturer"
    return df["y"].astype(int), "label_only_fallback"


def sample_weights(df: pd.DataFrame, mode: str | None) -> np.ndarray | None:
    if mode is None:
        return None
    if mode == "diagnosis_x_manufacturer":
        key = df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)
    elif mode == "manufacturer":
        key = df["Manufacturer"].astype(str)
    else:
        raise ValueError(mode)
    counts = key.value_counts()
    w = key.map(lambda x: len(key) / (len(counts) * counts[x])).to_numpy(dtype=float)
    # Normalize mean to one for numeric stability.
    return w / np.mean(w)


def make_estimator(candidate_id: str, params: dict[str, Any], fold: int) -> Any:
    seed = SEED + fold
    if candidate_id.startswith("logreg_l2"):
        return Pipeline(
            [
                ("pre", preprocessing(params["mu_cols"])),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        C=float(params["C"]),
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if candidate_id == "logreg_elasticnet":
        return Pipeline(
            [
                ("pre", preprocessing(params["mu_cols"])),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=float(params["C"]),
                        l1_ratio=float(params["l1_ratio"]),
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if candidate_id == "linear_svm_dx_mfr_weight":
        return Pipeline(
            [
                ("pre", preprocessing(params["mu_cols"])),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        C=float(params["C"]),
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if candidate_id == "sgd_logistic_dx_mfr_weight":
        return Pipeline(
            [
                ("pre", preprocessing(params["mu_cols"])),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="elasticnet",
                        alpha=float(params["alpha"]),
                        l1_ratio=float(params["l1_ratio"]),
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(candidate_id)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    description: str
    param_grid: list[dict[str, Any]]
    sample_weight_mode: str | None = None
    mfr_direction_k: int | None = None


def candidate_specs(mu_cols: list[str]) -> list[Candidate]:
    c_grid = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    specs = [
        Candidate("logreg_l2_baseline", "Logistic L2 baseline", [{"C": c} for c in c_grid]),
        Candidate(
            "logreg_l2_dx_mfr_weight",
            "Logistic L2 with sample weights balanced by Diagnosis x Manufacturer",
            [{"C": c} for c in c_grid],
            sample_weight_mode="diagnosis_x_manufacturer",
        ),
        Candidate(
            "logreg_l2_mfr_weight",
            "Logistic L2 with sample weights balanced by Manufacturer",
            [{"C": c} for c in c_grid],
            sample_weight_mode="manufacturer",
        ),
        Candidate(
            "logreg_elasticnet",
            "Logistic elastic-net with nested tuning",
            [{"C": c, "l1_ratio": r} for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1] for r in [0.1, 0.3, 0.5, 0.7]],
        ),
        Candidate(
            "linear_svm_dx_mfr_weight",
            "Linear SVM with probability calibration and Diagnosis x Manufacturer weights",
            [{"C": c} for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]],
            sample_weight_mode="diagnosis_x_manufacturer",
        ),
        Candidate(
            "sgd_logistic_dx_mfr_weight",
            "SGD logistic with class/manufacturer weights",
            [{"alpha": a, "l1_ratio": r} for a in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3] for r in [0.0, 0.15, 0.5]],
            sample_weight_mode="diagnosis_x_manufacturer",
        ),
    ]
    for k in [1, 2, 3, 5, 10]:
        specs.append(
            Candidate(
                f"logreg_l2_remove_mfrdirs_k{k}",
                f"Logistic L2 after removing top {k} linear Manufacturer directions from z",
                [{"C": c} for c in c_grid],
                mfr_direction_k=k,
            )
        )
    return specs


def remove_manufacturer_directions(train: pd.DataFrame, test: pd.DataFrame, mu_cols: list[str], k: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tr = train.copy()
    te = test.copy()
    mfr = tr["Manufacturer"].astype(str)
    levels = sorted(mfr.unique().tolist())
    if len(levels) < 2:
        return tr, te, {"status": "skipped_single_manufacturer"}
    dummies = pd.get_dummies(mfr, drop_first=False).to_numpy(dtype=float)
    # Center columns so the SVD directions capture manufacturer-mean deviations.
    X = dummies - dummies.mean(axis=0, keepdims=True)
    Y = tr[mu_cols].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    _, _, vt = np.linalg.svd(coef, full_matrices=False)
    dirs = vt[: min(k, vt.shape[0])].T

    def project_out(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        Z = out[mu_cols].to_numpy(dtype=float)
        out.loc[:, mu_cols] = Z - (Z @ dirs) @ dirs.T
        return out

    return project_out(tr), project_out(te), {"status": "fit_ok", "k": k, "n_dirs_removed": int(dirs.shape[1])}


def tune_candidate(
    cand: Candidate,
    fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    inner_folds: int,
) -> dict[str, Any]:
    mu_cols = mu_columns(train_df)
    train_use = train_df.copy()
    test_use = test_df.copy()
    transform_meta: dict[str, Any] = {}
    if cand.mfr_direction_k is not None:
        train_use, test_use, transform_meta = remove_manufacturer_directions(train_use, test_use, mu_cols, cand.mfr_direction_k)

    y_train = train_use["y"].astype(int).to_numpy()
    y_test = test_use["y"].astype(int).to_numpy()
    key, cv_context = strat_key(train_use, inner_folds)
    splits = list(StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=SEED + 100 + fold).split(np.zeros(len(train_use)), key))
    x_cols = mu_cols + ["Age", "Sex"]
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    grid_rows: list[dict[str, Any]] = []
    for params0 in cand.param_grid:
        params = dict(params0)
        params["mu_cols"] = mu_cols
        aucs = []
        for tr_idx, va_idx in splits:
            est = make_estimator(cand.candidate_id, params, fold)
            sw = sample_weights(train_use.iloc[tr_idx], cand.sample_weight_mode)
            fit_kwargs = {"model__sample_weight": sw} if sw is not None else {}
            est.fit(train_use.iloc[tr_idx][x_cols], y_train[tr_idx], **fit_kwargs)
            score = score_estimator(est, train_use.iloc[va_idx][x_cols])
            aucs.append(roc_auc_score(y_train[va_idx], score))
        mean_auc = float(np.mean(aucs))
        row = {"fold": fold, "candidate_id": cand.candidate_id, "mean_inner_auc": mean_auc, **params0}
        grid_rows.append(row)
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params0
    if best_params is None:
        raise RuntimeError(f"No params selected for {cand.candidate_id}")

    params = dict(best_params)
    params["mu_cols"] = mu_cols
    oof = np.zeros(len(train_use), dtype=float)
    for tr_idx, va_idx in splits:
        est = make_estimator(cand.candidate_id, params, fold)
        sw = sample_weights(train_use.iloc[tr_idx], cand.sample_weight_mode)
        fit_kwargs = {"model__sample_weight": sw} if sw is not None else {}
        est.fit(train_use.iloc[tr_idx][x_cols], y_train[tr_idx], **fit_kwargs)
        oof[va_idx] = score_estimator(est, train_use.iloc[va_idx][x_cols])
    final = make_estimator(cand.candidate_id, params, fold)
    sw_full = sample_weights(train_use, cand.sample_weight_mode)
    fit_kwargs = {"model__sample_weight": sw_full} if sw_full is not None else {}
    final.fit(train_use[x_cols], y_train, **fit_kwargs)
    test_raw = score_estimator(final, test_use[x_cols])
    return {
        "train": train_use,
        "test": test_use,
        "oof_raw": oof,
        "test_raw": test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "best_params": best_params,
        "best_inner_auc": best_score,
        "cv_context": cv_context,
        "grid": pd.DataFrame(grid_rows),
        "transform_meta": transform_meta,
    }


def score_estimator(est: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(x)[:, 1], dtype=float)
    raw = np.asarray(est.decision_function(x), dtype=float).ravel()
    return sigmoid(raw)


def evaluate_all(run_dir: Path, inner_folds: int) -> dict[str, pd.DataFrame]:
    pred_parts: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    grid_parts: list[pd.DataFrame] = []
    for fold in FOLDS:
        print(f"Fold {fold}: robust readouts", flush=True)
        train, test = load_fold(run_dir, fold)
        mu_cols = mu_columns(train)
        for cand in candidate_specs(mu_cols):
            print(f"  {cand.candidate_id}", flush=True)
            res = tune_candidate(cand, fold, train, test, inner_folds)
            grid_parts.append(res["grid"])
            for calib in ["raw", "oof_logitz", "oof_ecdf"]:
                cal_oof, cal_test, cal_note = calibrate_scores(calib, res["y_train"], res["oof_raw"], res["test_raw"])
                for sel in select_thresholds(res["y_train"], cal_oof):
                    thr = float(sel["threshold"])
                    y_pred = (cal_test >= thr).astype(int)
                    row = {
                        "fold": fold,
                        "candidate_id": cand.candidate_id,
                        "description": cand.description,
                        "calib_method": calib,
                        "threshold_strategy": sel["threshold_strategy"],
                        "threshold": thr,
                        "score_calibration_context": cal_note,
                        "best_inner_auc": float(res["best_inner_auc"]),
                        "best_params": json.dumps(res["best_params"], sort_keys=True),
                        "sample_weight_mode": cand.sample_weight_mode or "none",
                        **res["transform_meta"],
                    }
                    row.update(binary_metrics(res["y_test"], cal_test, y_pred))
                    fold_metric_rows.append(row)
                    pred = res["test"][["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y"]].copy()
                    pred = pred.rename(columns={"y": "y_true"})
                    pred["candidate_id"] = cand.candidate_id
                    pred["description"] = cand.description
                    pred["calib_method"] = calib
                    pred["threshold_strategy"] = sel["threshold_strategy"]
                    pred["threshold"] = thr
                    pred["y_score_raw"] = res["test_raw"]
                    pred["y_score"] = cal_test
                    pred["y_pred"] = y_pred
                    pred_parts.append(pred)
    return {
        "predictions": pd.concat(pred_parts, ignore_index=True, sort=False),
        "foldwise": pd.DataFrame(fold_metric_rows),
        "inner_grid": pd.concat(grid_parts, ignore_index=True, sort=False) if grid_parts else pd.DataFrame(),
    }


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["candidate_id", "description", "calib_method", "threshold_strategy"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def manufacturer_fpr(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in pred.groupby(["candidate_id", "calib_method", "threshold_strategy", "Manufacturer"], dropna=False):
        cand, calib, strategy, mfr = keys
        cn = sub[sub["y_true"] == 0]
        ad = sub[sub["y_true"] == 1]
        rows.append(
            {
                "candidate_id": cand,
                "calib_method": calib,
                "threshold_strategy": strategy,
                "Manufacturer": mfr,
                "n_cn": int(len(cn)),
                "fp_cn": int((cn["y_pred"] == 1).sum()),
                "fpr_cn": safe_div((cn["y_pred"] == 1).sum(), len(cn)),
                "specificity_cn": safe_div((cn["y_pred"] == 0).sum(), len(cn)),
                "n_ad": int(len(ad)),
                "fn_ad": int((ad["y_pred"] == 0).sum()),
                "fnr_ad": safe_div((ad["y_pred"] == 0).sum(), len(ad)),
                "sensitivity_ad": safe_div((ad["y_pred"] == 1).sum(), len(ad)),
            }
        )
    return pd.DataFrame(rows)


def score_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["candidate_id", "calib_method", "threshold_strategy", "Manufacturer", "ResearchGroup_Mapped"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        cand, calib, strategy, mfr, dx = keys
        s = sub["y_score"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate_id": cand,
                "calib_method": calib,
                "threshold_strategy": strategy,
                "Manufacturer": mfr,
                "diagnosis": dx,
                "n": int(len(s)),
                "mean": float(np.mean(s)),
                "std": float(np.std(s, ddof=0)),
                "min": float(np.min(s)),
                "p25": float(np.percentile(s, 25)),
                "median": float(np.median(s)),
                "p75": float(np.percentile(s, 75)),
                "max": float(np.max(s)),
            }
        )
    return pd.DataFrame(rows)


def score_association(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = pred[pred["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
    for (cand, calib), sub in target.groupby(["candidate_id", "calib_method"], dropna=False):
        sub = sub.copy()
        sub["DiagnosisAD"] = sub["y_true"].astype(int)
        sub["Age"] = pd.to_numeric(sub["Age"], errors="coerce")
        if smf is not None and len(sub) >= 20:
            try:
                fit = smf.ols("y_score ~ C(Manufacturer) + Age + C(Sex) + DiagnosisAD", data=sub).fit()
                for term, coef in fit.params.items():
                    if "Manufacturer" in term:
                        rows.append(
                            {
                                "candidate_id": cand,
                                "calib_method": calib,
                                "model": "score ~ Manufacturer + Age + Sex + Diagnosis",
                                "term": term,
                                "coef": float(coef),
                                "p_value": float(fit.pvalues.get(term, np.nan)),
                                "r_squared": float(fit.rsquared),
                                "n": int(fit.nobs),
                            }
                        )
            except Exception as exc:
                rows.append({"candidate_id": cand, "calib_method": calib, "model": "ols_failed", "term": str(exc)})
        for mfr, g in sub.groupby("Manufacturer", dropna=False):
            rows.append(
                {
                    "candidate_id": cand,
                    "calib_method": calib,
                    "model": "descriptive_score_by_manufacturer",
                    "term": str(mfr),
                    "coef": float(g["y_score"].mean()),
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def promoted_reference_predictions() -> pd.DataFrame:
    path = PROMOTED_OOF / "calib_predictions.csv"
    pred = pd.read_csv(path)
    sub = pred[
        pred["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pred["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & pred["calib_method"].astype(str).isin(["raw", "oof_logitz", "oof_ecdf"])
    ].copy()
    sub["candidate_id"] = "promoted_reference_reused"
    sub["description"] = "Promoted reference readout reused from existing OOF score-harmonization artifact"
    return sub[
        [
            "SubjectID",
            "tensor_idx",
            "ResearchGroup_Mapped",
            "Manufacturer",
            "Age",
            "Sex",
            "fold",
            "candidate_id",
            "description",
            "calib_method",
            "threshold_strategy",
            "threshold",
            "y_true",
            "y_score_raw",
            "y_score",
            "y_pred",
        ]
    ].copy()


def add_gate_columns(metrics: pd.DataFrame, fpr: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    ph = fpr[
        fpr["Manufacturer"].astype(str).str.lower().eq("philips")
        & fpr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ][["candidate_id", "calib_method", "fpr_cn", "fp_cn", "n_cn"]].rename(
        columns={"fpr_cn": "philips_cn_fpr", "fp_cn": "philips_cn_fp", "n_cn": "philips_cn_n"}
    )
    out = out.merge(ph, on=["candidate_id", "calib_method"], how="left")
    assoc_terms = assoc[assoc["term"].astype(str).str.contains("Manufacturer", na=False)].copy()
    if not assoc_terms.empty:
        mfr_p = assoc_terms.groupby(["candidate_id", "calib_method"])["p_value"].min().rename("min_manufacturer_score_assoc_p").reset_index()
        mfr_abs = assoc_terms.groupby(["candidate_id", "calib_method"])["coef"].apply(lambda x: float(np.nanmax(np.abs(x)))).rename("max_abs_manufacturer_score_coef").reset_index()
        out = out.merge(mfr_p, on=["candidate_id", "calib_method"], how="left").merge(mfr_abs, on=["candidate_id", "calib_method"], how="left")
    ref_assoc = out[
        out["candidate_id"].eq("promoted_reference_reused")
        & out["calib_method"].eq(PRIMARY_CALIB)
        & out["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    ref_assoc_abs = float(ref_assoc["max_abs_manufacturer_score_coef"].iloc[0]) if len(ref_assoc) and "max_abs_manufacturer_score_coef" in out.columns else np.nan
    out["auc_gate"] = out["auc"] > REF_AUC
    out["pr_auc_gate"] = out["pr_auc"] >= REF_PR_AUC
    out["ba_gate"] = out["balanced_accuracy"] >= REF_BA - 0.005
    out["sens_gate"] = out["sensitivity"] >= REF_SENS - 0.005
    out["f1_gate"] = out["f1"] >= REF_F1 - 0.005
    out["philips_fpr_gate"] = out["philips_cn_fpr"] <= REF_PHILIPS_FPR
    if "max_abs_manufacturer_score_coef" in out.columns and np.isfinite(ref_assoc_abs):
        out["manufacturer_score_assoc_gate"] = out["max_abs_manufacturer_score_coef"] <= ref_assoc_abs + 1e-9
    else:
        out["manufacturer_score_assoc_gate"] = False
    out["all_promotion_gates"] = (
        out["auc_gate"]
        & out["pr_auc_gate"]
        & out["ba_gate"]
        & out["sens_gate"]
        & out["f1_gate"]
        & out["philips_fpr_gate"]
        & out["manufacturer_score_assoc_gate"]
    )
    return out


def write_decision(outdir: Path, metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
    primary = primary.sort_values(["all_promotion_gates", "auc", "pr_auc"], ascending=[False, False, False])
    keep = [
        "candidate_id",
        "calib_method",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "philips_cn_fpr",
        "max_abs_manufacturer_score_coef",
        "all_promotion_gates",
    ]
    lines = [
        "# Robust Classifier-Only Promotion Gate",
        "",
        "Decision: **do_not_promote_any_new_readout**",
        "",
        "Promotion required AUC > 0.795155, PR-AUC >= 0.573934, no material BA/F1/Sensitivity loss, Philips CN FPR <= 0.454545, and no worse Manufacturer score association.",
        "",
        "## Primary Threshold Rows",
        "",
        primary[[c for c in keep if c in primary.columns]].to_markdown(index=False),
        "",
        "No test-fold labels were used for score calibration or threshold selection; all thresholds are derived from train/dev inner-OOF scores.",
    ]
    if primary["all_promotion_gates"].fillna(False).any():
        lines[2] = "Decision: **promotion_gate_passed_for_candidate_sensitivity_review**"
    (outdir / "promotion_gate_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "test_fold_threshold_fitting": False,
            "ephemeral_classifier_only_audit_models": True,
            "saved_audit_models": False,
        },
        "commands": [],
    }
    pyc = run_cmd([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())])
    command_log["commands"].append(pyc)
    if pyc["returncode"] != 0:
        raise RuntimeError("py_compile failed")
    required = [run_dir / "classifier_only_readout/latent_cache" / f"fold_{f}_{s}_latent_mu.csv" for f in FOLDS for s in ["trainDev", "test"]]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing latent cache files:\n" + "\n".join(missing))
    if args.dry_run:
        (outdir / "README.md").write_text(
            "# Dry Run\n\n"
            f"Run: `{run_dir}`\n\n"
            f"Found all {len(required)} latent-cache files. No classifiers run.\n",
            encoding="utf-8",
        )
        command_log["dry_run"] = True
        command_log["end_time"] = now_iso()
        (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return 0

    result = evaluate_all(run_dir, args.inner_folds)
    pred = result["predictions"]
    ref_pred = promoted_reference_predictions()
    pred_all = pd.concat([pred, ref_pred], ignore_index=True, sort=False)
    foldwise = result["foldwise"]
    metrics = pooled_metrics(pred_all)
    fpr = manufacturer_fpr(pred_all)
    score_dist = score_distribution(pred_all)
    assoc = score_association(pred_all)
    metrics_gate = add_gate_columns(metrics, fpr, assoc)

    write_table(outdir, "candidate_readout_metrics", metrics_gate, max_rows=300)
    write_table(outdir, "foldwise_metrics", foldwise, max_rows=300)
    write_table(outdir, "manufacturer_fpr", fpr, max_rows=300)
    write_table(outdir, "score_distribution_by_manufacturer", score_dist, max_rows=300)
    write_table(outdir, "score_manufacturer_association", assoc, max_rows=300)
    # Extra useful audit tables are CSV-only/MD-free requirements are already satisfied.
    result["inner_grid"].to_csv(outdir / "inner_cv_grid_scores.csv", index=False)
    pred_all.to_csv(outdir / "candidate_readout_predictions.csv", index=False)
    write_decision(outdir, metrics_gate)

    readme = [
        "# Promoted Latent384 Robust Classifier Readout Audit",
        "",
        f"Reference run: `{run_dir}`",
        "",
        "All candidate readouts use existing fold-wise latent mu + Age + Sex. Hyperparameters, score harmonization, and thresholds are fit from train/dev inner-CV/OOF predictions only.",
        "",
        "No VAE training, tensor modification, metadata modification, model artifact modification, or test-fold threshold fitting was performed.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log["end_time"] = now_iso()
    command_log["outputs"] = sorted(p.name for p in outdir.iterdir())
    command_log["decision"] = "do_not_promote_any_new_readout"
    if metrics_gate["all_promotion_gates"].fillna(False).any():
        command_log["decision"] = "promotion_gate_passed_for_candidate_sensitivity_review"
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
