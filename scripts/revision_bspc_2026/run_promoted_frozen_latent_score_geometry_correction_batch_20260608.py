#!/usr/bin/env python3
"""Frozen-latent score-geometry correction batch for the promoted model.

This audit uses only existing fold-wise latent caches and OOF predictions. It
does not train VAEs, touch tensors or metadata, score OASIS, or overwrite model
artifacts. All correction transforms, hyperparameters, and thresholds are
selected inside the train/dev side of each outer fold.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
PROMOTED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_CACHE = PROMOTED_RUN / "classifier_only_readout/latent_cache"
PROMOTED_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
CH1_OOF = RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration"
RESIDUALIZED_OOF = RESULTS / "promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602"
FOLDCOMBAT_OOF = RESULTS / "recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration"
OUT_DEFAULT = RESULTS / "promoted_frozen_latent_score_geometry_correction_batch_20260608"

FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = 5
SEED = 42
TARGET_SENS = 0.70
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

REF_AUC = 0.795155
REF_PR_AUC = 0.573934
REF_BA = 0.725979
REF_SENS = 0.731959
REF_F1 = 0.563492
REF_PHILIPS_CN_FPR = 0.4545


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    description: str
    family: str
    param_grid: list[dict[str, Any]]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_cmd(cmd: Sequence[str]) -> dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": list(cmd),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


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


def normalize_dx(v: Any) -> str:
    s = str(v).strip().upper()
    if s in {"CN", "CONTROL", "0"}:
        return "CN"
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA", "1"}:
        return "AD"
    return str(v)


def mu_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("mu_")]
    if not cols:
        raise ValueError("No mu_* columns found in latent cache")
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def load_fold(fold: int, split: str) -> pd.DataFrame:
    path = PROMOTED_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
    df = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    df["Manufacturer"] = df["Manufacturer"].fillna("UNKNOWN").astype(str)
    df["Sex"] = df["Sex"].fillna("UNKNOWN").astype(str)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if df["Age"].isna().any():
        bad = sorted(df.loc[df["Age"].isna(), "SubjectID"].astype(str).tolist())
        raise ValueError(f"Missing Age in {path}: {bad}")
    return df.reset_index(drop=True)


def primary_oof_rows(oof_dir: Path, *, harmonization_method: str | None = None) -> pd.DataFrame:
    pred_path = oof_dir / "calib_predictions.csv"
    if not pred_path.exists() and (oof_dir / "harmonized_stageb_predictions.csv").exists():
        pred_path = oof_dir / "harmonized_stageb_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    df = pd.read_csv(pred_path)
    mask = (
        df.get("model_name", "").astype(str).eq(PRIMARY_MODEL)
        & df.get("feature_set", "").astype(str).eq(PRIMARY_FEATURES)
        & df.get("calib_method", "").astype(str).eq(PRIMARY_CALIB)
        & df.get("threshold_strategy", "").astype(str).eq(PRIMARY_THRESHOLD)
    )
    if harmonization_method is not None:
        mask &= df.get("harmonization_method", "").astype(str).eq(harmonization_method)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No primary OOF rows found in {pred_path}")
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].map(normalize_dx)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise").astype(int)
    out["y_score"] = pd.to_numeric(out["y_score"], errors="raise")
    out["y_score_raw"] = pd.to_numeric(out["y_score_raw"], errors="coerce")
    return out.reset_index(drop=True)


def load_base_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    promoted = primary_oof_rows(PROMOTED_OOF)
    ch1 = primary_oof_rows(CH1_OOF)
    keep = ["SubjectID", "y_score", "y_score_raw", "y_true", "fold"]
    promoted = promoted[keep].rename(
        columns={"y_score": "promoted_oof_ecdf_score", "y_score_raw": "promoted_oof_raw_score"}
    )
    ch1 = ch1[keep].rename(columns={"y_score": "ch1_oof_ecdf_score", "y_score_raw": "ch1_oof_raw_score"})
    return promoted, ch1


def add_base_scores(df: pd.DataFrame, promoted: pd.DataFrame, ch1: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(promoted[["SubjectID", "promoted_oof_ecdf_score", "promoted_oof_raw_score"]], on="SubjectID", how="left")
    out = out.merge(ch1[["SubjectID", "ch1_oof_ecdf_score", "ch1_oof_raw_score"]], on="SubjectID", how="left")
    missing = out[
        out[["promoted_oof_ecdf_score", "ch1_oof_ecdf_score"]].isna().any(axis=1)
    ]["SubjectID"].tolist()
    if missing:
        raise ValueError(f"Missing base OOF scores for subjects: {missing[:20]}")
    return out


def strat_key(df: pd.DataFrame, n_splits: int) -> pd.Series:
    key = df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)
    if key.value_counts().min() >= n_splits:
        return key
    return df["y"].astype(int).astype(str)


def inner_splits(df: pd.DataFrame, n_splits: int, fold: int) -> list[tuple[np.ndarray, np.ndarray]]:
    key = strat_key(df, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + 1000 + fold)
    return list(cv.split(np.zeros(len(df)), key))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50, 50)))


def score_logit(scores: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def ecdf_transform(reference_scores: np.ndarray, target_scores: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference_scores, dtype=float))
    return np.searchsorted(ref, np.asarray(target_scores, dtype=float), side="right") / max(len(ref), 1)


def calibrate_scores(method: str, oof_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    oof = np.asarray(oof_scores, dtype=float)
    test = np.asarray(test_scores, dtype=float)
    if method == "raw":
        return oof.copy(), test.copy(), "raw frozen-feature score"
    if method == "oof_logitz":
        loof = score_logit(oof)
        ltest = score_logit(test)
        mu = float(np.mean(loof))
        sd = max(float(np.std(loof, ddof=1)), 1e-8)
        return sigmoid((loof - mu) / sd), sigmoid((ltest - mu) / sd), "inner OOF logit-z calibration"
    if method == "oof_ecdf":
        ranks = pd.Series(oof).rank(method="average").to_numpy(dtype=float)
        cal_oof = (ranks - 0.5) / len(oof)
        return cal_oof, ecdf_transform(oof, test), "inner OOF ECDF calibration"
    raise ValueError(method)


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], y_pred: Iterable[int]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    p = np.asarray(list(y_pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    row = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, p, zero_division=0)),
        "predicted_ad_rate": float(np.mean(p)),
    }
    return row


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.r_[0.0, 0.5, 1.0, s], 12))


def threshold_at_target_sens(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    rows = []
    for thr in threshold_grid(scores):
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        rows.append(
            {
                "threshold": float(thr),
                "inner_sensitivity": sens,
                "inner_specificity": spec,
                "inner_balanced_accuracy": 0.5 * (sens + spec),
            }
        )
    tbl = pd.DataFrame(rows)
    eligible = tbl[tbl["inner_sensitivity"] >= TARGET_SENS]
    if eligible.empty:
        row = tbl.sort_values(["inner_sensitivity", "inner_specificity", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        row = eligible.sort_values(
            ["inner_specificity", "inner_sensitivity", "inner_balanced_accuracy", "threshold"],
            ascending=[False, False, False, False],
        ).iloc[0]
    return row.to_dict()


def metadata_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["Sex_M"] = df["Sex"].astype(str).str.upper().str.startswith("M").astype(float)
    return out


def feature_frame(df: pd.DataFrame, feature_kind: str) -> pd.DataFrame:
    mu_cols = mu_columns(df)
    if feature_kind == "z_only":
        return df[mu_cols].copy()
    if feature_kind == "z_plus_age_sex":
        meta = metadata_numeric(df)
        return pd.concat([df[mu_cols].reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    if feature_kind == "stack_scores":
        meta = metadata_numeric(df)
        score_cols = df[["promoted_oof_ecdf_score", "ch1_oof_ecdf_score"]].copy()
        return pd.concat([score_cols.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    raise ValueError(feature_kind)


def fit_logreg(train: pd.DataFrame, y_train: np.ndarray, params: dict[str, Any], sample_weight: np.ndarray | None = None) -> Pipeline:
    x = feature_frame(train, params["feature_kind"])
    model = LogisticRegression(
        C=float(params["C"]),
        penalty="l2",
        solver="liblinear",
        max_iter=10000,
        class_weight=None if sample_weight is not None else "balanced",
        random_state=SEED,
    )
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    pipe.fit(x, y_train, **kwargs)
    return pipe


def logreg_scores(model: Pipeline, df: pd.DataFrame, feature_kind: str) -> np.ndarray:
    return np.asarray(model.predict_proba(feature_frame(df, feature_kind))[:, 1], dtype=float)


def dx_mfr_weights(df: pd.DataFrame, philips_cn_multiplier: float, ge_ad_multiplier: float) -> np.ndarray:
    key = df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)
    counts = key.value_counts()
    weights = key.map(lambda x: len(key) / (len(counts) * counts[x])).to_numpy(dtype=float)
    is_philips_cn = (df["ResearchGroup_Mapped"].eq("CN") & df["Manufacturer"].eq("Philips")).to_numpy()
    is_ge_ad = (df["ResearchGroup_Mapped"].eq("AD") & df["Manufacturer"].eq("GE")).to_numpy()
    weights[is_philips_cn] *= float(philips_cn_multiplier)
    weights[is_ge_ad] *= float(ge_ad_multiplier)
    return weights / np.mean(weights)


def manufacturer_direction_basis(train: pd.DataFrame, k: int) -> tuple[np.ndarray, dict[str, Any]]:
    mu_cols = mu_columns(train)
    levels = sorted(train["Manufacturer"].astype(str).unique().tolist())
    if k <= 0 or len(levels) < 2:
        return np.zeros((len(mu_cols), 0)), {"status": "no_removal", "k": int(k), "n_dirs": 0, "levels": levels}
    X = pd.get_dummies(train["Manufacturer"].astype(str), drop_first=False).reindex(columns=levels).to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    Y = train[mu_cols].to_numpy(dtype=float)
    Y = Y - Y.mean(axis=0, keepdims=True)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    _, _, vt = np.linalg.svd(coef, full_matrices=False)
    dirs = vt[: min(k, vt.shape[0])].T
    return dirs, {"status": "fit_ok", "k": int(k), "n_dirs": int(dirs.shape[1]), "levels": levels}


def apply_direction_removal(df: pd.DataFrame, dirs: np.ndarray) -> pd.DataFrame:
    if dirs.size == 0:
        return df.copy()
    out = df.copy()
    mu_cols = mu_columns(out)
    z = out[mu_cols].to_numpy(dtype=float)
    out.loc[:, mu_cols] = z - (z @ dirs) @ dirs.T
    return out


def score_shift_map(train: pd.DataFrame, alpha: float) -> dict[str, Any]:
    cn = train[train["ResearchGroup_Mapped"].eq("CN")]
    global_cn_median = float(cn["promoted_oof_ecdf_score"].median())
    shifts: dict[str, float] = {}
    counts: dict[str, int] = {}
    for mfr, sub in cn.groupby("Manufacturer"):
        counts[str(mfr)] = int(len(sub))
        shifts[str(mfr)] = float(sub["promoted_oof_ecdf_score"].median() - global_cn_median)
    return {"alpha": float(alpha), "global_cn_median": global_cn_median, "shifts": shifts, "counts": counts}


def apply_score_shift(df: pd.DataFrame, fit: dict[str, Any]) -> np.ndarray:
    scores = df["promoted_oof_ecdf_score"].to_numpy(dtype=float).copy()
    alpha = float(fit["alpha"])
    shifts = fit["shifts"]
    for i, mfr in enumerate(df["Manufacturer"].astype(str)):
        scores[i] -= alpha * float(shifts.get(mfr, 0.0))
    return np.clip(scores, 0.0, 1.0)


def candidate_specs() -> list[Candidate]:
    c_grid = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    return [
        Candidate(
            "promoted_z_only_logreg_l2",
            "Promoted latent mu only; nested logreg_l2; no Age/Sex.",
            "logreg",
            [{"feature_kind": "z_only", "C": c} for c in c_grid],
        ),
        Candidate(
            "promoted_z_plus_age_sex_logreg_l2_reweighted_diag_mfr",
            "Promoted latent mu + Age/Sex with Diagnosis x Manufacturer cell weights and Philips-CN/GE-AD multipliers.",
            "reweighted_logreg",
            [
                {
                    "feature_kind": "z_plus_age_sex",
                    "C": c,
                    "philips_cn_multiplier": p,
                    "ge_ad_multiplier": g,
                }
                for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
                for p in [1.0, 1.5, 2.0, 3.0]
                for g in [1.0, 1.5, 2.0, 3.0]
            ],
        ),
        Candidate(
            "promoted_score_mfr_cn_median_centering",
            "Manufacturer-specific CN median score centering, shrinkage alpha tuned inside inner CV.",
            "score_shift",
            [{"alpha": a} for a in [0.0, 0.25, 0.5, 0.75, 1.0]],
        ),
        Candidate(
            "promoted_latent_mfr_direction_removal",
            "Remove manufacturer-predictive latent directions, then fit logreg_l2 with Age/Sex.",
            "mfr_direction_removal",
            [{"feature_kind": "z_plus_age_sex", "C": c, "k": k} for k in [0, 1, 2, 3, 5] for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]],
        ),
        Candidate(
            "promoted_plus_ch1_constrained_score_readout",
            "Strongly regularized meta-logreg on promoted score, ch1-only score, Age, and Sex.",
            "logreg",
            [{"feature_kind": "stack_scores", "C": c} for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]],
        ),
    ]


def oof_for_params(train_df: pd.DataFrame, cand: Candidate, params: dict[str, Any], splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = train_df["y"].to_numpy(dtype=int)
    oof = np.full(len(train_df), np.nan, dtype=float)
    details: list[dict[str, Any]] = []
    for split_id, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)
        if cand.family == "score_shift":
            fit = score_shift_map(tr, float(params["alpha"]))
            oof[va_idx] = apply_score_shift(va, fit)
            details.append({"inner_split": split_id, "score_shift_fit": fit})
            continue
        if cand.family == "mfr_direction_removal":
            dirs, meta = manufacturer_direction_basis(tr, int(params["k"]))
            tr_fit = apply_direction_removal(tr, dirs)
            va_fit = apply_direction_removal(va, dirs)
            model = fit_logreg(tr_fit, tr_fit["y"].to_numpy(dtype=int), params)
            oof[va_idx] = logreg_scores(model, va_fit, params["feature_kind"])
            details.append({"inner_split": split_id, "direction_fit": meta})
            continue
        sample_weight = None
        if cand.family == "reweighted_logreg":
            sample_weight = dx_mfr_weights(
                tr,
                float(params["philips_cn_multiplier"]),
                float(params["ge_ad_multiplier"]),
            )
        model = fit_logreg(tr, y[tr_idx], params, sample_weight=sample_weight)
        oof[va_idx] = logreg_scores(model, va, params["feature_kind"])
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF score generation failed for {cand.candidate_id} {params}")
    return oof, details


def final_test_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, cand: Candidate, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    y = train_df["y"].to_numpy(dtype=int)
    if cand.family == "score_shift":
        fit = score_shift_map(train_df, float(params["alpha"]))
        return apply_score_shift(test_df, fit), {"score_shift_fit": fit}
    if cand.family == "mfr_direction_removal":
        dirs, meta = manufacturer_direction_basis(train_df, int(params["k"]))
        tr_fit = apply_direction_removal(train_df, dirs)
        te_fit = apply_direction_removal(test_df, dirs)
        model = fit_logreg(tr_fit, tr_fit["y"].to_numpy(dtype=int), params)
        return logreg_scores(model, te_fit, params["feature_kind"]), {"direction_fit": meta}
    sample_weight = None
    if cand.family == "reweighted_logreg":
        sample_weight = dx_mfr_weights(
            train_df,
            float(params["philips_cn_multiplier"]),
            float(params["ge_ad_multiplier"]),
        )
    model = fit_logreg(train_df, y, params, sample_weight=sample_weight)
    return logreg_scores(model, test_df, params["feature_kind"]), {}


def choose_params(train_df: pd.DataFrame, cand: Candidate, fold: int) -> dict[str, Any]:
    splits = inner_splits(train_df, INNER_FOLDS, fold)
    y = train_df["y"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for params in cand.param_grid:
        oof, _ = oof_for_params(train_df, cand, params, splits)
        row = {
            "fold": fold,
            "candidate_id": cand.candidate_id,
            "params_json": json.dumps(params, sort_keys=True),
            "inner_oof_auc": float(roc_auc_score(y, oof)),
            "inner_oof_pr_auc": float(average_precision_score(y, oof)),
        }
        rows.append(row)
        if best is None or (row["inner_oof_auc"], row["inner_oof_pr_auc"]) > (
            best["inner_oof_auc"],
            best["inner_oof_pr_auc"],
        ):
            best = {"params": params, "inner_oof_scores": oof, **row}
    assert best is not None
    return {"selected": best, "grid": pd.DataFrame(rows), "splits": splits}


def evaluate_candidates() -> dict[str, pd.DataFrame]:
    promoted_scores, ch1_scores = load_base_scores()
    pred_parts: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    grid_parts: list[pd.DataFrame] = []
    shift_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        train = add_base_scores(load_fold(fold, "trainDev"), promoted_scores, ch1_scores)
        test = add_base_scores(load_fold(fold, "test"), promoted_scores, ch1_scores)
        for cand in candidate_specs():
            print(f"fold {fold}: {cand.candidate_id}", flush=True)
            selected = choose_params(train, cand, fold)
            grid_parts.append(selected["grid"])
            params = selected["selected"]["params"]
            # Recompute selected inner OOF so transform summaries match selected params.
            inner_oof, inner_details = oof_for_params(train, cand, params, selected["splits"])
            test_raw, final_meta = final_test_scores(train, test, cand, params)
            if cand.family == "score_shift":
                fit = final_meta["score_shift_fit"]
                for mfr, shift in fit["shifts"].items():
                    shift_rows.append(
                        {
                            "fold": fold,
                            "candidate_id": cand.candidate_id,
                            "selected_alpha": fit["alpha"],
                            "Manufacturer": mfr,
                            "cn_count_train_dev": fit["counts"].get(mfr, 0),
                            "global_cn_median_train_dev": fit["global_cn_median"],
                            "mfr_cn_shift": shift,
                            "applied_shift": fit["alpha"] * shift,
                        }
                    )
            if cand.family == "mfr_direction_removal":
                direction_meta = final_meta["direction_fit"]
                shift_rows.append(
                    {
                        "fold": fold,
                        "candidate_id": cand.candidate_id,
                        "selected_k": direction_meta.get("k"),
                        "n_dirs_removed": direction_meta.get("n_dirs"),
                        "manufacturer_levels": json.dumps(direction_meta.get("levels", [])),
                    }
                )
            for calib in ["raw", "oof_logitz", "oof_ecdf"]:
                cal_oof, cal_test, calib_note = calibrate_scores(calib, inner_oof, test_raw)
                thr_info = threshold_at_target_sens(train["y"].to_numpy(dtype=int), cal_oof)
                thr = float(thr_info["threshold"])
                y_pred = (cal_test >= thr).astype(int)
                fold_row = {
                    "fold": fold,
                    "candidate_id": cand.candidate_id,
                    "description": cand.description,
                    "calib_method": calib,
                    "threshold_strategy": PRIMARY_THRESHOLD,
                    "threshold": thr,
                    "calibration_context": calib_note,
                    "selected_params_json": json.dumps(params, sort_keys=True),
                    "inner_oof_auc": float(selected["selected"]["inner_oof_auc"]),
                    "inner_oof_pr_auc": float(selected["selected"]["inner_oof_pr_auc"]),
                    **{k: v for k, v in thr_info.items() if k != "threshold"},
                }
                fold_row.update(binary_metrics(test["y"], cal_test, y_pred))
                fold_metric_rows.append(fold_row)
                pred = test[
                    [
                        "SubjectID",
                        "tensor_idx",
                        "ResearchGroup_Mapped",
                        "Manufacturer",
                        "Age",
                        "Sex",
                        "fold",
                        "y",
                    ]
                ].copy()
                pred = pred.rename(columns={"y": "y_true"})
                pred["candidate_id"] = cand.candidate_id
                pred["description"] = cand.description
                pred["calib_method"] = calib
                pred["threshold_strategy"] = PRIMARY_THRESHOLD
                pred["threshold"] = thr
                pred["y_score_raw"] = test_raw
                pred["y_score"] = cal_test
                pred["y_pred"] = y_pred
                pred_parts.append(pred)
            hyper_rows.append(
                {
                    "fold": fold,
                    "candidate_id": cand.candidate_id,
                    "description": cand.description,
                    "family": cand.family,
                    "selected_params_json": json.dumps(params, sort_keys=True),
                    "inner_oof_auc": float(selected["selected"]["inner_oof_auc"]),
                    "inner_oof_pr_auc": float(selected["selected"]["inner_oof_pr_auc"]),
                    "n_grid_rows": int(len(selected["grid"])),
                    "n_inner_details": int(len(inner_details)),
                }
            )
    return {
        "predictions": pd.concat(pred_parts, ignore_index=True, sort=False),
        "foldwise_metrics": pd.DataFrame(fold_metric_rows),
        "selected_hyperparameters": pd.DataFrame(hyper_rows),
        "inner_grid": pd.concat(grid_parts, ignore_index=True, sort=False),
        "score_shift_correction_summary": pd.DataFrame(shift_rows),
    }


def manufacturer_eta2(df: pd.DataFrame, score_col: str = "y_score") -> dict[str, float]:
    scores = df[score_col].to_numpy(dtype=float)
    grand = float(np.mean(scores))
    total = float(np.sum((scores - grand) ** 2))
    between = 0.0
    for _, sub in df.groupby("Manufacturer"):
        vals = sub[score_col].to_numpy(dtype=float)
        between += len(vals) * float((np.mean(vals) - grand) ** 2)
    means = df.groupby("Manufacturer")[score_col].mean()
    return {
        "score_manufacturer_eta2": safe_div(between, total),
        "score_manufacturer_max_mean_diff": float(means.max() - means.min()) if len(means) else float("nan"),
    }


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["candidate_id", "description", "calib_method", "threshold_strategy"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        row.update(manufacturer_eta2(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def manufacturer_errors(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["candidate_id", "calib_method", "threshold_strategy", "Manufacturer"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
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
                "cn_fp": int((cn["y_pred"] == 1).sum()),
                "cn_fpr": safe_div((cn["y_pred"] == 1).sum(), len(cn)),
                "n_ad": int(len(ad)),
                "ad_fn": int((ad["y_pred"] == 0).sum()),
                "ad_fnr": safe_div((ad["y_pred"] == 0).sum(), len(ad)),
            }
        )
    return pd.DataFrame(rows)


def primary_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["calib_method"].eq(PRIMARY_CALIB)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()


def reference_metric_row(label: str, pred: pd.DataFrame) -> dict[str, Any]:
    row = {
        "candidate_id": label,
        "description": "reference",
        "calib_method": PRIMARY_CALIB,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    row.update(binary_metrics(pred["y_true"], pred["y_score"], pred["y_pred"]))
    row.update(manufacturer_eta2(pred))
    return row


def load_reference_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    refs = []
    ref_preds = []
    sources = [
        ("reference_promoted_ch102_latent384_beta3p75", primary_oof_rows(PROMOTED_OOF), "promoted [1,0,2] primary OOF-ECDF"),
        ("reference_ch1only_latent384_beta3p75", primary_oof_rows(CH1_OOF), "ch1-only primary OOF-ECDF"),
        (
            "reference_residualized_manufacturer_stageB",
            primary_oof_rows(RESIDUALIZED_OOF, harmonization_method="residualize_mfr_preserve_age_sex"),
            "residualized Manufacturer Stage B sensitivity",
        ),
        ("reference_foldcombat_mfr_age_sex", primary_oof_rows(FOLDCOMBAT_OOF), "foldwise ComBat input sensitivity"),
    ]
    for label, df, desc in sources:
        pred = df[
            [
                "SubjectID",
                "tensor_idx",
                "ResearchGroup_Mapped",
                "Manufacturer",
                "Age",
                "Sex",
                "fold",
                "y_true",
                "y_score",
                "y_pred",
            ]
        ].copy()
        pred["candidate_id"] = label
        pred["description"] = desc
        pred["calib_method"] = PRIMARY_CALIB
        pred["threshold_strategy"] = PRIMARY_THRESHOLD
        refs.append(reference_metric_row(label, pred).update({"description": desc}))
        row = reference_metric_row(label, pred)
        row["description"] = desc
        refs[-1] = row
        ref_preds.append(pred)
    return pd.DataFrame(refs), pd.concat(ref_preds, ignore_index=True)


def add_gate_columns(metrics: pd.DataFrame, manufacturer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mfr_primary = manufacturer[
        manufacturer["calib_method"].eq(PRIMARY_CALIB)
        & manufacturer["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, r in mfr_primary.iterrows():
        lookup[(str(r["candidate_id"]), str(r["Manufacturer"]))] = r.to_dict()
    for _, r in metrics.iterrows():
        cand = str(r["candidate_id"])
        philips = lookup.get((cand, "Philips"), {})
        ge = lookup.get((cand, "GE"), {})
        out = r.to_dict()
        out["philips_cn_fp"] = philips.get("cn_fp", np.nan)
        out["philips_cn_n"] = philips.get("n_cn", np.nan)
        out["philips_cn_fpr"] = philips.get("cn_fpr", np.nan)
        out["ge_ad_fn"] = ge.get("ad_fn", np.nan)
        out["ge_ad_n"] = ge.get("n_ad", np.nan)
        out["ge_ad_fnr"] = ge.get("ad_fnr", np.nan)
        is_candidate = not cand.startswith("reference_")
        out["gate_auc"] = bool((r["auc"] > REF_AUC) or (r["auc"] >= REF_AUC - 0.005 and out["philips_cn_fpr"] < REF_PHILIPS_CN_FPR))
        out["gate_pr_auc"] = bool(r["pr_auc"] >= REF_PR_AUC)
        out["gate_ba_f1"] = bool((r["balanced_accuracy"] >= REF_BA - 1e-12) and (r["f1"] >= REF_F1 - 1e-12))
        out["gate_philips_cn_fpr"] = bool(out["philips_cn_fpr"] < REF_PHILIPS_CN_FPR)
        out["gate_ge_ad_fnr"] = bool(out["ge_ad_fnr"] <= lookup.get(("reference_promoted_ch102_latent384_beta3p75", "GE"), {}).get("ad_fnr", np.inf))
        out["gate_no_oasis_use"] = True
        promotes = is_candidate and out["gate_auc"] and out["gate_pr_auc"] and out["gate_ba_f1"] and out["gate_philips_cn_fpr"] and out["gate_ge_ad_fnr"]
        if promotes:
            decision = "promote_correction_candidate"
        elif is_candidate and out["philips_cn_fpr"] < REF_PHILIPS_CN_FPR and r["pr_auc"] >= REF_PR_AUC:
            decision = "sensitivity_only_improves_philips_fpr"
        elif is_candidate:
            decision = "do_not_promote"
        else:
            decision = "reference"
        out["decision"] = decision
        rows.append(out)
    return pd.DataFrame(rows)


def subject_error_table(candidate_pred: pd.DataFrame, ref_pred: pd.DataFrame) -> pd.DataFrame:
    promoted = ref_pred[ref_pred["candidate_id"].eq("reference_promoted_ch102_latent384_beta3p75")].copy()
    promoted = promoted.rename(columns={"y_score": "promoted_score", "y_pred": "promoted_pred"})
    base_cols = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "fold",
        "y_true",
        "promoted_score",
        "promoted_pred",
    ]
    promoted = promoted[base_cols]
    parts: list[pd.DataFrame] = []
    for cand, sub in primary_subset(candidate_pred).groupby("candidate_id"):
        merged = promoted.merge(
            sub[["SubjectID", "y_score", "y_pred"]].rename(columns={"y_score": "candidate_score", "y_pred": "candidate_pred"}),
            on="SubjectID",
            how="left",
        )
        merged["candidate_id"] = cand
        merged["promoted_error"] = merged["promoted_pred"].ne(merged["y_true"])
        merged["candidate_error"] = merged["candidate_pred"].ne(merged["y_true"])
        merged["error_change"] = np.select(
            [
                merged["promoted_error"] & ~merged["candidate_error"],
                ~merged["promoted_error"] & merged["candidate_error"],
                merged["promoted_error"] & merged["candidate_error"],
            ],
            ["fixed_by_candidate", "introduced_by_candidate", "both_wrong"],
            default="both_correct",
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True, sort=False)


def write_final_decision(outdir: Path, metrics: pd.DataFrame) -> None:
    primary = metrics[
        metrics["calib_method"].eq(PRIMARY_CALIB)
        & metrics["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    candidate_primary = primary[~primary["candidate_id"].astype(str).str.startswith("reference_")]
    promoted = candidate_primary[candidate_primary["decision"].eq("promote_correction_candidate")]
    lines = [
        "# Final Decision",
        "",
        "Scope: frozen-latent, inner-CV-only score-geometry correction batch for the promoted [1,0,2] latent384 beta3.75 model.",
        "",
        "Guardrails held: no VAE training, no tensor or metadata modification, no model artifact overwrite, no OASIS scoring, and no OASIS threshold/calibration fitting.",
        "",
    ]
    if promoted.empty:
        lines.append("Decision: **no frozen-latent correction candidate is promoted**.")
    else:
        lines.append("Decision: **at least one frozen-latent correction candidate passed the predefined correction gate**.")
    lines.extend(
        [
            "",
            "Promotion/correction gates:",
            f"- AUC > {REF_AUC:.6f}, or no material AUC loss with clearly better Philips CN FPR.",
            f"- PR-AUC >= {REF_PR_AUC:.6f}.",
            f"- BA >= {REF_BA:.6f} and F1 >= {REF_F1:.6f}.",
            f"- Philips CN FPR < {REF_PHILIPS_CN_FPR:.4f}.",
            "- GE AD FNR not worse than promoted reference.",
            "- No OASIS use.",
            "",
            "Primary OOF-ECDF candidate results:",
        ]
    )
    for _, r in candidate_primary.sort_values(["decision", "auc", "pr_auc"], ascending=[True, False, False]).iterrows():
        lines.append(
            f"- `{r['candidate_id']}`: {r['decision']}; "
            f"AUC={r['auc']:.6f}, PR-AUC={r['pr_auc']:.6f}, BA={r['balanced_accuracy']:.6f}, "
            f"Sens={r['sensitivity']:.6f}, Spec={r['specificity']:.6f}, F1={r['f1']:.6f}, "
            f"Philips CN FPR={r['philips_cn_fpr']:.4f}, GE AD FNR={r['ge_ad_fnr']:.4f}."
        )
    (outdir / "final_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = [
        {
            "timestamp": now_iso(),
            "event": "start",
            "argv": ["run_promoted_frozen_latent_score_geometry_correction_batch_20260608.py"],
            "output_dir": str(out),
            "guardrails": [
                "no VAE training",
                "no tensor modification",
                "no metadata modification",
                "no model artifact overwrite",
                "no OASIS scoring",
                "no OASIS threshold/calibration fitting",
                "all correction choices inside train/dev inner CV only",
            ],
        }
    ]

    required = [
        PROMOTED_CACHE,
        PROMOTED_OOF / "calib_predictions.csv",
        CH1_OOF / "calib_predictions.csv",
        RESIDUALIZED_OOF / "harmonized_stageb_predictions.csv",
        FOLDCOMBAT_OOF / "calib_predictions.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifact(s): {missing}")

    if args.dry_run:
        rows = [
            {"candidate_id": c.candidate_id, "family": c.family, "n_grid_rows": len(c.param_grid), "description": c.description}
            for c in candidate_specs()
        ]
        write_table(out, "candidate_metrics", pd.DataFrame(rows))
        (out / "final_decision.md").write_text("Dry-run only. No candidate evaluation was performed.\n", encoding="utf-8")
        command_log.append({"timestamp": now_iso(), "event": "dry_run_complete", "n_candidates": len(rows)})
        (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return

    outputs = evaluate_candidates()
    pred = outputs["predictions"]
    foldwise = outputs["foldwise_metrics"]
    pooled = pooled_metrics(pred)
    mfr = manufacturer_errors(pred)
    ref_metrics, ref_pred = load_reference_rows()
    ref_mfr = manufacturer_errors(ref_pred)
    candidate_primary = primary_subset(pooled)
    ref_primary = ref_metrics.copy()
    combined_primary = pd.concat([ref_primary, candidate_primary], ignore_index=True, sort=False)
    combined_mfr = pd.concat([ref_mfr, mfr], ignore_index=True, sort=False)
    combined_primary = add_gate_columns(combined_primary, combined_mfr)
    subj_errors = subject_error_table(pred, ref_pred)

    write_table(out, "candidate_metrics", combined_primary)
    write_table(out, "foldwise_metrics", foldwise)
    write_table(out, "manufacturer_error_by_candidate", combined_mfr)
    write_table(out, "selected_hyperparameters", outputs["selected_hyperparameters"])
    write_table(out, "score_shift_correction_summary", outputs["score_shift_correction_summary"])
    write_table(out, "promoted_vs_candidate_subject_errors", subj_errors, max_rows=400)
    write_table(out, "inner_grid_search_results", outputs["inner_grid"], max_rows=400)

    write_final_decision(out, combined_primary)
    readme = [
        "# Promoted Frozen-Latent Score-Geometry Correction Batch",
        "",
        "This batch evaluates leakage-safe classifier-only and score-only corrections over existing promoted latent caches and existing ch1/promoted OOF scores.",
        "",
        "No VAE retraining, tensor modification, metadata modification, model artifact overwrite, OASIS scoring, or OASIS threshold/calibration fitting was performed.",
        "",
        "Primary rows use `oof_ecdf` with `inner_oof_target_sens_ge_0p70_max_spec`.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    command_log.append(
        {
            "timestamp": now_iso(),
            "event": "completed",
            "n_prediction_rows": int(len(pred)),
            "n_candidate_primary_rows": int(len(candidate_primary)),
            "outputs": [
                "candidate_metrics.csv",
                "foldwise_metrics.csv",
                "manufacturer_error_by_candidate.csv",
                "selected_hyperparameters.csv",
                "score_shift_correction_summary.csv",
                "promoted_vs_candidate_subject_errors.csv",
                "final_decision.md",
            ],
        }
    )
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
