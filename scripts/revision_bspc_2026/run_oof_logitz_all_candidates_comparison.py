#!/usr/bin/env python3
"""OOF logit-z calibration comparison across all four candidate runs.

Candidates:
  1. locked_v5p1b        (latent256, beta2.5, horizon4480)
  2. recover035_latent256 (latent256, beta2.5, horizon4480)
  3. recover035_latent384_beta2p5  (latent384, beta2.5, horizon10000)
  4. recover035_latent384_beta3p75 (latent384, beta3.75, horizon10000)

Method (same for all):
  - logreg_l2 (original C-grid [0.001, 0.01, 0.1, 1.0])
  - z_plus_age_sex
  - raw baseline vs OOF logit-z calibration
  - threshold: inner_oof_target_sens_ge_0p70_max_spec

Calibration parameters learned from inner-CV OOF trainDev scores only.
No outer-test labels used for calibration or threshold fitting.
No VAE retraining. No tensor/metadata/model-output modification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUTPUT_DIR = RESULTS / "oof_logitz_all_candidates_comparison"

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]
TARGET_SENSITIVITY = 0.70
INNER_FOLDS = 5
FOLDS = [1, 2, 3, 4, 5]
SEED = 42

CALIB_METHODS = ["raw", "oof_logitz"]
MODEL_NAME = "logreg_l2_original"
FEATURE_SET = "z_plus_age_sex"
THRESHOLD_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"


CANDIDATES: List[Dict[str, Any]] = [
    {
        "label": "locked_v5p1b",
        "latent_cache_dir": RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep" / "latent_cache",
        "description": "v5.1b horizon4480/cycles56 FULL [1,0,2] latent256 beta2.5",
    },
    {
        "label": "recover035_latent256",
        "latent_cache_dir": RESULTS / "recover035_full5x5" / "classifier_only_readout" / "latent_cache",
        "description": "recover035 FULL5x5 latent256 beta2.5 horizon4480",
    },
    {
        "label": "recover035_latent384_beta2p5",
        "latent_cache_dir": RESULTS / "recover035_latent384_T80_h10000_p560_full5x5" / "classifier_only_readout" / "latent_cache",
        "description": "recover035 latent384 beta2.5 T80 h10000 p560",
    },
    {
        "label": "recover035_latent384_beta3p75",
        "latent_cache_dir": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5" / "classifier_only_readout" / "latent_cache",
        "description": "recover035 latent384 beta3.75 T80 h10000 p560",
    },
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return view.to_markdown(index=False) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


# ── Metrics ───────────────────────────────────────────────────────────────────

def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out = {
        "n": int(len(y_true)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    brier = float(np.mean((y_score - y_true) ** 2))
    out["brier"] = brier
    return out


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        frac_pos = float(y_true[mask].mean())
        mean_pred = float(y_score[mask].mean())
        ece += abs(frac_pos - mean_pred) * mask.sum() / n
    return float(ece)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(mu_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("latent", Pipeline([("scaler", StandardScaler())]), mu_cols),
        ("age", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["Age"]),
        ("sex", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]), ["Sex"]),
    ], remainder="drop", sparse_threshold=0.0)


def build_pipeline(fold: int) -> Tuple[Pipeline, Dict[str, list]]:
    return (
        Pipeline([
            ("pre", "passthrough"),
            ("model", LogisticRegression(
                penalty="l2", solver="lbfgs", class_weight="balanced",
                max_iter=5000, random_state=SEED + fold,
            )),
        ]),
        {"model__C": ORIGINAL_C_GRID},
    )


# ── Threshold selection ───────────────────────────────────────────────────────

def select_target_sens_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score[np.isfinite(y_score)])))
    best_thresh = 0.5
    best_spec = -1.0
    for thr in candidates:
        pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        if sens >= TARGET_SENSITIVITY and spec > best_spec:
            best_spec = spec
            best_thresh = float(thr)
    if best_spec < 0:
        # target not reached — fall back to max sensitivity
        for thr in sorted(candidates, reverse=True):
            pred = (y_score >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            sens = safe_div(tp, tp + fn)
            spec = safe_div(tn, tn + fp)
            if sens > 0:
                best_thresh = float(thr)
                break
    return best_thresh


# ── Calibration ──────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def apply_oof_logitz(
    oof_scores: np.ndarray,
    test_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    eps = 1e-6
    oof_l = np.log(np.clip(oof_scores, eps, 1 - eps) / (1 - np.clip(oof_scores, eps, 1 - eps)))
    test_l = np.log(np.clip(test_scores, eps, 1 - eps) / (1 - np.clip(test_scores, eps, 1 - eps)))
    mu = float(np.mean(oof_l))
    sigma = max(float(np.std(oof_l, ddof=1)), 1e-8)
    cal_oof = _sigmoid((oof_l - mu) / sigma)
    cal_test = _sigmoid((test_l - mu) / sigma)
    return cal_oof, cal_test, {"oof_logit_mean": round(mu, 6), "oof_logit_std": round(sigma, 6)}


# ── Inner stratification ──────────────────────────────────────────────────────

def inner_strat_key(df: pd.DataFrame) -> Any:
    cols = ["ResearchGroup_Mapped", "Manufacturer"]
    key = df[cols].fillna("UNK").astype(str).apply(lambda r: "_".join(r), axis=1)
    if int(key.value_counts().min()) < INNER_FOLDS:
        return df["y"].astype(int)
    return key


# ── Per-fold run ──────────────────────────────────────────────────────────────

def run_fold(
    fold: int,
    cache_dir: Path,
    candidate_label: str,
) -> List[Dict[str, Any]]:
    train_df = pd.read_csv(cache_dir / f"fold_{fold}_trainDev_latent_mu.csv")
    test_df = pd.read_csv(cache_dir / f"fold_{fold}_test_latent_mu.csv")

    mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
    feature_cols = mu_cols + ["Age", "Sex"]

    y_train = train_df["y"].astype(int).to_numpy()
    y_test = test_df["y"].astype(int).to_numpy()

    x_train = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    pre = make_preprocessor(mu_cols)
    base_pipe, grid = build_pipeline(fold)
    pipe = clone(base_pipe)
    pipe.steps[0] = ("pre", pre)

    inner_key = inner_strat_key(train_df)
    inner_cv = list(StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 30
    ).split(np.zeros(len(train_df)), inner_key))

    search = GridSearchCV(pipe, param_grid=grid, scoring="roc_auc", cv=inner_cv,
                          n_jobs=2, refit=True, error_score=np.nan)
    search.fit(x_train, y_train)
    best = search.best_estimator_
    best_C = search.best_params_.get("model__C", float("nan"))
    inner_auc = float(search.best_score_)

    # Inner OOF scores (same splits)
    oof_proba = cross_val_predict(
        clone(best), x_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=2
    )
    oof_scores_raw = oof_proba[:, 1].astype(float)

    # Outer test scores (raw)
    test_scores_raw = best.predict_proba(x_test)[:, 1].astype(float)

    oof_range_raw = float(np.ptp(oof_scores_raw))
    test_range_raw = float(np.ptp(test_scores_raw))

    print(
        f"    fold={fold}  best_C={best_C}  inner_auc={inner_auc:.4f}"
        f"  oof_range={oof_range_raw:.4f}  test_range={test_range_raw:.4f}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []

    for calib in CALIB_METHODS:
        if calib == "raw":
            cal_oof = oof_scores_raw.copy()
            cal_test = test_scores_raw.copy()
            meta: Dict[str, Any] = {}
        else:  # oof_logitz
            cal_oof, cal_test, meta = apply_oof_logitz(oof_scores_raw, test_scores_raw)

        thresh = select_target_sens_threshold(y_train, cal_oof)
        y_pred = (cal_test >= thresh).astype(int)

        m = binary_metrics(y_test, cal_test, y_pred)
        ece = compute_ece(y_test, cal_test)

        base = {
            "candidate": candidate_label,
            "fold": fold,
            "calib_method": calib,
            "best_C": best_C,
            "inner_auc": inner_auc,
            "threshold": thresh,
            "oof_range_raw": oof_range_raw,
            "oof_range_calib": float(np.ptp(cal_oof)),
            "test_range_raw": test_range_raw,
            "test_range_calib": float(np.ptp(cal_test)),
            "ece_10bins": ece,
        }
        base.update(meta)
        base.update(m)

        # Per-subject prediction rows
        p = test_df[["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]].copy()
        p["candidate"] = candidate_label
        p["fold"] = fold
        p["calib_method"] = calib
        p["threshold"] = thresh
        p["y_true"] = y_test
        p["y_score_raw"] = test_scores_raw
        p["y_score"] = cal_test
        p["y_pred"] = y_pred

        base["_pred_df"] = p
        rows.append(base)

    return rows


# ── Per-candidate run ─────────────────────────────────────────────────────────

def run_candidate(cand: Dict[str, Any]) -> Tuple[List[Dict], List[pd.DataFrame]]:
    label = cand["label"]
    cache_dir = cand["latent_cache_dir"]

    print(f"\n[{label}]  {cand['description']}", flush=True)
    for fold in FOLDS:
        for split in ["trainDev", "test"]:
            p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

    fold_rows: List[Dict] = []
    pred_dfs: List[pd.DataFrame] = []

    for fold in FOLDS:
        rows = run_fold(fold, cache_dir, label)
        for r in rows:
            pred_dfs.append(r.pop("_pred_df"))
            fold_rows.append(r)

    return fold_rows, pred_dfs


# ── Aggregation ───────────────────────────────────────────────────────────────

def pooled_metrics(pred_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cand, calib), sub in pred_all.groupby(["candidate", "calib_method"], sort=False):
        y_true = sub["y_true"].to_numpy()
        y_score = sub["y_score"].to_numpy()
        y_pred = sub["y_pred"].to_numpy()
        m = binary_metrics(y_true, y_score, y_pred)
        m["ece_10bins"] = compute_ece(y_true, y_score)
        m["candidate"] = cand
        m["calib_method"] = calib
        m["promotes"] = bool(
            m.get("auc", 0) > LOCKED_AUC and m.get("pr_auc", 0) >= LOCKED_PR_AUC
        )
        rows.append(m)
    col_order = [
        "candidate", "calib_method",
        "n", "n_cn", "n_ad",
        "auc", "pr_auc", "balanced_accuracy",
        "sensitivity", "specificity", "f1",
        "tp", "fp", "fn", "tn",
        "brier", "ece_10bins", "promotes",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in col_order if c in df.columns]]


def foldwise_metrics_table(foldwise_rows: List[Dict]) -> pd.DataFrame:
    cols = [
        "candidate", "fold", "calib_method",
        "best_C", "inner_auc",
        "auc", "pr_auc", "balanced_accuracy",
        "sensitivity", "specificity", "f1",
        "tp", "fp", "fn", "tn",
        "brier", "ece_10bins",
        "threshold", "oof_range_raw", "test_range_raw", "test_range_calib",
    ]
    df = pd.DataFrame(foldwise_rows)
    return df[[c for c in cols if c in df.columns]]


def fold4_table(foldwise_df: pd.DataFrame) -> pd.DataFrame:
    sub = foldwise_df[foldwise_df["fold"] == 4].copy()
    cols = [
        "candidate", "calib_method",
        "auc", "pr_auc", "balanced_accuracy",
        "sensitivity", "specificity", "f1",
        "tp", "fp", "fn", "tn",
        "brier", "ece_10bins", "best_C",
    ]
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def pooled_vs_foldwise_gap(foldwise_df: pd.DataFrame, pooled_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cand, calib), fw_sub in foldwise_df.groupby(["candidate", "calib_method"], sort=False):
        fw_auc = float(fw_sub["auc"].mean())
        fw_pr = float(fw_sub["pr_auc"].mean())
        pool_mask = (pooled_df["candidate"] == cand) & (pooled_df["calib_method"] == calib)
        pool_sub = pooled_df[pool_mask]
        pooled_auc = float(pool_sub["auc"].iloc[0]) if not pool_sub.empty else float("nan")
        pooled_pr = float(pool_sub["pr_auc"].iloc[0]) if not pool_sub.empty else float("nan")
        rows.append({
            "candidate": cand,
            "calib_method": calib,
            "pooled_auc": pooled_auc,
            "foldwise_mean_auc": fw_auc,
            "gap_fw_minus_pooled": fw_auc - pooled_auc,
            "pooled_pr_auc": pooled_pr,
            "foldwise_mean_pr_auc": fw_pr,
            "promotes": bool(pooled_auc > LOCKED_AUC and pooled_pr >= LOCKED_PR_AUC),
        })
    return pd.DataFrame(rows)


def brier_ece_table(pooled_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate", "calib_method", "brier", "ece_10bins"]
    df = pooled_df[[c for c in cols if c in pooled_df.columns]].copy()
    return df.sort_values(["candidate", "calib_method"]).reset_index(drop=True)


def philips_cn_fp_table(pred_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cand, calib), sub in pred_all.groupby(["candidate", "calib_method"], sort=False):
        cn_mask = sub["y_true"] == 0
        cn_sub = sub[cn_mask]
        for mfr_label, mfr_upper in [("Philips", "PHILIPS"), ("GE", "GE"), ("SIEMENS", "SIEMENS")]:
            mfr_cn = cn_sub[cn_sub["Manufacturer"].astype(str).str.upper() == mfr_upper]
            if len(mfr_cn) == 0:
                continue
            n_fp = int((mfr_cn["y_pred"] == 1).sum())
            n_cn = len(mfr_cn)
            rows.append({
                "candidate": cand,
                "calib_method": calib,
                "manufacturer": mfr_label,
                "n_cn": n_cn,
                "fp_cn": n_fp,
                "fpr_cn": round(n_fp / n_cn, 4),
                "spec_cn": round(1 - n_fp / n_cn, 4),
            })
    return pd.DataFrame(rows).sort_values(["manufacturer", "candidate", "calib_method"]).reset_index(drop=True)


def ge_ad_fn_table(pred_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cand, calib), sub in pred_all.groupby(["candidate", "calib_method"], sort=False):
        ge_ad = sub[(sub["Manufacturer"].astype(str).str.upper() == "GE") & (sub["y_true"] == 1)]
        fn = ge_ad[ge_ad["y_pred"] == 0]
        tp = ge_ad[ge_ad["y_pred"] == 1]
        n_total = len(ge_ad)
        n_fn = len(fn)
        n_tp = len(tp)
        sens = safe_div(n_tp, n_total)
        rows.append({
            "candidate": cand,
            "calib_method": calib,
            "n_ge_ad": n_total,
            "tp_ge_ad": n_tp,
            "fn_ge_ad": n_fn,
            "sensitivity_ge_ad": round(sens, 4),
        })
    return pd.DataFrame(rows).sort_values(["candidate", "calib_method"]).reset_index(drop=True)


def score_range_by_fold_table(foldwise_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate", "fold", "calib_method",
        "oof_range_raw", "test_range_raw", "test_range_calib",
    ]
    df = foldwise_df[[c for c in cols if c in foldwise_df.columns]]
    return df.sort_values(["candidate", "calib_method", "fold"]).reset_index(drop=True)


def raw_vs_logitz_delta(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: for each candidate, show raw AUC, logitz AUC, delta."""
    raw = pooled_df[pooled_df["calib_method"] == "raw"].set_index("candidate")
    logitz = pooled_df[pooled_df["calib_method"] == "oof_logitz"].set_index("candidate")

    rows = []
    for cand in raw.index:
        if cand not in logitz.index:
            continue
        r = raw.loc[cand]
        lz = logitz.loc[cand]
        rows.append({
            "candidate": cand,
            "auc_raw": r["auc"],
            "auc_logitz": lz["auc"],
            "delta_auc": lz["auc"] - r["auc"],
            "pr_auc_raw": r["pr_auc"],
            "pr_auc_logitz": lz["pr_auc"],
            "delta_pr_auc": lz["pr_auc"] - r["pr_auc"],
            "ba_raw": r["balanced_accuracy"],
            "ba_logitz": lz["balanced_accuracy"],
            "f1_raw": r["f1"],
            "f1_logitz": lz["f1"],
            "brier_raw": r["brier"],
            "brier_logitz": lz["brier"],
            "promotes_raw": r.get("promotes", False),
            "promotes_logitz": lz.get("promotes", False),
        })
    return pd.DataFrame(rows)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    outdir: Path,
    pooled: pd.DataFrame,
    gap: pd.DataFrame,
    delta: pd.DataFrame,
    fold4: pd.DataFrame,
    philips: pd.DataFrame,
    ge_fn: pd.DataFrame,
    brier_ece: pd.DataFrame,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# OOF Logit-Z Calibration Comparison — All Candidates",
        "",
        f"Generated: {now}",
        "",
        "## Candidates",
    ]
    for c in CANDIDATES:
        lines.append(f"- **{c['label']}**: {c['description']}")

    lines += [
        "",
        "## Method",
        "- logreg_l2 (C-grid: [0.001, 0.01, 0.1, 1.0], class_weight=balanced)",
        "- Feature set: z_plus_age_sex",
        "- Calibrations: raw (baseline) vs OOF logit-z",
        "- Threshold: inner_oof_target_sens_ge_0p70_max_spec",
        "- Calibration parameters (logit mean/std) learned from inner-CV OOF trainDev only",
        "- No VAE retraining. No outer-test label leakage.",
        "",
        f"Promotion gate: AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} (both simultaneously)",
        "",
        "## Raw vs OOF Logit-Z Delta",
        "_For each candidate: AUC and PR-AUC before and after calibration._",
        "",
    ]
    lines.append(md_table(delta))

    lines += ["", "## Pooled Metrics (all candidates, both calibrations)", ""]
    lines.append(md_table(pooled))

    lines += ["", "## Pooled vs Foldwise AUC Gap", ""]
    lines.append(md_table(gap))

    lines += ["", "## Fold 4 Metrics (raw vs logitz, all candidates)", ""]
    lines.append(md_table(fold4))

    lines += ["", "## Philips CN False Positives (pooled)", ""]
    philips_target = philips[philips["manufacturer"] == "Philips"]
    lines.append(md_table(philips_target) if not philips_target.empty else "_No Philips CN rows._\n")

    lines += ["", "## GE AD False Negatives (pooled)", ""]
    lines.append(md_table(ge_fn))

    lines += ["", "## Brier / ECE Comparison (pooled)", ""]
    lines.append(md_table(brier_ece))

    lines += [
        "",
        "## Read-only guarantee",
        "Did not retrain VAE. Did not fit thresholds on outer test data.",
        "Did not modify tensors, metadata, ledger, or any existing run outputs.",
    ]
    (outdir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    # Validate all cache paths first
    print("OOF logit-z calibration comparison — all candidates")
    for cand in CANDIDATES:
        cache_dir = cand["latent_cache_dir"]
        missing = []
        for fold in FOLDS:
            for split in ["trainDev", "test"]:
                p = cache_dir / f"fold_{fold}_{split}_latent_mu.csv"
                if not p.exists():
                    missing.append(str(p))
        if missing:
            print(f"MISSING files for {cand['label']}:")
            for m in missing:
                print(f"  {m}")
            return 1
        print(f"  {cand['label']}: all 10 cache files found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_rows: List[Dict] = []
    all_pred_dfs: List[pd.DataFrame] = []

    for cand in CANDIDATES:
        fold_rows, pred_dfs = run_candidate(cand)
        all_fold_rows.extend(fold_rows)
        all_pred_dfs.extend(pred_dfs)

    pred_all = pd.concat(all_pred_dfs, ignore_index=True, sort=False)
    foldwise_df = foldwise_metrics_table(all_fold_rows)

    pooled = pooled_metrics(pred_all)
    gap = pooled_vs_foldwise_gap(foldwise_df, pooled)
    fold4 = fold4_table(foldwise_df)
    philips = philips_cn_fp_table(pred_all)
    ge_fn = ge_ad_fn_table(pred_all)
    brier_ece = brier_ece_table(pooled)
    delta = raw_vs_logitz_delta(pooled)

    # Write tables
    write_table(OUTPUT_DIR, "pooled_metrics", pooled)
    write_table(OUTPUT_DIR, "foldwise_metrics", foldwise_df)
    write_table(OUTPUT_DIR, "fold4_metrics", fold4)
    write_table(OUTPUT_DIR, "pooled_vs_foldwise_gap", gap)
    write_table(OUTPUT_DIR, "philips_cn_fp", philips)
    write_table(OUTPUT_DIR, "ge_ad_fn", ge_fn)
    write_table(OUTPUT_DIR, "brier_ece_comparison", brier_ece)
    write_table(OUTPUT_DIR, "raw_vs_logitz_delta", delta)
    write_table(OUTPUT_DIR, "score_range_by_fold", score_range_by_fold_table(foldwise_df))
    pred_all.to_csv(OUTPUT_DIR / "calib_predictions_all.csv", index=False)

    write_report(OUTPUT_DIR, pooled, gap, delta, fold4, philips, ge_fn, brier_ece)

    write_json(OUTPUT_DIR / "command_log.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": [c["label"] for c in CANDIDATES],
        "model": MODEL_NAME,
        "feature_set": FEATURE_SET,
        "calib_methods": CALIB_METHODS,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "c_grid": ORIGINAL_C_GRID,
        "training_launched": False,
        "threshold_fitting_on_outer_test": False,
        "calibration_params_source": "inner_cv_oof_traindev_only",
        "locked_auc_gate": LOCKED_AUC,
        "locked_pr_auc_gate": LOCKED_PR_AUC,
    })

    # Console summary
    print("\n=== Raw vs OOF logit-z delta ===")
    for _, row in delta.iterrows():
        print(
            f"  {row['candidate']:<35}"
            f"  AUC: {row['auc_raw']:.4f} → {row['auc_logitz']:.4f} ({row['delta_auc']:+.4f})"
            f"  PR-AUC: {row['pr_auc_raw']:.4f} → {row['pr_auc_logitz']:.4f} ({row['delta_pr_auc']:+.4f})"
            f"  promotes: {row['promotes_logitz']}"
        )

    print(f"\nOutput: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
