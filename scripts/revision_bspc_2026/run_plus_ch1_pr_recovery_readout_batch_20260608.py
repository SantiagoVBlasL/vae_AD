#!/usr/bin/env python3
"""Second frozen-latent two-source readout batch for PR-AUC recovery.

The batch uses only existing promoted/ch1 OOF score features plus Age/Sex.
All meta-readout hyperparameters, score transforms, calibration, and
thresholds are selected inside each outer train/dev split. No VAE training,
tensor/metadata modification, OASIS scoring, or model-artifact overwrite is
performed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
PROMISING_AUDIT = RESULTS / "promising_frozen_latent_candidate_deep_audit_20260608"
FIRST_BATCH = RESULTS / "promoted_frozen_latent_score_geometry_correction_batch_20260608"
OUT_DEFAULT = RESULTS / "plus_ch1_pr_recovery_readout_batch_20260608"

PROMOTED_CACHE = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache"
PROMOTED_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration/calib_predictions.csv"
CH1_OOF = RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv"

FOLDS = [1, 2, 3, 4, 5]
INNER_FOLDS = 5
SEED = 42
TARGET_SENS = 0.70
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934
PROMOTED_BA = 0.725979
PROMOTED_F1 = 0.563492
PROMOTED_PHILIPS_CN_FPR = 0.4545
PREVIOUS_PLUS_CH1_AUC = 0.805790

C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    feature_transform: str
    selection_rule: str
    description: str


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
    return {"cmd": list(cmd), "returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr}


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


def primary_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No primary OOF rows found in {path}")
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise").astype(int)
    out["y_score"] = pd.to_numeric(out["y_score"], errors="raise")
    return out.reset_index(drop=True)


def load_base_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    promoted = primary_oof(PROMOTED_OOF)[["SubjectID", "y_score", "y_true", "fold"]].rename(
        columns={"y_score": "promoted_score"}
    )
    ch1 = primary_oof(CH1_OOF)[["SubjectID", "y_score", "y_true", "fold"]].rename(columns={"y_score": "ch1_score"})
    return promoted, ch1


def load_fold(fold: int, split: str, promoted: pd.DataFrame, ch1: pd.DataFrame) -> pd.DataFrame:
    path = PROMOTED_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=lambda c: not c.startswith("mu_"))
    df["SubjectID"] = df["SubjectID"].astype(str)
    df = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Sex_M"] = df["Sex"].astype(str).str.upper().str.startswith("M").astype(float)
    df["Manufacturer"] = df["Manufacturer"].astype(str)
    df = df.merge(promoted[["SubjectID", "promoted_score"]], on="SubjectID", how="left")
    df = df.merge(ch1[["SubjectID", "ch1_score"]], on="SubjectID", how="left")
    if df[["promoted_score", "ch1_score"]].isna().any().any():
        bad = df.loc[df[["promoted_score", "ch1_score"]].isna().any(axis=1), "SubjectID"].tolist()
        raise ValueError(f"Missing base scores for fold={fold} split={split}: {bad[:20]}")
    return df.reset_index(drop=True)


def candidates() -> list[Candidate]:
    return [
        Candidate(
            "plus_ch1_meta_logreg_pr_auc_selected",
            "score",
            "pr_auc",
            "Meta-logreg on promoted score, ch1 score, Age, Sex; select C by inner PR-AUC.",
        ),
        Candidate(
            "plus_ch1_meta_logreg_composite_selected",
            "score",
            "composite",
            "Meta-logreg on promoted score, ch1 score, Age, Sex; select C by 0.4*AUC + 0.4*PR-AUC + 0.2*BA.",
        ),
        Candidate(
            "plus_ch1_meta_logreg_pr_constrained_auc",
            "score",
            "pr_constrained_auc",
            "Meta-logreg on promoted score, ch1 score, Age, Sex; require inner PR-AUC >= promoted inner PR-AUC if possible.",
        ),
        Candidate(
            "plus_ch1_meta_logreg_rank_features",
            "rank",
            "pr_constrained_auc",
            "Meta-logreg on train-fitted ECDF/rank transformed promoted and ch1 scores plus Age/Sex.",
        ),
        Candidate(
            "plus_ch1_meta_logreg_logit_features",
            "logit",
            "pr_constrained_auc",
            "Meta-logreg on logit-transformed promoted and ch1 scores plus Age/Sex.",
        ),
    ]


def strat_key(df: pd.DataFrame, n_splits: int) -> pd.Series:
    key = df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)
    if key.value_counts().min() >= n_splits:
        return key
    return df["y"].astype(str)


def inner_splits(train: pd.DataFrame, outer_fold: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + outer_fold * 17)
    return list(cv.split(np.zeros(len(train)), strat_key(train, INNER_FOLDS)))


def ecdf_fit_transform(train_values: np.ndarray, target_values: np.ndarray, *, train_self: bool = False) -> np.ndarray:
    train_values = np.asarray(train_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    if train_self and len(train_values) == len(target_values) and np.allclose(train_values, target_values):
        ranks = pd.Series(train_values).rank(method="average").to_numpy(dtype=float)
        return (ranks - 0.5) / len(train_values)
    sorted_train = np.sort(train_values)
    return np.searchsorted(sorted_train, target_values, side="right") / max(len(sorted_train), 1)


def logit(values: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def feature_matrix(fit_df: pd.DataFrame, target_df: pd.DataFrame, transform: str, *, fit_self: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=target_df.index)
    if transform == "score":
        out["promoted_score"] = target_df["promoted_score"].to_numpy(dtype=float)
        out["ch1_score"] = target_df["ch1_score"].to_numpy(dtype=float)
    elif transform == "rank":
        out["promoted_score_rank"] = ecdf_fit_transform(
            fit_df["promoted_score"].to_numpy(dtype=float),
            target_df["promoted_score"].to_numpy(dtype=float),
            train_self=fit_self,
        )
        out["ch1_score_rank"] = ecdf_fit_transform(
            fit_df["ch1_score"].to_numpy(dtype=float),
            target_df["ch1_score"].to_numpy(dtype=float),
            train_self=fit_self,
        )
    elif transform == "logit":
        out["promoted_score_logit"] = logit(target_df["promoted_score"].to_numpy(dtype=float))
        out["ch1_score_logit"] = logit(target_df["ch1_score"].to_numpy(dtype=float))
    else:
        raise ValueError(transform)
    out["Age"] = target_df["Age"].to_numpy(dtype=float)
    out["Sex_M"] = target_df["Sex_M"].to_numpy(dtype=float)
    return out


def fit_meta(train_df: pd.DataFrame, params: dict[str, Any], transform: str) -> Pipeline:
    x = feature_matrix(train_df, train_df, transform, fit_self=True)
    y = train_df["y"].to_numpy(dtype=int)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(params["C"]),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def predict_meta(model: Pipeline, fit_df: pd.DataFrame, target_df: pd.DataFrame, transform: str, *, fit_self: bool = False) -> np.ndarray:
    x = feature_matrix(fit_df, target_df, transform, fit_self=fit_self)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def calibrate_ecdf(oof_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = pd.Series(oof_scores).rank(method="average").to_numpy(dtype=float)
    oof_ecdf = (ranks - 0.5) / len(oof_scores)
    test_ecdf = ecdf_fit_transform(oof_scores, test_scores)
    return oof_ecdf, test_ecdf


def threshold_at_target_sens(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    rows = []
    for thr in np.unique(np.round(np.r_[0.0, 0.5, 1.0, scores], 12)):
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        rows.append({"threshold": float(thr), "sens": sens, "spec": spec, "ba": 0.5 * (sens + spec)})
    tbl = pd.DataFrame(rows)
    eligible = tbl[tbl["sens"] >= TARGET_SENS]
    if eligible.empty:
        row = tbl.sort_values(["sens", "spec", "ba", "threshold"], ascending=[False, False, False, False]).iloc[0]
    else:
        row = eligible.sort_values(["spec", "sens", "ba", "threshold"], ascending=[False, False, False, False]).iloc[0]
    return {
        "threshold": float(row["threshold"]),
        "inner_sensitivity": float(row["sens"]),
        "inner_specificity": float(row["spec"]),
        "inner_balanced_accuracy": float(row["ba"]),
    }


def binary_metrics(y_true: Iterable[int], score: Iterable[float], pred: Iterable[int]) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(score), dtype=float)
    p = np.asarray(list(pred), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, p, zero_division=0)),
    }


def oof_for_c(train: pd.DataFrame, candidate: Candidate, c: float, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    y = train["y"].to_numpy(dtype=int)
    oof = np.full(len(train), np.nan, dtype=float)
    for tr_idx, va_idx in splits:
        tr = train.iloc[tr_idx].reset_index(drop=True)
        va = train.iloc[va_idx].reset_index(drop=True)
        model = fit_meta(tr, {"C": c}, candidate.feature_transform)
        oof[va_idx] = predict_meta(model, tr, va, candidate.feature_transform)
    if np.isnan(oof).any():
        raise RuntimeError(f"NaN OOF scores for {candidate.candidate_id} C={c}")
    return oof


def choose_c(train: pd.DataFrame, candidate: Candidate, outer_fold: int) -> tuple[dict[str, Any], pd.DataFrame]:
    y = train["y"].to_numpy(dtype=int)
    splits = inner_splits(train, outer_fold)
    promoted_inner_pr = float(average_precision_score(y, train["promoted_score"].to_numpy(dtype=float)))
    rows: list[dict[str, Any]] = []
    best_scores: dict[float, np.ndarray] = {}
    for c in C_GRID:
        oof_raw = oof_for_c(train, candidate, c, splits)
        oof_ecdf, _ = calibrate_ecdf(oof_raw, oof_raw)
        thr = threshold_at_target_sens(y, oof_ecdf)
        pred = (oof_ecdf >= thr["threshold"]).astype(int)
        row = {
            "fold": outer_fold,
            "candidate_id": candidate.candidate_id,
            "feature_transform": candidate.feature_transform,
            "selection_rule": candidate.selection_rule,
            "C": c,
            "inner_auc": float(roc_auc_score(y, oof_raw)),
            "inner_pr_auc": float(average_precision_score(y, oof_raw)),
            "inner_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "inner_threshold": thr["threshold"],
            "promoted_inner_pr_auc_reference": promoted_inner_pr,
        }
        row["composite_score"] = 0.4 * row["inner_auc"] + 0.4 * row["inner_pr_auc"] + 0.2 * row["inner_balanced_accuracy"]
        row["meets_promoted_inner_pr_constraint"] = bool(row["inner_pr_auc"] >= promoted_inner_pr)
        rows.append(row)
        best_scores[c] = oof_raw
    grid = pd.DataFrame(rows)
    if candidate.selection_rule == "pr_auc":
        chosen = grid.sort_values(["inner_pr_auc", "inner_auc", "inner_balanced_accuracy", "C"], ascending=[False, False, False, True]).iloc[0]
    elif candidate.selection_rule == "composite":
        chosen = grid.sort_values(["composite_score", "inner_pr_auc", "inner_auc", "C"], ascending=[False, False, False, True]).iloc[0]
    elif candidate.selection_rule == "pr_constrained_auc":
        eligible = grid[grid["meets_promoted_inner_pr_constraint"]]
        if eligible.empty:
            chosen = grid.sort_values(["inner_pr_auc", "inner_auc", "inner_balanced_accuracy", "C"], ascending=[False, False, False, True]).iloc[0]
            fallback = "fallback_best_pr_auc_no_c_met_constraint"
        else:
            chosen = eligible.sort_values(["inner_auc", "inner_pr_auc", "inner_balanced_accuracy", "C"], ascending=[False, False, False, True]).iloc[0]
            fallback = "constraint_satisfied_select_best_auc"
    else:
        raise ValueError(candidate.selection_rule)
    c = float(chosen["C"])
    selected = chosen.to_dict()
    selected["selection_context"] = locals().get("fallback", candidate.selection_rule)
    selected["inner_oof_raw_scores"] = best_scores[c]
    return selected, grid


def final_fold_eval(train: pd.DataFrame, test: pd.DataFrame, candidate: Candidate, selected: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    c = float(selected["C"])
    model = fit_meta(train, {"C": c}, candidate.feature_transform)
    test_raw = predict_meta(model, train, test, candidate.feature_transform)
    oof_raw = np.asarray(selected["inner_oof_raw_scores"], dtype=float)
    oof_ecdf, test_ecdf = calibrate_ecdf(oof_raw, test_raw)
    thr = threshold_at_target_sens(train["y"].to_numpy(dtype=int), oof_ecdf)
    pred = (test_ecdf >= float(thr["threshold"])).astype(int)
    out = test[["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y"]].copy()
    out = out.rename(columns={"y": "y_true"})
    out["candidate_id"] = candidate.candidate_id
    out["feature_transform"] = candidate.feature_transform
    out["selection_rule"] = candidate.selection_rule
    out["calib_method"] = PRIMARY_CALIB
    out["threshold_strategy"] = PRIMARY_THRESHOLD
    out["threshold"] = float(thr["threshold"])
    out["y_score_raw"] = test_raw
    out["y_score"] = test_ecdf
    out["y_pred"] = pred

    metric = {
        "fold": int(test["fold"].iloc[0]),
        "candidate_id": candidate.candidate_id,
        "feature_transform": candidate.feature_transform,
        "selection_rule": candidate.selection_rule,
        "selected_C": c,
        "selection_context": selected["selection_context"],
        "inner_auc": selected["inner_auc"],
        "inner_pr_auc": selected["inner_pr_auc"],
        "inner_balanced_accuracy": selected["inner_balanced_accuracy"],
        "promoted_inner_pr_auc_reference": selected["promoted_inner_pr_auc_reference"],
        **thr,
    }
    metric.update(binary_metrics(test["y"], test_ecdf, pred))

    x_train = feature_matrix(train, train, candidate.feature_transform, fit_self=True)
    coef = model.named_steps["model"].coef_.ravel()
    coef_row = {"fold": int(test["fold"].iloc[0]), "candidate_id": candidate.candidate_id, "selected_C": c}
    coef_row.update({f"coef_standardized_{name}": float(val) for name, val in zip(x_train.columns, coef)})
    return out, metric, coef_row


def evaluate() -> dict[str, pd.DataFrame]:
    promoted, ch1 = load_base_scores()
    pred_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    grid_parts: list[pd.DataFrame] = []
    hyper_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        train = load_fold(fold, "trainDev", promoted, ch1)
        test = load_fold(fold, "test", promoted, ch1)
        for cand in candidates():
            print(f"fold {fold}: {cand.candidate_id}", flush=True)
            selected, grid = choose_c(train, cand, fold)
            pred, fold_metric, coef = final_fold_eval(train, test, cand, selected)
            pred_parts.append(pred)
            fold_rows.append(fold_metric)
            grid_parts.append(grid)
            coef_rows.append(coef)
            hp = {k: v for k, v in selected.items() if k != "inner_oof_raw_scores"}
            hp["selected_params_json"] = json.dumps({"C": float(selected["C"]), "feature_transform": cand.feature_transform}, sort_keys=True)
            hyper_rows.append(hp)
    return {
        "predictions": pd.concat(pred_parts, ignore_index=True, sort=False),
        "foldwise_metrics": pd.DataFrame(fold_rows),
        "inner_grid_search_results": pd.concat(grid_parts, ignore_index=True, sort=False),
        "selected_hyperparameters": pd.DataFrame(hyper_rows),
        "meta_coefficients_by_fold": pd.DataFrame(coef_rows),
    }


def model_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, sub in pred.groupby("candidate_id", sort=False):
        row = {"candidate_id": model_id}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def load_reference_predictions() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, path in [
        ("reference_promoted_ch102_latent384_beta3p75", PROMOTED_OOF),
        ("reference_ch1only_latent384_beta3p75", CH1_OOF),
    ]:
        df = primary_oof(path)
        sub = df[["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y_true", "y_score", "y_pred"]].copy()
        sub = sub.rename(columns={"y_score": "y_score", "y_pred": "y_pred"})
        sub["candidate_id"] = label
        sub["calib_method"] = PRIMARY_CALIB
        sub["threshold_strategy"] = PRIMARY_THRESHOLD
        parts.append(sub)

    prev = pd.read_csv(FIRST_BATCH / "promoted_vs_candidate_subject_errors.csv")
    prev = prev[prev["candidate_id"].eq("promoted_plus_ch1_constrained_score_readout")].copy()
    prev = prev[
        ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y_true", "candidate_score", "candidate_pred"]
    ].rename(columns={"candidate_score": "y_score", "candidate_pred": "y_pred"})
    prev["candidate_id"] = "reference_previous_plus_ch1_auc_selected"
    prev["calib_method"] = PRIMARY_CALIB
    prev["threshold_strategy"] = PRIMARY_THRESHOLD
    parts.append(prev)
    refs = pd.concat(parts, ignore_index=True, sort=False)
    refs["y_true"] = pd.to_numeric(refs["y_true"], errors="raise").astype(int)
    refs["y_pred"] = pd.to_numeric(refs["y_pred"], errors="raise").astype(int)
    refs["y_score"] = pd.to_numeric(refs["y_score"], errors="raise")
    return refs


def manufacturer_errors(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in pred.groupby(["candidate_id", "Manufacturer"], dropna=False):
        cand, mfr = keys
        cn = sub[sub["y_true"] == 0]
        ad = sub[sub["y_true"] == 1]
        rows.append(
            {
                "candidate_id": cand,
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


def fixed_pr_levels(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cand, sub in pred.groupby("candidate_id", sort=False):
        y = sub["y_true"].to_numpy(dtype=int)
        score = sub["y_score"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, score)
        for level in [0.5, 0.6, 0.7]:
            eligible = precision[recall >= level]
            rows.append(
                {
                    "candidate_id": cand,
                    "query": f"precision_at_recall_ge_{level:.1f}",
                    "target": level,
                    "value": float(np.max(eligible)) if len(eligible) else np.nan,
                }
            )
        for level in [0.5, 0.6]:
            eligible = recall[precision >= level]
            rows.append(
                {
                    "candidate_id": cand,
                    "query": f"recall_at_precision_ge_{level:.1f}",
                    "target": level,
                    "value": float(np.max(eligible)) if len(eligible) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def topk_enrichment(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cand, sub in pred.groupby("candidate_id", sort=False):
        prevalence = float(sub["y_true"].mean())
        ranked = sub.sort_values("y_score", ascending=False)
        for k in [20, 40, 60, 80]:
            top = ranked.head(k)
            n_ad = int(top["y_true"].sum())
            rows.append(
                {
                    "candidate_id": cand,
                    "k": k,
                    "ad_in_top_k": n_ad,
                    "cn_in_top_k": int(k - n_ad),
                    "precision_at_k": safe_div(n_ad, k),
                    "lift_vs_prevalence": safe_div(safe_div(n_ad, k), prevalence),
                    "philips_cn_in_top_k": int(((top["y_true"] == 0) & top["Manufacturer"].eq("Philips")).sum()),
                    "ge_ad_in_top_k": int(((top["y_true"] == 1) & top["Manufacturer"].eq("GE")).sum()),
                }
            )
    return pd.DataFrame(rows)


def score_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in pred.groupby(["candidate_id", "ResearchGroup_Mapped", "Manufacturer"], dropna=False):
        cand, dx, mfr = keys
        s = sub["y_score"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate_id": cand,
                "diagnosis": dx,
                "Manufacturer": mfr,
                "n": int(len(sub)),
                "mean": float(np.mean(s)),
                "std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
                "median": float(np.median(s)),
                "p25": float(np.percentile(s, 25)),
                "p75": float(np.percentile(s, 75)),
                "predicted_ad_rate": float(sub["y_pred"].mean()),
            }
        )
    return pd.DataFrame(rows)


def add_gate(metrics: pd.DataFrame, mfr: pd.DataFrame) -> pd.DataFrame:
    lookup = {(r["candidate_id"], r["Manufacturer"]): r.to_dict() for _, r in mfr.iterrows()}
    rows: list[dict[str, Any]] = []
    prom_ge_fnr = lookup.get(("reference_promoted_ch102_latent384_beta3p75", "GE"), {}).get("ad_fnr", np.inf)
    for _, r in metrics.iterrows():
        cand = str(r["candidate_id"])
        phil = lookup.get((cand, "Philips"), {})
        ge = lookup.get((cand, "GE"), {})
        out = r.to_dict()
        out["philips_cn_fp"] = phil.get("cn_fp", np.nan)
        out["philips_cn_n"] = phil.get("n_cn", np.nan)
        out["philips_cn_fpr"] = phil.get("cn_fpr", np.nan)
        out["ge_ad_fn"] = ge.get("ad_fn", np.nan)
        out["ge_ad_n"] = ge.get("n_ad", np.nan)
        out["ge_ad_fnr"] = ge.get("ad_fnr", np.nan)
        out["gate_auc"] = bool(r["auc"] > PROMOTED_AUC)
        out["gate_auc_ideal_previous_plus_ch1"] = bool(r["auc"] >= PREVIOUS_PLUS_CH1_AUC)
        out["gate_pr_auc"] = bool(r["pr_auc"] >= PROMOTED_PR_AUC)
        out["gate_ba_f1"] = bool(r["balanced_accuracy"] >= PROMOTED_BA and r["f1"] >= PROMOTED_F1)
        out["gate_philips_cn_fpr"] = bool(out["philips_cn_fpr"] <= PROMOTED_PHILIPS_CN_FPR)
        out["gate_ge_ad_fnr"] = bool(out["ge_ad_fnr"] <= prom_ge_fnr + 1e-12)
        out["gate_no_oasis_use"] = True
        is_new = cand.startswith("plus_ch1_meta_")
        promotes = is_new and out["gate_auc"] and out["gate_pr_auc"] and out["gate_ba_f1"] and out["gate_philips_cn_fpr"] and out["gate_ge_ad_fnr"]
        if promotes:
            out["decision"] = "promote_frozen_readout_candidate"
        elif is_new and out["gate_auc"] and out["gate_pr_auc"]:
            out["decision"] = "sensitivity_only_partial_gate"
        elif is_new:
            out["decision"] = "do_not_promote"
        else:
            out["decision"] = "reference"
        rows.append(out)
    return pd.DataFrame(rows)


def write_final_decision(outdir: Path, metrics: pd.DataFrame) -> None:
    new = metrics[metrics["candidate_id"].astype(str).str.startswith("plus_ch1_meta_")].copy()
    promoted = new[new["decision"].eq("promote_frozen_readout_candidate")]
    best_auc = new.sort_values(["auc", "pr_auc"], ascending=False).iloc[0]
    best_pr = new.sort_values(["pr_auc", "auc"], ascending=False).iloc[0]
    lines = [
        "# Final Decision",
        "",
        "Scope: second frozen-latent two-source PR-recovery readout batch. No VAE training, no OASIS scoring, no OASIS threshold/calibration fitting, and no tensor/metadata/model artifact modification were performed.",
        "",
    ]
    if promoted.empty:
        lines.append("Decision: **no second-batch two-source readout candidate is promoted**.")
    else:
        lines.append("Decision: **at least one second-batch two-source readout candidate passed the frozen-readout gate**.")
    lines.extend(
        [
            "",
            "Best AUC candidate: "
            f"`{best_auc['candidate_id']}` AUC={best_auc['auc']:.6f}, PR-AUC={best_auc['pr_auc']:.6f}, "
            f"BA={best_auc['balanced_accuracy']:.6f}, F1={best_auc['f1']:.6f}, Philips CN FPR={best_auc['philips_cn_fpr']:.4f}.",
            "Best PR-AUC candidate: "
            f"`{best_pr['candidate_id']}` AUC={best_pr['auc']:.6f}, PR-AUC={best_pr['pr_auc']:.6f}, "
            f"BA={best_pr['balanced_accuracy']:.6f}, F1={best_pr['f1']:.6f}, Philips CN FPR={best_pr['philips_cn_fpr']:.4f}.",
            "",
            "Promotion gates:",
            f"- AUC > {PROMOTED_AUC:.6f}, ideally >= previous plus-ch1 {PREVIOUS_PLUS_CH1_AUC:.6f}.",
            f"- PR-AUC >= {PROMOTED_PR_AUC:.6f}.",
            f"- BA/F1 >= promoted ({PROMOTED_BA:.6f}/{PROMOTED_F1:.6f}).",
            f"- Philips CN FPR <= {PROMOTED_PHILIPS_CN_FPR:.4f}.",
            "- GE AD FNR not materially worse than promoted.",
            "- No OASIS use.",
            "",
            "Candidate summary:",
        ]
    )
    for _, r in new.sort_values(["decision", "auc", "pr_auc"], ascending=[True, False, False]).iterrows():
        lines.append(
            f"- `{r['candidate_id']}`: {r['decision']}; AUC={r['auc']:.6f}, PR-AUC={r['pr_auc']:.6f}, "
            f"BA={r['balanced_accuracy']:.6f}, Sens={r['sensitivity']:.6f}, Spec={r['specificity']:.6f}, "
            f"F1={r['f1']:.6f}, Philips CN FPR={r['philips_cn_fpr']:.4f}, GE AD FNR={r['ge_ad_fnr']:.4f}."
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
            "output_dir": str(out),
            "guardrails": [
                "no VAE training",
                "no tensor modification",
                "no metadata modification",
                "no model artifact overwrite",
                "no OASIS scoring",
                "no OASIS threshold/calibration fitting",
                "no outer-test labels for hyperparameter, threshold, or calibration selection",
            ],
        }
    ]

    required = [
        PROMISING_AUDIT / "final_recommendation.md",
        FIRST_BATCH / "promoted_vs_candidate_subject_errors.csv",
        PROMOTED_OOF,
        CH1_OOF,
        PROMOTED_CACHE / "fold_1_trainDev_latent_mu.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")

    if args.dry_run:
        write_table(
            out,
            "candidate_metrics",
            pd.DataFrame([{"candidate_id": c.candidate_id, "feature_transform": c.feature_transform, "selection_rule": c.selection_rule, "description": c.description} for c in candidates()]),
        )
        (out / "final_decision.md").write_text("Dry-run only. No candidate evaluation was performed.\n", encoding="utf-8")
        command_log.append({"timestamp": now_iso(), "event": "dry_run_complete", "n_candidates": len(candidates())})
        (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return

    outputs = evaluate()
    candidate_pred = outputs["predictions"]
    ref_pred = load_reference_predictions()
    all_pred = pd.concat([ref_pred, candidate_pred], ignore_index=True, sort=False)
    metrics = model_metrics(all_pred)
    mfr = manufacturer_errors(all_pred)
    metrics = add_gate(metrics, mfr)

    write_table(out, "candidate_metrics", metrics)
    write_table(out, "foldwise_metrics", outputs["foldwise_metrics"])
    write_table(out, "selected_hyperparameters", outputs["selected_hyperparameters"])
    write_table(out, "inner_grid_search_results", outputs["inner_grid_search_results"], max_rows=500)
    write_table(out, "meta_coefficients_by_fold", outputs["meta_coefficients_by_fold"])
    write_table(out, "manufacturer_error_by_candidate", mfr)
    write_table(out, "precision_recall_fixed_levels", fixed_pr_levels(all_pred))
    write_table(out, "topk_ad_enrichment", topk_enrichment(all_pred))
    write_table(out, "score_distribution_by_manufacturer", score_distribution(all_pred), max_rows=400)
    write_table(out, "predictions", all_pred, max_rows=400)
    write_final_decision(out, metrics)

    readme = [
        "# Plus-Ch1 PR-Recovery Frozen Readout Batch",
        "",
        "This batch tests second-pass leakage-safe meta-logreg selection strategies over existing promoted/ch1 OOF score features plus Age/Sex.",
        "",
        "No VAE training, tensor/metadata modification, model artifact overwrite, OASIS scoring, or OASIS threshold/calibration fitting was performed.",
        "",
        "Primary metrics use OOF-ECDF calibrated outer predictions with the inner-OOF target-sensitivity threshold rule.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log.append(
        {
            "timestamp": now_iso(),
            "event": "completed",
            "n_candidate_prediction_rows": int(len(candidate_pred)),
            "n_reference_prediction_rows": int(len(ref_pred)),
            "outputs": [
                "candidate_metrics.csv",
                "foldwise_metrics.csv",
                "selected_hyperparameters.csv",
                "manufacturer_error_by_candidate.csv",
                "precision_recall_fixed_levels.csv",
                "topk_ad_enrichment.csv",
                "score_distribution_by_manufacturer.csv",
                "final_decision.md",
            ],
        }
    )
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
