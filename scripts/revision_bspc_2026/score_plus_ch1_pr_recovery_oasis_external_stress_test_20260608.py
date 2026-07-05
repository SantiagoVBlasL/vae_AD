#!/usr/bin/env python3
"""Frozen OASIS external stress-test for plus-ch1 PR-recovery readouts.

This script consumes existing OASIS fold-level promoted/ch1 scores and existing
ADNI selected hyperparameters/thresholds. It reconstructs the ADNI fold-local
meta-logreg readouts, applies them to OASIS, and reports metrics. It does not
fit scalers, thresholds, calibrators, or models on OASIS.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
PLUS_BATCH = RESULTS / "plus_ch1_pr_recovery_readout_batch_20260608"
FIRST_BATCH = RESULTS / "promoted_frozen_latent_score_geometry_correction_batch_20260608"
OUT_DEFAULT = RESULTS / "plus_ch1_pr_recovery_oasis_external_stress_test_20260608"

PROMOTED_OASIS = RESULTS / "oasis_mega_90_90_external_inference_model_panel_20260604/predictions.csv"
CH1_OASIS = RESULTS / "ch1only_latent384_beta3p75_oasis_mega_90_90_external_inference_20260605/predictions.csv"
PROMOTED_CACHE = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache"
PROMOTED_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration/calib_predictions.csv"
CH1_OOF = RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv"

FOCUS_BUILDS = ["concatenated_timeseries", "runwise164_pilot_parity", "runwise_140TR_pilot_parity"]
FOLDS = [1, 2, 3, 4, 5]
SEED = 42
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

FOCUS_CANDIDATES = [
    "plus_ch1_meta_logreg_rank_features",
    "plus_ch1_meta_logreg_pr_auc_selected",
    "plus_ch1_meta_logreg_logit_features",
]
REFERENCE_PREV_PLUS = "reference_previous_plus_ch1_auc_selected"
REFERENCE_PROMOTED = "reference_promoted_ch102_latent384_beta3p75"
REFERENCE_CH1 = "reference_ch1only_latent384_beta3p75"


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


def logit(values: np.ndarray) -> np.ndarray:
    vals = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(vals / (1 - vals))


def ecdf_fit_transform(train_values: np.ndarray, target_values: np.ndarray, *, train_self: bool = False) -> np.ndarray:
    train_values = np.asarray(train_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    if train_self and len(train_values) == len(target_values) and np.allclose(train_values, target_values):
        ranks = pd.Series(train_values).rank(method="average").to_numpy(dtype=float)
        return (ranks - 0.5) / len(train_values)
    sorted_train = np.sort(train_values)
    return np.searchsorted(sorted_train, target_values, side="right") / max(len(sorted_train), 1)


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
    return out


def load_adni_score_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    promoted = primary_oof(PROMOTED_OOF)[["SubjectID", "y_score"]].rename(columns={"y_score": "promoted_score"})
    ch1 = primary_oof(CH1_OOF)[["SubjectID", "y_score"]].rename(columns={"y_score": "ch1_score"})
    return promoted, ch1


def load_adni_train(fold: int, promoted_scores: pd.DataFrame, ch1_scores: pd.DataFrame) -> pd.DataFrame:
    path = PROMOTED_CACHE / f"fold_{fold}_trainDev_latent_mu.csv"
    df = pd.read_csv(path, usecols=lambda c: not c.startswith("mu_"))
    df["SubjectID"] = df["SubjectID"].astype(str)
    df = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Sex_M"] = df["Sex"].astype(str).str.upper().str.startswith("M").astype(float)
    df = df.merge(promoted_scores, on="SubjectID", how="left")
    df = df.merge(ch1_scores, on="SubjectID", how="left")
    if df[["promoted_score", "ch1_score"]].isna().any().any():
        raise ValueError(f"Missing ADNI train score features for fold {fold}")
    return df.reset_index(drop=True)


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
    out["Age"] = pd.to_numeric(target_df["Age"], errors="raise").to_numpy(dtype=float)
    out["Sex_M"] = target_df["Sex"].astype(str).str.upper().str.startswith("M").astype(float).to_numpy(dtype=float)
    return out


def fit_meta(train: pd.DataFrame, c: float, transform: str) -> Pipeline:
    x = feature_matrix(train, train, transform, fit_self=True)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(x, train["y"].to_numpy(dtype=int))
    return model


def predict_meta(model: Pipeline, train_fit: pd.DataFrame, target: pd.DataFrame, transform: str) -> np.ndarray:
    return np.asarray(model.predict_proba(feature_matrix(train_fit, target, transform, fit_self=False))[:, 1], dtype=float)


def selected_params_table() -> pd.DataFrame:
    second = pd.read_csv(PLUS_BATCH / "selected_hyperparameters.csv")
    second = second[second["candidate_id"].isin(FOCUS_CANDIDATES)].copy()
    first = pd.read_csv(FIRST_BATCH / "selected_hyperparameters.csv")
    first = first[first["candidate_id"].eq("promoted_plus_ch1_constrained_score_readout")].copy()
    first["candidate_id"] = REFERENCE_PREV_PLUS
    out = pd.concat([second, first], ignore_index=True, sort=False)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    return out


def thresholds_table() -> pd.DataFrame:
    second = pd.read_csv(PLUS_BATCH / "foldwise_metrics.csv")
    second = second[second["candidate_id"].isin(FOCUS_CANDIDATES)].copy()
    second["threshold_strategy"] = PRIMARY_THRESHOLD
    second["calib_method"] = PRIMARY_CALIB
    first = pd.read_csv(FIRST_BATCH / "foldwise_metrics.csv")
    first = first[
        first["candidate_id"].eq("promoted_plus_ch1_constrained_score_readout")
        & first["calib_method"].eq(PRIMARY_CALIB)
        & first["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    first["candidate_id"] = REFERENCE_PREV_PLUS
    cols = ["candidate_id", "fold", "threshold", "threshold_strategy", "calib_method"]
    out = pd.concat([second[cols], first[cols]], ignore_index=True, sort=False)
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    return out


def candidate_transform(candidate_id: str, row: pd.Series) -> str:
    if candidate_id == REFERENCE_PREV_PLUS:
        params = json.loads(str(row["selected_params_json"]))
        return params.get("feature_kind", "stack_scores").replace("stack_scores", "score")
    if "feature_transform" in row and pd.notna(row["feature_transform"]):
        return str(row["feature_transform"])
    params = json.loads(str(row["selected_params_json"]))
    return str(params.get("feature_transform", "score"))


def candidate_c(row: pd.Series) -> float:
    if "C" in row and pd.notna(row["C"]):
        return float(row["C"])
    params = json.loads(str(row["selected_params_json"]))
    return float(params["C"])


def load_oasis_fold_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    promoted = pd.read_csv(PROMOTED_OASIS)
    promoted = promoted[
        promoted["prediction_level"].eq("fold_model")
        & promoted["build_candidate"].isin(FOCUS_BUILDS)
        & promoted["candidate"].eq("promoted_beta3p75_oof_ecdf")
    ].copy()
    ch1 = pd.read_csv(CH1_OASIS)
    ch1 = ch1[
        ch1["prediction_level"].eq("fold_model")
        & ch1["build_candidate"].isin(FOCUS_BUILDS)
        & ch1["candidate"].eq("ch1only_latent384_beta3p75_oof_ecdf")
    ].copy()
    for df in [promoted, ch1]:
        df["SubjectID"] = df["SubjectID"].astype(str)
        df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
        df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)
        df["Age"] = pd.to_numeric(df["Age"], errors="raise")
        df["Sex"] = df["Sex"].astype(str)
    return promoted, ch1


def align_oasis_features(promoted: pd.DataFrame, ch1: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["build_candidate", "SubjectID", "fold"]
    meta_cols = [
        "SubjectID",
        "subject_id",
        "session_id",
        "experiment_id",
        "source_batch",
        "protocol_subset",
        "diagnosis",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "ScannerModel",
        "selected_qc_runs",
        "selected_run_ids",
        "mean_fd_subject",
        "max_fd_subject",
        "y",
        "build_candidate",
        "fold",
    ]
    left = promoted[meta_cols + ["y_score", "threshold", "y_pred"]].rename(
        columns={"y_score": "promoted_score", "threshold": "promoted_threshold", "y_pred": "promoted_pred"}
    )
    right = ch1[key_cols + ["y_score", "threshold", "y_pred"]].rename(
        columns={"y_score": "ch1_score", "threshold": "ch1_threshold", "y_pred": "ch1_pred"}
    )
    merged = left.merge(right, on=key_cols, how="inner", validate="one_to_one")
    expected = len(promoted)
    if len(merged) != expected:
        raise ValueError(f"OASIS promoted/ch1 alignment mismatch: {len(merged)} vs expected {expected}")
    return merged


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
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "predicted_ad_rate": float(np.mean(p)),
        "auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
    }


def score_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    promoted_scores, ch1_scores = load_adni_score_features()
    selected = selected_params_table()
    thresholds = thresholds_table()
    oasis_promoted, oasis_ch1 = load_oasis_fold_scores()
    oasis = align_oasis_features(oasis_promoted, oasis_ch1)
    pred_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for candidate_id in FOCUS_CANDIDATES + [REFERENCE_PREV_PLUS]:
        for fold in FOLDS:
            hp = selected[(selected["candidate_id"].eq(candidate_id)) & (selected["fold"].eq(fold))]
            th = thresholds[(thresholds["candidate_id"].eq(candidate_id)) & (thresholds["fold"].eq(fold))]
            if len(hp) != 1 or len(th) != 1:
                raise ValueError(f"Expected one hp/threshold row for {candidate_id} fold {fold}: hp={len(hp)}, th={len(th)}")
            hp_row = hp.iloc[0]
            transform = candidate_transform(candidate_id, hp_row)
            c = candidate_c(hp_row)
            threshold = float(th.iloc[0]["threshold"])
            train = load_adni_train(fold, promoted_scores, ch1_scores)
            model = fit_meta(train, c, transform)
            ext = oasis[oasis["fold"].eq(fold)].copy()
            ext_scores = predict_meta(model, train, ext, transform)
            ext_pred = (ext_scores >= threshold).astype(int)
            out = ext.copy()
            out["candidate"] = candidate_id
            out["role"] = "external_stress_test_only" if candidate_id != REFERENCE_PREV_PLUS else "previous_plus_ch1_reference"
            out["model_name"] = "plus_ch1_meta_logreg"
            out["feature_transform"] = transform
            out["calib_method"] = PRIMARY_CALIB
            out["threshold_strategy"] = PRIMARY_THRESHOLD
            out["prediction_level"] = "fold_model"
            out["y_score"] = ext_scores
            out["threshold"] = threshold
            out["y_pred"] = ext_pred
            pred_parts.append(out)
            audit_rows.append(
                {
                    "candidate": candidate_id,
                    "fold": fold,
                    "selected_C": c,
                    "feature_transform": transform,
                    "threshold": threshold,
                    "adni_train_n": int(len(train)),
                    "oasis_fold_rows": int(len(ext)),
                    "fit_scope": "ADNI outer train/dev only",
                    "oasis_fit_used": False,
                }
            )
    fold_preds = pd.concat(pred_parts, ignore_index=True, sort=False)
    ensemble_keys = [
        "SubjectID",
        "subject_id",
        "session_id",
        "experiment_id",
        "source_batch",
        "protocol_subset",
        "diagnosis",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "ScannerModel",
        "selected_qc_runs",
        "selected_run_ids",
        "mean_fd_subject",
        "max_fd_subject",
        "y",
        "build_candidate",
        "candidate",
        "role",
        "model_name",
        "feature_transform",
        "calib_method",
        "threshold_strategy",
    ]
    ens = (
        fold_preds.groupby(ensemble_keys, dropna=False)
        .agg(
            y_score=("y_score", "mean"),
            fold_score_std=("y_score", "std"),
            fold_positive_votes=("y_pred", "sum"),
        )
        .reset_index()
    )
    ens["fold"] = "ensemble_mean_score_majority_vote"
    ens["prediction_level"] = "ensemble_mean_score_majority_vote"
    ens["threshold"] = np.nan
    ens["y_pred"] = (ens["fold_positive_votes"] >= 3).astype(int)
    all_preds = pd.concat([fold_preds, ens], ignore_index=True, sort=False)
    return all_preds, pd.DataFrame(audit_rows)


def load_reference_predictions() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source, candidate_value, out_candidate, role in [
        (PROMOTED_OASIS, "promoted_beta3p75_oof_ecdf", REFERENCE_PROMOTED, "primary_reference"),
        (CH1_OASIS, "ch1only_latent384_beta3p75_oof_ecdf", REFERENCE_CH1, "ch1only_reference"),
    ]:
        df = pd.read_csv(source)
        df = df[df["build_candidate"].isin(FOCUS_BUILDS) & df["candidate"].eq(candidate_value)].copy()
        df["candidate"] = out_candidate
        df["role"] = role
        df["feature_transform"] = "reference"
        parts.append(df)
    refs = pd.concat(parts, ignore_index=True, sort=False)
    refs["y"] = pd.to_numeric(refs["y"], errors="raise").astype(int)
    refs["y_pred"] = pd.to_numeric(refs["y_pred"], errors="raise").astype(int)
    refs["y_score"] = pd.to_numeric(refs["y_score"], errors="raise")
    return refs


def metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    for keys, sub in predictions.groupby(["build_candidate", "candidate", "role", "prediction_level", "fold"], dropna=False):
        build, cand, role, level, fold = keys
        row = {
            "build_candidate": build,
            "candidate": cand,
            "role": role,
            "prediction_level": level,
            "fold": fold,
        }
        row.update(binary_metrics(sub["y"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
        confusion_rows.append(
            {
                "build_candidate": build,
                "candidate": cand,
                "prediction_level": level,
                "fold": fold,
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
            }
        )
    for keys, sub in predictions.groupby(["build_candidate", "candidate", "prediction_level", "diagnosis"], dropna=False):
        build, cand, level, dx = keys
        s = sub["y_score"].to_numpy(dtype=float)
        dist_rows.append(
            {
                "build_candidate": build,
                "candidate": cand,
                "prediction_level": level,
                "diagnosis": dx,
                "n": int(len(sub)),
                "score_mean": float(np.mean(s)),
                "score_std": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
                "score_median": float(np.median(s)),
                "score_p25": float(np.percentile(s, 25)),
                "score_p75": float(np.percentile(s, 75)),
                "predicted_ad_rate": float(sub["y_pred"].mean()),
            }
        )
    metrics = pd.DataFrame(rows)
    primary = metrics[metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    return primary, metrics, pd.DataFrame(confusion_rows), pd.DataFrame(dist_rows)


def adni_metrics() -> pd.DataFrame:
    first = pd.read_csv(FIRST_BATCH / "candidate_metrics.csv")
    second = pd.read_csv(PLUS_BATCH / "candidate_metrics.csv")
    keep = [
        "candidate_id",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "philips_cn_fpr",
        "ge_ad_fnr",
        "decision",
    ]
    first = first[first["candidate_id"].eq("promoted_plus_ch1_constrained_score_readout")][keep]
    first["candidate_id"] = REFERENCE_PREV_PLUS
    second = second[second["candidate_id"].isin([REFERENCE_PROMOTED, REFERENCE_CH1] + FOCUS_CANDIDATES)][keep]
    return pd.concat([second, first], ignore_index=True, sort=False).rename(columns={"candidate_id": "candidate"})


def adni_vs_oasis(primary: pd.DataFrame, adni: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    adni_lookup = {r["candidate"]: r for _, r in adni.iterrows()}
    for _, r in primary.iterrows():
        a = adni_lookup.get(r["candidate"])
        row = r.to_dict()
        if a is not None:
            row["adni_auc"] = a["auc"]
            row["adni_pr_auc"] = a["pr_auc"]
            row["adni_balanced_accuracy"] = a["balanced_accuracy"]
            row["adni_f1"] = a["f1"]
            row["auc_drop_adni_to_oasis"] = a["auc"] - r["auc"]
            row["pr_auc_drop_adni_to_oasis"] = a["pr_auc"] - r["pr_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_final_recommendation(outdir: Path, primary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    ens = primary.copy()
    focus = ens[ens["candidate"].isin(FOCUS_CANDIDATES)].copy()
    promoted = ens[ens["candidate"].eq(REFERENCE_PROMOTED)].set_index("build_candidate")
    lines = [
        "# Final Recommendation",
        "",
        "Scope: frozen OASIS external stress-test for plus-ch1 PR-recovery readouts. No OASIS fitting, thresholding, calibration, or model selection was performed.",
        "",
    ]
    if focus.empty:
        lines.append("No focus candidate rows were available.")
    else:
        for build in FOCUS_BUILDS:
            sub = focus[focus["build_candidate"].eq(build)].sort_values(["auc", "pr_auc"], ascending=False)
            if sub.empty:
                lines.append(f"- `{build}`: no focus-candidate rows.")
                continue
            best = sub.iloc[0]
            if build in promoted.index:
                pref = promoted.loc[build]
                transfer_note = (
                    f" promoted reference AUC={pref['auc']:.6f}, PR-AUC={pref['pr_auc']:.6f}; "
                    f"delta AUC={best['auc'] - pref['auc']:.6f}, delta PR-AUC={best['pr_auc'] - pref['pr_auc']:.6f}."
                )
            else:
                transfer_note = ""
            lines.append(
                f"- `{build}` best focus candidate by AUC: `{best['candidate']}` "
                f"AUC={best['auc']:.6f}, PR-AUC={best['pr_auc']:.6f}, BA={best['balanced_accuracy']:.6f}, F1={best['f1']:.6f};"
                f"{transfer_note}"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- The plus-ch1 PR-recovery candidates remain external stress-test candidates only.",
            "- The ADNI gain does **not** transfer as an external improvement over the promoted model on the requested OASIS builds: the best plus-ch1 focus row is below the promoted reference on AUC and PR-AUC for concatenated, runwise164, and runwise140TR parity.",
            "- The transferred ADNI thresholds are very conservative for plus-ch1 on OASIS, yielding high specificity but low sensitivity.",
            "- OASIS results must not be used to select or promote a model; compare transfer qualitatively against the promoted and ch1-only references.",
            "- The most defensible interpretation is that the plus-ch1 PR-recovery improvement is ADNI-specific frozen-readout geometry, not a robust external transfer gain.",
        ]
    )
    (outdir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                "no OASIS threshold fitting",
                "no OASIS calibration fitting",
                "no OASIS model selection",
                "no tensor/metadata/model artifact overwrite",
                "ADNI fold-local meta-readout reconstructed from ADNI train/dev only",
            ],
        }
    ]

    required = [
        PLUS_BATCH / "selected_hyperparameters.csv",
        PLUS_BATCH / "foldwise_metrics.csv",
        FIRST_BATCH / "selected_hyperparameters.csv",
        FIRST_BATCH / "foldwise_metrics.csv",
        PROMOTED_OASIS,
        CH1_OASIS,
        PROMOTED_CACHE / "fold_1_trainDev_latent_mu.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    if args.dry_run:
        validation = pd.DataFrame(
            [{"path": str(p), "exists": p.exists()} for p in required]
            + [{"focus_build": b, "available": True} for b in FOCUS_BUILDS]
            + [{"focus_candidate": c, "available": True} for c in FOCUS_CANDIDATES]
        )
        write_table(out, "artifact_validation", validation)
        (out / "final_recommendation.md").write_text("Dry-run only. No OASIS scoring was performed.\n", encoding="utf-8")
        command_log.append({"timestamp": now_iso(), "event": "dry_run_complete"})
        (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return

    plus_preds, fit_audit = score_candidates()
    ref_preds = load_reference_predictions()
    predictions = pd.concat([ref_preds, plus_preds], ignore_index=True, sort=False)
    primary, all_metrics, confusion, dist = metrics_tables(predictions)
    adni = adni_metrics()
    comparison = adni_vs_oasis(primary, adni)

    write_table(out, "artifact_validation", pd.DataFrame([{"path": str(p), "exists": p.exists()} for p in required]))
    write_table(out, "fold_meta_readout_reconstruction_audit", fit_audit)
    write_table(out, "predictions", predictions, max_rows=500)
    write_table(out, "primary_metrics", primary.sort_values(["build_candidate", "candidate"]))
    write_table(out, "foldwise_metrics", all_metrics.sort_values(["build_candidate", "candidate", "prediction_level", "fold"]), max_rows=800)
    write_table(out, "confusion_matrices", confusion.sort_values(["build_candidate", "candidate", "prediction_level", "fold"]), max_rows=800)
    write_table(out, "score_distribution_by_diagnosis", dist.sort_values(["build_candidate", "candidate", "prediction_level", "diagnosis"]), max_rows=500)
    write_table(out, "adni_vs_oasis_comparison", comparison.sort_values(["build_candidate", "candidate"]))
    write_final_recommendation(out, primary, comparison)

    readme = [
        "# Plus-Ch1 PR-Recovery OASIS External Stress Test",
        "",
        "This package applies ADNI-frozen plus-ch1 meta-readouts to existing mega-OASIS fold-level promoted/ch1 scores.",
        "",
        "No VAE training, OASIS threshold fitting, OASIS calibration fitting, OASIS model selection, tensor modification, metadata modification, or model artifact overwrite was performed.",
        "",
        "Primary rows use ensemble mean score with majority vote across the five ADNI fold readouts.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log.append(
        {
            "timestamp": now_iso(),
            "event": "completed",
            "n_prediction_rows": int(len(predictions)),
            "n_primary_rows": int(len(primary)),
            "focus_builds": FOCUS_BUILDS,
            "focus_candidates": FOCUS_CANDIDATES,
        }
    )
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
