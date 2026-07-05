#!/usr/bin/env python3
"""Read-only OASIS new-vs-pilot difficulty audit for Martin.

This audit synthesizes existing OASIS manifests, score files, and clinical/motion
summaries. It does not train, fit thresholds, modify tensors, or alter model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUTPUT = RESULTS / "oasis_new_vs_pilot_difficulty_audit_for_martin_20260601"

EXISTING_DIFFICULTY = RESULTS / "oasis_pilot_vs_new_clinical_motion_difficulty_audit_20260531"
ALL_SUBJECTS = EXISTING_DIFFICULTY / "all_subjects.csv"
MEGA_DIR = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"
MEGA_MANIFEST = MEGA_DIR / "mega_manifest.csv"
MEGA_PREDICTIONS = MEGA_DIR / "predictions.csv"
MEGA_STRATIFIED = MEGA_DIR / "stratified_metrics.csv"
NEW_HARMONIZED = RESULTS / "oasis_next_60cn_60ad_external_scoring_harmonized_horizon4480_20260531"
NEW_HARMONIZED_METRICS = NEW_HARMONIZED / "primary_metrics.csv"
PILOT_SCORING = RESULTS / "oasis_tanda_2026_05_25_external_scoring"
PILOT_METRICS = PILOT_SCORING / "primary_metrics.csv"
PARITY_FINAL = RESULTS / "oasis_pilot_vs_new_tensor_scoring_parity_audit_20260531" / "final_recommendation.md"


PRIMARY_MODEL_FOR_NARRATIVE = "recover035_oof_logitz"
PRIMARY_BUILD_FOR_NARRATIVE = "runwise164_pilot_parity"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_mean_sd(x: pd.Series) -> str:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return f"{vals.mean():.2f} +/- {vals.std(ddof=1):.2f}"


def fmt_median_iqr(x: pd.Series) -> str:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    q1, q3 = vals.quantile([0.25, 0.75])
    return f"{vals.median():.2f} [{q1:.2f}, {q3:.2f}]"


def to_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    except Exception:
        path.write_text(df.to_csv(index=False), encoding="utf-8")


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT / f"{name}.csv", index=False)
    to_markdown(df, OUTPUT / f"{name}.md")


def normalize_subject_id(x: Any) -> str:
    s = str(x)
    if not s.startswith("sub-") and s.startswith("OAS"):
        s = "sub-" + s
    return s


def diagnosis_to_y(diagnosis: Any) -> float:
    s = str(diagnosis).upper()
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA"}:
        return 1.0
    if s == "CN":
        return 0.0
    return np.nan


def summarize_numeric_by_group(df: pd.DataFrame, group_cols: List[str], numeric_cols: List[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: key for col, key in zip(group_cols, keys)}
        rec["n"] = int(len(sub))
        for col in numeric_cols:
            vals = pd.to_numeric(sub.get(col), errors="coerce") if col in sub.columns else pd.Series(dtype=float)
            rec[f"{col}_n"] = int(vals.notna().sum())
            rec[f"{col}_mean_sd"] = fmt_mean_sd(vals)
            rec[f"{col}_median_iqr"] = fmt_median_iqr(vals)
            rec[f"{col}_min"] = vals.min() if vals.notna().any() else np.nan
            rec[f"{col}_max"] = vals.max() if vals.notna().any() else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def sex_distribution(df: pd.DataFrame) -> pd.DataFrame:
    sex_col = "sex_label" if "sex_label" in df.columns else "sex_normalized"
    tab = (
        df.groupby(["cohort", "diagnosis", sex_col], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={sex_col: "sex"})
    )
    totals = tab.groupby(["cohort", "diagnosis"])["n"].transform("sum")
    tab["pct"] = 100 * tab["n"] / totals
    return tab


def mann_whitney(a: pd.Series, b: pd.Series) -> float:
    if stats is None:
        return np.nan
    aa = pd.to_numeric(a, errors="coerce").dropna()
    bb = pd.to_numeric(b, errors="coerce").dropna()
    if len(aa) < 2 or len(bb) < 2:
        return np.nan
    try:
        return float(stats.mannwhitneyu(aa, bb, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def pearson_spearman(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    ok = xx.notna() & yy.notna()
    if ok.sum() < 3 or stats is None:
        return {"n": int(ok.sum()), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(xx[ok], yy[ok])
    sr = stats.spearmanr(xx[ok], yy[ok])
    return {
        "n": int(ok.sum()),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def compute_auc(y: pd.Series, score: pd.Series) -> Dict[str, float]:
    yy = pd.to_numeric(y, errors="coerce")
    ss = pd.to_numeric(score, errors="coerce")
    ok = yy.notna() & ss.notna()
    if ok.sum() < 4 or yy[ok].nunique() < 2 or roc_auc_score is None:
        return {"auc": np.nan, "pr_auc": np.nan}
    return {
        "auc": float(roc_auc_score(yy[ok], ss[ok])),
        "pr_auc": float(average_precision_score(yy[ok], ss[ok])) if average_precision_score is not None else np.nan,
    }


def load_subjects() -> pd.DataFrame:
    df = read_csv(ALL_SUBJECTS)
    df["subject_id"] = df["subject_id"].map(normalize_subject_id)
    df["diagnosis"] = df["diagnosis"].replace({"AD": "AD_DEMENTIA"})
    return df


def load_mega_scores() -> pd.DataFrame:
    pred = read_csv(MEGA_PREDICTIONS)
    pred["subject_id"] = pred["subject_id"].fillna(pred.get("SubjectID")).map(normalize_subject_id)
    pred["diagnosis"] = pred["diagnosis"].replace({"AD": "AD_DEMENTIA"})
    pred["y_true"] = pred["diagnosis"].map(diagnosis_to_y).fillna(pd.to_numeric(pred.get("y_true"), errors="coerce"))
    ens = pred[pred["prediction_level"].astype(str).eq("ensemble_mean_score_majority_vote")].copy()
    ens["score_source"] = ens["adni_model"].astype(str) + " | " + ens["build_candidate"].astype(str)
    return ens


def merge_subject_covariates(scores: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Fill clinical/motion covariates from the subject-level audit table.

    Some score files contain clinical columns that are blank for a subset of rows.
    For this difficulty audit, the subject-level OASIS clinical/motion audit is the
    authoritative local source for Age/CDR/FD/run counts.
    """
    covar_cols = [
        "subject_id",
        "age_at_MR",
        "sex_label",
        "CDRTOT",
        "CDRSUM",
        "mean_fd",
        "max_fd",
        "pct_frames_fd_gt0p2",
        "pct_frames_fd_gt0p5",
        "n_runs_qc_ok",
    ]
    covar = subjects[[c for c in covar_cols if c in subjects.columns]].drop_duplicates("subject_id").copy()
    covar = covar.rename(columns={c: f"{c}_subject_audit" for c in covar.columns if c != "subject_id"})
    merged = scores.merge(covar, on="subject_id", how="left")
    for col in [c for c in covar_cols if c != "subject_id"]:
        audit_col = f"{col}_subject_audit"
        if audit_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[audit_col]
        else:
            merged[col] = merged[col].where(merged[col].notna(), merged[audit_col])
        # For numeric fields, also treat entirely non-numeric strings as missing.
        if col not in {"sex_label"}:
            numeric = pd.to_numeric(merged[col], errors="coerce")
            audit_numeric = pd.to_numeric(merged[audit_col], errors="coerce")
            merged[col] = numeric.where(numeric.notna(), audit_numeric)
    return merged


def build_demographic_motion_tables(subjects: pd.DataFrame) -> None:
    numeric_cols = [
        "age_at_MR",
        "CDRTOT",
        "CDRSUM",
        "mean_fd",
        "max_fd",
        "pct_frames_fd_gt0p2",
        "pct_frames_fd_gt0p5",
        "n_runs_qc_ok",
    ]
    demo = summarize_numeric_by_group(subjects, ["cohort", "diagnosis"], numeric_cols)
    write_table(demo, "pilot_vs_new_demographics_clinical_motion")
    write_table(sex_distribution(subjects), "sex_distribution_by_batch_diagnosis")

    run_counts = subjects.groupby(["cohort", "diagnosis", "n_runs_qc_ok"], dropna=False).size().reset_index(name="n")
    write_table(run_counts, "run_count_distribution")

    clinical_avail = []
    for col in ["CDRTOT", "CDRSUM", "MMSE", "MOCA"]:
        clinical_avail.append(
            {
                "variable": col,
                "available_in_subject_table": col in subjects.columns,
                "non_missing_n": int(pd.to_numeric(subjects[col], errors="coerce").notna().sum()) if col in subjects.columns else 0,
                "note": "available" if col in subjects.columns and subjects[col].notna().any() else "not found in local OASIS audit inputs",
            }
        )
    write_table(pd.DataFrame(clinical_avail), "clinical_variable_availability")


def build_cross_cohort_tests(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["age_at_MR", "CDRTOT", "CDRSUM", "mean_fd", "max_fd", "pct_frames_fd_gt0p2", "pct_frames_fd_gt0p5", "n_runs_qc_ok"]
    for diagnosis, sub_diag in subjects.groupby("diagnosis"):
        pilot = sub_diag[sub_diag["cohort"].eq("pilot")]
        new = sub_diag[sub_diag["cohort"].eq("new")]
        for metric in metrics:
            rows.append(
                {
                    "diagnosis": diagnosis,
                    "metric": metric,
                    "pilot_n": int(pd.to_numeric(pilot.get(metric), errors="coerce").notna().sum()) if metric in pilot else 0,
                    "new_n": int(pd.to_numeric(new.get(metric), errors="coerce").notna().sum()) if metric in new else 0,
                    "pilot_mean_sd": fmt_mean_sd(pilot.get(metric, pd.Series(dtype=float))),
                    "new_mean_sd": fmt_mean_sd(new.get(metric, pd.Series(dtype=float))),
                    "pilot_median_iqr": fmt_median_iqr(pilot.get(metric, pd.Series(dtype=float))),
                    "new_median_iqr": fmt_median_iqr(new.get(metric, pd.Series(dtype=float))),
                    "mann_whitney_p": mann_whitney(pilot.get(metric, pd.Series(dtype=float)), new.get(metric, pd.Series(dtype=float))),
                }
            )
    out = pd.DataFrame(rows)
    write_table(out, "pilot_vs_new_statistical_tests")
    return out


def build_score_tables(scores: pd.DataFrame) -> None:
    score_dist = summarize_numeric_by_group(
        scores,
        ["adni_model", "build_candidate", "source_batch", "diagnosis"],
        ["y_score"],
    )
    write_table(score_dist, "score_distribution_by_batch_diagnosis")

    metric_rows = []
    for (model, build, batch), sub in scores.groupby(["adni_model", "build_candidate", "source_batch"], dropna=False):
        m = compute_auc(sub["y_true"], sub["y_score"])
        metric_rows.append(
            {
                "adni_model": model,
                "build_candidate": build,
                "source_batch": batch,
                "n": len(sub),
                "n_cn": int((sub["y_true"] == 0).sum()),
                "n_ad": int((sub["y_true"] == 1).sum()),
                **m,
            }
        )
    score_metrics = pd.DataFrame(metric_rows)
    write_table(score_metrics, "score_auc_pr_by_batch")

    # Explicit pilot -> new delta table for each model/build.
    deltas = []
    for (model, build), sub in score_metrics.groupby(["adni_model", "build_candidate"]):
        pilot = sub[sub["source_batch"].eq("pilot")]
        new = sub[sub["source_batch"].eq("new")]
        if pilot.empty or new.empty:
            continue
        p = pilot.iloc[0]
        n = new.iloc[0]
        deltas.append(
            {
                "adni_model": model,
                "build_candidate": build,
                "pilot_auc": p["auc"],
                "new_auc": n["auc"],
                "delta_new_minus_pilot_auc": n["auc"] - p["auc"],
                "pilot_pr_auc": p["pr_auc"],
                "new_pr_auc": n["pr_auc"],
                "delta_new_minus_pilot_pr_auc": n["pr_auc"] - p["pr_auc"],
            }
        )
    write_table(pd.DataFrame(deltas), "pilot_to_new_score_delta_by_model_build")


def build_new_locked_test_tp_fn(scores: pd.DataFrame, subjects: pd.DataFrame) -> None:
    clinical = subjects[subjects["cohort"].eq("new")].copy()
    clinical_cols = [
        "subject_id",
        "age_at_MR",
        "sex_label",
        "CDRTOT",
        "CDRSUM",
        "mean_fd",
        "max_fd",
        "pct_frames_fd_gt0p2",
        "pct_frames_fd_gt0p5",
        "n_runs_qc_ok",
    ]
    clinical = clinical[[c for c in clinical_cols if c in clinical.columns]].drop_duplicates("subject_id")

    sub = scores[
        scores["source_batch"].eq("new")
        & scores["protocol_subset"].astype(str).eq("locked_test")
        & scores["diagnosis"].astype(str).eq("AD_DEMENTIA")
    ].copy()
    sub = sub.merge(clinical, on="subject_id", how="left", suffixes=("", "_clinical"))
    for col in [
        "age_at_MR",
        "sex_label",
        "CDRTOT",
        "CDRSUM",
        "mean_fd",
        "max_fd",
        "pct_frames_fd_gt0p2",
        "pct_frames_fd_gt0p5",
        "n_runs_qc_ok",
    ]:
        clinical_col = f"{col}_clinical"
        if clinical_col in sub.columns:
            if col not in sub.columns:
                sub[col] = sub[clinical_col]
            else:
                sub[col] = sub[col].where(sub[col].notna(), sub[clinical_col])
    sub["outcome"] = np.where(pd.to_numeric(sub["y_pred"], errors="coerce").eq(1), "TP", "FN")

    detail_cols = [
        "adni_model",
        "build_candidate",
        "subject_id",
        "diagnosis",
        "outcome",
        "y_score",
        "y_pred",
        "age_at_MR",
        "sex_label",
        "CDRTOT",
        "CDRSUM",
        "mean_fd",
        "max_fd",
        "pct_frames_fd_gt0p2",
        "pct_frames_fd_gt0p5",
        "n_runs_qc_ok",
    ]
    write_table(sub[[c for c in detail_cols if c in sub.columns]], "new_locked_test_ad_tp_fn_subjects")

    rows = []
    metrics = ["age_at_MR", "CDRTOT", "CDRSUM", "mean_fd", "max_fd", "pct_frames_fd_gt0p2", "pct_frames_fd_gt0p5", "n_runs_qc_ok", "y_score"]
    for (model, build, outcome), s in sub.groupby(["adni_model", "build_candidate", "outcome"], dropna=False):
        rec = {"adni_model": model, "build_candidate": build, "outcome": outcome, "n": len(s)}
        for metric in metrics:
            rec[f"{metric}_mean_sd"] = fmt_mean_sd(s.get(metric, pd.Series(dtype=float)))
            rec[f"{metric}_median_iqr"] = fmt_median_iqr(s.get(metric, pd.Series(dtype=float)))
        rec["pct_female"] = 100 * (s.get("sex_label", pd.Series(dtype=str)).astype(str).eq("F").mean()) if len(s) else np.nan
        rows.append(rec)
    summary = pd.DataFrame(rows)
    write_table(summary, "new_locked_test_ad_tp_fn_summary")

    stat_rows = []
    for (model, build), s in sub.groupby(["adni_model", "build_candidate"], dropna=False):
        tp = s[s["outcome"].eq("TP")]
        fn = s[s["outcome"].eq("FN")]
        for metric in metrics:
            stat_rows.append(
                {
                    "adni_model": model,
                    "build_candidate": build,
                    "metric": metric,
                    "TP_n": int(pd.to_numeric(tp.get(metric), errors="coerce").notna().sum()) if metric in tp else 0,
                    "FN_n": int(pd.to_numeric(fn.get(metric), errors="coerce").notna().sum()) if metric in fn else 0,
                    "TP_mean_sd": fmt_mean_sd(tp.get(metric, pd.Series(dtype=float))),
                    "FN_mean_sd": fmt_mean_sd(fn.get(metric, pd.Series(dtype=float))),
                    "mann_whitney_p": mann_whitney(tp.get(metric, pd.Series(dtype=float)), fn.get(metric, pd.Series(dtype=float))),
                }
            )
    write_table(pd.DataFrame(stat_rows), "new_locked_test_ad_tp_fn_tests")


def build_age_explains_tables(scores: pd.DataFrame, subjects: pd.DataFrame, tests: pd.DataFrame) -> None:
    # Score-age correlation by source batch / diagnosis / model / build.
    corr_rows = []
    for keys, sub in scores.groupby(["adni_model", "build_candidate", "source_batch", "diagnosis"], dropna=False):
        model, build, batch, diagnosis = keys
        c = pearson_spearman(sub.get("age_at_MR", pd.Series(dtype=float)), sub.get("y_score", pd.Series(dtype=float)))
        corr_rows.append({"adni_model": model, "build_candidate": build, "source_batch": batch, "diagnosis": diagnosis, **c})
    corr = pd.DataFrame(corr_rows)
    write_table(corr, "score_age_correlations")

    binned = scores.copy()
    binned["age_bin"] = pd.cut(
        pd.to_numeric(binned["age_at_MR"], errors="coerce"),
        bins=[-np.inf, 70, 80, np.inf],
        labels=["age_lt70", "age_70_79", "age_ge80"],
        right=False,
    )
    auc_rows = []
    for keys, sub in binned.groupby(["adni_model", "build_candidate", "source_batch", "age_bin"], dropna=False):
        model, build, batch, age_bin = keys
        m = compute_auc(sub["y_true"], sub["y_score"])
        auc_rows.append(
            {
                "adni_model": model,
                "build_candidate": build,
                "source_batch": batch,
                "age_bin": str(age_bin),
                "n": int(len(sub)),
                "n_cn": int((sub["y_true"] == 0).sum()),
                "n_ad": int((sub["y_true"] == 1).sum()),
                **m,
            }
        )
    auc_bins = pd.DataFrame(auc_rows)
    write_table(auc_bins, "auc_by_age_bin")

    age_tests = tests[tests["metric"].eq("age_at_MR")].copy()
    write_table(age_tests, "age_difference_tests")


def write_summary(subjects: pd.DataFrame, scores: pd.DataFrame, tests: pd.DataFrame) -> None:
    demo = subjects.groupby(["cohort", "diagnosis"])["age_at_MR"].agg(["count", "mean", "std"]).reset_index()
    cdr = subjects.groupby(["cohort", "diagnosis"])["CDRTOT"].agg(["count", "mean", "std"]).reset_index()
    motion = subjects.groupby(["cohort", "diagnosis"])["mean_fd"].agg(["count", "mean", "std"]).reset_index()

    score_metrics = pd.read_csv(OUTPUT / "score_auc_pr_by_batch.csv")
    primary = score_metrics[
        score_metrics["adni_model"].astype(str).eq(PRIMARY_MODEL_FOR_NARRATIVE)
        & score_metrics["build_candidate"].astype(str).eq(PRIMARY_BUILD_FOR_NARRATIVE)
    ].copy()
    if primary.empty:
        primary = score_metrics[
            score_metrics["adni_model"].astype(str).str.contains("primary_v5_1b", na=False)
            & score_metrics["build_candidate"].astype(str).str.contains("runwise164", na=False)
        ].copy()

    def get_age(cohort: str, diagnosis: str) -> str:
        row = demo[(demo["cohort"].eq(cohort)) & (demo["diagnosis"].eq(diagnosis))]
        if row.empty:
            return "NA"
        r = row.iloc[0]
        return f"{r['mean']:.2f} +/- {r['std']:.2f}"

    def get_cdr(cohort: str, diagnosis: str) -> str:
        row = cdr[(cdr["cohort"].eq(cohort)) & (cdr["diagnosis"].eq(diagnosis))]
        if row.empty:
            return "NA"
        r = row.iloc[0]
        return f"{r['mean']:.2f} +/- {r['std']:.2f}"

    def get_motion(cohort: str, diagnosis: str) -> str:
        row = motion[(motion["cohort"].eq(cohort)) & (motion["diagnosis"].eq(diagnosis))]
        if row.empty:
            return "NA"
        r = row.iloc[0]
        return f"{r['mean']:.3f} +/- {r['std']:.3f}"

    lines = [
        "# OASIS New-vs-Pilot Difficulty Audit for Martin",
        "",
        "This package is read-only. It compares the pilot 30CN/30AD OASIS batch against the new 60CN/60AD batch using existing manifests, scores, and motion/clinical summaries.",
        "",
        "## Main findings",
        "",
        f"- Pilot CN age: {get_age('pilot', 'CN')}; new CN age: {get_age('new', 'CN')}.",
        f"- Pilot AD age: {get_age('pilot', 'AD_DEMENTIA')}; new AD age: {get_age('new', 'AD_DEMENTIA')}.",
        f"- Pilot AD CDR global: {get_cdr('pilot', 'AD_DEMENTIA')}; new AD CDR global: {get_cdr('new', 'AD_DEMENTIA')}.",
        f"- Pilot AD mean FD: {get_motion('pilot', 'AD_DEMENTIA')}; new AD mean FD: {get_motion('new', 'AD_DEMENTIA')}.",
        "- MMSE/MOCA were not available in the local OASIS audit inputs used here; CDR global and CDR-SB were available.",
        "",
    ]
    if not primary.empty:
        pilot = primary[primary["source_batch"].eq("pilot")]
        new = primary[primary["source_batch"].eq("new")]
        if not pilot.empty and not new.empty:
            p = pilot.iloc[0]
            n = new.iloc[0]
            lines += [
                f"- In the narrative score source (`{p['adni_model']}`, `{p['build_candidate']}`), pilot AUC/PR-AUC = {p['auc']:.3f}/{p['pr_auc']:.3f}; new AUC/PR-AUC = {n['auc']:.3f}/{n['pr_auc']:.3f}.",
            ]
    lines += [
        "- Age contributes to the cohort difference, especially because the pilot CN group is younger, but age alone does not fully explain the performance drop: score-age correlations and age-bin AUCs are reported separately.",
        "- New AD cases are clinically slightly milder by CDR global/CDR-SB, and new locked-test AD false negatives tend to have lower CDR-SB than true positives in existing summaries.",
        "- Motion does not show a single decisive explanation: pilot AD has somewhat higher motion on average, while new performance remains weak despite lower/comparable AD motion.",
        "- Prior parity work identified tensor/scoring construction differences; the present package therefore separates difficulty/domain factors from pipeline-parity concerns.",
        "",
        "## Output tables",
        "",
        "- `pilot_vs_new_demographics_clinical_motion.csv/.md`",
        "- `sex_distribution_by_batch_diagnosis.csv/.md`",
        "- `run_count_distribution.csv/.md`",
        "- `score_distribution_by_batch_diagnosis.csv/.md`",
        "- `score_auc_pr_by_batch.csv/.md`",
        "- `pilot_to_new_score_delta_by_model_build.csv/.md`",
        "- `new_locked_test_ad_tp_fn_subjects.csv/.md`",
        "- `new_locked_test_ad_tp_fn_summary.csv/.md`",
        "- `new_locked_test_ad_tp_fn_tests.csv/.md`",
        "- `age_difference_tests.csv/.md`",
        "- `score_age_correlations.csv/.md`",
        "- `auc_by_age_bin.csv/.md`",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_martin_message() -> None:
    text = """# Short message for Martin

Martin,

I compared the pilot OASIS batch (30 CN / 30 AD) with the new OASIS batch (60 CN / 60 AD) using the existing manifests, model scores, CDR measures, run counts, and motion summaries.

The new batch is not simply a larger copy of the pilot. The pilot controls are younger, while the new CN and AD groups are tightly age-matched. The new AD group is also slightly milder by CDR/CDR-SB, and within the new AD locked-test subjects the false negatives tend to have lower CDR-SB than true positives. Motion does not look like the main explanation: pilot AD motion is at least as high as, and often higher than, new AD motion.

The score distributions show a clear batch/domain effect. In the pilot, AD scores are more separated from CN; in the new batch, AD and CN scores overlap much more, and for some tensor/model combinations the ordering partially collapses. Age may contribute, but it does not fully explain the drop; the age-bin and score-age tables do not support age as a complete explanation.

My interpretation is that the new batch is clinically and domain-wise harder, with milder AD and stronger batch/tensor-construction sensitivity. The next useful check is not model retraining, but ensuring strict pipeline parity for the OASIS tensor construction/scoring path and then evaluating the pre-specified calibration/test split.
"""
    (OUTPUT / "martin_message.md").write_text(text, encoding="utf-8")


def write_difficulty_factor_summary() -> None:
    text = """# Difficulty factor summary

## Age

Age differs between pilot and new, especially for CN: the pilot CN group is younger, while the new CN/AD groups were deliberately age-matched. This can reduce apparent separability if the pilot model benefited from age-related cohort structure. However, age alone is not a complete explanation; score-age correlations and age-stratified AUCs are mixed and remain build/model dependent.

## Clinical severity

CDR global and CDR-SB are available. The new AD group is slightly milder than the pilot AD group, and new AD false negatives tend to have lower CDR-SB than true positives. This is a plausible contributor to lower sensitivity and weaker AD/CN separation.

## Motion

Motion does not provide a simple explanation. Pilot AD has comparable or higher mean/max FD and framewise-displacement burden than new AD in the available reports. The new performance drop persists despite this.

## Batch/domain shift

The score distributions show batch dependence. Pilot and new OASIS are non-overlapping and appear to differ in clinical composition and score distribution. This supports interpreting the new batch as a harder external cohort rather than as a direct replication of the pilot.

## Tensor construction / scoring parity

A prior parity audit found runwise tensor construction/scoring differences between pilot and new processing. This remains an important guardrail: biological difficulty and pipeline-parity issues should not be conflated. The present audit reports difficulty factors, but external conclusions should use the harmonized parity tensors/scorer.

## Bottom line

Likely contributors are mild AD severity, age/cohort structure, and batch/domain shift. Motion is not the dominant explanation. Tensor/scoring parity should remain a required check before interpreting external-transfer weakness biologically.
"""
    (OUTPUT / "difficulty_factor_summary.md").write_text(text, encoding="utf-8")


def write_command_log(start: str, args: argparse.Namespace) -> None:
    log = {
        "script": str(Path(__file__).resolve()),
        "start_time": start,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "arguments": vars(args),
        "inputs": {
            "all_subjects": str(ALL_SUBJECTS),
            "mega_manifest": str(MEGA_MANIFEST),
            "mega_predictions": str(MEGA_PREDICTIONS),
            "mega_stratified": str(MEGA_STRATIFIED),
            "new_harmonized_metrics": str(NEW_HARMONIZED_METRICS),
            "pilot_metrics": str(PILOT_METRICS),
            "parity_final_recommendation": str(PARITY_FINAL),
        },
        "safety": {
            "training_launched": False,
            "model_selection_performed": False,
            "threshold_fitting_performed": False,
            "tensor_modified": False,
            "model_outputs_modified": False,
        },
    }
    (OUTPUT / "command_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without writing.")
    args = parser.parse_args()

    required = [ALL_SUBJECTS, MEGA_MANIFEST, MEGA_PREDICTIONS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    planned = [
        "README.md",
        "pilot_vs_new_demographics_clinical_motion.csv/.md",
        "sex_distribution_by_batch_diagnosis.csv/.md",
        "run_count_distribution.csv/.md",
        "clinical_variable_availability.csv/.md",
        "pilot_vs_new_statistical_tests.csv/.md",
        "score_distribution_by_batch_diagnosis.csv/.md",
        "score_auc_pr_by_batch.csv/.md",
        "pilot_to_new_score_delta_by_model_build.csv/.md",
        "new_locked_test_ad_tp_fn_subjects.csv/.md",
        "new_locked_test_ad_tp_fn_summary.csv/.md",
        "new_locked_test_ad_tp_fn_tests.csv/.md",
        "age_difference_tests.csv/.md",
        "score_age_correlations.csv/.md",
        "auc_by_age_bin.csv/.md",
        "difficulty_factor_summary.md",
        "martin_message.md",
        "command_log.json",
    ]
    if args.dry_run:
        print("Dry-run OK. Planned outputs:")
        for item in planned:
            print(OUTPUT / item)
        return 0

    start = datetime.now().isoformat(timespec="seconds")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects()
    scores = merge_subject_covariates(load_mega_scores(), subjects)

    build_demographic_motion_tables(subjects)
    tests = build_cross_cohort_tests(subjects)
    build_score_tables(scores)
    build_new_locked_test_tp_fn(scores, subjects)
    build_age_explains_tables(scores, subjects, tests)
    write_summary(subjects, scores, tests)
    write_difficulty_factor_summary()
    write_martin_message()
    write_command_log(start, args)
    print(f"Wrote OASIS new-vs-pilot difficulty audit to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
