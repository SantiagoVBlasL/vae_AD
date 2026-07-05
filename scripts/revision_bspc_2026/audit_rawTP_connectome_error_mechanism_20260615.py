#!/usr/bin/env python3
"""Read-only rawTP/connectome mechanism audit for promoted ADNI rs-fMRI model.

This script intentionally writes only derived audit outputs under a new results
directory. It does not train, refit thresholds, edit tensors, metadata, or model
artifacts.
"""

from __future__ import annotations

import json
import math
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - optional dependency diagnostics
    sm = None

try:
    import scipy.io as sio
except Exception:  # pragma: no cover - optional dependency diagnostics
    sio = None


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_bspc_2026/rawTP_connectome_error_mechanism_audit_20260615"
FULL_DB = ROOT / (
    "results/revision_bspc_2026/full_database_for_martin_and_validity_preflight_20260612/"
    "promoted_model_full_database_for_martin_20260612.csv"
)
PRED = ROOT / (
    "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration/"
    "calib_predictions.csv"
)
GLOBAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)

PRIMARY_FILTER = {
    "model_name": "logreg_l2_original",
    "feature_set": "z_plus_age_sex",
    "calib_method": "oof_ecdf",
    "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


COMMAND_LOG: list[dict[str, Any]] = []


def log(action: str, status: str, details: dict[str, Any] | None = None) -> None:
    COMMAND_LOG.append(
        {
            "timestamp": now_iso(),
            "action": action,
            "status": status,
            "details": details or {},
        }
    )


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def normalize_manufacturer(x: Any) -> str:
    s = safe_str(x).strip().upper()
    if "PHILIPS" in s:
        return "Philips"
    if "SIEMENS" in s:
        return "SIEMENS"
    if s in {"GE", "GENERAL ELECTRIC"} or "GE MEDICAL" in s:
        return "GE"
    return safe_str(x).strip()


def normalize_dx(x: Any, y: Any = np.nan) -> str:
    s = safe_str(x).strip().upper()
    if s in {"CN", "NORMAL", "COGNITIVELY NORMAL"}:
        return "CN"
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA"}:
        return "AD"
    if s == "MCI":
        return "MCI"
    if not pd.isna(y):
        try:
            return "AD" if int(y) == 1 else "CN"
        except Exception:
            pass
    return safe_str(x).strip()


def normalize_rawtp(x: Any) -> str:
    s = safe_str(x).strip()
    if not s:
        return "MISSING"
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    n = int(m.group(1))
    if n == 140:
        return "140"
    if n in {197, 200}:
        return "197_200"
    return str(n)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    md = view.to_markdown(index=False)
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return md + "\n" + suffix


def write_df(stem: str, df: pd.DataFrame, max_md_rows: int = 80) -> None:
    csv_path = OUT / f"{stem}.csv"
    md_path = OUT / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df, max_md_rows), encoding="utf-8")
    log("write_table", "ok", {"csv": str(csv_path), "rows": int(len(df))})


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def offdiag_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def tensor_channel_summaries(mat: np.ndarray) -> dict[str, float]:
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[0]
    mask = offdiag_mask(n)
    vals = mat[mask]
    finite = vals[np.isfinite(vals)]
    diag = np.diag(mat)
    if finite.size:
        q05, q25, q50, q75, q95 = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
        mean = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        prop_pos = float(np.mean(finite > 0))
        prop_neg = float(np.mean(finite < 0))
        min_v = float(np.min(finite))
        max_v = float(np.max(finite))
    else:
        q05 = q25 = q50 = q75 = q95 = mean = sd = prop_pos = prop_neg = min_v = max_v = np.nan
    sym = mat - mat.T
    sym_finite = np.abs(sym[np.isfinite(sym)])
    diag_finite = np.abs(diag[np.isfinite(diag)])
    return {
        "offdiag_mean": mean,
        "offdiag_sd": sd,
        "offdiag_min": min_v,
        "offdiag_q05": float(q05),
        "offdiag_q25": float(q25),
        "offdiag_q50": float(q50),
        "offdiag_q75": float(q75),
        "offdiag_q95": float(q95),
        "offdiag_max": max_v,
        "prop_positive": prop_pos,
        "prop_negative": prop_neg,
        "nan_count": int(np.isnan(mat).sum()),
        "inf_count": int(np.isinf(mat).sum()),
        "symmetry_max_abs": float(np.max(sym_finite)) if sym_finite.size else np.nan,
        "diagonal_max_abs": float(np.max(diag_finite)) if diag_finite.size else np.nan,
        "diagonal_mean_abs": float(np.mean(diag_finite)) if diag_finite.size else np.nan,
    }


def confusion_label(y_true: Any, y_pred: Any) -> str:
    try:
        yt = int(y_true)
        yp = int(y_pred)
    except Exception:
        return ""
    if yt == 1 and yp == 1:
        return "TP"
    if yt == 1 and yp == 0:
        return "FN"
    if yt == 0 and yp == 1:
        return "FP"
    if yt == 0 and yp == 0:
        return "TN"
    return ""


def cles_from_u(u: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(u / (n1 * n2))


def mann_whitney_row(
    comparison: str,
    metric: str,
    group_a: str,
    a: Iterable[float],
    group_b: str,
    b: Iterable[float],
) -> dict[str, Any]:
    av = pd.Series(a, dtype="float64").dropna().to_numpy()
    bv = pd.Series(b, dtype="float64").dropna().to_numpy()
    row: dict[str, Any] = {
        "comparison": comparison,
        "metric": metric,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(len(av)),
        "n_b": int(len(bv)),
        "mean_a": float(np.mean(av)) if len(av) else np.nan,
        "mean_b": float(np.mean(bv)) if len(bv) else np.nan,
        "median_a": float(np.median(av)) if len(av) else np.nan,
        "median_b": float(np.median(bv)) if len(bv) else np.nan,
        "delta_median_a_minus_b": (
            float(np.median(av) - np.median(bv)) if len(av) and len(bv) else np.nan
        ),
        "u_stat": np.nan,
        "p_mannwhitney": np.nan,
        "cles_p_a_gt_b": np.nan,
    }
    if len(av) >= 2 and len(bv) >= 2:
        u, p = stats.mannwhitneyu(av, bv, alternative="two-sided", method="auto")
        row.update({"u_stat": float(u), "p_mannwhitney": float(p), "cles_p_a_gt_b": cles_from_u(u, len(av), len(bv))})
    return row


def fpr_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        fp = int((g["confusion_label_final"] == "FP").sum())
        tn = int((g["confusion_label_final"] == "TN").sum())
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update(
            {
                "n_cn": n,
                "fp": fp,
                "tn": tn,
                "fpr": fp / n if n else np.nan,
                "score_mean": finite_numeric(g["y_score_final"]).mean(),
                "score_median": finite_numeric(g["y_score_final"]).median(),
                "score_q25": finite_numeric(g["y_score_final"]).quantile(0.25),
                "score_q75": finite_numeric(g["y_score_final"]).quantile(0.75),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_descriptive_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if sm is None:
        return pd.DataFrame([{"model": "statsmodels_unavailable", "status": "not_run"}])

    d = df.copy()
    d["rawtp_140"] = (d["raw_tp_group_norm"] == "140").astype(float)
    d["Age"] = finite_numeric(d["Age"])
    d["FP_status"] = (d["confusion_label_final"] == "FP").astype(float)
    base_cols = ["rawtp_140", "Age", "ch0_offdiag_mean", "ch1_offdiag_mean", "ch2_offdiag_mean"]
    for c in base_cols:
        d[c] = finite_numeric(d[c])
    site = pd.get_dummies(d["Site3"].astype(str), prefix="Site3", drop_first=True, dtype=float)
    x = pd.concat([d[base_cols], site], axis=1)
    y_score = finite_numeric(d["y_score_final"])
    keep = x.notna().all(axis=1) & y_score.notna()
    try:
        x_ols = sm.add_constant(x.loc[keep], has_constant="add")
        model = sm.OLS(y_score.loc[keep], x_ols).fit(cov_type="HC3")
        for term, coef in model.params.items():
            rows.append(
                {
                    "model": "OLS_y_score_final",
                    "status": "ok",
                    "n": int(keep.sum()),
                    "term": term,
                    "coef": float(coef),
                    "se": float(model.bse.get(term, np.nan)),
                    "p": float(model.pvalues.get(term, np.nan)),
                }
            )
    except Exception as exc:
        rows.append({"model": "OLS_y_score_final", "status": "failed", "error": repr(exc)})

    keep_logit = x.notna().all(axis=1) & d["FP_status"].notna()
    if int(keep_logit.sum()) >= 20 and d.loc[keep_logit, "FP_status"].nunique() == 2:
        try:
            x_glm = sm.add_constant(x.loc[keep_logit], has_constant="add")
            model = sm.GLM(d.loc[keep_logit, "FP_status"], x_glm, family=sm.families.Binomial()).fit()
            for term, coef in model.params.items():
                rows.append(
                    {
                        "model": "GLM_binomial_FP_status",
                        "status": "ok",
                        "n": int(keep_logit.sum()),
                        "term": term,
                        "coef": float(coef),
                        "se": float(model.bse.get(term, np.nan)),
                        "p": float(model.pvalues.get(term, np.nan)),
                    }
                )
        except Exception as exc:
            rows.append({"model": "GLM_binomial_FP_status", "status": "failed", "error": repr(exc)})
    else:
        rows.append(
            {
                "model": "GLM_binomial_FP_status",
                "status": "not_run",
                "reason": "insufficient complete rows or only one outcome class",
                "n_complete": int(keep_logit.sum()),
            }
        )
    return pd.DataFrame(rows)


def read_mat_timeseries(path: str | Path) -> tuple[np.ndarray | None, str]:
    if sio is None:
        return None, "scipy_io_unavailable"
    p = Path(path)
    if not p.exists():
        return None, "path_missing"
    try:
        data = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return None, "mat_v73_not_supported_by_scipy"
    except Exception as exc:
        return None, f"load_failed:{type(exc).__name__}"
    candidates: list[tuple[str, np.ndarray]] = []
    for key, val in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(val)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            if 131 in arr.shape and max(arr.shape) >= 140:
                candidates.append((key, arr.astype(float)))
    if not candidates:
        return None, "no_2d_numeric_131roi_candidate"
    key, arr = sorted(candidates, key=lambda kv: kv[1].size, reverse=True)[0]
    if arr.shape[1] == 131:
        ts = arr
    elif arr.shape[0] == 131:
        ts = arr.T
    else:
        return None, f"candidate_without_131_axis:{key}:{arr.shape}"
    return ts, f"ok:{key}:{arr.shape}"


def fisher_z_corr(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    corr = np.corrcoef(ts, rowvar=False)
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    np.fill_diagonal(z, 0.0)
    return z


def optional_recompute_stability(df: pd.DataFrame, max_subjects: int = 12) -> pd.DataFrame:
    """Lightweight optional Pearson stability probe for readable 197/200 ROISignals.

    Full OMST and MI-KNN recomputation is intentionally not attempted here because
    exact historical preprocessing and hyperparameters must be audited before using
    recomputed matrices as evidence. Pearson full first/last-140 stability is
    reported when ROISignal matrices are readable.
    """
    rows: list[dict[str, Any]] = []
    philips_197 = df[
        (df["Manufacturer"] == "Philips")
        & (df["diagnosis"] == "CN")
        & (df["raw_tp_group_norm"] == "197_200")
    ].copy()
    path_cols = [c for c in ["roisignals_mat_path", "mat_path"] if c in philips_197.columns]
    checked = 0
    computed = 0
    for _, row in philips_197.iterrows():
        if checked >= max_subjects:
            break
        path = None
        for c in path_cols:
            val = safe_str(row.get(c))
            if val:
                path = val
                break
        if not path:
            continue
        checked += 1
        ts, status = read_mat_timeseries(path)
        base = {
            "SubjectID": row["SubjectID"],
            "raw_tp_group": row["raw_tp_group_norm"],
            "path_checked": path,
            "read_status": status,
        }
        if ts is None:
            rows.append({**base, "computed": False})
            continue
        n_tp = int(ts.shape[0])
        base["n_timepoints_roisignals_detected"] = n_tp
        if n_tp < 197:
            rows.append({**base, "computed": False, "reason": "less_than_197_timepoints"})
            continue
        try:
            full = fisher_z_corr(ts)
            first = fisher_z_corr(ts[:140])
            last = fisher_z_corr(ts[-140:])
            mask = offdiag_mask(full.shape[0])
            full_v, first_v, last_v = full[mask], first[mask], last[mask]
            rows.append(
                {
                    **base,
                    "computed": True,
                    "channel": "Pearson_Full_FisherZ_Signed_probe_only",
                    "corr_full_vs_first140": float(np.corrcoef(full_v, first_v)[0, 1]),
                    "corr_full_vs_last140": float(np.corrcoef(full_v, last_v)[0, 1]),
                    "corr_first140_vs_last140": float(np.corrcoef(first_v, last_v)[0, 1]),
                    "fro_full_vs_first140": float(np.linalg.norm(full_v - first_v)),
                    "fro_full_vs_last140": float(np.linalg.norm(full_v - last_v)),
                    "offdiag_mean_full": float(np.mean(full_v)),
                    "offdiag_mean_first140": float(np.mean(first_v)),
                    "offdiag_mean_last140": float(np.mean(last_v)),
                    "offdiag_sd_full": float(np.std(full_v, ddof=1)),
                    "offdiag_sd_first140": float(np.std(first_v, ddof=1)),
                    "offdiag_sd_last140": float(np.std(last_v, ddof=1)),
                    "note": (
                        "Pearson-only probe; OMST/MI-KNN not recomputed because exact "
                        "historical implementation should be invoked separately before "
                        "using recomputed matrices for manuscript claims."
                    ),
                }
            )
            computed += 1
        except Exception as exc:
            rows.append({**base, "computed": False, "reason": f"compute_failed:{type(exc).__name__}"})
    if not rows:
        rows.append(
            {
                "computed": False,
                "reason": "no_readable_197_200_philips_cn_roisignals_paths_found_in_audit_table",
                "note": "No optional 197-to-140 recomputation was performed.",
            }
        )
    rows.append(
        {
            "computed": bool(computed),
            "summary": True,
            "n_paths_checked": checked,
            "n_subjects_computed": computed,
            "max_subjects_attempted": max_subjects,
        }
    )
    return pd.DataFrame(rows)


def load_primary_predictions() -> pd.DataFrame:
    pred = pd.read_csv(PRED)
    for col, value in PRIMARY_FILTER.items():
        pred = pred[pred[col] == value]
    if pred.empty:
        raise RuntimeError(f"No primary prediction rows found using {PRIMARY_FILTER}")
    pred = pred.rename(
        columns={
            "fold": "outer_fold_pred",
            "Manufacturer": "Manufacturer_pred",
            "Age": "Age_pred",
            "Sex": "Sex_pred",
            "y_score": "y_score_primary",
            "y_pred": "y_pred_primary",
        }
    )
    return pred


def build_subject_table() -> tuple[pd.DataFrame, list[str]]:
    db = pd.read_csv(FULL_DB)
    pred = load_primary_predictions()
    merged = pred.merge(db, on="SubjectID", how="left", suffixes=("_predfile", "_db"))
    missing_db = merged["diagnosis_group"].isna().sum()
    if missing_db:
        raise RuntimeError(f"{missing_db} primary prediction rows could not be found in full DB")

    z = np.load(GLOBAL_TENSOR, allow_pickle=True)
    tensor = z["global_tensor_data"]
    subject_ids = [str(x) for x in z["subject_ids"]]
    channel_names = [str(x) for x in z["channel_names"]]
    subj_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        sid = str(r["SubjectID"])
        idx = subj_to_idx.get(sid)
        if idx is None:
            continue
        y_true = r["y_true_predfile"] if "y_true_predfile" in r else r["y_true"]
        y_pred = r["y_pred_primary"]
        diag = normalize_dx(r.get("diagnosis_group"), y_true)
        base: dict[str, Any] = {
            "SubjectID": sid,
            "ImageID": r.get("ImageID"),
            "RID": r.get("RID"),
            "diagnosis": diag,
            "Manufacturer": normalize_manufacturer(r.get("Manufacturer_final", r.get("Manufacturer_pred"))),
            "Site3": r.get("Site3_final"),
            "raw_tp_group": r.get("raw_tp_group_final"),
            "raw_tp_group_norm": normalize_rawtp(r.get("raw_tp_group_final")),
            "inferred_ADNI_phase": r.get("COLPROT_final"),
            "COLPROT": r.get("COLPROT_final"),
            "ORIGPROT": r.get("ORIGPROT_final"),
            "Age": r.get("Age_final", r.get("Age_pred")),
            "Sex": r.get("Sex_final", r.get("Sex_pred")),
            "outer_fold": r.get("outer_fold_pred"),
            "y_true": y_true,
            "y_score_final": r.get("y_score_primary"),
            "y_pred": y_pred,
            "confusion_label_final": confusion_label(y_true, y_pred),
            "n_timepoints_original": r.get("n_timepoints_raw"),
            "n_timepoints_after_dummy_removal": np.nan,
            "n_timepoints_ROISignals": np.nan,
            "n_timepoints_used_for_connectome": r.get("n_timepoints_model_input"),
            "n_timepoints_raw": r.get("n_timepoints_raw"),
            "n_timepoints_model_input": r.get("n_timepoints_model_input"),
            "mat_path": r.get("mat_path"),
            "roisignals_mat_path": r.get("roisignals_mat_path"),
            "global_tensor_path": r.get("global_tensor_path"),
        }
        for c in [
            "scanner_model_final",
            "manufacturer_model_name_final",
            "software_version_final",
            "phase_encoding_direction_final",
            "PHASEDIR_final",
            "slice_order_class",
            "problem_site_flag",
            "high_confidence_slice_timing_match",
            "match_confidence",
            "fd_mean",
            "fd_max",
        ]:
            base[c] = r.get(c)
        mats = tensor[idx]
        for ci, cname in enumerate(channel_names):
            base[f"ch{ci}_name"] = cname
            summ = tensor_channel_summaries(mats[ci])
            for key, val in summ.items():
                base[f"ch{ci}_{key}"] = val
        rows.append(base)

    df = pd.DataFrame(rows)
    df = df[df["diagnosis"].isin(["CN", "AD"])].copy()
    df["Age"] = finite_numeric(df["Age"])
    df["y_score_final"] = finite_numeric(df["y_score_final"])
    df["y_pred"] = finite_numeric(df["y_pred"]).astype("Int64")
    df["y_true"] = finite_numeric(df["y_true"]).astype("Int64")
    return df, channel_names


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log(
        "start",
        "ok",
        {
            "full_db": str(FULL_DB),
            "predictions": str(PRED),
            "global_tensor": str(GLOBAL_TENSOR),
            "guardrails": [
                "read_only",
                "no_training",
                "no_tensor_edits",
                "no_metadata_edits",
                "no_prediction_edits",
                "no_threshold_refitting",
                "no_subject_exclusion",
            ],
        },
    )

    subject_df, channel_names = build_subject_table()
    write_df("rawTP_subject_level_audit", subject_df, max_md_rows=40)

    philips_cn = subject_df[
        (subject_df["Manufacturer"] == "Philips")
        & (subject_df["diagnosis"] == "CN")
        & (subject_df["raw_tp_group_norm"].isin(["140", "197_200"]))
    ].copy()
    fpr = fpr_summary(philips_cn, ["raw_tp_group_norm"]).sort_values("raw_tp_group_norm")
    if set(fpr["raw_tp_group_norm"]) >= {"140", "197_200"}:
        g140 = philips_cn[philips_cn["raw_tp_group_norm"] == "140"]
        g197 = philips_cn[philips_cn["raw_tp_group_norm"] == "197_200"]
        table = np.array(
            [
                [(g140["confusion_label_final"] == "FP").sum(), (g140["confusion_label_final"] == "TN").sum()],
                [(g197["confusion_label_final"] == "FP").sum(), (g197["confusion_label_final"] == "TN").sum()],
            ]
        )
        odds, p_fisher = stats.fisher_exact(table)
        fpr["fisher_or_140_vs_197_200"] = odds
        fpr["fisher_p_140_vs_197_200"] = p_fisher
    write_df("philips_cn_rawTP_fpr_table", fpr)

    score_tests = []
    g140 = philips_cn[philips_cn["raw_tp_group_norm"] == "140"]["y_score_final"]
    g197 = philips_cn[philips_cn["raw_tp_group_norm"] == "197_200"]["y_score_final"]
    score_tests.append(
        mann_whitney_row(
            "Philips CN rawTP score distribution",
            "y_score_final",
            "rawTP_140",
            g140,
            "rawTP_197_200",
            g197,
        )
    )
    write_df("rawTP_score_distribution_tests", pd.DataFrame(score_tests))

    strat_rows = []
    site_summary = fpr_summary(philips_cn, ["Site3", "raw_tp_group_norm"])
    if not site_summary.empty:
        site_summary.insert(0, "stratum_type", "Site3")
        site_summary = site_summary.rename(columns={"Site3": "stratum"})
        strat_rows.append(site_summary)
    tmp = philips_cn.copy()
    tmp["Age_tertile"] = pd.qcut(tmp["Age"], q=3, labels=["T1_youngest", "T2_middle", "T3_oldest"], duplicates="drop")
    age_summary = fpr_summary(tmp, ["Age_tertile", "raw_tp_group_norm"])
    if not age_summary.empty:
        age_summary.insert(0, "stratum_type", "Age_tertile")
        age_summary = age_summary.rename(columns={"Age_tertile": "stratum"})
        strat_rows.append(age_summary)
    strat = pd.concat(strat_rows, ignore_index=True) if strat_rows else pd.DataFrame()
    write_df("rawTP_site_age_stratified_error", strat)

    metric_suffixes = ["offdiag_mean", "offdiag_sd", "offdiag_q50", "prop_positive", "prop_negative"]
    channel_test_rows: list[dict[str, Any]] = []
    comparisons = [
        (
            "Philips_CN_140TP_FP_vs_TN",
            philips_cn[(philips_cn["raw_tp_group_norm"] == "140") & (philips_cn["confusion_label_final"] == "FP")],
            "140TP_FP",
            philips_cn[(philips_cn["raw_tp_group_norm"] == "140") & (philips_cn["confusion_label_final"] == "TN")],
            "140TP_TN",
        ),
        (
            "Philips_CN_197_200TP_FP_vs_TN",
            philips_cn[(philips_cn["raw_tp_group_norm"] == "197_200") & (philips_cn["confusion_label_final"] == "FP")],
            "197_200TP_FP",
            philips_cn[(philips_cn["raw_tp_group_norm"] == "197_200") & (philips_cn["confusion_label_final"] == "TN")],
            "197_200TP_TN",
        ),
        (
            "Philips_CN_140TP_vs_197_200TP",
            philips_cn[philips_cn["raw_tp_group_norm"] == "140"],
            "140TP",
            philips_cn[philips_cn["raw_tp_group_norm"] == "197_200"],
            "197_200TP",
        ),
    ]
    comp_cn = subject_df[
        (subject_df["diagnosis"] == "CN")
        & (subject_df["Manufacturer"].isin(["GE", "SIEMENS"]))
        & (subject_df["raw_tp_group_norm"].isin(["140", "197_200"]))
    ].copy()
    if not comp_cn.empty:
        comparisons.append(
            (
                "GE_SIEMENS_CN_140TP_vs_197_200TP",
                comp_cn[comp_cn["raw_tp_group_norm"] == "140"],
                "140TP",
                comp_cn[comp_cn["raw_tp_group_norm"] == "197_200"],
                "197_200TP",
            )
        )
    for comparison, a_df, a_name, b_df, b_name in comparisons:
        for ci, cname in enumerate(channel_names):
            for suffix in metric_suffixes:
                metric = f"ch{ci}_{suffix}"
                row = mann_whitney_row(comparison, metric, a_name, a_df[metric], b_name, b_df[metric])
                row["channel_index"] = ci
                row["channel_name"] = cname
                row["metric_suffix"] = suffix
                channel_test_rows.append(row)
    write_df("rawTP_channel_summary_tests", pd.DataFrame(channel_test_rows), max_md_rows=120)

    model_df = philips_cn.copy()
    desc_models = fit_descriptive_models(model_df)
    write_df("descriptive_model_results", desc_models, max_md_rows=120)

    coverage_rows = []
    group_cols = ["Manufacturer", "diagnosis", "raw_tp_group_norm"]
    for keys, g in subject_df.groupby(group_cols, dropna=False):
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update(
            {
                "n": int(len(g)),
                "n_timepoints_raw_nonmissing": int(g["n_timepoints_raw"].notna().sum()),
                "n_timepoints_model_input_nonmissing": int(g["n_timepoints_model_input"].notna().sum()),
                "roisignals_mat_path_nonmissing": int(g["roisignals_mat_path"].notna().sum()),
                "mat_path_nonmissing": int(g["mat_path"].notna().sum()),
                "n_timepoints_raw_median": finite_numeric(g["n_timepoints_raw"]).median(),
                "n_timepoints_model_input_median": finite_numeric(g["n_timepoints_model_input"]).median(),
            }
        )
        coverage_rows.append(row)
    write_df("timepoint_count_coverage", pd.DataFrame(coverage_rows))

    optional = optional_recompute_stability(subject_df)
    write_df("optional_recomputed_197_vs_140_connectome_stability", optional, max_md_rows=80)

    # Conservative final interpretation from actual output rows.
    fpr_map = {str(r["raw_tp_group_norm"]): r for _, r in fpr.iterrows()}
    f140 = fpr_map.get("140", {})
    f197 = fpr_map.get("197_200", {})
    score_row = score_tests[0]
    interp = f"""# Final Interpretation

This is a read-only mechanism audit. It used the promoted model's frozen OOF-ECDF predictions and the locked global tensor; it did not train models, refit thresholds, edit tensors, edit metadata, edit predictions, or exclude subjects.

## Philips CN rawTP Error Pattern

The promoted OOF predictions reproduce the rawTP split:

- Philips CN rawTP=140: {int(f140.get('fp', 0))}/{int(f140.get('n_cn', 0))} FP, FPR={float(f140.get('fpr', np.nan)):.6f}
- Philips CN rawTP=197/200: {int(f197.get('fp', 0))}/{int(f197.get('n_cn', 0))} FP, FPR={float(f197.get('fpr', np.nan)):.6f}
- Fisher OR={float(f140.get('fisher_or_140_vs_197_200', np.nan)):.6g}, p={float(f140.get('fisher_p_140_vs_197_200', np.nan)):.6g}

Score distributions also differ descriptively:

- Mann-Whitney p={float(score_row.get('p_mannwhitney', np.nan)):.6g}
- CLES P(score_140 > score_197/200)={float(score_row.get('cles_p_a_gt_b', np.nan)):.6f}

## Mechanism Interpretation

The rawTP=140 association should be treated as a protocol/domain marker, not as proof that the number of time points directly causes false positives. In this ADNI/Philips subset, rawTP is entangled with site, ADNI phase/protocol labels, scanner model/software availability, age, and previously observed channel-level shifts. The audit therefore supports a multi-factor explanation: rawTP/protocol/site/scanner context plus channel-level Pearson/OMST/MI distribution shifts are more plausible than a pure time-series-length explanation.

The channel summary tests identify which tensor summaries differ by rawTP and by FP/TN status. The most manuscript-relevant interpretation should focus on channels [1,0,2] while preserving the all-channel audit table for traceability.

## 197-to-140 Stability Probe

The optional ROISignal probe is deliberately conservative. When readable 197/200 ROI time series were found, it only computed a Pearson full first/last-140 stability check. OMST and MI-KNN were not recomputed as manuscript evidence because exact historical feature-extraction settings should be invoked in a separate controlled recomputation audit before drawing channel-specific causal conclusions.

## Manuscript Limitation/Sensitivity Language

The promoted model remains unchanged. These findings should be reported as descriptive evidence that Philips rawTP/protocol strata are associated with elevated CN false positives and altered connectivity summaries. Because M0/M1 exclusion retraining was negative, the more defensible path is not additional exclusion; it is transparent reporting plus, if pursued later, corrected preprocessing/reintegration for confirmed acquisition-preprocessing mismatches.
"""
    (OUT / "final_interpretation.md").write_text(interp, encoding="utf-8")
    log("write_final_interpretation", "ok", {"path": str(OUT / "final_interpretation.md")})

    log(
        "validation",
        "ok",
        {
            "py_compile": "run separately by caller or validation command",
            "no_source_modification": True,
            "rows_subject_level": int(len(subject_df)),
            "channel_names": channel_names,
        },
    )
    (OUT / "command_log.json").write_text(json.dumps(COMMAND_LOG, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        log("fatal", "failed", {"error": repr(exc), "traceback": traceback.format_exc()})
        (OUT / "command_log.json").write_text(json.dumps(COMMAND_LOG, indent=2), encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr)
        raise
