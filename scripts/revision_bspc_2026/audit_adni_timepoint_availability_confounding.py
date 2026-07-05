#!/usr/bin/env python3
"""Read-only ADNI timepoint availability and confounding audit.

This script audits the ROI-signal timepoint counts available before the locked
140-TR homogenization step for the v5.1b ADNI model. It only reads existing
metadata/QC/fold files and writes a new audit package.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_timepoint_availability_confounding_audit"

TRAINING_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
LOCKED_RUN = ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
PREDICTIONS = LOCKED_RUN / "classifier_only_readout/classifier_sweep_predictions.csv"

BASE_QC = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors/full_extraction_qc.csv"
)
INCREMENTAL_QC_BY_BATCH = {
    "20260513_bandpass_batch1": Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_v5_1_batch20260513_incremental_no_pybandpass/incremental_qc_summary.csv"
    ),
    "20260514_bandpass_batch2": Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_v5_1_batch20260514_incremental_no_pybandpass/incremental_qc_summary.csv"
    ),
    "20260514b_bandpass_batch3": Path(
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_v5_1_batch20260514b_incremental_no_pybandpass/incremental_qc_summary.csv"
    ),
}

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
LOCKED_N_TR = 140


def parse_shape(shape_value: Any) -> tuple[float, float]:
    match = re.search(r"\((\d+)\s*,\s*(\d+)\)", str(shape_value))
    if not match:
        return (np.nan, np.nan)
    return (float(match.group(1)), float(match.group(2)))


def sitecode(subject_id: Any) -> str:
    match = re.match(r"^(\d{3})_S_\d{4}$", str(subject_id))
    return match.group(1) if match else ""


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def flatten_subjects(values: Iterable[Any]) -> str:
    vals = [str(v) for v in values if pd.notna(v)]
    return ";".join(sorted(set(vals), key=vals.index))


def to_markdown(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.write_text(df.to_markdown(index=index) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, basename: str, index: bool = False) -> None:
    csv_path = OUT_DIR / f"{basename}.csv"
    md_path = OUT_DIR / f"{basename}.md"
    df.to_csv(csv_path, index=index)
    to_markdown(df, md_path, index=index)


def quantile_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "original_n_timepoints",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                "n_subjects": int(len(sub)),
                "n_with_timepoints": int(vals.shape[0]),
                "mean_n_TR": float(vals.mean()) if len(vals) else np.nan,
                "sd_n_TR": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "median_n_TR": float(vals.median()) if len(vals) else np.nan,
                "p25_n_TR": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "p75_n_TR": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "min_n_TR": float(vals.min()) if len(vals) else np.nan,
                "max_n_TR": float(vals.max()) if len(vals) else np.nan,
                "n_gt_140": int((vals > LOCKED_N_TR).sum()) if len(vals) else 0,
                "pct_gt_140": float((vals > LOCKED_N_TR).mean()) if len(vals) else np.nan,
                "n_retained_160": int((vals >= 160).sum()) if len(vals) else 0,
                "pct_retained_160": float((vals >= 160).mean()) if len(vals) else np.nan,
                "n_retained_180": int((vals >= 180).sum()) if len(vals) else 0,
                "pct_retained_180": float((vals >= 180).mean()) if len(vals) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_qc_sources() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base = safe_read_csv(BASE_QC)
    raw_n, raw_rois = zip(*base["raw_shape"].map(parse_shape))
    base_qc = pd.DataFrame(
        {
            "SubjectID": base["SubjectID"],
            "qc_source": "full_extraction_qc",
            "qc_batch": "v5_dparsf10000_no_pybandpass",
            "roi_signal_path": base.get("signal_path", pd.Series(index=base.index, dtype=object)),
            "raw_shape": base.get("raw_shape", pd.Series(index=base.index, dtype=object)),
            "original_n_timepoints": raw_n,
            "n_rois_raw_qc": raw_rois,
            "qc_status": base.get("status", pd.Series(index=base.index, dtype=object)),
            "load_status": base.get("load_status", pd.Series(index=base.index, dtype=object)),
            "processed_shape_qc": base.get("processed_shape", pd.Series(index=base.index, dtype=object)),
            "reduced_shape_qc": base.get("reduced_shape", pd.Series(index=base.index, dtype=object)),
            "reduced_finite_fraction_qc": base.get(
                "reduced_finite_fraction", pd.Series(index=base.index, dtype=object)
            ),
            "processed_nan_count_qc": base.get(
                "processed_nan_count", pd.Series(index=base.index, dtype=object)
            ),
            "tensor_nan_count_qc": base.get("tensor_nan_count", pd.Series(index=base.index, dtype=object)),
        }
    )
    frames.append(base_qc)

    for batch, path in INCREMENTAL_QC_BY_BATCH.items():
        inc = safe_read_csv(path)
        raw_n, raw_rois = zip(*inc["raw_shape"].map(parse_shape))
        frames.append(
            pd.DataFrame(
                {
                    "SubjectID": inc["SubjectID"],
                    "qc_source": "incremental_qc_summary",
                    "qc_batch": batch,
                    "roi_signal_path": inc.get(
                        "selected_path", pd.Series(index=inc.index, dtype=object)
                    ),
                    "raw_shape": inc.get("raw_shape", pd.Series(index=inc.index, dtype=object)),
                    "original_n_timepoints": raw_n,
                    "n_rois_raw_qc": raw_rois,
                    "qc_status": inc.get("status", pd.Series(index=inc.index, dtype=object)),
                    "load_status": inc.get("load_status", pd.Series(index=inc.index, dtype=object)),
                    "processed_shape_qc": inc.get(
                        "processed_shape", pd.Series(index=inc.index, dtype=object)
                    ),
                    "reduced_shape_qc": inc.get(
                        "reduced_shape", pd.Series(index=inc.index, dtype=object)
                    ),
                    "reduced_finite_fraction_qc": inc.get(
                        "reduced_finite_fraction", pd.Series(index=inc.index, dtype=object)
                    ),
                    "processed_nan_count_qc": inc.get(
                        "processed_nan_count", pd.Series(index=inc.index, dtype=object)
                    ),
                    "tensor_nan_count_qc": inc.get(
                        "tensor_nan_count", pd.Series(index=inc.index, dtype=object)
                    ),
                }
            )
        )

    all_qc = pd.concat(frames, ignore_index=True)
    all_qc["roi_signal_path_exists"] = all_qc["roi_signal_path"].map(
        lambda p: bool(Path(str(p)).exists()) if pd.notna(p) and str(p) else False
    )
    return all_qc


def choose_qc_for_metadata(meta: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    qc_by_subject_batch: dict[tuple[str, str], dict[str, Any]] = {}
    qc_by_subject_any: dict[str, dict[str, Any]] = {}
    for _, row in qc.iterrows():
        rec = row.to_dict()
        sid = str(row["SubjectID"])
        batch = str(row["qc_batch"])
        qc_by_subject_batch[(sid, batch)] = rec
        qc_by_subject_any.setdefault(sid, rec)

    selected_rows: list[dict[str, Any]] = []
    for _, row in meta.iterrows():
        sid = str(row["SubjectID"])
        source_batch = str(row.get("source_batch", ""))
        rec = qc_by_subject_batch.get((sid, source_batch))
        if rec is None and source_batch == "v5_dparsf10000_no_pybandpass":
            rec = qc_by_subject_batch.get((sid, "v5_dparsf10000_no_pybandpass"))
        if rec is None:
            rec = qc_by_subject_any.get(sid, {})
        selected_rows.append(rec)

    selected = pd.DataFrame(selected_rows)
    out = pd.concat([meta.reset_index(drop=True), selected.add_prefix("raw_").reset_index(drop=True)], axis=1)
    out.rename(
        columns={
            "raw_original_n_timepoints": "original_n_timepoints",
            "raw_n_rois_raw_qc": "n_rois_raw_qc",
            "raw_roi_signal_path": "roi_signal_path",
            "raw_raw_shape": "raw_shape",
        },
        inplace=True,
    )
    return out


def annotate_folds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["classifier_outer_fold"] = np.nan
    df["classifier_pool_role"] = "not_classifier_pool"
    df["vae_pool_folds"] = ""
    df["vae_actual_train_folds"] = ""
    df["vae_internal_val_folds"] = ""

    pred = safe_read_csv(PREDICTIONS)
    primary = pred[
        (pred["model_name"] == PRIMARY_MODEL) & (pred["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    fold_map = primary.drop_duplicates("SubjectID").set_index("SubjectID")["fold"].to_dict()
    df["classifier_outer_fold"] = df["SubjectID"].map(fold_map)
    df.loc[df["ResearchGroup_Mapped"].isin(["CN", "AD"]), "classifier_pool_role"] = "classifier_cn_ad"
    df.loc[df["ResearchGroup_Mapped"].eq("MCI"), "classifier_pool_role"] = "vae_only_mci"

    tensor_to_subject = df.set_index("tensor_index")["SubjectID"].to_dict()
    vae_pool: dict[str, list[int]] = {}
    vae_train: dict[str, list[int]] = {}
    vae_val: dict[str, list[int]] = {}
    for fold in range(1, 6):
        fold_dir = LOCKED_RUN / f"fold_{fold}"
        pool_path = fold_dir / "vae_training_pool_tensor_idx.npy"
        train_local_path = fold_dir / "vae_actual_train_idx_local_to_pool.npy"
        val_local_path = fold_dir / "vae_internal_val_idx_local_to_pool.npy"
        if not (pool_path.exists() and train_local_path.exists() and val_local_path.exists()):
            continue
        pool_tensor_idx = np.load(pool_path)
        train_local = np.load(train_local_path)
        val_local = np.load(val_local_path)
        for tidx in pool_tensor_idx:
            sid = tensor_to_subject.get(int(tidx))
            if sid:
                vae_pool.setdefault(sid, []).append(fold)
        for local_idx in train_local:
            if int(local_idx) >= len(pool_tensor_idx):
                continue
            sid = tensor_to_subject.get(int(pool_tensor_idx[int(local_idx)]))
            if sid:
                vae_train.setdefault(sid, []).append(fold)
        for local_idx in val_local:
            if int(local_idx) >= len(pool_tensor_idx):
                continue
            sid = tensor_to_subject.get(int(pool_tensor_idx[int(local_idx)]))
            if sid:
                vae_val.setdefault(sid, []).append(fold)

    df["vae_pool_folds"] = df["SubjectID"].map(lambda s: flatten_subjects(vae_pool.get(str(s), [])))
    df["vae_actual_train_folds"] = df["SubjectID"].map(
        lambda s: flatten_subjects(vae_train.get(str(s), []))
    )
    df["vae_internal_val_folds"] = df["SubjectID"].map(lambda s: flatten_subjects(vae_val.get(str(s), [])))
    df["in_vae_pool_any_fold"] = df["vae_pool_folds"].ne("")
    return df


def make_subject_table() -> pd.DataFrame:
    meta = safe_read_csv(TRAINING_METADATA)
    qc = load_qc_sources()
    df = choose_qc_for_metadata(meta, qc)
    df = annotate_folds(df)
    df["SiteCode"] = df["SubjectID"].map(sitecode)
    df["AgeBin"] = pd.cut(
        pd.to_numeric(df["Age"], errors="coerce"),
        bins=[-np.inf, 64.999, 69.999, 74.999, 79.999, 84.999, np.inf],
        labels=["<65", "65-69", "70-74", "75-79", "80-84", "85+"],
    ).astype(str)
    df["locked_n_timepoints_used"] = LOCKED_N_TR
    df["original_n_timepoints"] = pd.to_numeric(df["original_n_timepoints"], errors="coerce")
    df["gt_140_available"] = df["original_n_timepoints"] > LOCKED_N_TR
    df["retained_first_140"] = df["original_n_timepoints"] >= 140
    df["retained_first_160"] = df["original_n_timepoints"] >= 160
    df["retained_first_180"] = df["original_n_timepoints"] >= 180
    df["all_available_variable_length_possible"] = df["original_n_timepoints"].notna()
    df["motion_qc_fields_available"] = False
    df["motion_qc_note"] = "No FD/DVARS/motion-summary columns were found in the local v5.1b metadata/QC inputs."

    preferred_cols = [
        "SubjectID",
        "tensor_index",
        "ResearchGroup_Mapped",
        "Diagnosis",
        "classifier_pool_role",
        "classifier_outer_fold",
        "in_vae_pool_any_fold",
        "vae_pool_folds",
        "vae_actual_train_folds",
        "vae_internal_val_folds",
        "SiteCode",
        "Site3",
        "Manufacturer",
        "Age",
        "AgeBin",
        "Sex",
        "ImageID",
        "Visit",
        "source_batch",
        "source_label",
        "tensor_source",
        "roi_signal_path",
        "raw_roi_signal_path_exists",
        "raw_shape",
        "original_n_timepoints",
        "locked_n_timepoints_used",
        "gt_140_available",
        "retained_first_140",
        "retained_first_160",
        "retained_first_180",
        "all_available_variable_length_possible",
        "n_rois_raw_qc",
        "raw_qc_source",
        "raw_qc_batch",
        "raw_qc_status",
        "raw_load_status",
        "raw_processed_shape_qc",
        "raw_reduced_shape_qc",
        "raw_reduced_finite_fraction_qc",
        "raw_processed_nan_count_qc",
        "raw_tensor_nan_count_qc",
        "finite_fraction",
        "scale_label",
        "spectral_class",
        "dicom_series_ok",
        "python_bandpass_applied",
        "motion_qc_fields_available",
        "motion_qc_note",
    ]
    existing = [c for c in preferred_cols if c in df.columns]
    return df[existing].copy()


def make_age_sex_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = [
        ("Sex", ["Sex"]),
        ("AgeBin", ["AgeBin"]),
        ("Diagnosis_Sex", ["ResearchGroup_Mapped", "Sex"]),
        ("Diagnosis_AgeBin", ["ResearchGroup_Mapped", "AgeBin"]),
    ]
    for level, cols in specs:
        tab = quantile_summary(subjects, cols, "original_n_timepoints")
        tab.insert(0, "grouping_level", level)
        frames.append(tab)
    return pd.concat(frames, ignore_index=True)


def make_fold_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        test = subjects[subjects["classifier_outer_fold"].eq(fold)].copy()
        if len(test):
            row = quantile_summary(test, [], "original_n_timepoints")
            row.insert(0, "fold", fold)
            row.insert(1, "pool_scope", "classifier_test")
            rows.append(row)
        train = []
        val = []
        pool = []
        for _, rec in subjects.iterrows():
            if str(fold) in str(rec.get("vae_pool_folds", "")).split(";"):
                pool.append(rec)
            if str(fold) in str(rec.get("vae_actual_train_folds", "")).split(";"):
                train.append(rec)
            if str(fold) in str(rec.get("vae_internal_val_folds", "")).split(";"):
                val.append(rec)
        for label, records in [
            ("vae_training_pool", pool),
            ("vae_actual_train", train),
            ("vae_internal_val", val),
        ]:
            if records:
                sub = pd.DataFrame(records)
                row = quantile_summary(sub, [], "original_n_timepoints")
                row.insert(0, "fold", fold)
                row.insert(1, "pool_scope", label)
                rows.append(row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_retention(subjects: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("first_140_locked_baseline", 140, "fixed_length"),
        ("first_160", 160, "fixed_length"),
        ("first_180", 180, "fixed_length"),
        ("all_available_variable_length", None, "variable_length"),
    ]
    rows: list[dict[str, Any]] = []
    pools = {
        "vae_pool_all_training_ready": subjects,
        "classifier_pool_cn_ad": subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])],
    }
    for candidate, threshold, mode in candidates:
        for pool_name, pool in pools.items():
            if mode == "variable_length":
                retained = pool[pool["original_n_timepoints"].notna()]
            else:
                retained = pool[pool["original_n_timepoints"] >= float(threshold)]
            row: dict[str, Any] = {
                "candidate_length": candidate,
                "mode": mode,
                "required_n_TR": threshold if threshold is not None else "subject_specific_all_available",
                "pool": pool_name,
                "n_total": int(len(pool)),
                "n_retained": int(len(retained)),
                "pct_retained": float(len(retained) / len(pool)) if len(pool) else np.nan,
            }
            for dx in ["CN", "AD", "MCI"]:
                total = int((pool["ResearchGroup_Mapped"] == dx).sum())
                kept = int((retained["ResearchGroup_Mapped"] == dx).sum())
                row[f"{dx}_total"] = total
                row[f"{dx}_retained"] = kept
                row[f"{dx}_pct_retained"] = float(kept / total) if total else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def run_statistical_tests(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover - environment dependent
        stats = None
        rows.append(
            {
                "test_name": "scipy_import",
                "scope": "environment",
                "status": "failed",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect": "",
                "details": str(exc),
            }
        )

    clf = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf["original_n_timepoints"] = pd.to_numeric(clf["original_n_timepoints"], errors="coerce")
    cn = clf.loc[clf["ResearchGroup_Mapped"].eq("CN"), "original_n_timepoints"].dropna()
    ad = clf.loc[clf["ResearchGroup_Mapped"].eq("AD"), "original_n_timepoints"].dropna()
    rows.append(
        {
            "test_name": "descriptive_CN_vs_AD",
            "scope": "classifier_pool",
            "status": "ok",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect": float(ad.mean() - cn.mean()) if len(ad) and len(cn) else np.nan,
            "details": (
                f"CN n={len(cn)} mean={cn.mean():.3f}; AD n={len(ad)} mean={ad.mean():.3f}; "
                "effect is AD-CN mean n_TR"
            ),
        }
    )
    if stats is not None and len(cn) > 1 and len(ad) > 1:
        t_stat, t_p = stats.ttest_ind(ad, cn, equal_var=False, nan_policy="omit")
        u_stat, u_p = stats.mannwhitneyu(ad, cn, alternative="two-sided")
        rows.extend(
            [
                {
                    "test_name": "welch_t_CN_vs_AD",
                    "scope": "classifier_pool",
                    "status": "ok",
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "effect": float(ad.mean() - cn.mean()),
                    "details": "Diagnosis association with original n_TR, AD-CN mean difference.",
                },
                {
                    "test_name": "mann_whitney_CN_vs_AD",
                    "scope": "classifier_pool",
                    "status": "ok",
                    "statistic": float(u_stat),
                    "p_value": float(u_p),
                    "effect": float(ad.median() - cn.median()),
                    "details": "Diagnosis association with original n_TR, AD-CN median difference.",
                },
            ]
        )

    if stats is not None:
        for group_col, scope in [
            ("ResearchGroup_Mapped", "vae_pool_all_training_ready"),
            ("Manufacturer", "vae_pool_all_training_ready"),
            ("Manufacturer", "classifier_pool"),
            ("SiteCode", "classifier_pool_sites_with_at_least_2_subjects"),
        ]:
            data = subjects if "vae_pool" in scope else clf
            groups = []
            labels = []
            for label, sub in data.groupby(group_col, dropna=False):
                vals = pd.to_numeric(sub["original_n_timepoints"], errors="coerce").dropna()
                min_n = 2 if group_col != "SiteCode" else 2
                if len(vals) >= min_n:
                    groups.append(vals)
                    labels.append(str(label))
            if len(groups) >= 2:
                stat, p_val = stats.kruskal(*groups)
                rows.append(
                    {
                        "test_name": f"kruskal_by_{group_col}",
                        "scope": scope,
                        "status": "ok",
                        "statistic": float(stat),
                        "p_value": float(p_val),
                        "effect": "",
                        "details": "Groups tested: " + ",".join(labels[:30]),
                    }
                )

    # Diagnosis model adjusted for age, sex, and manufacturer. Statsmodels gives
    # interpretable p-values if available; sklearn fallback reports coefficients.
    model_df = clf[["ResearchGroup_Mapped", "original_n_timepoints", "Age", "Sex", "Manufacturer"]].dropna()
    if len(model_df) > 20 and model_df["ResearchGroup_Mapped"].nunique() == 2:
        y = model_df["ResearchGroup_Mapped"].eq("AD").astype(int)
        X = pd.get_dummies(
            model_df[["original_n_timepoints", "Age", "Sex", "Manufacturer"]],
            columns=["Sex", "Manufacturer"],
            drop_first=True,
            dtype=float,
        )
        X = X.astype(float)
        try:
            import statsmodels.api as sm

            X_sm = sm.add_constant(X, has_constant="add")
            fit = sm.Logit(y, X_sm).fit(disp=False, maxiter=200)
            coef = float(fit.params.get("original_n_timepoints", np.nan))
            p_val = float(fit.pvalues.get("original_n_timepoints", np.nan))
            rows.append(
                {
                    "test_name": "logit_AD_vs_CN_adjusted",
                    "scope": "classifier_pool",
                    "status": "ok",
                    "statistic": coef,
                    "p_value": p_val,
                    "effect": float(math.exp(coef)) if np.isfinite(coef) else np.nan,
                    "details": (
                        "Diagnosis ~ n_TR + Age + Sex + Manufacturer; effect is odds ratio per one TR. "
                        f"n={len(model_df)}"
                    ),
                }
            )
        except Exception as exc:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import make_pipeline

                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
                pipe.fit(X, y)
                coef = float(pipe.named_steps["logisticregression"].coef_[0][list(X.columns).index("original_n_timepoints")])
                rows.append(
                    {
                        "test_name": "logit_AD_vs_CN_adjusted_sklearn",
                        "scope": "classifier_pool",
                        "status": "ok_no_pvalue",
                        "statistic": coef,
                        "p_value": np.nan,
                        "effect": coef,
                        "details": f"Statsmodels failed: {exc}. Coefficient is standardized sklearn coefficient.",
                    }
                )
            except Exception as inner_exc:
                rows.append(
                    {
                        "test_name": "logit_AD_vs_CN_adjusted",
                        "scope": "classifier_pool",
                        "status": "failed",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect": "",
                        "details": f"statsmodels failed: {exc}; sklearn failed: {inner_exc}",
                    }
                )

    return pd.DataFrame(rows)


def determine_recommendation(
    subjects: pd.DataFrame, retention: pd.DataFrame, tests: pd.DataFrame
) -> tuple[str, str]:
    coverage = float(subjects["original_n_timepoints"].notna().mean())
    if coverage < 0.95:
        return (
            "not_possible_missing_raw_timeseries",
            f"Only {coverage:.1%} of training-ready subjects have traceable original timepoint counts.",
        )

    p_lookup = {
        row["test_name"]: row.get("p_value", np.nan)
        for _, row in tests.iterrows()
        if pd.notna(row.get("p_value", np.nan))
    }
    diagnosis_p = min(
        [
            p_lookup.get("welch_t_CN_vs_AD", np.nan),
            p_lookup.get("mann_whitney_CN_vs_AD", np.nan),
            p_lookup.get("logit_AD_vs_CN_adjusted", np.nan),
        ],
        default=np.nan,
    )
    if np.isnan(diagnosis_p):
        diagnosis_p = 1.0
    manufacturer_p = p_lookup.get("kruskal_by_Manufacturer", np.nan)
    site_p = p_lookup.get("kruskal_by_SiteCode", np.nan)

    clf_retention = retention[retention["pool"].eq("classifier_pool_cn_ad")].set_index("candidate_length")
    fixed_160_ok = False
    fixed_180_ok = False
    for label in ["first_160", "first_180"]:
        if label in clf_retention.index:
            row = clf_retention.loc[label]
            cn_ret = float(row.get("CN_pct_retained", np.nan))
            ad_ret = float(row.get("AD_pct_retained", np.nan))
            fixed_ok = (
                float(row.get("pct_retained", 0.0)) >= 0.90
                and np.isfinite(cn_ret)
                and np.isfinite(ad_ret)
                and abs(cn_ret - ad_ret) <= 0.05
            )
            if label == "first_160":
                fixed_160_ok = fixed_ok
            else:
                fixed_180_ok = fixed_ok

    confounded = (
        diagnosis_p < 0.05
        or (pd.notna(manufacturer_p) and manufacturer_p < 0.05)
        or (pd.notna(site_p) and site_p < 0.05)
    )
    if confounded and (fixed_160_ok or fixed_180_ok):
        return (
            "fixed_longer_length_preferred",
            "Original n_TR is confounded, so variable all-timepoint connectomes are unsafe; "
            "a fixed longer length is the safer controlled test if retention remains balanced.",
        )
    if confounded:
        return (
            "not_safe_due_to_confounding",
            "Original n_TR is associated with diagnosis, Manufacturer, or SiteCode and fixed longer-length "
            "retention is not cleanly balanced enough to recommend a rebuild now.",
        )
    return (
        "all_timepoints_safe_to_test",
        "No strong diagnosis/manufacturer/site timepoint-count confounding was detected in this audit.",
    )


def write_final_recommendation(
    subjects: pd.DataFrame,
    retention: pd.DataFrame,
    tests: pd.DataFrame,
    recommendation: str,
    rationale: str,
) -> None:
    source_counts = (
        subjects.groupby(["source_batch", "raw_shape"], dropna=False)
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["source_batch", "raw_shape"])
    )
    classifier = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    cn = classifier[classifier["ResearchGroup_Mapped"].eq("CN")]["original_n_timepoints"].dropna()
    ad = classifier[classifier["ResearchGroup_Mapped"].eq("AD")]["original_n_timepoints"].dropna()

    fixed_160 = retention[
        (retention["pool"] == "classifier_pool_cn_ad") & (retention["candidate_length"] == "first_160")
    ]
    fixed_180 = retention[
        (retention["pool"] == "classifier_pool_cn_ad") & (retention["candidate_length"] == "first_180")
    ]
    text = [
        "# ADNI Timepoint Availability and Confounding Audit",
        "",
        f"Final recommendation: **{recommendation}**.",
        "",
        rationale,
        "",
        "## Key Findings",
        "",
        f"- Training-ready cohort audited: {len(subjects)} subjects "
        f"({subjects['ResearchGroup_Mapped'].value_counts().to_dict()}).",
        f"- Classifier pool audited: {len(classifier)} CN/AD subjects "
        f"({classifier['ResearchGroup_Mapped'].value_counts().to_dict()}).",
        f"- Traceable original ROI-signal timepoint counts: "
        f"{subjects['original_n_timepoints'].notna().sum()}/{len(subjects)}.",
        f"- Locked pipeline used exactly {LOCKED_N_TR} TR for connectome construction.",
        f"- CN mean original n_TR: {cn.mean():.2f}; AD mean original n_TR: {ad.mean():.2f}; "
        f"AD-CN mean difference: {ad.mean() - cn.mean():.2f}.",
        "- No local FD/DVARS/motion-summary columns were found in the v5.1b metadata/QC inputs; "
        "available QC fields are signal shape, finite fraction, NaN counts, and tensor build status.",
        "",
        "## Fixed-Length Retention",
        "",
    ]
    for label, table in [("160 TR", fixed_160), ("180 TR", fixed_180)]:
        if len(table):
            row = table.iloc[0]
            text.append(
                f"- {label}: retains {int(row['n_retained'])}/{int(row['n_total'])} classifier subjects "
                f"({row['pct_retained']:.1%}); CN {int(row['CN_retained'])}/{int(row['CN_total'])}, "
                f"AD {int(row['AD_retained'])}/{int(row['AD_total'])}."
            )
    text.extend(
        [
            "",
            "## Source/Shape Inventory",
            "",
            source_counts.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "Using all available timepoints would intentionally introduce variable acquisition length into "
            "the connectome estimates. That can improve Pearson/MI stability, but it can also encode scanner, "
            "site, or cohort-acquisition differences. A scientifically defensible rebuild should therefore "
            "only proceed if it keeps temporal length fixed or if timepoint count is demonstrably not associated "
            "with diagnosis/site/manufacturer. The locked 140-TR model remains the non-confounded reference.",
        ]
    )
    (OUT_DIR / "final_recommendation.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command_log: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "training_metadata": str(TRAINING_METADATA),
            "locked_run": str(LOCKED_RUN),
            "predictions": str(PREDICTIONS),
            "base_qc": str(BASE_QC),
            "incremental_qc": {k: str(v) for k, v in INCREMENTAL_QC_BY_BATCH.items()},
        },
        "mode": "read_only_audit",
        "writes_only": str(OUT_DIR),
    }

    subjects = make_subject_table()
    write_table(subjects, "subject_timepoint_table")

    diagnosis = quantile_summary(subjects, ["ResearchGroup_Mapped"], "original_n_timepoints")
    write_table(diagnosis, "timepoints_by_diagnosis")

    by_mfr = quantile_summary(subjects, ["Manufacturer"], "original_n_timepoints")
    by_mfr.insert(0, "grouping_level", "Manufacturer")
    by_mfr_site = quantile_summary(subjects, ["Manufacturer", "SiteCode"], "original_n_timepoints")
    by_mfr_site.insert(0, "grouping_level", "Manufacturer_SiteCode")
    write_table(pd.concat([by_mfr, by_mfr_site], ignore_index=True), "timepoints_by_manufacturer_site")

    age_sex = make_age_sex_summary(subjects)
    write_table(age_sex, "timepoints_by_age_sex_bins")

    fold_summary = make_fold_summary(subjects)
    write_table(fold_summary, "timepoints_by_fold")

    retention = make_retention(subjects)
    write_table(retention, "candidate_length_retention")

    tests = run_statistical_tests(subjects)
    write_table(tests, "confounding_tests")

    recommendation, rationale = determine_recommendation(subjects, retention, tests)
    write_final_recommendation(subjects, retention, tests, recommendation, rationale)

    command_log.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": [
                "subject_timepoint_table.csv/.md",
                "timepoints_by_diagnosis.csv/.md",
                "timepoints_by_manufacturer_site.csv/.md",
                "timepoints_by_age_sex_bins.csv/.md",
                "timepoints_by_fold.csv/.md",
                "candidate_length_retention.csv/.md",
                "confounding_tests.csv/.md",
                "final_recommendation.md",
                "command_log.json",
            ],
            "recommendation": recommendation,
            "n_subjects": int(len(subjects)),
            "n_classifier_pool": int(subjects["ResearchGroup_Mapped"].isin(["CN", "AD"]).sum()),
            "n_traceable_timepoints": int(subjects["original_n_timepoints"].notna().sum()),
        }
    )
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package to {OUT_DIR}")
    print(f"Recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
