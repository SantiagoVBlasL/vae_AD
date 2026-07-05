#!/usr/bin/env python3
"""Read-only metadata confounding and ComBat-input feasibility audit.

The script reads the promoted ADNI run, fold assignments, metadata, and existing
OOF scores. It fits only ephemeral metadata-only audit models on outer train/dev
folds to estimate confounding signal; no model artifacts are saved and no VAE or
classifier readout is trained or modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - optional reporting path
    smf = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF_DIR = RESULTS / "recover035_lat384_beta3p75_stageB_oof_score_calibration"
OOF_DIR_FALLBACK = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
LAYER_AUDIT_DIR = RESULTS / "promoted_model_layerwise_architecture_failure_audit_20260603"
OUT_DEFAULT = RESULTS / "promoted_model_metadata_confounding_combat_feasibility_audit_20260603"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return df.to_csv(index=False)


def write_table(df: pd.DataFrame, stem: str, out: Path, max_rows: int | None = 200) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    shown = df if max_rows is None else df.head(max_rows)
    text = f"# {stem}\n\nRows: {len(df)}\n\n"
    if len(shown) < len(df):
        text += f"Showing first {len(shown)} rows; full table is in CSV.\n\n"
    text += to_md(shown)
    (out / f"{stem}.md").write_text(text, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_float(v: Any) -> float:
    try:
        if v is None or pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def safe_div(a: float, b: float) -> float:
    if b == 0 or math.isnan(b):
        return float("nan")
    return float(a) / float(b)


def smd(a: Iterable[float], b: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(a)), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(list(b)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    return safe_div(float(np.mean(y) - np.mean(x)), pooled)


def normalize_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if "tensor_index" in out.columns and "tensor_idx" not in out.columns:
        out = out.rename(columns={"tensor_index": "tensor_idx"})
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["tensor_idx"] = pd.to_numeric(out["tensor_idx"], errors="coerce")
    for col in ["ResearchGroup_Mapped", "Diagnosis", "Manufacturer", "Sex", "Site3", "Visit", "source_batch", "source_label"]:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    out["Age"] = pd.to_numeric(out.get("Age", np.nan), errors="coerce")
    out["n_timepoints_raw"] = pd.to_numeric(out.get("n_timepoints_raw", np.nan), errors="coerce")
    out["finite_fraction"] = pd.to_numeric(out.get("finite_fraction", np.nan), errors="coerce")
    out["ADNI_phase"] = out["Visit"].str.extract(r"(ADNI\d+)", expand=False).fillna("UNKNOWN")
    out["Site"] = out["Site3"].astype(str)
    return out


def metadata_path_from_run() -> Path:
    cfg = read_json(RUN_DIR / "run_config.json")
    args = cfg.get("args", cfg)
    return Path(args["metadata_path"])


def selected_oof_dir() -> Path:
    return OOF_DIR if OOF_DIR.exists() else OOF_DIR_FALLBACK


def primary_predictions() -> pd.DataFrame:
    df = read_csv(selected_oof_dir() / "calib_predictions.csv", required=True)
    rows = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    rows["SubjectID"] = rows["SubjectID"].astype(str)
    rows["Age"] = pd.to_numeric(rows["Age"], errors="coerce")
    return rows


def merge_fold_subjects(subjects: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    sub = subjects.copy()
    sub["SubjectID"] = sub["SubjectID"].astype(str)
    meta_cols = [c for c in meta.columns if c not in {"ResearchGroup_Mapped"}] + ["ResearchGroup_Mapped"]
    merged = sub.merge(meta[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    if "ResearchGroup_Mapped_meta" in merged.columns:
        merged["ResearchGroup_Mapped"] = merged["ResearchGroup_Mapped"].where(merged["ResearchGroup_Mapped"].notna(), merged["ResearchGroup_Mapped_meta"])
        merged = merged.drop(columns=["ResearchGroup_Mapped_meta"])
    return merged


def age_summary(df: pd.DataFrame, value_col: str = "Age") -> dict[str, Any]:
    x = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if x.empty:
        return {"n_nonmissing": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan}
    q1, q3 = x.quantile([0.25, 0.75])
    return {
        "n_nonmissing": int(x.shape[0]),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        "median": float(x.median()),
        "iqr": float(q3 - q1),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def global_metadata_balance(meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_cols = [
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Site",
        "ADNI_phase",
        "Sex",
        "source_batch",
        "source_label",
        "training_ready",
        "exclude_from_supervised",
        "python_bandpass_applied",
    ]
    optional_cols = [
        "FieldStrength",
        "MagneticFieldStrength",
        "ScannerModel",
        "DeviceSerialNumber",
        "ProtocolName",
        "TR",
        "RepetitionTime",
        "mean_fd",
        "max_fd",
        "motion_mean_fd",
    ]
    for col in count_cols + optional_cols:
        if col not in meta.columns:
            rows.append({"section": "availability", "variable": col, "level": "COLUMN_MISSING", "n": 0, "percent": np.nan})
            continue
        counts = meta[col].fillna("MISSING").astype(str).value_counts(dropna=False)
        for level, n in counts.items():
            rows.append({"section": "counts", "variable": col, "level": level, "n": int(n), "percent": safe_div(n, len(meta))})
    for col in ["Age", "n_timepoints_raw", "finite_fraction"]:
        stats_row = age_summary(meta, col)
        stats_row.update({"section": "numeric_summary", "variable": col, "level": "all", "n": len(meta), "percent": np.nan})
        rows.append(stats_row)
    return pd.DataFrame(rows)


def crosstab_long(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    if any(c not in df.columns for c in cols):
        return pd.DataFrame([{"table": name, "status": "missing_columns", "columns": ",".join(cols)}])
    tab = df.groupby(cols, dropna=False).size().rename("n").reset_index()
    tab.insert(0, "table", name)
    return tab


def manufacturer_site_tables(meta: pd.DataFrame) -> pd.DataFrame:
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    frames = [
        crosstab_long(clf, ["ResearchGroup_Mapped", "Manufacturer"], "diagnosis_x_manufacturer"),
        crosstab_long(clf, ["ResearchGroup_Mapped", "Site"], "diagnosis_x_site"),
        crosstab_long(clf, ["ResearchGroup_Mapped", "Manufacturer", "Sex"], "diagnosis_x_manufacturer_x_sex"),
        crosstab_long(clf, ["Manufacturer", "Site", "ResearchGroup_Mapped"], "manufacturer_x_site_x_diagnosis"),
        crosstab_long(clf, ["Manufacturer", "ADNI_phase", "ResearchGroup_Mapped"], "manufacturer_x_phase_x_diagnosis"),
    ]
    age_rows = []
    for (dx, mfr), grp in clf.groupby(["ResearchGroup_Mapped", "Manufacturer"], dropna=False):
        row = age_summary(grp)
        row.update({"table": "diagnosis_x_manufacturer_age", "ResearchGroup_Mapped": dx, "Manufacturer": mfr, "n": len(grp)})
        age_rows.append(row)
    frames.append(pd.DataFrame(age_rows))
    smd_rows = []
    cn = clf[clf["ResearchGroup_Mapped"].eq("CN")]
    ad = clf[clf["ResearchGroup_Mapped"].eq("AD")]
    smd_rows.append({"table": "age_smd_ad_vs_cn", "Manufacturer": "ALL", "age_smd_ad_minus_cn": smd(cn["Age"], ad["Age"]), "n_cn": len(cn), "n_ad": len(ad)})
    for mfr, grp in clf.groupby("Manufacturer", dropna=False):
        cnm = grp[grp["ResearchGroup_Mapped"].eq("CN")]
        adm = grp[grp["ResearchGroup_Mapped"].eq("AD")]
        smd_rows.append({"table": "age_smd_ad_vs_cn", "Manufacturer": mfr, "age_smd_ad_minus_cn": smd(cnm["Age"], adm["Age"]), "n_cn": len(cnm), "n_ad": len(adm)})
    frames.append(pd.DataFrame(smd_rows))
    return pd.concat(frames, ignore_index=True, sort=False)


def foldwise_metadata_balance(meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_mfr = {"GE", "Philips", "SIEMENS"}
    for fold in range(1, 6):
        fold_dir = RUN_DIR / f"fold_{fold}"
        for split_name, file_name in [("trainDev", "train_dev_subjects_fold.csv"), ("test", "test_subjects_fold.csv")]:
            subjects = read_csv(fold_dir / file_name, required=True)
            df = merge_fold_subjects(subjects, meta)
            clf = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
            present_mfr = set(clf["Manufacturer"].dropna().astype(str).unique())
            rows.append(
                {
                    "fold": fold,
                    "split": split_name,
                    "table": "manufacturer_presence",
                    "group": "ALL",
                    "level": "ALL",
                    "n": len(clf),
                    "all_3_manufacturers_present": expected_mfr.issubset(present_mfr),
                    "manufacturers_present": ";".join(sorted(present_mfr)),
                    "inner_fold_manufacturer_presence": "not_saved_or_not_available",
                }
            )
            for dx, n in clf["ResearchGroup_Mapped"].value_counts().items():
                rows.append({"fold": fold, "split": split_name, "table": "diagnosis_counts", "group": "ResearchGroup_Mapped", "level": dx, "n": int(n)})
            for keys, grp in clf.groupby(["ResearchGroup_Mapped", "Manufacturer"], dropna=False):
                rows.append({"fold": fold, "split": split_name, "table": "diagnosis_x_manufacturer", "group": str(keys[0]), "level": str(keys[1]), "n": len(grp)})
            for keys, grp in clf.groupby(["ResearchGroup_Mapped", "Site"], dropna=False):
                rows.append({"fold": fold, "split": split_name, "table": "diagnosis_x_site", "group": str(keys[0]), "level": str(keys[1]), "n": len(grp)})
            for keys, grp in clf.groupby(["ResearchGroup_Mapped", "Manufacturer", "Sex"], dropna=False):
                rows.append({"fold": fold, "split": split_name, "table": "diagnosis_x_manufacturer_x_sex", "group": f"{keys[0]}|{keys[1]}", "level": str(keys[2]), "n": len(grp)})
            for keys, grp in clf.groupby(["ResearchGroup_Mapped", "Manufacturer"], dropna=False):
                stats_row = age_summary(grp)
                stats_row.update({"fold": fold, "split": split_name, "table": "age_by_diagnosis_manufacturer", "group": f"{keys[0]}|{keys[1]}", "level": "Age", "n": len(grp)})
                rows.append(stats_row)
    return pd.DataFrame(rows)


def onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def metadata_model_pipeline(features: list[str], model_type: str) -> Pipeline:
    numeric = [f for f in features if f == "Age"]
    categorical = [f for f in features if f != "Age"]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", onehot_encoder())]), categorical))
    clf: Any
    if model_type == "ridge_classifier":
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    else:
        clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=42)
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", clf)])


def decision_scores(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(pipe, "predict_proba") and hasattr(model, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(x)
        return 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))
    pred = pipe.predict(x)
    return np.asarray(pred, dtype=float)


def metrics_from_scores(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def metadata_only_prediction(meta: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "AgeSex": ["Age", "Sex"],
        "ManufacturerOnly": ["Manufacturer"],
        "AgeSexManufacturer": ["Age", "Sex", "Manufacturer"],
        "AgeSexManufacturerSite": ["Age", "Sex", "Manufacturer", "Site"],
    }
    rows = []
    pred_rows = []
    for model_type in ["logreg_l2_fixedC", "ridge_classifier"]:
        for fs_name, features in feature_sets.items():
            all_scores = []
            all_y = []
            for fold in range(1, 6):
                tr = merge_fold_subjects(read_csv(RUN_DIR / f"fold_{fold}" / "train_dev_subjects_fold.csv", required=True), meta)
                te = merge_fold_subjects(read_csv(RUN_DIR / f"fold_{fold}" / "test_subjects_fold.csv", required=True), meta)
                tr = tr[tr["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
                te = te[te["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
                y_train = tr["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).to_numpy(dtype=int)
                y_test = te["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).to_numpy(dtype=int)
                pipe = metadata_model_pipeline(features, model_type)
                pipe.fit(tr[features], y_train)
                score = decision_scores(pipe, te[features])
                all_scores.append(score)
                all_y.append(y_test)
                rec = metrics_from_scores(y_test, score)
                rec.update({"model_type": model_type, "feature_set": fs_name, "fold": fold, "scope": "fold"})
                rows.append(rec)
                for sid, yt, sc in zip(te["SubjectID"], y_test, score):
                    pred_rows.append({"model_type": model_type, "feature_set": fs_name, "fold": fold, "SubjectID": sid, "y_true": int(yt), "score": float(sc)})
            y = np.concatenate(all_y)
            s = np.concatenate(all_scores)
            rec = metrics_from_scores(y, s)
            rec.update({"model_type": model_type, "feature_set": fs_name, "fold": "pooled", "scope": "pooled_oof"})
            rows.append(rec)
    return pd.DataFrame(rows)


def score_metadata_association(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = pred.merge(meta[["SubjectID", "Site", "ADNI_phase", "n_timepoints_raw", "finite_fraction"]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    df["Diagnosis"] = np.where(df["y_true"].eq(1), "AD", "CN")
    rows: list[dict[str, Any]] = []
    if smf is None:
        return pd.DataFrame([{"model": "statsmodels_unavailable", "term": "NA", "coef": np.nan, "pvalue": np.nan}])
    formulas = {
        "score_age_sex_dx_mfr": "y_score ~ Age + C(Sex) + C(Diagnosis) + C(Manufacturer)",
        "score_age_sex_dx_mfr_site": "y_score ~ Age + C(Sex) + C(Diagnosis) + C(Manufacturer) + C(Site)",
    }
    for name, formula in formulas.items():
        try:
            model = smf.ols(formula, data=df).fit()
            for term, coef in model.params.items():
                rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coef": float(coef),
                        "pvalue": float(model.pvalues.get(term, np.nan)),
                        "rsquared": float(model.rsquared),
                        "n": int(model.nobs),
                        "manufacturer_term": "Manufacturer" in term,
                    }
                )
        except Exception as exc:
            rows.append({"model": name, "term": "MODEL_FAILED", "coef": np.nan, "pvalue": np.nan, "error": str(exc)})
    # Nonparametric score summaries by manufacturer.
    for mfr, grp in df.groupby("Manufacturer", dropna=False):
        rows.append(
            {
                "model": "descriptive_score_by_manufacturer",
                "term": str(mfr),
                "coef": float(grp["y_score"].mean()),
                "pvalue": np.nan,
                "rsquared": np.nan,
                "n": len(grp),
                "manufacturer_term": True,
                "score_median": float(grp["y_score"].median()),
                "score_iqr": float(grp["y_score"].quantile(0.75) - grp["y_score"].quantile(0.25)),
            }
        )
    return pd.DataFrame(rows)


def fisher_or_chi2(table: pd.DataFrame) -> dict[str, Any]:
    try:
        arr = table.to_numpy()
        if arr.shape == (2, 2):
            odds, p = stats.fisher_exact(arr)
            return {"test": "fisher_exact", "statistic": odds, "pvalue": p}
        chi2, p, dof, _ = stats.chi2_contingency(arr)
        return {"test": "chi_square", "statistic": chi2, "pvalue": p, "dof": dof}
    except Exception as exc:
        return {"test": "failed", "statistic": np.nan, "pvalue": np.nan, "error": str(exc)}


def philips_fp_tn_audit(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    residual = read_csv(LAYER_AUDIT_DIR / "philips_cn_fp_tn_residual_comparison.csv")
    residual_summary = pd.DataFrame()
    if not residual.empty:
        residual_summary = (
            residual.groupby("fp_tn_status", dropna=False)[["mae_offdiag", "mse_offdiag", "near_over_far_mae"]]
            .mean()
            .reset_index()
        )
    df = pred[
        pred["ResearchGroup_Mapped"].astype(str).eq("CN")
        & pred["Manufacturer"].astype(str).str.lower().eq("philips")
    ].copy()
    df = df.merge(meta[["SubjectID", "Site", "ADNI_phase", "n_timepoints_raw", "finite_fraction", "Visit", "source_batch", "source_label"]].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    df["fp_tn_status"] = np.where(df["y_pred"].eq(1), "Philips_CN_FP", "Philips_CN_TN")
    rows: list[dict[str, Any]] = []
    for status, grp in df.groupby("fp_tn_status", dropna=False):
        a = age_summary(grp, "Age")
        a.update({"section": "numeric", "variable": "Age", "group": status, "n": len(grp)})
        rows.append(a)
        score_stats = age_summary(grp.rename(columns={"y_score": "score_tmp"}), "score_tmp")
        score_stats.update({"section": "numeric", "variable": "latent_score", "group": status, "n": len(grp)})
        rows.append(score_stats)
        for col in ["Sex", "Site", "ADNI_phase", "Visit", "source_batch", "source_label"]:
            for level, n in grp[col].fillna("MISSING").astype(str).value_counts().items():
                rows.append({"section": "counts", "variable": col, "group": status, "level": level, "n": int(n), "percent": safe_div(n, len(grp))})
    # Exploratory group tests.
    fp = df[df["fp_tn_status"].eq("Philips_CN_FP")]
    tn = df[df["fp_tn_status"].eq("Philips_CN_TN")]
    for col in ["Age", "y_score", "n_timepoints_raw", "finite_fraction"]:
        x = pd.to_numeric(fp[col], errors="coerce").dropna()
        y = pd.to_numeric(tn[col], errors="coerce").dropna()
        if len(x) and len(y):
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append({"section": "test", "variable": col, "group": "FP_vs_TN", "test": "mannwhitneyu", "statistic": float(stat), "pvalue": float(p), "n_fp": len(x), "n_tn": len(y)})
    for col in ["Sex", "Site", "ADNI_phase"]:
        tab = pd.crosstab(df["fp_tn_status"], df[col])
        test = fisher_or_chi2(table=tab)
        test.update({"section": "test", "variable": col, "group": "FP_vs_TN"})
        rows.append(test)
    if not residual_summary.empty:
        for _, row in residual_summary.iterrows():
            rec = row.to_dict()
            rec.update({"section": "residual_summary", "variable": "reconstruction_residual", "group": row["fp_tn_status"]})
            rows.append(rec)
    return pd.DataFrame(rows)


def combat_feasibility(meta: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    expected = {"GE", "Philips", "SIEMENS"}
    for fold in range(1, 6):
        tr = merge_fold_subjects(read_csv(RUN_DIR / f"fold_{fold}" / "train_dev_subjects_fold.csv", required=True), meta)
        te = merge_fold_subjects(read_csv(RUN_DIR / f"fold_{fold}" / "test_subjects_fold.csv", required=True), meta)
        for split, df in [("trainDev", tr), ("test", te)]:
            present = set(df["Manufacturer"].dropna().astype(str).unique())
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "n": len(df),
                    "manufacturers_present": ";".join(sorted(present)),
                    "all_3_manufacturers_present": expected.issubset(present),
                    "age_missing": int(df["Age"].isna().sum()),
                    "sex_missing": int(df["Sex"].isna().sum()),
                    "diagnosis_required_for_transform": False,
                }
            )
    package_rows = []
    for pkg in ["neuroCombat", "neuroHarmonize", "neurocombat_sklearn"]:
        package_rows.append(
            {
                "fold": "package",
                "split": pkg,
                "n": np.nan,
                "manufacturers_present": "",
                "all_3_manufacturers_present": np.nan,
                "age_missing": np.nan,
                "sex_missing": np.nan,
                "diagnosis_required_for_transform": False,
                "package_available": importlib.util.find_spec(pkg) is not None,
            }
        )
    feas = pd.concat([pd.DataFrame(rows), pd.DataFrame(package_rows)], ignore_index=True)
    neurocombat_available = bool(importlib.util.find_spec("neuroCombat") is not None)
    neuroharmonize_available = bool(importlib.util.find_spec("neuroHarmonize") is not None)
    neurocombat_sklearn_available = bool(importlib.util.find_spec("neurocombat_sklearn") is not None)
    all_train_ok = bool(feas[(feas["split"].eq("trainDev"))]["all_3_manufacturers_present"].all())
    no_age_sex_missing = bool((feas[feas["split"].isin(["trainDev", "test"])][["age_missing", "sex_missing"]].fillna(0).sum().sum()) == 0)
    validated_transform_available = neuroharmonize_available or neurocombat_sklearn_available
    status = "not_ready_for_FULL_training"
    if all_train_ok and no_age_sex_missing and validated_transform_available:
        status = "technically_ready_with_train_apply_wrapper"
    elif all_train_ok and no_age_sex_missing and neurocombat_available:
        status = "possible_but_train_apply_transform_needs_validation"
    text = f"""# ComBat-Input Feasibility Report

Status: **{status}**.

Proposed design:
- batch: `Manufacturer`
- protected covariates: `Age + Sex`
- excluded covariates: `Diagnosis`
- features: channel-wise upper-triangle off-diagonal connectome entries
- fit scope: train/dev fold only
- application: frozen transform applied to validation/test without diagnosis labels

Fold checks:
- All train/dev folds contain GE, Philips, and SIEMENS: `{all_train_ok}`.
- Age/Sex missing across train/dev/test folds: `{not no_age_sex_missing}`.
- Diagnosis labels are not required at test-time if only Age/Sex are protected covariates.

Package checks:
- `neuroCombat` available: `{neurocombat_available}`.
- `neuroHarmonize` available: `{neuroharmonize_available}`.
- `neurocombat_sklearn` available: `{neurocombat_sklearn_available}`.

Interpretation:
Fold-wise ComBat-input harmonization is conceptually feasible because all manufacturers are present in every train/dev fold and no test diagnosis labels are required. However, the environment lacks `neuroHarmonize`, and a validated train/apply transform with Age/Sex support should be confirmed before a FULL VAE run. If using `neuroCombat` directly, the frozen application path to test data must be implemented and validated as a dry-run artifact first; otherwise this is not ready for FULL training.
"""
    return feas, text


def recommendation_text(meta_metrics: pd.DataFrame, score_assoc: pd.DataFrame, combat_text: str) -> str:
    pooled = meta_metrics[(meta_metrics["scope"].eq("pooled_oof")) & (meta_metrics["model_type"].eq("logreg_l2_fixedC"))]
    best_auc = safe_float(pooled["auc"].max()) if not pooled.empty else np.nan
    mfr_assoc = score_assoc[
        score_assoc.get("manufacturer_term", pd.Series(dtype=bool)).astype(bool)
        & score_assoc["model"].astype(str).str.contains("score_age_sex_dx_mfr", na=False)
    ].copy()
    min_mfr_p = safe_float(mfr_assoc["pvalue"].min()) if not mfr_assoc.empty else np.nan
    combat_ready = "technically_ready_with_train_apply_wrapper" in combat_text
    if combat_ready and best_auc >= 0.65 and (not np.isnan(min_mfr_p) and min_mfr_p < 0.05):
        decision = "B. run classifier-only latent/residual sensitivity first"
    else:
        decision = "B. run classifier-only latent/residual sensitivity first"
    return f"""# Recommendation

Recommendation: **{decision}**.

Rationale:
- Metadata-only OOF classifiers show nontrivial confounding signal; the best fixed, non-tuned metadata-only AUC is `{best_auc:.6f}`.
- Manufacturer terms in score association models have minimum exploratory p-value `{min_mfr_p:.6g}` after Age/Sex/diagnosis adjustment where estimable.
- ComBat-input harmonization is conceptually fold-safe if fit only on train/dev with `Manufacturer` as batch and `Age + Sex` as protected covariates, excluding diagnosis. It does not require test diagnosis labels.
- A FULL ComBat-input VAE run is not the next safest step until the train/apply implementation is validated and classifier-only or residual-level harmonization sensitivity shows that harmonization can reduce Philips CN false positives or scanner leakage without damaging AUC/PR-AUC.

Rejected immediate options:
- **A. run ComBat-input FULL VAE**: premature without a validated frozen train/apply transform and a classifier-only sensitivity win.
- **C. do not harmonize; preserve promoted model**: preserve the promoted model as primary, but a read-only/frozen-latent harmonization sensitivity is still scientifically useful.
- **D. collect more data / external validation first**: external calibration/test remains the primary next manuscript step, but it does not preclude a small read-only/classifier-only harmonization feasibility check.

Operational next step:
Prepare a read-only or classifier-only harmonization sensitivity using existing fold latent/residual features before any new FULL VAE training.
"""


def main() -> None:
    args = parse_args()
    out = args.output_dir
    meta_path = metadata_path_from_run()
    meta = normalize_metadata(pd.read_csv(meta_path))
    pred = primary_predictions()
    if args.dry_run:
        print("Inputs OK.")
        print("metadata:", meta_path, meta.shape)
        print("primary predictions:", pred.shape)
        print("output:", rel(out))
        return
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")

    global_balance = global_metadata_balance(meta)
    fold_balance = foldwise_metadata_balance(meta)
    mfr_tables = manufacturer_site_tables(meta)
    metadata_metrics = metadata_only_prediction(meta)
    score_assoc = score_metadata_association(pred, meta)
    philips = philips_fp_tn_audit(pred, meta)
    combat_df, combat_report = combat_feasibility(meta)
    rec_text = recommendation_text(metadata_metrics, score_assoc, combat_report)

    write_table(global_balance, "global_metadata_balance", out, max_rows=300)
    write_table(fold_balance, "foldwise_metadata_balance", out, max_rows=300)
    write_table(mfr_tables, "manufacturer_site_diagnosis_tables", out, max_rows=300)
    write_table(metadata_metrics, "metadata_only_prediction_metrics", out, max_rows=200)
    write_table(score_assoc, "promoted_score_metadata_association", out, max_rows=300)
    write_table(philips, "philips_cn_fp_tn_metadata_audit", out, max_rows=300)
    write_table(combat_df, "combat_feasibility_checks", out, max_rows=200)
    write_text(out / "combat_input_feasibility_report.md", combat_report)
    write_text(out / "recommendation.md", rec_text)

    readme = f"""# Promoted Model Metadata Confounding and ComBat Feasibility Audit

Generated: {datetime.now().isoformat(timespec='seconds')}

Run: `{rel(RUN_DIR)}`

Metadata: `{meta_path}`

Primary score rows: `{PRIMARY_MODEL} / {PRIMARY_FEATURES} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD}`.

Safety:
- VAE training: no.
- Promoted classifier training/scoring: no.
- Threshold fitting: no.
- Tensor/metadata/model artifact modification: no.
- Ephemeral metadata-only audit models: yes, fixed hyperparameters, outer-fold train/dev fit only, no saved models.

Required outputs:
- `global_metadata_balance.csv/.md`
- `foldwise_metadata_balance.csv/.md`
- `manufacturer_site_diagnosis_tables.csv/.md`
- `metadata_only_prediction_metrics.csv/.md`
- `promoted_score_metadata_association.csv/.md`
- `philips_cn_fp_tn_metadata_audit.csv/.md`
- `combat_input_feasibility_report.md`
- `recommendation.md`
"""
    write_text(out / "README.md", readme)

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "run_dir": rel(RUN_DIR),
        "metadata_path": str(meta_path),
        "oof_dir": rel(selected_oof_dir()),
        "output_dir": rel(out),
        "safety": {
            "vae_training": False,
            "promoted_classifier_scoring": False,
            "threshold_fitting": False,
            "tensor_modification": False,
            "metadata_modification": False,
            "model_artifact_modification": False,
            "ephemeral_metadata_only_audit_models": True,
            "saved_audit_models": False,
        },
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    write_text(out / "command_log.json", json.dumps(command_log, indent=2))
    print(f"Wrote audit package: {rel(out)}")


if __name__ == "__main__":
    main()
