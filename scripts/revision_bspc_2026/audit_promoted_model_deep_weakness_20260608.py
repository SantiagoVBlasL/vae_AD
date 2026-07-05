#!/usr/bin/env python3
"""Deep weakness audit for promoted [1,0,2] latent384 beta3.75 model.

This script reads existing OOF predictions, latent caches, rate-distortion, and
scanner/latent QC artifacts. It writes a descriptive audit package only.

Guardrails:
- no VAE training;
- no OASIS scoring or OASIS calibration/threshold fitting;
- no tensor/metadata/model artifact modification;
- no model artifact overwrite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/revision_bspc_2026/promoted_model_deep_weakness_audit_20260608"

PROMOTED_RUN = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_OOF = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
PROMOTED_LATENT_CACHE = PROMOTED_RUN / "classifier_only_readout/latent_cache"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


@dataclass(frozen=True)
class PredictionSource:
    label: str
    path: Path
    kind: str = "calib_predictions"
    method_filter: str | None = None


SOURCES = [
    PredictionSource("promoted", PROMOTED_OOF / "calib_predictions.csv"),
    PredictionSource(
        "ch1only",
        ROOT
        / "results/revision_bspc_2026/recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
    ),
    PredictionSource(
        "foldcombat",
        ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration/calib_predictions.csv",
    ),
    PredictionSource(
        "chmeanloss",
        ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_chmeanloss_stageB_oof_score_calibration/calib_predictions.csv",
    ),
    PredictionSource(
        "ch12_beta2p75",
        ROOT
        / "results/revision_bspc_2026/recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
    ),
    PredictionSource(
        "residualized_mfr",
        ROOT
        / "results/revision_bspc_2026/promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602/harmonized_stageb_predictions.csv",
        kind="harmonized",
        method_filter="residualize_mfr_preserve_age_sex",
    ),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_table(df: pd.DataFrame, stem: str, max_rows: int = 500) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{stem}.csv", index=False)
    view = df.head(max_rows)
    try:
        md = view.to_markdown(index=False)
    except Exception:
        md = view.to_string(index=False)
    if len(df) > max_rows:
        md += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    (OUT / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    if den == 0 or pd.isna(num) or pd.isna(den):
        return float("nan")
    return float(num) / float(den)


def metrics_from_scores(y: Iterable[int], score: Iterable[float], pred: Iterable[int]) -> Dict[str, float]:
    y_arr = np.asarray(list(y), dtype=int)
    s_arr = np.asarray(list(score), dtype=float)
    p_arr = np.asarray(list(pred), dtype=int)
    tn = int(((y_arr == 0) & (p_arr == 0)).sum())
    fp = int(((y_arr == 0) & (p_arr == 1)).sum())
    fn = int(((y_arr == 1) & (p_arr == 0)).sum())
    tp = int(((y_arr == 1) & (p_arr == 1)).sum())
    return {
        "n": int(len(y_arr)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "auc": roc_auc_score(y_arr, s_arr) if len(np.unique(y_arr)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_arr, s_arr) if len(np.unique(y_arr)) == 2 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_arr, p_arr) if len(np.unique(y_arr)) == 2 else np.nan,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1": f1_score(y_arr, p_arr) if len(np.unique(y_arr)) == 2 else np.nan,
    }


def choose_threshold_target_sens(y: np.ndarray, score: np.ndarray, target: float = 0.70) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.5
    # precision/recall arrays are one longer than thresholds; align threshold i
    candidates: List[Tuple[float, float, float]] = []
    for thr, rec in zip(thresholds, recall[:-1]):
        pred = (score >= thr).astype(int)
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        spec = safe_div(tn, tn + fp)
        candidates.append((float(thr), float(rec), float(spec)))
    eligible = [c for c in candidates if c[1] >= target]
    if eligible:
        return max(eligible, key=lambda x: (x[2], x[1]))[0]
    return max(candidates, key=lambda x: (x[1], x[2]))[0]


def load_primary_predictions(src: PredictionSource) -> pd.DataFrame:
    if not src.path.exists():
        return pd.DataFrame()
    df = pd.read_csv(src.path)
    if src.kind == "harmonized" and src.method_filter is not None:
        df = df[df["harmonization_method"] == src.method_filter].copy()
    df = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURE)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    if df.empty:
        return df
    keep = [
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
        "threshold",
    ]
    df = df[keep].copy()
    df = df.rename(
        columns={
            "y_score": f"{src.label}_score",
            "y_pred": f"{src.label}_prediction",
            "threshold": f"{src.label}_threshold",
        }
    )
    return df


def error_type(y: int, pred: int) -> str:
    if y == 1 and pred == 1:
        return "TP"
    if y == 0 and pred == 0:
        return "TN"
    if y == 0 and pred == 1:
        return "FP"
    if y == 1 and pred == 0:
        return "FN"
    return "unknown"


def subject_level_error_table() -> pd.DataFrame:
    base = load_primary_predictions(SOURCES[0])
    if base.empty:
        raise RuntimeError("Missing promoted primary predictions")
    base = base.rename(
        columns={
            "promoted_score": "promoted_score",
            "promoted_prediction": "promoted_prediction",
            "promoted_threshold": "promoted_threshold",
        }
    )
    out = base.copy()
    for src in SOURCES[1:]:
        comp = load_primary_predictions(src)
        if comp.empty:
            out[f"{src.label}_score"] = np.nan
            out[f"{src.label}_prediction"] = np.nan
            out[f"{src.label}_threshold"] = np.nan
            continue
        cols = ["SubjectID", f"{src.label}_score", f"{src.label}_prediction", f"{src.label}_threshold"]
        out = out.merge(comp[cols], on="SubjectID", how="left")
    out["promoted_error_type"] = [error_type(int(y), int(p)) for y, p in zip(out["y_true"], out["promoted_prediction"])]
    out["promoted_score_margin"] = out["promoted_score"] - out["promoted_threshold"]
    out["promoted_abs_margin"] = out["promoted_score_margin"].abs()
    out["high_confidence_false_positive"] = (out["promoted_error_type"] == "FP") & (out["promoted_score_margin"] >= 0.20)
    out["high_confidence_false_negative"] = (out["promoted_error_type"] == "FN") & (out["promoted_score_margin"] <= -0.20)
    out["borderline_near_threshold"] = out["promoted_abs_margin"] <= 0.05
    for label in ["ch1only", "residualized_mfr", "foldcombat", "chmeanloss", "ch12_beta2p75"]:
        pred_col = f"{label}_prediction"
        if pred_col in out.columns:
            out[f"{label}_error_type"] = [
                error_type(int(y), int(p)) if not pd.isna(p) else ""
                for y, p in zip(out["y_true"], out[pred_col])
            ]
    out["corrected_by_ch1only_missed_by_promoted"] = out["promoted_error_type"].isin(["FP", "FN"]) & out["ch1only_error_type"].isin(["TP", "TN"])
    out["corrected_by_promoted_missed_by_ch1only"] = out["ch1only_error_type"].isin(["FP", "FN"]) & out["promoted_error_type"].isin(["TP", "TN"])
    return out


def high_confidence_errors(subjects: pd.DataFrame) -> pd.DataFrame:
    return subjects[
        subjects["high_confidence_false_positive"]
        | subjects["high_confidence_false_negative"]
        | subjects["borderline_near_threshold"]
        | subjects["corrected_by_ch1only_missed_by_promoted"]
        | subjects["corrected_by_promoted_missed_by_ch1only"]
    ].sort_values(["promoted_error_type", "promoted_abs_margin"], ascending=[True, False])


def manufacturer_score_shift(subjects: pd.DataFrame) -> pd.DataFrame:
    global_cn_median = float(subjects.loc[subjects["y_true"] == 0, "promoted_score"].median())
    rows: List[Dict[str, Any]] = []
    for (mfr, y), grp in subjects.groupby(["Manufacturer", "y_true"]):
        y_name = "AD" if int(y) == 1 else "CN"
        row: Dict[str, Any] = {
            "Manufacturer": mfr,
            "diagnosis": y_name,
            "n": int(len(grp)),
            "mean_score": float(grp["promoted_score"].mean()),
            "std_score": float(grp["promoted_score"].std(ddof=1)) if len(grp) > 1 else np.nan,
            "median_score": float(grp["promoted_score"].median()),
            "q10": float(grp["promoted_score"].quantile(0.10)),
            "q25": float(grp["promoted_score"].quantile(0.25)),
            "q75": float(grp["promoted_score"].quantile(0.75)),
            "q90": float(grp["promoted_score"].quantile(0.90)),
            "shift_vs_global_cn_median": float(grp["promoted_score"].median()) - global_cn_median,
            "global_cn_median": global_cn_median,
        }
        rows.append(row)
    for mfr, grp in subjects.groupby("Manufacturer"):
        if grp["y_true"].nunique() == 2:
            pred = (grp["promoted_score"] >= grp["promoted_threshold"]).astype(int)
            m = metrics_from_scores(grp["y_true"], grp["promoted_score"], pred)
            rows.append(
                {
                    "Manufacturer": mfr,
                    "diagnosis": "CN_vs_AD",
                    "n": int(len(grp)),
                    "mean_score": np.nan,
                    "std_score": np.nan,
                    "median_score": np.nan,
                    "q10": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "q90": np.nan,
                    "shift_vs_global_cn_median": np.nan,
                    "global_cn_median": global_cn_median,
                    "manufacturer_auc": m["auc"],
                    "manufacturer_pr_auc": m["pr_auc"],
                    "manufacturer_ba": m["balanced_accuracy"],
                    "manufacturer_f1": m["f1"],
                }
            )
    return pd.DataFrame(rows)


def foldwise_weakness(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold, grp in subjects.groupby("fold"):
        m = metrics_from_scores(grp["y_true"], grp["promoted_score"], grp["promoted_prediction"])
        philips_cn = grp[(grp["Manufacturer"] == "Philips") & (grp["y_true"] == 0)]
        ge_ad = grp[(grp["Manufacturer"] == "GE") & (grp["y_true"] == 1)]
        rd = pd.read_csv(PROMOTED_RUN / f"fold_{fold}/fold_{fold}_rate_distortion.csv")
        best = rd.loc[rd["L_val_betaMax"].idxmin()]
        mi = pd.read_csv(PROMOTED_RUN / f"fold_{fold}/fold_{fold}_test_latent_info_summary.csv")
        mi_pivot = mi.pivot_table(columns="variable", values="mi_sum_nats", aggfunc="first")
        leak = pd.read_csv(PROMOTED_RUN / f"fold_{fold}/fold_{fold}_test_scanner_leakage_summary.csv").iloc[0]
        rows.append(
            {
                "fold": int(fold),
                **m,
                "philips_cn_n": int(len(philips_cn)),
                "philips_cn_fp": int((philips_cn["promoted_prediction"] == 1).sum()),
                "philips_cn_fpr": safe_div(int((philips_cn["promoted_prediction"] == 1).sum()), int(len(philips_cn))),
                "ge_ad_n": int(len(ge_ad)),
                "ge_ad_fn": int((ge_ad["promoted_prediction"] == 0).sum()),
                "ge_ad_fnr": safe_div(int((ge_ad["promoted_prediction"] == 0).sum()), int(len(ge_ad))),
                "scanner_leakage_test_latent_ba": float(leak["acc_site_latent"]),
                "beta_KLD_over_D": safe_div(float(best["beta"]) * float(best["R_val_nats"]), float(best["D_val"])),
                "MI_Z_Y_nats": float(mi_pivot.get("Y_target", pd.Series([np.nan])).iloc[0]),
                "MI_Z_Manufacturer_nats": float(mi_pivot.get("Manufacturer", pd.Series([np.nan])).iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out["MI_Manufacturer_over_MI_Y"] = out["MI_Z_Manufacturer_nats"] / out["MI_Z_Y_nats"]
    return out


def latent_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("mu_")]


def latent_geometry(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for fold in sorted(subjects["fold"].unique()):
        train = pd.read_csv(PROMOTED_LATENT_CACHE / f"fold_{int(fold)}_trainDev_latent_mu.csv")
        test = pd.read_csv(PROMOTED_LATENT_CACHE / f"fold_{int(fold)}_test_latent_mu.csv")
        test = test.merge(
            subjects[["SubjectID", "promoted_error_type", "promoted_score", "promoted_threshold"]],
            on="SubjectID",
            how="left",
        )
        cols = latent_feature_cols(test)
        x_train = train[cols].to_numpy(float)
        x_test = test[cols].to_numpy(float)
        y_train = train["y"].to_numpy(int)
        cn_centroid = x_train[y_train == 0].mean(axis=0)
        ad_centroid = x_train[y_train == 1].mean(axis=0)
        lw = LedoitWolf().fit(x_train)
        pca = PCA(n_components=2, random_state=42).fit(x_train)
        test_pca = pca.transform(x_test)
        diag_sil = silhouette_score(x_test, test["y"].to_numpy(int)) if test["y"].nunique() > 1 else np.nan
        mfr_sil = silhouette_score(x_test, test["Manufacturer"].astype(str)) if test["Manufacturer"].nunique() > 1 else np.nan
        dist_cn = np.linalg.norm(x_test - cn_centroid, axis=1)
        dist_ad = np.linalg.norm(x_test - ad_centroid, axis=1)
        maha = lw.mahalanobis(x_test)
        test = test.assign(
            pca1=test_pca[:, 0],
            pca2=test_pca[:, 1],
            dist_to_train_cn_centroid=dist_cn,
            dist_to_train_ad_centroid=dist_ad,
            dist_ad_minus_cn=dist_ad - dist_cn,
            mahalanobis_to_train_support=maha,
        )
        focus = test[
            ((test["Manufacturer"] == "Philips") & (test["ResearchGroup_Mapped"] == "CN") & (test["promoted_error_type"] == "FP"))
            | ((test["Manufacturer"] == "GE") & (test["ResearchGroup_Mapped"] == "AD") & (test["promoted_error_type"] == "FN"))
        ].copy()
        for _, rec in focus.iterrows():
            detail_rows.append(
                {
                    "fold": int(fold),
                    "SubjectID": rec["SubjectID"],
                    "ResearchGroup_Mapped": rec["ResearchGroup_Mapped"],
                    "Manufacturer": rec["Manufacturer"],
                    "promoted_error_type": rec["promoted_error_type"],
                    "promoted_score": rec["promoted_score"],
                    "promoted_threshold": rec["promoted_threshold"],
                    "dist_to_train_cn_centroid": rec["dist_to_train_cn_centroid"],
                    "dist_to_train_ad_centroid": rec["dist_to_train_ad_centroid"],
                    "dist_ad_minus_cn": rec["dist_ad_minus_cn"],
                    "mahalanobis_to_train_support": rec["mahalanobis_to_train_support"],
                    "pca1_visualization_only": rec["pca1"],
                    "pca2_visualization_only": rec["pca2"],
                }
            )
        rows.append(
            {
                "fold": int(fold),
                "diagnosis_silhouette_test_latent": diag_sil,
                "manufacturer_silhouette_test_latent": mfr_sil,
                "pca_explained_var_1": float(pca.explained_variance_ratio_[0]),
                "pca_explained_var_2": float(pca.explained_variance_ratio_[1]),
                "philips_cn_fp_n": int(len(focus[(focus["Manufacturer"] == "Philips") & (focus["ResearchGroup_Mapped"] == "CN")])),
                "ge_ad_fn_n": int(len(focus[(focus["Manufacturer"] == "GE") & (focus["ResearchGroup_Mapped"] == "AD")])),
                "philips_cn_fp_mean_dist_ad_minus_cn": float(
                    focus[(focus["Manufacturer"] == "Philips") & (focus["ResearchGroup_Mapped"] == "CN")]["dist_ad_minus_cn"].mean()
                )
                if not focus.empty
                else np.nan,
                "ge_ad_fn_mean_dist_ad_minus_cn": float(
                    focus[(focus["Manufacturer"] == "GE") & (focus["ResearchGroup_Mapped"] == "AD")]["dist_ad_minus_cn"].mean()
                )
                if not focus.empty
                else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    write_table(detail, "latent_geometry_error_subject_distances")
    return summary


def inner_cv_predictions_age_sex_only() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    c_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    for fold in range(1, 6):
        train = pd.read_csv(PROMOTED_LATENT_CACHE / f"fold_{fold}_trainDev_latent_mu.csv")
        test = pd.read_csv(PROMOTED_LATENT_CACHE / f"fold_{fold}_test_latent_mu.csv")
        train = train[train["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
        test = test[test["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
        y = train["y"].to_numpy(int)
        strata = train["ResearchGroup_Mapped"].astype(str) + "_" + train["Manufacturer"].astype(str)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + fold)
        best_auc = -np.inf
        best_c = c_grid[0]
        best_oof = None
        for c in c_grid:
            oof = np.zeros(len(train), dtype=float)
            for tr_idx, va_idx in inner.split(train, strata):
                pipe = make_age_sex_pipe(c)
                pipe.fit(train.iloc[tr_idx][["Age", "Sex"]], y[tr_idx])
                oof[va_idx] = pipe.predict_proba(train.iloc[va_idx][["Age", "Sex"]])[:, 1]
            auc = roc_auc_score(y, oof)
            if auc > best_auc:
                best_auc = auc
                best_c = c
                best_oof = oof
        thr = choose_threshold_target_sens(y, best_oof)
        final = make_age_sex_pipe(best_c)
        final.fit(train[["Age", "Sex"]], y)
        score = final.predict_proba(test[["Age", "Sex"]])[:, 1]
        pred = (score >= thr).astype(int)
        out = test[["SubjectID", "tensor_idx", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y"]].copy()
        out = out.rename(columns={"y": "y_true"})
        out["feature_set"] = "age_sex_only"
        out["best_C"] = best_c
        out["inner_auc"] = best_auc
        out["threshold"] = thr
        out["y_score"] = score
        out["y_pred"] = pred
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def make_age_sex_pipe(c: float) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("age", StandardScaler(), ["Age"]),
            ("sex", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), ["Sex"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            ("model", LogisticRegression(C=c, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000)),
        ]
    )


def metadata_contribution() -> pd.DataFrame:
    oof = pd.read_csv(PROMOTED_OOF / "calib_pooled_metrics.csv")
    rows: List[Dict[str, Any]] = []
    for feature in ["z_only", "z_plus_age_sex"]:
        row = oof[
            (oof["model_name"] == PRIMARY_MODEL)
            & (oof["feature_set"] == feature)
            & (oof["calib_method"] == PRIMARY_CALIB)
            & (oof["threshold_strategy"] == PRIMARY_THRESHOLD)
        ].iloc[0]
        rows.append(
            {
                "feature_set": feature,
                "source": "existing_promoted_oof_ecdf",
                "auc": row["auc"],
                "pr_auc": row["pr_auc"],
                "balanced_accuracy": row["balanced_accuracy"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "f1": row["f1"],
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
            }
        )
    age_pred = inner_cv_predictions_age_sex_only()
    m = metrics_from_scores(age_pred["y_true"], age_pred["y_score"], age_pred["y_pred"])
    rows.append({"feature_set": "age_sex_only", "source": "nested_age_sex_only_audit", **m})
    write_table(age_pred, "metadata_age_sex_only_predictions")
    out = pd.DataFrame(rows)
    z = out.set_index("feature_set")
    if "z_plus_age_sex" in z.index and "z_only" in z.index:
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            delta = float(z.loc["z_plus_age_sex", metric]) - float(z.loc["z_only", metric])
            out.loc[out["feature_set"] == "z_plus_age_sex", f"delta_vs_z_only_{metric}"] = delta
    return out


def promoted_vs_ch1_complementarity(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for category, mask in {
        "both_correct": subjects["promoted_error_type"].isin(["TP", "TN"]) & subjects["ch1only_error_type"].isin(["TP", "TN"]),
        "both_wrong": subjects["promoted_error_type"].isin(["FP", "FN"]) & subjects["ch1only_error_type"].isin(["FP", "FN"]),
        "ch1_correct_promoted_wrong": subjects["corrected_by_ch1only_missed_by_promoted"],
        "promoted_correct_ch1_wrong": subjects["corrected_by_promoted_missed_by_ch1only"],
    }.items():
        grp = subjects[mask]
        rows.append(
            {
                "category": category,
                "n": int(len(grp)),
                "CN": int((grp["y_true"] == 0).sum()),
                "AD": int((grp["y_true"] == 1).sum()),
                "Philips": int((grp["Manufacturer"] == "Philips").sum()),
                "GE": int((grp["Manufacturer"] == "GE").sum()),
                "SIEMENS": int((grp["Manufacturer"] == "SIEMENS").sum()),
                "mean_promoted_score": float(grp["promoted_score"].mean()) if len(grp) else np.nan,
                "mean_ch1only_score": float(grp["ch1only_score"].mean()) if len(grp) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_recommendations(subjects: pd.DataFrame, manuf: pd.DataFrame, foldwise: pd.DataFrame, metadata: pd.DataFrame) -> None:
    philips_cn = manuf[(manuf["Manufacturer"] == "Philips") & (manuf["diagnosis"] == "CN")]
    philips_shift = float(philips_cn["shift_vs_global_cn_median"].iloc[0]) if not philips_cn.empty else np.nan
    worst_fold = foldwise.sort_values("auc").iloc[0]
    zdelta = metadata[metadata["feature_set"] == "z_plus_age_sex"].iloc[0]
    text = f"""# Correction Candidate Recommendations

The promoted model weakness is not a single fold-only defect. Fold `{int(worst_fold['fold'])}`
has the weakest AUC in this audit, but Philips CN false positives and manufacturer score
shift are present across folds.

Key observations:
- Philips CN median score shift versus global CN median: `{philips_shift:.6f}`.
- High-confidence promoted false positives: `{int(subjects['high_confidence_false_positive'].sum())}`.
- High-confidence promoted false negatives: `{int(subjects['high_confidence_false_negative'].sum())}`.
- Borderline cases within 0.05 of threshold: `{int(subjects['borderline_near_threshold'].sum())}`.
- z+Age/Sex vs z-only delta AUC: `{float(zdelta.get('delta_vs_z_only_auc', np.nan)):.6f}`;
  delta PR-AUC: `{float(zdelta.get('delta_vs_z_only_pr_auc', np.nan)):.6f}`.

Leakage-safe candidates that remain scientifically plausible:
1. Classifier-only manufacturer-aware weighting or residualization can be retained as a
   sensitivity, but prior residualization reduced Philips FPR/leakage at the cost of AUC/BA.
   It should not be promoted unless an inner-CV-only variant improves ranking and operating
   metrics simultaneously.
2. Report manufacturer-specific operating behavior explicitly rather than tuning thresholds
   post hoc. A manufacturer-specific threshold would require a predeclared nested procedure
   and enough sample size; it is not justified from these test-fold errors alone.
3. If further internal work is unavoidable, prefer leakage-safe classifier-only experiments
   from frozen latents over new VAE training: sample weighting by Diagnosis x Manufacturer,
   calibrated linear models, or fold-local removal of a small number of manufacturer
   directions with all choices made inside inner CV.
4. Do not use OASIS to select among these corrections. OASIS remains an external stress test
   or future calibration/test target only.

Not recommended:
- Global threshold tuning on pooled OOF predictions.
- Manufacturer-specific thresholds fitted on outer-test errors.
- Excluding Philips CN or GE AD subjects based on error phenotype.
- Additional VAE training unless a predeclared, leakage-safe mechanism has strong evidence
  from classifier-only sensitivity.
"""
    (OUT / "correction_candidate_recommendations.md").write_text(text, encoding="utf-8")

    final = """# Final Interpretation

The promoted [1,0,2] latent384 beta3.75 model remains the current primary model,
but its correctable-looking weakness is manufacturer-linked score geometry rather
than simple underfitting of the diagnostic axis. Philips CN subjects are shifted
upward in score, GE AD false negatives remain a recurring asymmetric error mode,
and scanner/manufacturer separability is reduced but not eliminated in z.

The ch1-only model corrects some promoted errors and has stronger internal AUC,
but it does not cleanly solve the operating profile or external-stress-test concerns.
Residualized and foldwise-harmonized sensitivities show that manufacturer signal
can be reduced leakage-safely, but the AD/CN ranking and Philips CN FPR trade-off
does not cleanly improve. The chmeanloss and ch12 beta2.75 branches also do not
provide a promotable multichannel correction.

The next defensible step is not OASIS-based model selection or post-hoc threshold
tuning. If any follow-up is pursued, it should be a frozen-latent, inner-CV-only
classifier sensitivity aimed at manufacturer-aware operating behavior, with OASIS
held out for external stress testing or a future locked calibration/test protocol.
"""
    (OUT / "final_interpretation.md").write_text(final, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = subject_level_error_table()
    write_table(subjects, "subject_level_error_table")
    write_table(high_confidence_errors(subjects), "high_confidence_errors")
    manuf = manufacturer_score_shift(subjects)
    write_table(manuf, "manufacturer_score_shift")
    foldwise = foldwise_weakness(subjects)
    write_table(foldwise, "foldwise_weakness_summary")
    geom = latent_geometry(subjects)
    write_table(geom, "latent_geometry_error_summary")
    metadata = metadata_contribution()
    write_table(metadata, "metadata_contribution_summary")
    comp = promoted_vs_ch1_complementarity(subjects)
    write_table(comp, "promoted_vs_ch1_error_complementarity")
    write_recommendations(subjects, manuf, foldwise, metadata)
    command_log = {
        "created_utc": now_utc(),
        "promoted_run": rel(PROMOTED_RUN),
        "promoted_oof": rel(PROMOTED_OOF),
        "output_dir": rel(OUT),
        "prediction_sources": [{"label": s.label, "path": rel(s.path), "available": s.path.exists()} for s in SOURCES],
        "guardrails": [
            "no VAE training",
            "no OASIS scoring",
            "no OASIS threshold/calibration fitting",
            "no tensor modification",
            "no metadata modification",
            "no model artifact overwrite",
        ],
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
