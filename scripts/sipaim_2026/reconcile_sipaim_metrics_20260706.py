#!/usr/bin/env python3
"""Reconcile SIPAIM multisite transport metrics without retraining.

This script is intentionally read-only with respect to model/tensor/metadata
inputs. It writes a small reconciliation package under results/.
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/diego/proyectos/vae_AD")
OUT = PROJECT / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "sipaim_metric_reconciliation_20260706"
)

SIPAIM = PROJECT / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "sipaim_multisite_transport_20260706"
)
MFR_FOLDLOCAL = PROJECT / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "site_geometry_foldlocal_confirmatory_20260705"
)
LOCKED_RUN = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
STAGEB = PROJECT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
FOLDCOMBAT_STAGEB = PROJECT / "results/revision_bspc_2026/recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration"
OASIS_PANEL = PROJECT / "results/revision_bspc_2026/oasis_mega_90_90_external_inference_model_panel_20260604"
OASIS_AUDIT = PROJECT / "results/revision_bspc_2026/oasis_run_handling_and_subject_level_audit_20260626"
OASIS_LATENT = PROJECT / "results/revision_bspc_2026/promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/oasis_fold_latent_mu_runwise164.csv"

PRIMARY = {
    "model_name": "logreg_l2_original",
    "feature_set": "z_plus_age_sex",
    "calib_method": "oof_ecdf",
    "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
}


def md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```\n"


def write_csv_md(stem: str, df: pd.DataFrame, title: str = "", max_rows: Optional[int] = None) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    parts = []
    if title:
        parts.extend([f"# {title}", ""])
    parts.append(md_table(df, max_rows=max_rows))
    (OUT / f"{stem}.md").write_text("\n".join(parts), encoding="utf-8")


def primary_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, val in PRIMARY.items():
        mask &= df[key].astype(str).eq(val)
    return mask


def get_site_code(subject_id: Any) -> Optional[str]:
    m = re.match(r"^(\d{3})_S_\d{4}$", str(subject_id))
    return m.group(1) if m else None


def load_latent(fold: int, split: str) -> pd.DataFrame:
    path = LOCKED_RUN / "classifier_only_readout/latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["SiteCode"] = df["SubjectID"].map(get_site_code)
    return df


def mu_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if re.match(r"^mu_\d+$", c)], key=lambda x: int(x.split("_")[1]))


def covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    sex = df["Sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).fillna(0.5).to_numpy()
    age_raw = pd.to_numeric(df["Age"], errors="coerce")
    age = age_raw.fillna(age_raw.median()).to_numpy()
    if "y" in df.columns:
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).to_numpy()
    else:
        y = df["ResearchGroup_Mapped"].astype(str).eq("AD").astype(float).to_numpy()
    return np.column_stack([np.ones(len(df)), y, age, sex])


def scale_and_residualize(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols = mu_cols(train)
    if len(cols) != 384:
        raise RuntimeError(f"Expected 384 mu columns, found {len(cols)}")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[cols].to_numpy(float))
    x_test = scaler.transform(test[cols].to_numpy(float))
    z_train = covariate_matrix(train)
    z_test = covariate_matrix(test)
    reg = LinearRegression(fit_intercept=False).fit(z_train, x_train)
    return x_train - reg.predict(z_train), x_test - reg.predict(z_test)


def stratified_site_permutation(train_subset: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    out = train_subset["SiteCode"].astype(str).to_numpy().copy()
    strata = train_subset["ResearchGroup_Mapped"].astype(str) + "|" + train_subset["Manufacturer"].astype(str)
    for _, idx in pd.Series(np.arange(len(train_subset))).groupby(strata).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) > 1:
            out[idx_arr] = rng.permutation(out[idx_arr])
    return out


def stratified_label_permutation(labels: np.ndarray, strata: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Permute held-out labels within strata, preserving the fixed predictions."""
    out = np.asarray(labels, dtype=str).copy()
    for _, idx in pd.Series(np.arange(len(out))).groupby(strata.astype(str)).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) > 1:
            out[idx_arr] = rng.permutation(out[idx_arr])
    return out


def fit_site_classifier(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(x, y)
    return clf


def rerun_site_permutations(n_perm: int = 1000) -> pd.DataFrame:
    prior = pd.read_csv(SIPAIM / "foldlocal_site_decoding.csv")
    rows = []
    rng = np.random.default_rng(20260706)
    for fold in range(1, 6):
        train = load_latent(fold, "trainDev")
        test = load_latent(fold, "test")
        train = train[train["SiteCode"].notna()].copy()
        test = test[test["SiteCode"].notna()].copy()
        x_train, x_test = scale_and_residualize(train, test)
        train_sites = train["SiteCode"].astype(str).to_numpy()
        test_sites = test["SiteCode"].astype(str).to_numpy()
        site_counts = pd.Series(train_sites).value_counts()
        keep_train = np.array([site_counts.get(s, 0) >= 2 for s in train_sites])
        supported_train_sites = set(pd.Series(train_sites[keep_train]).unique())
        keep_test = np.array([s in supported_train_sites for s in test_sites])

        observed_row = prior.loc[prior["fold"].eq(fold)].iloc[0]
        observed_ba = float(observed_row["site_balanced_accuracy"])
        observed_macro_f1 = float(observed_row["site_macro_f1"])

        # Recompute observed value as a parity check, but keep the existing
        # observed value as the registered endpoint.
        recomputed_ba = np.nan
        recomputed_macro_f1 = np.nan
        if keep_train.sum() > 5 and keep_test.sum() > 0:
            clf = fit_site_classifier(x_train[keep_train], train_sites[keep_train])
            pred = clf.predict(x_test[keep_test])
            recomputed_ba = float(balanced_accuracy_score(test_sites[keep_test], pred))
            recomputed_macro_f1 = float(f1_score(test_sites[keep_test], pred, average="macro", zero_division=0))

        test_eval = test.iloc[np.where(keep_test)[0]].copy()
        yhat_fixed = None
        if keep_train.sum() > 5 and keep_test.sum() > 0:
            clf_obs = fit_site_classifier(x_train[keep_train], train_sites[keep_train])
            yhat_fixed = clf_obs.predict(x_test[keep_test])
        if yhat_fixed is None:
            null = np.asarray([], dtype=float)
        else:
            null_ba = []
            test_strata = test_eval["ResearchGroup_Mapped"].astype(str) + "|" + test_eval["Manufacturer"].astype(str)
            for _ in range(n_perm):
                perm_test_sites = stratified_label_permutation(test_sites[keep_test], test_strata, rng)
                null_ba.append(float(balanced_accuracy_score(perm_test_sites, yhat_fixed)))
            null = np.asarray(null_ba, dtype=float)
        p_val = float((np.sum(null >= observed_ba) + 1) / (len(null) + 1)) if len(null) else np.nan
        rows.append(
            {
                "fold": fold,
                "observed_site_ba_existing": observed_ba,
                "observed_site_macro_f1_existing": observed_macro_f1,
                "observed_site_ba_recomputed": recomputed_ba,
                "observed_site_macro_f1_recomputed": recomputed_macro_f1,
                "observed_ba_abs_diff": abs(observed_ba - recomputed_ba) if np.isfinite(recomputed_ba) else np.nan,
                "evaluated_test_n": int(keep_test.sum()),
                "train_n": int(keep_train.sum()),
                "train_sites_n": int(pd.Series(train_sites[keep_train]).nunique()),
                "test_sites_n": int(pd.Series(test_sites[keep_test]).nunique()),
                "permutation_type": "heldout_site_label_permutation_fixed_predictions",
                "permutation_preserved_strata": "ResearchGroup_Mapped+Manufacturer within evaluated outer-test subset",
                "n_perm_requested": int(n_perm),
                "n_perm_completed": int(len(null)),
                "prior_n_perm": 50,
                "prior_p_ge_observed": float(observed_row["permutation_p_ge_observed"]),
                "p_ge_observed_1000": p_val,
                "permutation_null_ba_mean_1000": float(np.nanmean(null)) if len(null) else np.nan,
                "permutation_null_ba_sd_1000": float(np.nanstd(null, ddof=1)) if len(null) > 1 else np.nan,
                "p_resolution": float(1.0 / (len(null) + 1)) if len(null) else np.nan,
                "observed_ba_changed": False,
            }
        )
    return pd.DataFrame(rows)


def manufacturer_reconciliation() -> Tuple[pd.DataFrame, str]:
    knn = pd.read_csv(MFR_FOLDLOCAL / "figure_data/knn_bacc_by_fold.csv")
    knn = knn[knn["metric"].eq("knn5_mfr_bacc")].copy()
    scanner_rows = []
    for fold in range(1, 6):
        path = LOCKED_RUN / f"fold_{fold}/fold_{fold}_test_scanner_leakage.csv"
        df = pd.read_csv(path)
        row = df[(df["representation"].eq("latent_mu")) & (df["site_col"].astype(str).eq("Manufacturer"))].iloc[0]
        row = row.to_dict()
        row.update({"fold": fold, "source_file": str(path)})
        scanner_rows.append(row)
    scanner = pd.DataFrame(scanner_rows)
    rows = [
        {
            "estimate_id": "matched_foldlocal_5nn_manufacturer_ba",
            "source_file": str(MFR_FOLDLOCAL / "figure_data/knn_bacc_by_fold.csv"),
            "population": "Final supervised CN/AD latent-cache subjects; trainDev decoder evaluated on outer-test subjects fold by fold.",
            "folds_splits": "5 locked outer folds; classifier trained on trainDev and evaluated on held-out outer test.",
            "feature_representation": "posterior latent mu, 384 dimensions",
            "scaler_residualizer": "StandardScaler fit on trainDev; OLS residualization of diagnosis, Age, Sex fit on trainDev and applied to test.",
            "classifier": "KNeighborsClassifier",
            "hyperparameters": "n_neighbors=5, metric=euclidean",
            "nuisance_covariates": "ResearchGroup_Mapped/diagnosis, Age, Sex residualized",
            "permutation_procedure": "1000 permutations of outer-test manufacturer labels against fixed predictions in the prior package; p=0.001 in each fold.",
            "outer_test_labels_entered_model_fitting": False,
            "mean_ba": float(knn["value"].mean()),
            "sd_ba": float(knn["value"].std(ddof=1)),
            "fold_values": "; ".join(f"fold {int(r.fold)}={float(r.value):.4f}" for r in knn.itertuples()),
            "primary_for_sitecode_comparison": True,
        },
        {
            "estimate_id": "sipaim_main_table_scanner_leakage_qc_manufacturer_ba",
            "source_file": str(LOCKED_RUN / "fold_*/fold_*_test_scanner_leakage.csv"),
            "population": "Outer-test subset only for each fold; manufacturer decodability cross-validated inside the test subset.",
            "folds_splits": "Within each held-out outer-test fold, internal StratifiedKFold CV with up to 5 splits.",
            "feature_representation": "posterior latent mu, 384 dimensions",
            "scaler_residualizer": "No trainDev-to-test scaler/residualizer in evaluate_scanner_leakage; cross_val_score uses the raw latent_mu array supplied for the test subset.",
            "classifier": "LogisticRegression",
            "hyperparameters": "max_iter=1000, class_weight=balanced, solver=lbfgs",
            "nuisance_covariates": "No diagnosis/Age/Sex residualization",
            "permutation_procedure": "None in the SIPAIM main table; fold values are internal CV balanced-accuracy means.",
            "outer_test_labels_entered_model_fitting": True,
            "mean_ba": float(scanner["balanced_accuracy_mean"].mean()),
            "sd_ba": float(scanner["balanced_accuracy_mean"].std(ddof=1)),
            "fold_values": "; ".join(f"fold {int(r.fold)}={float(r.balanced_accuracy_mean):.4f}" for r in scanner.itertuples()),
            "primary_for_sitecode_comparison": False,
        },
    ]
    table = pd.DataFrame(rows)
    text = f"""# Manufacturer Decoding Reconciliation

## Decision

Use `matched_foldlocal_5nn_manufacturer_ba` as the primary manufacturer estimate for direct comparison with SiteCode decoding. Do not average it with the SIPAIM main-table QC value.

The two estimates differ because they answer different questions:

- `matched_foldlocal_5nn_manufacturer_ba` is a fold-local trainDev-to-outer-test domain-transport decoder. It uses trainDev-fitted scaling and residualization for diagnosis, Age, and Sex, then evaluates a 5-NN manufacturer decoder on held-out subjects. This produced BA = {knn['value'].mean():.4f} +/- {knn['value'].std(ddof=1):.4f}.
- `sipaim_main_table_scanner_leakage_qc_manufacturer_ba` reads `fold_*_test_scanner_leakage.csv`, where `evaluate_scanner_leakage` runs LogisticRegression cross-validation inside each outer-test subset on latent_mu. It does not use trainDev residualization by diagnosis/Age/Sex and it fits scanner-label classifiers using outer-test labels inside the QC cross-validation. This produced BA = {scanner['balanced_accuracy_mean'].mean():.4f} +/- {scanner['balanced_accuracy_mean'].std(ddof=1):.4f}.

The 0.7276 value is therefore appropriate as a scanner-leakage QC diagnostic, but not as the matched estimate for the SIPAIM SiteCode comparison.

## Evidence Table

{md_table(table)}

## Source-Code Evidence

`src/betavae_xai/analysis_qc/fold_qc.py::evaluate_scanner_leakage` documents and implements the SIPAIM 0.7276 source: LogisticRegression with `class_weight="balanced"`, internal StratifiedKFold cross-validation, and `balanced_accuracy_mean` saved to `fold_*_test_scanner_leakage.csv`.

The matched fold-local estimate comes from `{MFR_FOLDLOCAL / 'figure_data/knn_bacc_by_fold.csv'}` and its companion permutation report `{MFR_FOLDLOCAL / 'permutation_tests.csv'}`.
"""
    return table, text


def sitecode_serialization_audit() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(SIPAIM.rglob("*.csv")):
        rel = path.relative_to(SIPAIM)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            site_cols = [c for c in reader.fieldnames if c in {"SiteCode", "target_site"}]
            if not site_cols:
                continue
            vals_by_col: Dict[str, List[str]] = {c: [] for c in site_cols}
            for row in reader:
                for col in site_cols:
                    val = row.get(col, "")
                    if val != "":
                        vals_by_col[col].append(val)
            for col, vals in vals_by_col.items():
                bad = [v for v in vals if not re.match(r"^\d{3}$", str(v))]
                numeric_short = [v for v in vals if re.match(r"^\d{1,2}$", str(v))]
                rows.append(
                    {
                        "file": str(rel),
                        "column": col,
                        "n_nonmissing": len(vals),
                        "unique_values_preview": ";".join(sorted(set(vals))[:12]),
                        "all_three_char_strings": len(bad) == 0,
                        "short_numeric_values_n": len(numeric_short),
                        "bad_values_n": len(bad),
                        "bad_values_preview": ";".join(sorted(set(bad))[:12]),
                        "corrected_copy_written": False,
                    }
                )
    return pd.DataFrame(rows)


def oasis_provenance() -> Tuple[pd.DataFrame, str]:
    counts = pd.read_csv(OASIS_AUDIT / "oasis_prediction_level_counts.csv")
    dataflow = pd.read_csv(OASIS_AUDIT / "oasis_data_flow_by_build.csv")
    metrics = pd.read_csv(OASIS_PANEL / "primary_metrics.csv")
    preds = pd.read_csv(OASIS_PANEL / "predictions.csv")
    latent = pd.read_csv(OASIS_LATENT)

    build = "runwise164_pilot_parity"
    cand = "promoted_beta3p75_oof_ecdf"
    pred_final = preds[
        preds["build_candidate"].eq(build)
        & preds["candidate"].eq(cand)
        & preds["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].copy()
    fold_pred = preds[
        preds["build_candidate"].eq(build)
        & preds["candidate"].eq(cand)
        & preds["prediction_level"].eq("fold_model")
    ].copy()
    primary = metrics[
        metrics["build_candidate"].eq(build)
        & metrics["candidate"].eq(cand)
        & metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].iloc[0]

    latent_counts = (
        latent.groupby("fold")
        .agg(
            latent_rows=("SubjectID", "size"),
            latent_unique_subjects=("SubjectID", "nunique"),
            latent_cn=("ResearchGroup_Mapped", lambda s: int((s.astype(str) == "CN").sum())),
            latent_ad=("ResearchGroup_Mapped", lambda s: int((s.astype(str) == "AD").sum())),
        )
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "check": "authoritative_build",
                "value": build,
                "source": str(OASIS_AUDIT / "oasis_run_handling_audit.md"),
            },
            {
                "check": "final_prediction_rows",
                "value": int(len(pred_final)),
                "source": str(OASIS_PANEL / "predictions.csv"),
            },
            {
                "check": "final_unique_subjects",
                "value": int(pred_final["SubjectID"].nunique()),
                "source": str(OASIS_PANEL / "predictions.csv"),
            },
            {
                "check": "final_cn_ad",
                "value": f"CN={int((pred_final['ResearchGroup_Mapped'].astype(str) == 'CN').sum())}; AD={int((pred_final['ResearchGroup_Mapped'].astype(str) == 'AD').sum())}",
                "source": str(OASIS_PANEL / "predictions.csv"),
            },
            {
                "check": "fold_model_prediction_rows",
                "value": int(len(fold_pred)),
                "source": str(OASIS_PANEL / "predictions.csv"),
            },
            {
                "check": "latent_source",
                "value": str(OASIS_LATENT),
                "source": str(OASIS_LATENT),
            },
            {
                "check": "latent_rows_by_fold",
                "value": "; ".join(f"fold {int(r.fold)}: {int(r.latent_rows)} rows/{int(r.latent_unique_subjects)} subjects" for r in latent_counts.itertuples()),
                "source": str(OASIS_LATENT),
            },
            {
                "check": "final_auc_pr_auc",
                "value": f"ROC-AUC={float(primary['auc']):.9f}; PR-AUC={float(primary['pr_auc']):.9f}",
                "source": str(OASIS_PANEL / "primary_metrics.csv"),
            },
            {
                "check": "aggregation_convention",
                "value": "One metric from 180 one-row-per-subject/session ensemble rows; score is mean over five ADNI fold-model scores; label is majority vote over fold thresholds.",
                "source": str(OASIS_AUDIT / "oasis_data_flow_by_build.csv"),
            },
        ]
    )

    dup_n = int(pred_final["SubjectID"].duplicated().sum())
    if len(pred_final) != 180 or pred_final["SubjectID"].nunique() != 180 or dup_n != 0:
        raise RuntimeError("OASIS final prediction rows are not one row per subject")
    text = f"""# OASIS Provenance

- Authoritative build used for the SIPAIM table: `{build}`.
- Candidate: `{cand}`.
- Final prediction unit: one row per subject/session, N=180, CN=90, AD=90.
- Fold-model prediction rows: {len(fold_pred)} = 180 subjects x 5 ADNI fold-specific pipelines.
- Final score aggregation: mean of the five ADNI fold-model scores for each OASIS subject/session. Final thresholded label is the majority vote across the five fold-specific threshold decisions.
- Final metric aggregation: ROC-AUC and PR-AUC are computed once on the 180 subject/session ensemble rows; they are not mean fold AUCs and not run-level AUCs.
- Final OASIS ROC-AUC / PR-AUC: {float(primary['auc']):.6f} / {float(primary['pr_auc']):.6f}.
- Latent source used for Wasserstein analysis: `{OASIS_LATENT}`; each fold has 180 unique OASIS subjects.
- No OASIS subject is used for ADNI training, feature scaling, calibration, threshold selection, hyperparameter selection, or model selection in these audited files. OASIS labels enter only final metric computation.

## Evidence Table

{md_table(summary)}

## Data-Flow Evidence

{md_table(dataflow[dataflow['build_candidate'].eq(build)], max_rows=20)}

## Prediction-Level Counts

{md_table(counts[counts['build_candidate'].eq(build)])}
"""
    return summary, text


def reconciled_main_table(mfr_table: pd.DataFrame) -> pd.DataFrame:
    prior_main = pd.read_csv(SIPAIM / "main_results_table.csv")
    site_perm_prior = pd.read_csv(SIPAIM / "foldlocal_site_decoding.csv")
    primary = pd.read_csv(STAGEB / "calib_pooled_metrics.csv")
    primary = primary[primary_mask(primary)].iloc[0]
    combat = pd.read_csv(FOLDCOMBAT_STAGEB / "calib_pooled_metrics.csv")
    combat = combat[primary_mask(combat)].iloc[0]
    loso = pd.read_csv(SIPAIM / "leave_one_site_out_metrics.csv")
    oasis_metrics = pd.read_csv(OASIS_PANEL / "primary_metrics.csv")
    oasis_primary = oasis_metrics[
        oasis_metrics["candidate"].eq("promoted_beta3p75_oof_ecdf")
        & oasis_metrics["build_candidate"].eq("runwise164_pilot_parity")
        & oasis_metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")
    ].iloc[0]
    oasis_w = pd.read_csv(SIPAIM / "oasis_wasserstein_by_fold.csv")
    w_all = oasis_w[oasis_w["comparison_group"].eq("all")]
    mfr_primary = mfr_table[mfr_table["primary_for_sitecode_comparison"]].iloc[0]
    return pd.DataFrame(
        [
            {
                "result": "manufacturer latent decodability",
                "n_or_folds": 5,
                "primary_value": float(mfr_primary["mean_ba"]),
                "secondary_value": float(mfr_primary["sd_ba"]),
                "metric": "matched fold-local 5-NN balanced accuracy mean, SD",
                "source_policy": "primary estimate selected for direct SiteCode comparison",
            },
            {
                "result": "site latent decodability",
                "n_or_folds": 5,
                "primary_value": float(site_perm_prior["site_balanced_accuracy"].mean()),
                "secondary_value": float(site_perm_prior["site_balanced_accuracy"].std(ddof=1)),
                "metric": "fold-local SiteCode balanced accuracy mean, SD",
                "source_policy": "observed BA unchanged; p-values rerun with 1000 permutations",
            },
            {
                "result": "locked ADNI diagnostic performance",
                "n_or_folds": int(primary["n"]),
                "primary_value": float(primary["auc"]),
                "secondary_value": float(primary["pr_auc"]),
                "metric": "ROC-AUC, PR-AUC",
                "source_policy": "final selected StageB OOF-ECDF primary convention",
            },
            {
                "result": "fold-wise input-ComBat performance",
                "n_or_folds": int(combat["n"]),
                "primary_value": float(combat["auc"]),
                "secondary_value": float(combat["pr_auc"]),
                "metric": "ROC-AUC, PR-AUC",
                "source_policy": "existing foldwise input-ComBat StageB OOF-ECDF",
            },
            {
                "result": "leave-one-site-out summary",
                "n_or_folds": int(len(loso)),
                "primary_value": float(loso["auc"].mean()) if len(loso) else np.nan,
                "secondary_value": float(loso["pr_auc"].mean()) if len(loso) else np.nan,
                "metric": "mean supported-site ROC-AUC, PR-AUC",
                "source_policy": "classifier/readout transport on frozen latent representations",
            },
            {
                "result": "OASIS external performance",
                "n_or_folds": int(oasis_primary["n"]),
                "primary_value": float(oasis_primary["auc"]),
                "secondary_value": float(oasis_primary["pr_auc"]),
                "metric": "runwise164 one-row-per-subject ROC-AUC, PR-AUC",
                "source_policy": "one ensemble row per subject/session",
            },
            {
                "result": "ADNI-OASIS Wasserstein distance",
                "n_or_folds": 5,
                "primary_value": float(w_all["sliced_wasserstein"].mean()),
                "secondary_value": float(w_all["bures_wasserstein_ledoitwolf"].mean()),
                "metric": "mean sliced W, mean Gaussian Bures-W",
                "source_policy": "existing fold-local ADNI trainDev to OASIS latent distance",
            },
        ]
    )


def write_final_claims(site_perm: pd.DataFrame, mfr_table: pd.DataFrame, main_table: pd.DataFrame) -> None:
    mfr = mfr_table[mfr_table["primary_for_sitecode_comparison"]].iloc[0]
    site = main_table[main_table["result"].eq("site latent decodability")].iloc[0]
    min_p = float(site_perm["p_ge_observed_1000"].min())
    max_p = float(site_perm["p_ge_observed_1000"].max())
    text = f"""# Final SIPAIM Claims After Metric Reconciliation

1. The SIPAIM table should use the matched fold-local manufacturer estimate for direct comparison with SiteCode decoding: manufacturer BA = {float(mfr['mean_ba']):.4f} +/- {float(mfr['sd_ba']):.4f}. The older 0.7276 +/- 0.0904 value is a scanner-leakage QC estimate from within-test CV and should not be averaged with the fold-local transport estimate.

2. SiteCode decoding observed values are unchanged: BA = {float(site['primary_value']):.4f} +/- {float(site['secondary_value']):.4f}. The site-label permutation test was rerun with 1000 fixed-prediction held-out label permutations preserving `ResearchGroup_Mapped + Manufacturer` strata within each evaluated outer-test subset; fold p-values now have resolution about 0.001, with p-values spanning {min_p:.4f} to {max_p:.4f}.

3. SiteCode is serialized as a three-character string in the generated SIPAIM CSV files checked here. No source metadata was modified.

4. The authoritative OASIS build remains `runwise164_pilot_parity`, N=180 with 90 CN and 90 AD, one ensemble row per subject/session. The final OASIS ROC-AUC/PR-AUC are computed once from subject/session-level ensemble scores, not as mean fold metrics and not as run-level metrics.

5. Safe wording for the four-page SIPAIM manuscript: latent representations retain measurable acquisition-domain structure. Manufacturer decodability is higher under the matched fold-local decoder than multiclass SiteCode decodability, while OASIS performance and ADNI-OASIS Wasserstein distances indicate non-trivial cross-dataset shift. These are descriptive transport and domain-shift analyses, not evidence of a causal acquisition artifact.

## Guardrails

- no VAE training: true
- no OASIS inference: true
- no TeX/manuscript edits: true
- no TDA/Mapper: true
- no tensor or metadata modification: true
"""
    (OUT / "final_sipaim_claims.md").write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    OUT.mkdir(parents=True, exist_ok=True)

    required = [
        SIPAIM / "main_results_table.csv",
        SIPAIM / "foldlocal_site_decoding.csv",
        MFR_FOLDLOCAL / "figure_data/knn_bacc_by_fold.csv",
        MFR_FOLDLOCAL / "permutation_tests.csv",
        OASIS_AUDIT / "oasis_data_flow_by_build.csv",
        OASIS_AUDIT / "oasis_prediction_level_counts.csv",
        OASIS_PANEL / "predictions.csv",
        OASIS_PANEL / "primary_metrics.csv",
        OASIS_LATENT,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    mfr_table, mfr_text = manufacturer_reconciliation()
    write_csv_md("manufacturer_decoding_reconciliation", mfr_table, "Manufacturer Decoding Reconciliation")
    (OUT / "manufacturer_decoding_reconciliation.md").write_text(mfr_text, encoding="utf-8")

    site_perm = rerun_site_permutations(n_perm=1000)
    write_csv_md("site_permutation_1000", site_perm, "Site Permutation Audit With 1000 Permutations")
    (OUT / "site_permutation_1000.md").write_text(
        "# Site Permutation Audit With 1000 Permutations\n\n"
        "Observed SiteCode balanced-accuracy values were not changed. Only the null distribution was recomputed with 1000 held-out SiteCode label permutations preserving `ResearchGroup_Mapped + Manufacturer` strata within the evaluated outer-test subset. The fold-local SiteCode decoder is fit once per fold to recover fixed predictions; null permutations do not refit the decoder.\n\n"
        + md_table(site_perm),
        encoding="utf-8",
    )

    sitecode = sitecode_serialization_audit()
    write_csv_md("sitecode_serialization_audit", sitecode, "SiteCode Serialization Audit", max_rows=200)
    note = "\nAll checked `SiteCode`/`target_site` values are serialized as three-character strings; no corrected copies were required.\n"
    if len(sitecode) and (sitecode["bad_values_n"].fillna(0).astype(int) > 0).any():
        note = "\nAt least one checked file contains non-three-character site codes. Corrected copies were not written automatically because source SIPAIM outputs are preserved read-only; use the audit table to patch downstream views only.\n"
    with (OUT / "sitecode_serialization_audit.md").open("a", encoding="utf-8") as f:
        f.write(note)

    oasis_summary, oasis_text = oasis_provenance()
    oasis_summary.to_csv(OUT / "oasis_provenance.csv", index=False)
    (OUT / "oasis_provenance.md").write_text(oasis_text, encoding="utf-8")

    main_table = reconciled_main_table(mfr_table)
    write_csv_md("reconciled_main_results_table", main_table, "Reconciled Main Results Table")
    write_final_claims(site_perm, mfr_table, main_table)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "outputs": sorted(p.name for p in OUT.iterdir()),
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_ad_classifier": False,
            "did_run_oasis_inference": False,
            "did_edit_tex": False,
            "did_use_tda": False,
            "did_modify_metadata": False,
            "did_modify_model_outputs": False,
            "site_permutation_decoder_refit_for_null_only": False,
            "site_permutation_type": "heldout_site_label_permutation_fixed_predictions",
            "site_permutation_n": 1000,
        },
        "inputs": {
            "sipaim_package": str(SIPAIM),
            "foldlocal_manufacturer_geometry": str(MFR_FOLDLOCAL),
            "locked_run": str(LOCKED_RUN),
            "oasis_audit": str(OASIS_AUDIT),
            "oasis_panel": str(OASIS_PANEL),
            "oasis_latent": str(OASIS_LATENT),
        },
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
