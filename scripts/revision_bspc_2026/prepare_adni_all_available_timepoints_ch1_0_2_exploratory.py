#!/usr/bin/env python3
"""Prepare an exploratory ADNI all-available-timepoints branch preflight.

This script is intentionally read-only with respect to locked tensors,
metadata, ledgers, configs, and model outputs. It does not build a new tensor
and does not launch training. The output package documents the confounding
guardrails required before an all-timepoint [1,0,2] sensitivity branch can be
considered scientifically interpretable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    ROOT
    / "results/revision_bspc_2026/adni_all_available_timepoints_ch1_0_2_exploratory"
)
TIMEPOINT_AUDIT_DIR = (
    ROOT / "results/revision_bspc_2026/adni_timepoint_availability_confounding_audit"
)
LONGER_PREFLIGHT_DIR = (
    ROOT / "results/revision_bspc_2026/adni_longer_timeseries_connectome_preflight"
)
SUBJECT_TIMEPOINT_TABLE = TIMEPOINT_AUDIT_DIR / "subject_timepoint_table.csv"
CONFOUNDING_TESTS = TIMEPOINT_AUDIT_DIR / "confounding_tests.csv"
CANDIDATE_RETENTION = TIMEPOINT_AUDIT_DIR / "candidate_length_retention.csv"
LOCKED_CONFIG = (
    ROOT
    / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
)
LOCKED_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
LOCKED_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

CHANNELS_TO_USE = [1, 0, 2]
CHANNEL_INDEX_TO_NAME = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
}
BRANCH_NAME = "adni_all_available_timepoints_ch1_0_2_exploratory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the exploratory all-available-timepoints [1,0,2] preflight. "
            "No tensor build or model training is launched."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write the preflight package without build/training.",
    )
    parser.add_argument(
        "--confirm-build",
        action="store_true",
        help="Reserved guardrail flag. This preflight script still refuses real tensor builds.",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Reserved guardrail flag. This preflight script still refuses real training.",
    )
    return parser.parse_args()


def markdown_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.write_text(df.to_markdown(index=index) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, stem: str, index: bool = False) -> None:
    df.to_csv(OUT_DIR / f"{stem}.csv", index=index)
    markdown_table(df, OUT_DIR / f"{stem}.md", index=index)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_inputs() -> None:
    required = [
        SUBJECT_TIMEPOINT_TABLE,
        CONFOUNDING_TESTS,
        CANDIDATE_RETENTION,
        LOCKED_CONFIG,
        LOCKED_TENSOR,
        LOCKED_METADATA,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def build_design_matrix(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for predictor in predictors:
        if predictor == "n_TR":
            parts.append(
                pd.DataFrame(
                    {"n_TR": pd.to_numeric(df["original_n_timepoints"], errors="coerce")},
                    index=df.index,
                )
            )
        elif predictor == "Age":
            parts.append(
                pd.DataFrame(
                    {"Age": pd.to_numeric(df["Age"], errors="coerce")}, index=df.index
                )
            )
        elif predictor == "Sex":
            sex = df["Sex"].astype(str).str.strip().replace({"nan": np.nan})
            parts.append(pd.get_dummies(sex, prefix="Sex", drop_first=True, dtype=float))
        elif predictor == "Manufacturer":
            manufacturer = (
                df["Manufacturer"].astype(str).str.strip().replace({"nan": np.nan})
            )
            parts.append(
                pd.get_dummies(
                    manufacturer, prefix="Manufacturer", drop_first=True, dtype=float
                )
            )
        elif predictor == "SiteCode":
            site = df["SiteCode"].astype(str).str.zfill(3).replace({"nan": np.nan})
            parts.append(pd.get_dummies(site, prefix="SiteCode", drop_first=True, dtype=float))
        else:
            raise ValueError(f"Unsupported predictor: {predictor}")
    x = pd.concat(parts, axis=1)
    return x.apply(pd.to_numeric, errors="coerce")


def fit_statsmodels_logit(
    y: pd.Series, x: pd.DataFrame, formula_label: str
) -> tuple[str, float, float, float, str]:
    try:
        import statsmodels.api as sm

        complete = pd.concat([y.rename("y"), x], axis=1).dropna()
        y_fit = complete["y"].astype(int)
        x_fit = sm.add_constant(complete.drop(columns=["y"]), has_constant="add")
        model = sm.Logit(y_fit, x_fit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = model.fit(disp=False, maxiter=200)
        coef = safe_float(result.params.get("n_TR", np.nan))
        p_value = safe_float(result.pvalues.get("n_TR", np.nan))
        odds_ratio = math.exp(coef) if math.isfinite(coef) else math.nan
        return "ok", coef, odds_ratio, p_value, ""
    except Exception as exc:  # pragma: no cover - fallback path depends on local solver state
        return "failed", math.nan, math.nan, math.nan, f"{formula_label}: {exc}"


def sklearn_auc(y: pd.Series, x: pd.DataFrame) -> tuple[float, float, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        complete = pd.concat([y.rename("y"), x], axis=1).dropna()
        y_fit = complete["y"].astype(int).to_numpy()
        x_fit = complete.drop(columns=["y"]).to_numpy(dtype=float)
        if len(np.unique(y_fit)) < 2 or len(y_fit) < 10:
            return math.nan, math.nan, math.nan
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        )
        model.fit(x_fit, y_fit)
        in_sample_scores = model.predict_proba(x_fit)[:, 1]
        in_sample_auc = float(roc_auc_score(y_fit, in_sample_scores))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_predict(model, x_fit, y_fit, cv=cv, method="predict_proba")[
            :, 1
        ]
        cv_auc = float(roc_auc_score(y_fit, cv_scores))
        fold_aucs: list[float] = []
        for train_idx, test_idx in cv.split(x_fit, y_fit):
            model.fit(x_fit[train_idx], y_fit[train_idx])
            fold_scores = model.predict_proba(x_fit[test_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y_fit[test_idx], fold_scores)))
        return in_sample_auc, cv_auc, float(np.std(fold_aucs, ddof=1))
    except Exception:
        return math.nan, math.nan, math.nan


def build_ntr_diagnostics(subjects: pd.DataFrame) -> pd.DataFrame:
    clf = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf["target_ad"] = (clf["ResearchGroup_Mapped"] == "AD").astype(int)
    predictors_by_model = {
        "diagnosis ~ n_TR": ["n_TR"],
        "diagnosis ~ n_TR + Age + Sex + Manufacturer": [
            "n_TR",
            "Age",
            "Sex",
            "Manufacturer",
        ],
        "diagnosis ~ n_TR + SiteCode": ["n_TR", "SiteCode"],
    }
    rows: list[dict[str, Any]] = []
    for formula, predictors in predictors_by_model.items():
        x = build_design_matrix(clf, predictors)
        y = clf["target_ad"]
        complete = pd.concat([y.rename("y"), x], axis=1).dropna()
        status, coef, odds_ratio, p_value, warning = fit_statsmodels_logit(
            y, x, formula
        )
        auc_in, auc_cv, auc_cv_sd = sklearn_auc(y, x)
        rows.append(
            {
                "model": formula,
                "scope": "CN_vs_AD_classifier_pool_diagnostic_only",
                "n_total_rows": int(len(clf)),
                "n_complete_rows": int(len(complete)),
                "cn_count": int((clf["ResearchGroup_Mapped"] == "CN").sum()),
                "ad_count": int((clf["ResearchGroup_Mapped"] == "AD").sum()),
                "statsmodels_status": status,
                "n_TR_coef": coef,
                "n_TR_odds_ratio_per_TR": odds_ratio,
                "n_TR_p_value": p_value,
                "roc_auc_in_sample": auc_in,
                "roc_auc_cv5": auc_cv,
                "roc_auc_cv5_fold_sd": auc_cv_sd,
                "interpretation": (
                    "n_TR has diagnostic signal and is therefore a confounding guardrail, "
                    "not a proposed classifier feature."
                ),
                "warning": warning,
            }
        )
    prior = pd.read_csv(CONFOUNDING_TESTS)
    prior_row = prior[prior["test_name"] == "logit_AD_vs_CN_adjusted"]
    if not prior_row.empty:
        row = prior_row.iloc[0]
        rows.append(
            {
                "model": "prior_audit_logit_AD_vs_CN_adjusted",
                "scope": row.get("scope", "classifier_pool"),
                "n_total_rows": 396,
                "n_complete_rows": 396,
                "cn_count": 300,
                "ad_count": 96,
                "statsmodels_status": row.get("status", "ok"),
                "n_TR_coef": row.get("statistic", np.nan),
                "n_TR_odds_ratio_per_TR": row.get("effect", np.nan),
                "n_TR_p_value": row.get("p_value", np.nan),
                "roc_auc_in_sample": np.nan,
                "roc_auc_cv5": np.nan,
                "roc_auc_cv5_fold_sd": np.nan,
                "interpretation": "Prior locked audit result retained for manuscript traceability.",
                "warning": row.get("details", ""),
            }
        )
    return pd.DataFrame(rows)


def locked_tensor_qc(subjects: pd.DataFrame) -> pd.DataFrame:
    npz = np.load(LOCKED_TENSOR, allow_pickle=False)
    tensor = npz["global_tensor_data"]
    subject_ids = npz["subject_ids"].astype(str)
    channel_names = npz["channel_names"].astype(str)
    roi_names = npz["roi_names_in_order"].astype(str)
    offdiag = ~np.eye(tensor.shape[-1], dtype=bool)
    rows: list[dict[str, Any]] = []
    for channel_idx in CHANNELS_TO_USE:
        values = tensor[:, channel_idx]
        off_values = values[:, offdiag]
        rows.append(
            {
                "qc_item": "locked_140TR_channel_distribution",
                "branch": "locked_v5_1b_reference",
                "channel_index": channel_idx,
                "channel_name": channel_names[channel_idx],
                "n_subjects_tensor": int(tensor.shape[0]),
                "n_subjects_training_metadata": int(len(subjects)),
                "n_rois": int(tensor.shape[-1]),
                "finite_fraction_all_entries": float(np.isfinite(values).mean()),
                "finite_fraction_offdiag": float(np.isfinite(off_values).mean()),
                "mean_all_entries": float(np.nanmean(values)),
                "std_all_entries": float(np.nanstd(values)),
                "min_all_entries": float(np.nanmin(values)),
                "max_all_entries": float(np.nanmax(values)),
                "mean_offdiag": float(np.nanmean(off_values)),
                "std_offdiag": float(np.nanstd(off_values)),
                "target_len_ts": int(npz["target_len_ts"]),
                "roi_order": str(npz["roi_order_name"]),
                "python_bandpass_applied": bool(npz["python_bandpass_applied"]),
                "status": "reference_computed",
                "notes": "Reference locked 140TR tensor distribution; allTR tensor is intentionally not built.",
            }
        )
    rows.extend(
        [
            {
                "qc_item": "planned_allTR_channel_distribution_vs_locked",
                "branch": BRANCH_NAME,
                "channel_index": "1,0,2",
                "channel_name": "planned selected channels",
                "n_subjects_tensor": "planned_same_metadata_pool_where_possible",
                "n_subjects_training_metadata": int(len(subjects)),
                "n_rois": int(len(roi_names)),
                "finite_fraction_all_entries": np.nan,
                "finite_fraction_offdiag": np.nan,
                "mean_all_entries": np.nan,
                "std_all_entries": np.nan,
                "min_all_entries": np.nan,
                "max_all_entries": np.nan,
                "mean_offdiag": np.nan,
                "std_offdiag": np.nan,
                "target_len_ts": "subject_specific_all_available",
                "roi_order": "must match locked ADNI 131 ROI order",
                "python_bandpass_applied": False,
                "status": "pending_after_explicit_confirm_build",
                "notes": (
                    "After build, compare channel distributions against locked 140TR and compute "
                    "per-subject matrix distances."
                ),
            },
            {
                "qc_item": "planned_matrix_distance_140TR_vs_allTR",
                "branch": BRANCH_NAME,
                "channel_index": "1,0,2",
                "channel_name": "planned selected channels",
                "n_subjects_tensor": "pending",
                "n_subjects_training_metadata": int(len(subjects)),
                "n_rois": int(len(roi_names)),
                "finite_fraction_all_entries": np.nan,
                "finite_fraction_offdiag": np.nan,
                "mean_all_entries": np.nan,
                "std_all_entries": np.nan,
                "min_all_entries": np.nan,
                "max_all_entries": np.nan,
                "mean_offdiag": np.nan,
                "std_offdiag": np.nan,
                "target_len_ts": "subject_specific_all_available",
                "roi_order": "must match locked ADNI 131 ROI order",
                "python_bandpass_applied": False,
                "status": "pending_after_explicit_confirm_build",
                "notes": (
                    "Required post-build: Frobenius/correlation distance vs locked 140TR by subject "
                    "and association of distance/norm with original_n_timepoints."
                ),
            },
        ]
    )
    return pd.DataFrame(rows)


def branch_manifest(subjects: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    params = config["parameters"]
    cn_ad = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    rows = [
        {
            "stage": "preflight",
            "artifact_or_command": "prepare_adni_all_available_timepoints_ch1_0_2_exploratory.py --dry-run",
            "status": "completed_by_this_script",
            "requires_confirm_flag": False,
            "writes_outputs": "preflight reports only",
            "notes": "No tensor build or training.",
        },
        {
            "stage": "tensor_build",
            "artifact_or_command": "future explicit allTR tensor builder, branch-local output only",
            "status": "blocked_not_launched",
            "requires_confirm_flag": True,
            "writes_outputs": "new branch tensor + branch metadata with original_n_TR",
            "notes": (
                "Must preserve 131 ROI order, same subject pool where possible, and store "
                "original_n_TR per subject. Locked tensors remain untouched."
            ),
        },
        {
            "stage": "tensor_qc",
            "artifact_or_command": "post-build tensor QC against locked 140TR reference",
            "status": "pending_after_build",
            "requires_confirm_flag": False,
            "writes_outputs": "channel distributions, matrix distances, distance~n_TR tests",
            "notes": "Required before any training.",
        },
        {
            "stage": "stage_a_full_vae",
            "artifact_or_command": "scripts/run_vae_clf_ad_inference.py on branch tensor",
            "status": "blocked_not_launched",
            "requires_confirm_flag": True,
            "writes_outputs": "branch-local full 5x5 VAE fold artifacts",
            "notes": (
                f"Use locked v5.1b config with channels {params['channels_to_use']}, "
                f"outer={params['outer_folds']}, inner={params['inner_folds']}, "
                f"epochs={params['epochs_vae']}, cycles={params['cyclical_beta_n_cycles']}, "
                f"T0={params['lr_scheduler_T0']}."
            ),
        },
        {
            "stage": "stage_b_readout",
            "artifact_or_command": "classifier-only logreg_l2 with true inner-CV OOF thresholding",
            "status": "pending_after_stage_a",
            "requires_confirm_flag": False,
            "writes_outputs": "branch-local classifier-only readout",
            "notes": (
                "Primary threshold remains inner_oof_target_sens_ge_0p70_max_spec. "
                "Do not use Stage A classifier metrics for promotion."
            ),
        },
        {
            "stage": "posthoc_robustness",
            "artifact_or_command": "n_TR/site/manufacturer leakage and OASIS external scoring",
            "status": "pending_after_stage_b",
            "requires_confirm_flag": False,
            "writes_outputs": "robustness and external scoring reports",
            "notes": "Promotion requires internal and external robustness, not AUC alone.",
        },
    ]
    for row in rows:
        row.update(
            {
                "branch_name": BRANCH_NAME,
                "dataset_reference": "v5.1b locked metadata/tensor source",
                "planned_classifier_pool_n": int(len(cn_ad)),
                "planned_cn": int((cn_ad["ResearchGroup_Mapped"] == "CN").sum()),
                "planned_ad": int((cn_ad["ResearchGroup_Mapped"] == "AD").sum()),
                "planned_vae_pool_n": int(len(subjects)),
                "channels_to_use": "[1,0,2]",
                "channel_names": "; ".join(CHANNEL_INDEX_TO_NAME[i] for i in CHANNELS_TO_USE),
                "exploratory_label": "exploratory_confounding_stress_test",
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    subjects: pd.DataFrame, ntr_diag: pd.DataFrame, config: dict[str, Any], args: argparse.Namespace
) -> None:
    cn_ad = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    lines = [
        "# ADNI All-Available-Timepoints [1,0,2] Exploratory Branch",
        "",
        "This package is a **read-only/dry-run preflight** for an explicitly exploratory "
        "all-available-timepoints connectome branch. It does not build a tensor and does "
        "not launch training.",
        "",
        "## Branch Status",
        "",
        "- Branch name: `adni_all_available_timepoints_ch1_0_2_exploratory`.",
        "- Scientific status: `exploratory_confounding_stress_test`.",
        "- Locked 140TR tensors and model outputs were not modified.",
        "- Any future tensor must be written to a new branch only.",
        "- Any future model must be interpreted as a sensitivity analysis, not as a primary replacement based on AUC alone.",
        "",
        "## Planned Subject Pool",
        "",
        f"- Training-ready VAE pool from the prior audit: `{len(subjects)}` subjects.",
        f"- CN/AD classifier pool: `{len(cn_ad)}` subjects (`CN={int((cn_ad['ResearchGroup_Mapped'] == 'CN').sum())}`, `AD={int((cn_ad['ResearchGroup_Mapped'] == 'AD').sum())}`).",
        "- Planned channels: `[1,0,2]` = Pearson Full, Pearson OMST/GCE, MI kNN symmetric.",
        "- Planned original n_TR metadata: stored per subject in any branch metadata.",
        "",
        "## Locked Model Reference",
        "",
        f"- Config: `{LOCKED_CONFIG.relative_to(ROOT)}`.",
        f"- Outer/inner folds: `{config['parameters']['outer_folds']}x{config['parameters']['inner_folds']}`.",
        f"- VAE horizon: `{config['parameters']['epochs_vae']}` epochs, `{config['parameters']['cyclical_beta_n_cycles']}` cycles, `T0={config['parameters']['lr_scheduler_T0']}`.",
        "- Stage B readout remains classifier-only `logreg_l2` with true inner-CV OOF threshold selection.",
        "",
        "## Confounding Warning",
        "",
        "The prior audit found that original timepoint count is associated with diagnosis, Manufacturer, and SiteCode. "
        "The new n_TR diagnostic classifier table in this package is therefore a guardrail, not a proposed feature.",
        "",
        "## Generated Files",
        "",
        "- `preflight_report.md`",
        "- `ntr_diagnostic_classifier.csv/.md`",
        "- `tensor_qc.csv/.md`",
        "- `confounding_guardrail.md`",
        "- `run_manifest.csv/.md`",
        "- `command_log.json`",
    ]
    OUT_DIR.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preflight_report(args: argparse.Namespace, ntr_diag: pd.DataFrame) -> None:
    lines = [
        "# Preflight Report",
        "",
        f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`.",
        f"- Dry-run flag: `{args.dry_run}`.",
        "- Real tensor build launched: `no`.",
        "- Real VAE/classifier training launched: `no`.",
        "- Input data modified: `no`.",
        "- Locked tensors/configs/model outputs modified: `no`.",
        "",
        "## Validation Summary",
        "",
        "- Required input tables found.",
        "- Locked tensor readable and used only for reference 140TR channel distribution summaries.",
        "- n_TR diagnostic classifier table generated from existing subject-level timepoint audit.",
        "- Future all-timepoint tensor build remains blocked behind explicit confirmation and additional guardrails.",
        "",
        "## n_TR Diagnostic Summary",
        "",
    ]
    cols = [
        "model",
        "n_complete_rows",
        "n_TR_odds_ratio_per_TR",
        "n_TR_p_value",
        "roc_auc_cv5",
        "interpretation",
    ]
    lines.append(ntr_diag[cols].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Dry-Run Decision",
            "",
            "`PASS`: the exploratory branch preflight is prepared. No build or training was launched.",
        ]
    )
    OUT_DIR.joinpath("preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confounding_guardrail(subjects: pd.DataFrame, ntr_diag: pd.DataFrame) -> None:
    prior = pd.read_csv(CONFOUNDING_TESTS)
    retention = pd.read_csv(CANDIDATE_RETENTION)
    cn_ad = subjects[subjects["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    cn_mean = cn_ad.loc[cn_ad["ResearchGroup_Mapped"] == "CN", "original_n_timepoints"].mean()
    ad_mean = cn_ad.loc[cn_ad["ResearchGroup_Mapped"] == "AD", "original_n_timepoints"].mean()
    adjusted = prior[prior["test_name"] == "logit_AD_vs_CN_adjusted"].iloc[0]
    lines = [
        "# Confounding Guardrail",
        "",
        "This branch is explicitly marked as an exploratory/confounding-stress-test branch.",
        "",
        "## Known n_TR Confounding",
        "",
        f"- CN mean original n_TR: `{cn_mean:.2f}`.",
        f"- AD mean original n_TR: `{ad_mean:.2f}`.",
        f"- Prior adjusted logit `AD ~ n_TR + Age + Sex + Manufacturer`: p=`{float(adjusted['p_value']):.3g}`, OR per TR=`{float(adjusted['effect']):.4f}`.",
        "- n_TR is also associated with Manufacturer and SiteCode in the prior audit.",
        "",
        "## Required Promotion Guardrails",
        "",
        "This branch cannot be promoted based on internal AUC alone. It can only be considered as a sensitivity analysis if all of the following hold:",
        "",
        "1. AUC and PR-AUC improve over the locked model.",
        "2. BA, sensitivity, and F1 do not deteriorate.",
        "3. n_TR, site, and manufacturer leakage do not increase materially.",
        "4. n_TR is not more predictable from the latent representation than in the locked model.",
        "5. OASIS external ranking improves or at least does not worsen.",
        "6. The result is described as exploratory because the input acquisition length is diagnosis/site/manufacturer-confounded.",
        "",
        "## Fixed-Length Alternative Is Not Clean",
        "",
        "Fixed 160/180TR branches avoid variable-length connectomes but reduce the classifier pool and disproportionately drop AD subjects:",
        "",
        retention.to_markdown(index=False),
        "",
        "## Fresh Diagnostic Models",
        "",
        ntr_diag.to_markdown(index=False),
    ]
    OUT_DIR.joinpath("confounding_guardrail.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_command_log(args: argparse.Namespace, outputs: list[str]) -> None:
    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "argv": sys.argv,
        "dry_run": bool(args.dry_run),
        "confirm_build": bool(args.confirm_build),
        "confirm_training": bool(args.confirm_training),
        "actions": [
            "loaded prior ADNI timepoint availability/confounding audit",
            "computed n_TR diagnostic classifier summaries",
            "summarized locked 140TR tensor distributions for channels [1,0,2]",
            "wrote exploratory branch preflight outputs",
        ],
        "safety": {
            "tensor_build_launched": False,
            "training_launched": False,
            "locked_tensor_modified": False,
            "locked_metadata_modified": False,
            "ledger_modified": False,
            "model_outputs_modified": False,
        },
        "inputs": {
            "subject_timepoint_table": str(SUBJECT_TIMEPOINT_TABLE),
            "confounding_tests": str(CONFOUNDING_TESTS),
            "candidate_retention": str(CANDIDATE_RETENTION),
            "locked_config": str(LOCKED_CONFIG),
            "locked_tensor": str(LOCKED_TENSOR),
            "locked_metadata": str(LOCKED_METADATA),
            "longer_timeseries_preflight": str(LONGER_PREFLIGHT_DIR),
        },
        "outputs": outputs,
        "decision": "dry_run_preflight_pass_no_build_no_training",
    }
    (OUT_DIR / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.confirm_build or args.confirm_training:
        raise SystemExit(
            "This script is a preflight-only guardrail package and refuses real "
            "tensor builds/training. Create a dedicated launcher after explicit approval."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_inputs()

    subjects = pd.read_csv(SUBJECT_TIMEPOINT_TABLE)
    config = load_json(LOCKED_CONFIG)
    ntr_diag = build_ntr_diagnostics(subjects)
    tensor_qc = locked_tensor_qc(subjects)
    manifest = branch_manifest(subjects, config)

    write_table(ntr_diag, "ntr_diagnostic_classifier")
    write_table(tensor_qc, "tensor_qc")
    write_table(manifest, "run_manifest")
    write_readme(subjects, ntr_diag, config, args)
    write_preflight_report(args, ntr_diag)
    write_confounding_guardrail(subjects, ntr_diag)

    outputs = [
        str(OUT_DIR / "README.md"),
        str(OUT_DIR / "preflight_report.md"),
        str(OUT_DIR / "ntr_diagnostic_classifier.csv"),
        str(OUT_DIR / "ntr_diagnostic_classifier.md"),
        str(OUT_DIR / "tensor_qc.csv"),
        str(OUT_DIR / "tensor_qc.md"),
        str(OUT_DIR / "confounding_guardrail.md"),
        str(OUT_DIR / "run_manifest.csv"),
        str(OUT_DIR / "run_manifest.md"),
        str(OUT_DIR / "command_log.json"),
    ]
    write_command_log(args, outputs)
    print(f"Wrote exploratory all-timepoints preflight to {OUT_DIR}")
    print("No tensor build or training was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
