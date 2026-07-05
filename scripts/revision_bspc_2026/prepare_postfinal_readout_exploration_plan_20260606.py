#!/usr/bin/env python3
"""Prepare a read-only/preflight classifier-readout exploration plan.

The package produced by this script is a plan over already-trained latent
caches. It performs no classifier fitting, no VAE training, no OASIS scoring,
and no threshold/calibration fitting.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_OUTPUT = RESULTS_ROOT / "postfinal_readout_exploration_plan_20260606"

FOLDS = [1, 2, 3, 4, 5]
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

LATENT_SOURCES = [
    {
        "source_id": "promoted_ch102_latent384_beta3p75",
        "display_name": "promoted [1,0,2] latent384 beta3.75",
        "role": "required_primary_source",
        "run_dir": RESULTS_ROOT / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "latent_cache_dir": RESULTS_ROOT
        / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache",
        "oof_calibration_dir": RESULTS_ROOT / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim_expected": 384,
        "include_in_single_source_readouts": True,
        "include_in_two_source_stacking": True,
    },
    {
        "source_id": "ch1only_latent384_beta3p75",
        "display_name": "ch1-only latent384 beta3.75",
        "role": "required_parsimony_source",
        "run_dir": RESULTS_ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "latent_cache_dir": RESULTS_ROOT
        / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache",
        "oof_calibration_dir": RESULTS_ROOT
        / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "channels": "[1]",
        "latent_dim_expected": 384,
        "include_in_single_source_readouts": True,
        "include_in_two_source_stacking": True,
    },
    {
        "source_id": "latent512_beta3p75_optional",
        "display_name": "optional latent512 beta3.75",
        "role": "optional_capacity_source",
        "run_dir": RESULTS_ROOT / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
        "latent_cache_dir": RESULTS_ROOT
        / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache",
        "oof_calibration_dir": RESULTS_ROOT / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
        "channels": "[1,0,2]",
        "latent_dim_expected": 512,
        "include_in_single_source_readouts": True,
        "include_in_two_source_stacking": False,
    },
]

READOUTS = [
    {
        "readout_id": "logreg_l2_baseline",
        "readout_family": "single_source",
        "representation": "mu_plus_age_sex",
        "classifier": "logreg_l2",
        "inner_cv_grid": "C=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]; class_weight=balanced",
        "transform_fit_scope": "inner-train only; final outer model fit on outer train/dev only",
        "notes": "Reproduce baseline convention before comparing any richer readout.",
    },
    {
        "readout_id": "pca_logreg_l2",
        "readout_family": "single_source",
        "representation": "PCA(mu) + Age + Sex",
        "classifier": "logreg_l2",
        "inner_cv_grid": "n_components=[16,32,64,96,128,192] capped by latent_dim and inner n; C=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]",
        "transform_fit_scope": "PCA scaler and PCA fit inside each inner CV training split only",
        "notes": "Tests whether lower-dimensional latent projections reduce score noise/overfit.",
    },
    {
        "readout_id": "pls_logreg_l2",
        "readout_family": "single_source",
        "representation": "PLS(mu,y) + Age + Sex",
        "classifier": "logreg_l2",
        "inner_cv_grid": "n_components=[2,4,8,16,32] capped by inner n and latent_dim; C=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]",
        "transform_fit_scope": "PLS is supervised and must be fit only inside inner CV training split; final PLS fit only on outer train/dev",
        "notes": "Higher leakage risk if implemented incorrectly; keep as explicit nested-pipeline transformer.",
    },
    {
        "readout_id": "linear_svm",
        "readout_family": "single_source",
        "representation": "mu_plus_age_sex",
        "classifier": "linear_svm_calibrated",
        "inner_cv_grid": "C=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]; calibration inside train/dev only",
        "transform_fit_scope": "scaler and SVM fit inside inner CV; calibration from inner/outer train-dev only",
        "notes": "Compare margin readout without nonlinear kernel capacity.",
    },
    {
        "readout_id": "rbf_svm_restricted",
        "readout_family": "single_source",
        "representation": "mu_plus_age_sex",
        "classifier": "rbf_svm_calibrated_restricted",
        "inner_cv_grid": "C=[0.03,0.1,0.3,1.0]; gamma=['scale',0.0003,0.001,0.003]; calibration inside train/dev only",
        "transform_fit_scope": "all scaling, kernel hyperparameters, and calibration selected inside inner CV only",
        "notes": "Restricted grid only; promotion requires clear gain without Philips/scanner degradation.",
    },
    {
        "readout_id": "two_source_score_stacking_logreg",
        "readout_family": "two_source_stack",
        "representation": "inner-OOF scores from promoted [1,0,2] + ch1-only readouts",
        "classifier": "meta_logreg_l2",
        "inner_cv_grid": "base readout fixed to leakage-safe inner-OOF protocol; meta C=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]",
        "transform_fit_scope": "meta model fit only on inner-OOF scores from outer train/dev; outer test gets scores from base models fit on train/dev",
        "notes": "No outer-test labels for score stacking; no OASIS stacking/calibration.",
    },
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def to_md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def finite_ratio(df: pd.DataFrame, cols: list[str]) -> float:
    if not cols:
        return float("nan")
    arr = df[cols].to_numpy(dtype=float)
    return float(np.isfinite(arr).mean())


def validate_source(source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = Path(source["latent_cache_dir"])
    summary: dict[str, Any] = {
        "source_id": source["source_id"],
        "display_name": source["display_name"],
        "role": source["role"],
        "run_dir": rel(Path(source["run_dir"])),
        "latent_cache_dir": rel(cache),
        "oof_calibration_dir": rel(Path(source["oof_calibration_dir"])),
        "channels": source["channels"],
        "latent_dim_expected": source["latent_dim_expected"],
        "latent_cache_exists": cache.exists(),
        "oof_calibration_dir_exists": Path(source["oof_calibration_dir"]).exists(),
    }
    fold_rows: list[dict[str, Any]] = []
    all_ok = True
    latent_dims = []
    total_test_n = 0
    total_test_cn = 0
    total_test_ad = 0
    for fold in FOLDS:
        train_path = cache / f"fold_{fold}_trainDev_latent_mu.csv"
        test_path = cache / f"fold_{fold}_test_latent_mu.csv"
        row: dict[str, Any] = {
            "source_id": source["source_id"],
            "fold": fold,
            "train_cache": rel(train_path),
            "test_cache": rel(test_path),
            "train_exists": train_path.exists(),
            "test_exists": test_path.exists(),
        }
        if not train_path.exists() or not test_path.exists():
            row["status"] = "missing_cache"
            all_ok = False
            fold_rows.append(row)
            continue
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        mu_cols = [c for c in train.columns if c.startswith("mu_")]
        test_mu_cols = [c for c in test.columns if c.startswith("mu_")]
        latent_dim = len(mu_cols)
        latent_dims.append(latent_dim)
        train_subjects = set(train["SubjectID"].astype(str)) if "SubjectID" in train.columns else set()
        test_subjects = set(test["SubjectID"].astype(str)) if "SubjectID" in test.columns else set()
        overlap = sorted(train_subjects & test_subjects)
        y_train = pd.to_numeric(train.get("y"), errors="coerce") if "y" in train.columns else pd.Series(dtype=float)
        y_test = pd.to_numeric(test.get("y"), errors="coerce") if "y" in test.columns else pd.Series(dtype=float)
        n_train_cn = int((y_train == 0).sum())
        n_train_ad = int((y_train == 1).sum())
        n_test_cn = int((y_test == 0).sum())
        n_test_ad = int((y_test == 1).sum())
        total_test_n += len(test)
        total_test_cn += n_test_cn
        total_test_ad += n_test_ad
        status = "ok"
        if latent_dim != int(source["latent_dim_expected"]):
            status = "latent_dim_mismatch"
        if latent_dim != len(test_mu_cols):
            status = "train_test_latent_dim_mismatch"
        if overlap:
            status = "train_test_subject_overlap"
        if n_train_cn == 0 or n_train_ad == 0 or n_test_cn == 0 or n_test_ad == 0:
            status = "class_missing"
        if status != "ok":
            all_ok = False
        row.update(
            {
                "status": status,
                "n_trainDev": len(train),
                "n_test": len(test),
                "n_trainDev_cn": n_train_cn,
                "n_trainDev_ad": n_train_ad,
                "n_test_cn": n_test_cn,
                "n_test_ad": n_test_ad,
                "latent_dim": latent_dim,
                "train_mu_finite_ratio": finite_ratio(train, mu_cols),
                "test_mu_finite_ratio": finite_ratio(test, test_mu_cols),
                "age_missing_trainDev": int(pd.to_numeric(train.get("Age"), errors="coerce").isna().sum()) if "Age" in train.columns else np.nan,
                "age_missing_test": int(pd.to_numeric(test.get("Age"), errors="coerce").isna().sum()) if "Age" in test.columns else np.nan,
                "sex_missing_trainDev": int(train.get("Sex", pd.Series(dtype=object)).isna().sum()) if "Sex" in train.columns else np.nan,
                "sex_missing_test": int(test.get("Sex", pd.Series(dtype=object)).isna().sum()) if "Sex" in test.columns else np.nan,
                "train_test_subject_overlap_n": len(overlap),
            }
        )
        fold_rows.append(row)
    summary.update(
        {
            "folds_found": sum(1 for r in fold_rows if r.get("train_exists") and r.get("test_exists")),
            "all_5_folds_ready": bool(all_ok and len(fold_rows) == 5),
            "latent_dims_observed": ",".join(map(str, sorted(set(latent_dims)))) if latent_dims else "",
            "total_outer_test_n": total_test_n,
            "total_outer_test_cn": total_test_cn,
            "total_outer_test_ad": total_test_ad,
        }
    )
    return summary, fold_rows


def build_matrix(source_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    single_sources = source_inventory[
        (source_inventory["all_5_folds_ready"].astype(bool))
        & (source_inventory["include_in_single_source_readouts"].astype(bool))
    ]
    for _, src in single_sources.iterrows():
        for readout in READOUTS:
            if readout["readout_family"] != "single_source":
                continue
            rows.append(
                {
                    "candidate_id": f"{src['source_id']}__{readout['readout_id']}",
                    "latent_source_id": src["source_id"],
                    "latent_source_display": src["display_name"],
                    "readout_id": readout["readout_id"],
                    "readout_family": readout["readout_family"],
                    "representation": readout["representation"],
                    "classifier": readout["classifier"],
                    "inner_cv_grid": readout["inner_cv_grid"],
                    "transform_fit_scope": readout["transform_fit_scope"],
                    "threshold_strategy": PRIMARY_THRESHOLD,
                    "primary_metrics": "ADNI outer OOF AUC, PR-AUC, BA, sensitivity, specificity, F1",
                    "secondary_metrics": "Philips CN FPR; GE/SIEMENS/Philips FPR; score-Manufacturer association; scanner leakage where feature transform produces latent-like representation",
                    "oasis_policy": "Do not score OASIS until ADNI candidate is locked; then frozen inference only with ADNI-derived thresholds/calibrations.",
                    "notes": readout["notes"],
                }
            )
    stack_readout = next(r for r in READOUTS if r["readout_id"] == "two_source_score_stacking_logreg")
    ready_stack = source_inventory[
        (source_inventory["all_5_folds_ready"].astype(bool))
        & (source_inventory["include_in_two_source_stacking"].astype(bool))
    ]
    if {"promoted_ch102_latent384_beta3p75", "ch1only_latent384_beta3p75"}.issubset(set(ready_stack["source_id"])):
        rows.append(
            {
                "candidate_id": "promoted_plus_ch1only__two_source_score_stacking_logreg",
                "latent_source_id": "promoted_ch102_latent384_beta3p75 + ch1only_latent384_beta3p75",
                "latent_source_display": "promoted [1,0,2] + ch1-only",
                "readout_id": stack_readout["readout_id"],
                "readout_family": stack_readout["readout_family"],
                "representation": stack_readout["representation"],
                "classifier": stack_readout["classifier"],
                "inner_cv_grid": stack_readout["inner_cv_grid"],
                "transform_fit_scope": stack_readout["transform_fit_scope"],
                "threshold_strategy": PRIMARY_THRESHOLD,
                "primary_metrics": "ADNI outer OOF AUC, PR-AUC, BA, sensitivity, specificity, F1",
                "secondary_metrics": "Philips CN FPR; score-Manufacturer association by source and stacked score",
                "oasis_policy": "Only after ADNI stack is locked; OASIS receives frozen base fold scorers and frozen meta-logreg, no OASIS labels for fitting.",
                "notes": stack_readout["notes"],
            }
        )
    return pd.DataFrame(rows)


def write_docs(out: Path, source_df: pd.DataFrame, matrix: pd.DataFrame) -> None:
    readme = [
        "# Postfinal Classifier-Readout Exploration Plan",
        "",
        "Scope: classifier-only exploration over already-trained ADNI latent caches. This package is a preflight plan and artifact inventory only.",
        "",
        "Guardrails:",
        "- no VAE retraining",
        "- no tensor, metadata, ledger, or model artifact modification",
        "- all transforms, reducers, hyperparameter choices, calibration, and thresholds fit inside train/dev only",
        "- no outer-test labels used for transforms, hyperparameters, calibration, thresholding, or stacking",
        "- no OASIS model selection or OASIS threshold fitting",
        "- OASIS frozen inference only after an ADNI readout is locked",
        "",
        f"Ready latent sources: `{int(source_df['all_5_folds_ready'].sum())}/{len(source_df)}`.",
        f"Planned readout candidates: `{len(matrix)}`.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    leakage = [
        "# Leakage-Safe Nested Design",
        "",
        "For each outer fold, only that fold's train/dev latent cache is used for inner CV model selection.",
        "",
        "Single-source readouts:",
        "1. Split outer train/dev into inner folds stratified by diagnosis and tracking Manufacturer.",
        "2. Fit imputation/scaling/PCA/PLS/classifier only on inner-training rows.",
        "3. Generate inner-validation scores for hyperparameter and threshold selection.",
        "4. Select hyperparameters by inner OOF AUC/PR-AUC with predeclared tie-breaks.",
        "5. Select threshold using inner OOF target sensitivity >= 0.70 max specificity.",
        "6. Refit the selected transform/classifier on full outer train/dev.",
        "7. Apply once to the held-out outer test fold.",
        "",
        "PLS is supervised; it must live inside the sklearn Pipeline searched by inner CV. A globally fitted PLS transform is forbidden.",
        "",
        "Two-source stacking:",
        "1. Within each outer train/dev fold, generate inner-OOF scores for promoted and ch1-only sources.",
        "2. Fit the meta logistic regression only on those inner-OOF train/dev score pairs.",
        "3. Refit each base readout on full outer train/dev and score the outer test subjects.",
        "4. Apply the frozen meta logistic regression to the two outer-test scores.",
        "5. Select the operating threshold only from train/dev inner-OOF stacked scores.",
        "",
        "OASIS labels may be used only for final external metrics after the ADNI readout is locked.",
    ]
    (out / "leakage_safe_nested_design.md").write_text("\n".join(leakage) + "\n", encoding="utf-8")

    launch = [
        "# Future Execution Plan",
        "",
        "This package does not launch readout fitting. A future execution script should consume `experiment_matrix.csv` and write to a new output directory, for example:",
        "",
        "```bash",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/run_postfinal_readout_exploration_20260606.py \\",
        "  --matrix results/revision_bspc_2026/postfinal_readout_exploration_plan_20260606/experiment_matrix.csv \\",
        "  --output-dir results/revision_bspc_2026/postfinal_readout_exploration_20260606 \\",
        "  --inner-folds 5 \\",
        "  --seed 42 \\",
        "  --dry-run",
        "```",
        "",
        "Real classifier-only execution should require an explicit `--confirm-readout` flag. OASIS inference should be a separate later command and should require a locked ADNI candidate manifest.",
        "",
        "Expected execution outputs:",
        "- `candidate_readout_metrics.csv/.md`",
        "- `foldwise_metrics.csv/.md`",
        "- `selected_hyperparameters.csv/.md`",
        "- `thresholds_by_fold.csv/.md`",
        "- `philips_cn_fpr.csv/.md`",
        "- `score_distribution_by_manufacturer.csv/.md`",
        "- `scanner_leakage_or_score_association.csv/.md`",
        "- `promotion_gate_decision.md`",
        "- `command_log.json`",
    ]
    (out / "execution_plan.md").write_text("\n".join(launch) + "\n", encoding="utf-8")

    oasis = [
        "# OASIS Frozen Inference Policy",
        "",
        "OASIS must not be used during ADNI readout exploration.",
        "",
        "After one ADNI readout is locked, OASIS inference can be run only with:",
        "- frozen ADNI-trained fold transforms/classifiers",
        "- ADNI-derived score harmonization and thresholds",
        "- no OASIS threshold fitting",
        "- no OASIS calibration fitting",
        "- no OASIS-driven model selection",
        "",
        "Report OASIS only as external stress-test metrics for the locked ADNI readout.",
    ]
    (out / "oasis_frozen_inference_policy.md").write_text("\n".join(oasis) + "\n", encoding="utf-8")

    gate = [
        "# Promotion Gate",
        "",
        "A classifier-only readout may be considered only if it improves the ADNI readout without opening a scanner/site failure mode.",
        "",
        "Minimum gates:",
        "- ADNI OOF AUC > promoted reference AUC 0.795155, or PR-AUC materially improves without AUC loss.",
        "- PR-AUC >= promoted reference PR-AUC 0.573934 unless the readout is explicitly a sensitivity model.",
        "- BA/F1/sensitivity not materially worse than the promoted reference.",
        "- Philips CN FPR <= promoted reference 0.4545, or clearly improves relative to the relevant source baseline.",
        "- Score-Manufacturer association and scanner leakage not worse.",
        "- No leakage path detected in PCA/PLS/stacking implementation.",
        "",
        "OASIS can support external plausibility after ADNI lock-in, but cannot promote a readout by itself.",
    ]
    (out / "promotion_gate.md").write_text("\n".join(gate) + "\n", encoding="utf-8")

    schema = [
        "# Planned Output Schema",
        "",
        "Primary ADNI metrics:",
        "- model/source/readout identifiers",
        "- AUC, PR-AUC, BA, sensitivity, specificity, F1",
        "- confusion matrix",
        "- selected hyperparameters",
        "- selected threshold per fold",
        "",
        "Safety metrics:",
        "- Philips CN FP/FPR",
        "- GE/SIEMENS/Philips CN FPR",
        "- AD FNR by Manufacturer",
        "- score distribution by Manufacturer and diagnosis",
        "- score-Manufacturer association after diagnosis/Age/Sex adjustment",
        "",
        "OASIS deferred metrics:",
        "- concatenated, runwise164, and runwise140 AUC/PR-AUC",
        "- fixed ADNI threshold metrics only",
        "- no OASIS threshold/calibration fitting",
    ]
    (out / "planned_output_schema.md").write_text("\n".join(schema) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for source in LATENT_SOURCES:
        summary, folds = validate_source(source)
        summary.update(
            {
                "include_in_single_source_readouts": bool(source["include_in_single_source_readouts"]),
                "include_in_two_source_stacking": bool(source["include_in_two_source_stacking"]),
            }
        )
        inventory_rows.append(summary)
        fold_rows.extend(folds)
    source_df = pd.DataFrame(inventory_rows)
    fold_df = pd.DataFrame(fold_rows)
    matrix = build_matrix(source_df)
    preflight = pd.DataFrame(
        [
            {
                "check": "all_required_latent_sources_ready",
                "status": "PASS"
                if source_df[source_df["role"].str.startswith("required")]["all_5_folds_ready"].all()
                else "FAIL",
                "detail": "Promoted and ch1-only latent384 source caches must have all five trainDev/test folds.",
            },
            {
                "check": "optional_latent512_ready",
                "status": "PASS"
                if bool(source_df[source_df["source_id"].eq("latent512_beta3p75_optional")]["all_5_folds_ready"].iloc[0])
                else "SKIP",
                "detail": "Optional latent512 source included only if all folds are present.",
            },
            {
                "check": "two_source_stacking_feasible",
                "status": "PASS"
                if "promoted_plus_ch1only__two_source_score_stacking_logreg" in set(matrix["candidate_id"])
                else "FAIL",
                "detail": "Requires promoted and ch1-only trainDev/test caches for all folds.",
            },
            {
                "check": "oasis_deferred",
                "status": "PASS",
                "detail": "No OASIS scoring, threshold fitting, calibration, or model selection in this preflight package.",
            },
        ]
    )

    source_df.to_csv(out / "latent_source_inventory.csv", index=False)
    fold_df.to_csv(out / "latent_cache_fold_validation.csv", index=False)
    matrix.to_csv(out / "experiment_matrix.csv", index=False)
    preflight.to_csv(out / "preflight_status.csv", index=False)
    to_md(source_df, out / "latent_source_inventory.md")
    to_md(fold_df, out / "latent_cache_fold_validation.md")
    to_md(matrix, out / "experiment_matrix.md")
    to_md(preflight, out / "preflight_status.md")
    write_docs(out, source_df, matrix)

    command_log = {
        "created_at": datetime.now().isoformat(),
        "script": rel(Path(__file__)),
        "output_dir": rel(out),
        "guardrails": [
            "no VAE retraining",
            "no classifier fitting in preflight",
            "no OASIS scoring",
            "no OASIS threshold/calibration fitting",
            "no tensor/metadata/ledger/model artifact modification",
        ],
        "latent_sources": [
            {
                "source_id": row["source_id"],
                "latent_cache_dir": row["latent_cache_dir"],
                "all_5_folds_ready": bool(row["all_5_folds_ready"]),
            }
            for row in source_df.to_dict(orient="records")
        ],
        "planned_candidates": len(matrix),
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
