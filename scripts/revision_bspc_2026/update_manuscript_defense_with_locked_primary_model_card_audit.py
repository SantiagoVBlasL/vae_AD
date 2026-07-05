#!/usr/bin/env python3
"""Update the manuscript defense package with the locked primary model-card audit.

This script modifies only manuscript-defense package files. It does not train,
score, fit thresholds, select models, or modify tensor/metadata/ledger/model
output artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFENSE_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
AUDIT_DIR = RESULTS_ROOT / "locked_primary_model_deep_model_card_audit"

README = DEFENSE_DIR / "README.md"
FINAL_RECOMMENDATION = DEFENSE_DIR / "final_recommendation.md"
REVIEWER_TEXT = DEFENSE_DIR / "reviewer_response_ready_text.md"
COMMAND_LOG = DEFENSE_DIR / "command_log.json"
RESULT_MD = DEFENSE_DIR / "locked_primary_model_deep_model_card_result.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def marker_block(name: str, body: str) -> str:
    return f"<!-- BEGIN {name} -->\n{body.rstrip()}\n<!-- END {name} -->\n"


def upsert_marker(path: Path, name: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    begin = f"<!-- BEGIN {name} -->"
    end = f"<!-- END {name} -->"
    block = marker_block(name, body)
    if begin in text and end in text:
        pre = text.split(begin, 1)[0].rstrip()
        rest = text.split(begin, 1)[1].split(end, 1)[1].lstrip()
        text = f"{pre}\n\n{block}\n{rest}".rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def load_csv(name: str) -> pd.DataFrame:
    path = AUDIT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def summarize_manufacturer_errors(subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cn = subjects[subjects["ResearchGroup_Mapped"].eq("CN")].copy()
    ad = subjects[subjects["ResearchGroup_Mapped"].eq("AD")].copy()
    cn_rows = []
    for mfr, group in cn.groupby("Manufacturer"):
        n = int(len(group))
        fp = int(group["locked_error_type"].eq("FP").sum())
        cn_rows.append(
            {
                "Manufacturer": mfr,
                "CN_n": n,
                "CN_FP": fp,
                "CN_FP_rate": round(fp / n, 4) if n else "",
            }
        )
    ad_rows = []
    for mfr, group in ad.groupby("Manufacturer"):
        n = int(len(group))
        fn = int(group["locked_error_type"].eq("FN").sum())
        ad_rows.append(
            {
                "Manufacturer": mfr,
                "AD_n": n,
                "AD_FN": fn,
                "AD_FN_rate": round(fn / n, 4) if n else "",
            }
        )
    return pd.DataFrame(cn_rows), pd.DataFrame(ad_rows)


def build_result_doc(updated_utc: str) -> str:
    maturity = load_csv("vae_training_maturity_by_fold.csv")
    hp = load_csv("stageb_logreg_hyperparameters_by_fold.csv")
    hard = load_csv("hard_fold_diagnosis.csv")
    leakage = load_csv("scanner_manufacturer_leakage_by_fold.csv")
    threshold = load_csv("threshold_confusion_by_fold.csv")
    subjects = load_csv("subject_error_table.csv")

    inner = threshold[threshold["threshold_strategy"].eq("inner_oof_target_sens_ge_0p70_max_spec")].copy()
    weak = hard[hard["fold"].isin([1, 4])].copy()
    cn_fp, ad_fn = summarize_manufacturer_errors(subjects)
    test_leakage = leakage[leakage["scope"].eq("test")].copy()

    maturity_summary = pd.DataFrame(
        [
            {
                "folds": int(maturity["fold"].nunique()),
                "reached_max_epoch": int(maturity["reached_max_epoch"].sum()),
                "best_epoch_in_last_10_percent": int(maturity["best_epoch_in_last_10_percent"].sum()),
                "all_early_stopped": bool((~maturity["reached_max_epoch"]).all()),
                "median_epochs_after_best": float(maturity["epochs_after_best"].median()),
            }
        ]
    )
    hp_summary = pd.DataFrame(
        [
            {
                "folds": int(hp["fold"].nunique()),
                "selected_C_values": ", ".join(str(v) for v in hp["best_C"].tolist()),
                "lower_boundary_hits": int(hp["hit_lower_boundary"].sum()),
                "upper_boundary_hits": int(hp["hit_upper_boundary"].sum()),
                "grid": (
                    f"[{hp['grid_min_C'].iloc[0]}, {hp['grid_max_C'].iloc[0]}]"
                    if not hp.empty
                    else ""
                ),
            }
        ]
    )
    leakage_summary = pd.DataFrame(
        [
            {
                "scope": "test",
                "mean_raw_manufacturer_BA": round(float(test_leakage["acc_site_raw"].mean()), 4),
                "mean_latent_manufacturer_BA": round(float(test_leakage["acc_site_latent"].mean()), 4),
                "mean_latent_minus_raw": round(float(test_leakage["latent_minus_raw"].mean()), 4),
            }
        ]
    )

    weak_cols = [
        "fold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "ad_score_median",
        "cn_score_median",
        "score_median_gap_ad_minus_cn",
        "diagnosis",
        "interpretation",
    ]
    return f"""# Locked Primary Model Deep Model-Card Audit Result

Updated: `{updated_utc}`

Source audit: `{AUDIT_DIR.relative_to(PROJECT_ROOT)}/`

## VAE Training Maturity

{markdown_table(maturity_summary)}

All five VAE folds early-stopped. No fold reached the maximum epoch budget and no fold selected its best checkpoint in the final 10% of training. This argues against a longer-horizon rerun as a general internal optimization step.

## Stage B Logistic Regression Boundary

{markdown_table(hp_summary)}

The Stage B `logreg_l2` readout selected `C=0.001` in all folds. This is a real lower-bound signal, but it does not justify expanding the grid again because the prior ultra-regularized readout and frozen-latent classifier sweep were negative.

## Weak Fold Diagnosis

{markdown_table(weak[weak_cols])}

Fold 1 and Fold 4 are the weak folds. Fold 4 shows weak rank separation, not only threshold failure: both ROC-AUC/PR-AUC and the AD-CN median score gap are low.

Primary threshold fold metrics:

{markdown_table(inner[['fold', 'threshold', 'tn', 'fp', 'fn', 'tp', 'auc', 'pr_auc', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1']])}

## Error Asymmetry

CN false positives by Manufacturer:

{markdown_table(cn_fp)}

AD false negatives by Manufacturer:

{markdown_table(ad_fn)}

The main asymmetry remains elevated Philips CN false positives and elevated GE AD false negatives.

## Scanner/Manufacturer Separability

{markdown_table(leakage_summary)}

Scanner/manufacturer separability is reduced from raw connectivity to latent space on average, but it is not eliminated. This supports scanner-aware reporting and external calibration/test validation rather than additional internal tuning.

## Final Decision

No further internal optimization is justified. The appropriate next step is the pre-specified OASIS external calibration/test protocol.

## Safety

- No training.
- No scoring.
- No threshold fitting.
- No model selection.
- No tensor, metadata, ledger, config, or model-output modification.
"""


def update_readme() -> None:
    body = """## Locked Primary Model Deep Model-Card Audit

Audit path: `results/revision_bspc_2026/locked_primary_model_deep_model_card_audit/`

The locked primary model-card audit confirms that the final v5.1b horizon4480/cycles56 `[1,0,2]` model is not obviously undertrained: all five VAE folds early-stopped, `0/5` folds reached the maximum epoch budget, and `0/5` folds selected the best checkpoint in the last 10% of training.

Stage B `logreg_l2` selected `C=0.001` in `5/5` folds, but this does not justify expanding the C grid because the prior ultra-regularized readout and frozen-latent classifier sweeps were negative. Weak folds are Fold 1 and Fold 4; Fold 4 shows weak rank separation rather than only threshold failure. The dominant error asymmetry remains elevated Philips CN false positives and elevated GE AD false negatives. Scanner/manufacturer separability is reduced from raw connectivity to latent space, but not eliminated.

Decision: no further internal optimization; proceed with the OASIS external calibration/test protocol.
"""
    upsert_marker(README, "LOCKED_PRIMARY_MODEL_CARD_AUDIT", body)


def update_final_recommendation() -> None:
    body = """## Locked Primary Model Deep Model-Card Audit

The deep model-card audit supports stopping internal optimization. All five VAE folds early-stopped; `0/5` reached max epoch and `0/5` selected best epoch in the last 10% of the budget. This argues against another longer-horizon internal rerun.

Stage B `logreg_l2` selected `C=0.001` in `5/5` folds. This boundary hit is documented but does not justify expanding the C grid because a prior ultra-regularized readout lowered AUC/PR-AUC, and the frozen-latent classifier sweep did not produce a clean replacement.

Fold 1 and Fold 4 remain the weak folds. Fold 4 is not merely a threshold failure: rank separation is weak, with lower AUC/PR-AUC and small AD-CN median score separation. Error asymmetry remains elevated in Philips CN false positives and GE AD false negatives. Latent representations reduce scanner/manufacturer separability relative to raw connectivity but do not remove it.

Decision: no further internal AUC optimization is scientifically justified. Proceed with OASIS external calibration/test validation.
"""
    upsert_marker(FINAL_RECOMMENDATION, "LOCKED_PRIMARY_MODEL_CARD_AUDIT", body)


def update_reviewer_text() -> None:
    body = """## Reviewer Response: Deep Model-Card and Training Maturity Audit

We added a read-only deep model-card audit of the locked primary ADNI model to test whether further internal optimization was justified. The audit found that all five VAE folds early-stopped; no fold reached the maximum epoch budget, and no fold selected its best checkpoint in the last 10% of training. This argues against a longer-horizon rerun as a general remedy.

The Stage B logistic readout selected `C=0.001` in all five folds. We do not treat this boundary hit as a reason for further classifier tuning because a prior ultra-regularized readout explicitly tested stronger regularization and worsened test ranking, and a frozen-latent classifier sweep did not identify a clean replacement. The weak folds are Fold 1 and Fold 4. Fold 4 shows weak rank separation, not only threshold failure, so post-hoc threshold manipulation would not solve the main limitation.

The audit also confirms the main error asymmetry: Philips CN false positives and GE AD false negatives are elevated. Scanner/manufacturer separability is reduced from raw connectivity to latent space, but not eliminated. We therefore stop internal optimization and move to the pre-specified OASIS external calibration/test protocol rather than continuing to tune on ADNI.
"""
    upsert_marker(REVIEWER_TEXT, "LOCKED_PRIMARY_MODEL_CARD_AUDIT", body)


def update_command_log(updated_utc: str) -> None:
    log = json.loads(COMMAND_LOG.read_text(encoding="utf-8")) if COMMAND_LOG.exists() else {}
    log["last_updated_utc"] = updated_utc
    log["latest_update_script"] = str(Path(__file__).resolve())
    updates = log.setdefault("updates", [])
    updates.append(
        {
            "action": "add_locked_primary_model_deep_model_card_audit",
            "source_dir": str(AUDIT_DIR),
            "recorded_findings": {
                "all_vae_folds_early_stopped": True,
                "folds_reached_max_epoch": "0/5",
                "folds_best_epoch_last_10_percent": "0/5",
                "stageb_logreg_l2_C_lower_boundary": "5/5 selected C=0.001",
                "do_not_expand_C_grid_reason": "ultra_regularized_and_frozen_latent_sweeps_negative",
                "weak_folds": [1, 4],
                "fold4_interpretation": "weak_rank_separation_not_only_threshold_failure",
                "main_error_asymmetry": {
                    "philips_cn_false_positives_elevated": True,
                    "ge_ad_false_negatives_elevated": True,
                },
                "scanner_manufacturer_separability": "reduced_from_raw_to_latent_but_not_eliminated",
                "final_decision": "no_further_internal_optimization_proceed_with_oasis_external_calibration_test",
            },
            "training_launched": False,
            "scoring_launched": False,
            "threshold_fitting": False,
            "model_selection": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "model_outputs_modified": False,
            "updated_utc": updated_utc,
        }
    )
    COMMAND_LOG.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    updated_utc = now_utc()
    DEFENSE_DIR.mkdir(parents=True, exist_ok=True)
    for required in [
        "vae_training_maturity_by_fold.csv",
        "stageb_logreg_hyperparameters_by_fold.csv",
        "hard_fold_diagnosis.csv",
        "scanner_manufacturer_leakage_by_fold.csv",
        "threshold_confusion_by_fold.csv",
        "subject_error_table.csv",
    ]:
        if not (AUDIT_DIR / required).exists():
            raise FileNotFoundError(AUDIT_DIR / required)

    RESULT_MD.write_text(build_result_doc(updated_utc), encoding="utf-8")
    update_readme()
    update_final_recommendation()
    update_reviewer_text()
    update_command_log(updated_utc)

    print(f"Updated manuscript defense package with locked primary model-card audit: {DEFENSE_DIR}")


if __name__ == "__main__":
    main()
