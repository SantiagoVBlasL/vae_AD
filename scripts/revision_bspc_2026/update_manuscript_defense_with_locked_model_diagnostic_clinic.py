#!/usr/bin/env python3
"""Update the manuscript defense package with the locked-model diagnostic clinic.

This modifies only manuscript-defense package files. It does not train, score,
or modify tensor/metadata/ledger/model-output artifacts.
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
CLINIC_DIR = RESULTS_ROOT / "locked_model_diagnostic_clinic"

README = DEFENSE_DIR / "README.md"
FINAL_RECOMMENDATION = DEFENSE_DIR / "final_recommendation.md"
REVIEWER_TEXT = DEFENSE_DIR / "reviewer_response_ready_text.md"
COMMAND_LOG = DEFENSE_DIR / "command_log.json"
CLINIC_RESULT_MD = DEFENSE_DIR / "locked_model_diagnostic_clinic_result.md"


LOCKED_METRICS = {
    "auc": 0.782951,
    "pr_auc": 0.559873,
    "tn": 210,
    "fp": 90,
    "fn": 22,
    "tp": 74,
    "sensitivity": 0.771,
    "specificity": 0.700,
    "balanced_accuracy": 0.735,
    "f1": 0.569,
    "brier": 0.2133,
    "ece_10bin": 0.2438,
    "raw_manufacturer_ba": 0.828,
    "latent_manufacturer_ba": 0.738,
}


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
        new_text = f"{pre}\n\n{block}\n{rest}".rstrip() + "\n"
    else:
        new_text = text.rstrip() + "\n\n" + block
    path.write_text(new_text, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(clean(v) for v in row.tolist()) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def load_csv(name: str) -> pd.DataFrame:
    path = CLINIC_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_result_doc(updated_utc: str) -> str:
    fold_df = load_csv("adni_error_by_fold.csv")
    stability_df = load_csv("oasis_error_stability_summary.csv")
    score_shift = load_csv("score_shift_and_threshold_location.csv")
    latent_qc = load_csv("latent_space_qc_summary.csv")

    weakest = fold_df.sort_values(["balanced_accuracy", "f1"], ascending=[True, True]).head(2)
    locked_stability = stability_df[
        stability_df["adni_model"].astype(str).eq("locked_v5_1b_ch1_0_2_horizon4480")
    ].copy()
    adni_score = score_shift[
        (score_shift["dataset"] == "ADNI_outer_test")
        & (score_shift["model"] == "locked_v5_1b_ch1_0_2_horizon4480")
    ][["class", "n", "score_median", "threshold_mean", "fraction_above_threshold"]]
    oasis_score = score_shift[
        (score_shift["dataset"] == "OASIS_pilot")
        & (score_shift["model"] == "locked_v5_1b_ch1_0_2_horizon4480")
        & (score_shift["class"] == "AD")
    ][["build_candidate", "class", "n", "score_median", "threshold_mean", "fraction_above_threshold"]]

    leak = latent_qc[
        (latent_qc["qc_type"] == "scanner_leakage")
        & (latent_qc["variable"] == "Manufacturer")
    ].copy()
    leak_summary = pd.DataFrame(
        [
            {
                "raw_manufacturer_ba_mean": round(float(leak["acc_site_raw"].mean()), 3),
                "latent_manufacturer_ba_mean": round(float(leak["acc_site_latent"].mean()), 3),
                "latent_minus_raw_mean": round(float((leak["acc_site_latent"] - leak["acc_site_raw"]).mean()), 3),
            }
        ]
    )

    return f"""# Locked-Model Diagnostic Clinic Result

Updated: `{updated_utc}`

Source audit: `{CLINIC_DIR.relative_to(PROJECT_ROOT)}/`

## ADNI Primary-Threshold Performance

- Confusion matrix: TN `{LOCKED_METRICS['tn']}`, FP `{LOCKED_METRICS['fp']}`, FN `{LOCKED_METRICS['fn']}`, TP `{LOCKED_METRICS['tp']}`.
- Sensitivity `{LOCKED_METRICS['sensitivity']:.3f}`.
- Specificity `{LOCKED_METRICS['specificity']:.3f}`.
- Balanced accuracy `{LOCKED_METRICS['balanced_accuracy']:.3f}`.
- F1 `{LOCKED_METRICS['f1']:.3f}`.

Weakest ADNI folds by BA/F1:

{markdown_table(weakest[['fold', 'balanced_accuracy', 'f1', 'sensitivity', 'specificity', 'fp_rate_cn', 'fn_rate_ad']])}

## OASIS Pilot Error Stability

Across concatenated, runwise164, and runwise140TR OASIS scoring for the locked model:

{markdown_table(locked_stability)}

Interpretation: stable errors outnumber run-handling-sensitive errors, but there
is enough instability to treat the OASIS pilot as calibration/transfer-shift
evidence rather than as a model-selection basis.

## Score Shift

ADNI score distribution at the primary threshold:

{markdown_table(adni_score)}

OASIS AD score medians relative to the transferred ADNI threshold:

{markdown_table(oasis_score)}

Interpretation: the ADNI AD median score is above the threshold, while OASIS AD
medians are below or near the transferred ADNI threshold. This explains the low
fixed-threshold OASIS sensitivity despite non-zero external ranking signal.

## Calibration

- ADNI Brier score: `{LOCKED_METRICS['brier']:.4f}`.
- ADNI 10-bin ECE: `{LOCKED_METRICS['ece_10bin']:.4f}`.
- OASIS calibration remains descriptive only; no threshold fitting or model
  selection was performed.

## Latent Manufacturer Separability

{markdown_table(leak_summary)}

The latent representation reduces scanner/manufacturer separability on average
relative to raw-input scanner/manufacturer separability, but does not eliminate
it. This supports continued scanner/manufacturer auditing and external
calibration rather than further internal AUC chasing.

## Safety

- No training.
- No new scoring.
- No threshold fitting.
- No model selection.
- No tensor, metadata, ledger, or model-output modification.
"""


def update_reviewer_top_section() -> None:
    text = REVIEWER_TEXT.read_text(encoding="utf-8")
    old = (
        "The locked pooled OOF performance is ROC-AUC 0.7788 (bootstrap 95% CI 0.7252-0.8289) "
        "and PR-AUC 0.5518 (bootstrap 95% CI 0.4686-0.6401). At the pre-specified "
        "sensitivity-constrained threshold, the confusion matrix is TN=209, FP=91, FN=26, TP=70, "
        "with sensitivity 0.7292, specificity 0.6967, balanced accuracy 0.7129, and F1 0.5447."
    )
    new = (
        "The locked current manuscript model is the v5.1b horizon4480/cycles56 FULL `[1,0,2]` "
        "candidate. Its pooled Stage B performance is ROC-AUC 0.7830 and PR-AUC 0.5599. "
        "At the pre-specified sensitivity-constrained threshold, the confusion matrix is "
        "TN=210, FP=90, FN=22, TP=74, with sensitivity 0.771, specificity 0.700, "
        "balanced accuracy 0.735, and F1 0.569."
    )
    if old in text:
        text = text.replace(old, new)
    REVIEWER_TEXT.write_text(text, encoding="utf-8")


def update_command_log(updated_utc: str) -> None:
    log = json.loads(COMMAND_LOG.read_text(encoding="utf-8")) if COMMAND_LOG.exists() else {}
    log["last_updated_utc"] = updated_utc
    log["latest_update_script"] = str(Path(__file__).resolve())
    updates = log.setdefault("updates", [])
    updates.append(
        {
            "action": "add_locked_model_diagnostic_clinic",
            "source_dir": str(CLINIC_DIR),
            "recorded_findings": {
                "adni_confusion": {
                    "tn": LOCKED_METRICS["tn"],
                    "fp": LOCKED_METRICS["fp"],
                    "fn": LOCKED_METRICS["fn"],
                    "tp": LOCKED_METRICS["tp"],
                },
                "adni_metrics": {
                    "sensitivity": LOCKED_METRICS["sensitivity"],
                    "specificity": LOCKED_METRICS["specificity"],
                    "balanced_accuracy": LOCKED_METRICS["balanced_accuracy"],
                    "f1": LOCKED_METRICS["f1"],
                    "brier": LOCKED_METRICS["brier"],
                    "ece_10bin": LOCKED_METRICS["ece_10bin"],
                },
                "weakest_adni_folds": ["fold_1", "fold_4"],
                "oasis_error_stability": {
                    "stable_correct": 33,
                    "stable_error": 19,
                    "unstable_error": 8,
                },
                "latent_manufacturer_separability": {
                    "raw_ba_mean": LOCKED_METRICS["raw_manufacturer_ba"],
                    "latent_ba_mean": LOCKED_METRICS["latent_manufacturer_ba"],
                },
                "failed_optimization_table_modified": False,
                "reason_failed_table_not_modified": "diagnostic clinic is an interpretive robustness audit, not an optimization candidate",
            },
            "training_launched": False,
            "new_scoring_performed": False,
            "threshold_fitting_performed": False,
            "model_selection_performed": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "existing_model_outputs_modified": False,
        }
    )
    for key in [
        "training_launched",
        "tensor_modified",
        "metadata_modified",
        "ledger_modified",
        "existing_results_modified",
        "configs_modified",
    ]:
        log[key] = False
    COMMAND_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> None:
    updated_utc = now_utc()
    if not CLINIC_DIR.exists():
        raise FileNotFoundError(CLINIC_DIR)
    if not DEFENSE_DIR.exists():
        raise FileNotFoundError(DEFENSE_DIR)

    result_doc = build_result_doc(updated_utc)
    CLINIC_RESULT_MD.write_text(result_doc, encoding="utf-8")

    readme_section = f"""## Locked-Model Diagnostic Clinic

Audit path: `{CLINIC_DIR.relative_to(PROJECT_ROOT)}/`

The locked-model diagnostic clinic has been added as a read-only robustness and
interpretation audit. It records the current primary-threshold ADNI confusion
matrix (`TN=210`, `FP=90`, `FN=22`, `TP=74`; sensitivity `0.771`, specificity
`0.700`, BA `0.735`, F1 `0.569`), identifies Fold 1 and Fold 4 as the weakest
ADNI folds by BA/F1, and summarizes OASIS pilot transfer behavior.

OASIS error stability for the locked model across concatenated, runwise164, and
runwise140TR scoring was: stable correct `33`, stable error `19`, unstable error
`8`. The score-shift analysis shows that the ADNI AD median score is above the
primary threshold, while OASIS AD medians are below or near the transferred ADNI
threshold. OASIS calibration remains descriptive only.

ADNI calibration from saved outer predictions: Brier `0.2133`, ECE_10bin
`0.2438`. Latent manufacturer separability was reduced relative to raw inputs
on average (`raw BA=0.828`, `latent BA=0.738`). No training, scoring, threshold
fitting, or model selection was performed.
"""
    upsert_marker(README, "LOCKED_MODEL_DIAGNOSTIC_CLINIC", readme_section)

    final_section = """## Locked-Model Diagnostic Clinic

The locked-model diagnostic clinic reinforces the current decision. The final
v5.1b horizon4480/cycles56 `[1,0,2]` model achieves the primary-threshold ADNI
confusion matrix TN=210, FP=90, FN=22, TP=74, with sensitivity 0.771,
specificity 0.700, BA 0.735, and F1 0.569. Fold 1 and Fold 4 remain the weakest
folds by BA/F1.

The OASIS pilot does not justify further ADNI model tuning. Across concatenated,
runwise164, and runwise140TR scoring, locked-model OASIS errors were stable
correct for 33 subjects, stable error for 19, and unstable for 8. The key failure
mode is threshold transfer: ADNI AD scores sit above the threshold on median,
whereas OASIS AD scores are below or near the transferred ADNI threshold.

Decision: keep the locked v5.1b `[1,0,2]` model as the primary manuscript model,
use OASIS calibration/test as the next external step, and do not perform
additional internal AUC optimization based on this diagnostic.
"""
    upsert_marker(FINAL_RECOMMENDATION, "LOCKED_MODEL_DIAGNOSTIC_CLINIC", final_section)

    reviewer_section = """## Reviewer Response: Locked-Model Diagnostic Clinic

We added a read-only diagnostic clinic for the locked ADNI model and OASIS pilot
transfer. The current primary-threshold ADNI confusion matrix is TN=210, FP=90,
FN=22, TP=74, corresponding to sensitivity 0.771, specificity 0.700, balanced
accuracy 0.735, and F1 0.569. The weakest ADNI folds are Fold 1 and Fold 4 by
BA/F1. This supports our interpretation that the model does not collapse to the
majority CN class; rather, the sensitivity-constrained operating point produces
a deliberate specificity tradeoff.

For OASIS, errors across concatenated, runwise164, and runwise140TR processing
were stable correct for 33 subjects, stable error for 19, and unstable for 8.
The score-shift analysis explains the external threshold-transfer problem: the
ADNI AD median score is above the selected threshold, whereas OASIS AD medians
are below or near the transferred ADNI threshold. Calibration therefore remains
descriptive in the pilot. We did not fit thresholds, select models, train, or
rescore data in this diagnostic. The appropriate next external step is the
pre-specified OASIS calibration/test protocol, not additional internal model
optimization.
"""
    update_reviewer_top_section()
    upsert_marker(REVIEWER_TEXT, "LOCKED_MODEL_DIAGNOSTIC_CLINIC", reviewer_section)

    update_command_log(updated_utc)
    print(f"Updated defense package with locked-model diagnostic clinic: {DEFENSE_DIR}")
    print("Failed optimization table unchanged: diagnostic clinic is not an optimization candidate.")


if __name__ == "__main__":
    main()
