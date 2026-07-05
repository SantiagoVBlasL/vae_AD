#!/usr/bin/env python3
"""Update manuscript defense package with Manufacturer-conditioned FULL 5x5 result.

This updates documentation/summary tables only. It does not train, modify
input tensors/metadata/ledgers, or touch model-output folders.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFENSE_DIR = REPO / "results/revision_bspc_2026/adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
SOURCE_DIR = REPO / "results/revision_bspc_2026/conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"

LOCKED = {
    "label": "v5.1b horizon4480/cycles56 FULL [1,0,2]",
    "auc": 0.782951,
    "pr_auc": 0.559873,
    "balanced_accuracy": 0.735417,
    "sensitivity": 0.770833,
    "specificity": 0.700000,
    "f1": 0.569231,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt(x: float) -> str:
    return f"{x:.6f}"


def replace_or_append_section(path: Path, heading: str, content: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    content = content.rstrip() + "\n\n"
    pattern = re.compile(
        rf"(^## {re.escape(heading)}\n.*?)(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    section = f"## {heading}\n\n{content}"
    if pattern.search(text):
        text = pattern.sub(section, text)
    else:
        if text and not text.endswith("\n\n"):
            text = text.rstrip() + "\n\n"
        text += section
    path.write_text(text, encoding="utf-8")


def write_failed_table(row: dict[str, object]) -> pd.DataFrame:
    csv_path = DEFENSE_DIR / "failed_optimization_table.csv"
    df = pd.read_csv(csv_path)
    cleanmfr_mask = df["candidate"].eq("conditional_beta_vae_manufacturer_cleanmfr_fast3x3")
    if cleanmfr_mask.any():
        df.loc[cleanmfr_mask, "decision"] = (
            "FAST-stage diagnostic: all 4 cleanmfr candidates completed; Manufacturer conditioning reduced latent leakage "
            "but did not satisfy FAST promotion as a standalone result; superseded by corrected FULL 5x5 secondary deconfounding audit"
        )
    df = df[df["candidate"] != row["candidate"]].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    md = "# Failed / Non-Promoted Optimization Table\n\n" + df.to_markdown(index=False) + "\n"
    (DEFENSE_DIR / "failed_optimization_table.md").write_text(md, encoding="utf-8")
    return df


def main() -> int:
    DEFENSE_DIR.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(SOURCE_DIR / "primary_results.csv")
    leakage = pd.read_csv(SOURCE_DIR / "scanner_manufacturer_leakage_summary.csv")
    pred = pd.read_csv(SOURCE_DIR / "latent_covariate_predictability_summary.csv")
    pool = pd.read_csv(SOURCE_DIR / "classifier_pooled_metric_audit.csv")

    cond = primary.loc[primary["candidate_id"] == "ch1_0_2_decoder_only_manufacturer"].iloc[0].to_dict()
    base = primary.loc[primary["candidate_id"] == "ch1_0_2_baseline_unconditioned"].iloc[0].to_dict()
    cond_leak = leakage.loc[leakage["candidate_id"] == "ch1_0_2_decoder_only_manufacturer"].iloc[0].to_dict()
    base_leak = leakage.loc[leakage["candidate_id"] == "ch1_0_2_baseline_unconditioned"].iloc[0].to_dict()
    cond_pred = pred[(pred["candidate_id"] == "ch1_0_2_decoder_only_manufacturer") & (pred["target"] == "Manufacturer")].iloc[0].to_dict()
    base_pred = pred[(pred["candidate_id"] == "ch1_0_2_baseline_unconditioned") & (pred["target"] == "Manufacturer")].iloc[0].to_dict()
    pool_ok = bool(pool["pool_locked_ok"].all()) and set(pool["n"].astype(int)) == {396}

    row = {
        "candidate": "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked",
        "category": "conditional beta-VAE Manufacturer conditioning",
        "evaluation_stage": "FULL 5x5 secondary deconfounding audit",
        "decision": (
            "not promoted as primary: integrity PASS and matched Manufacturer-conditioned decoder improved AUC/PR-AUC "
            "and reduced latent leakage versus its matched baseline, but locked final remains stronger on AUC, BA, "
            "sensitivity, and F1; retain as secondary deconfounding audit"
        ),
        "auc": round(float(cond["auc"]), 6),
        "pr_auc": round(float(cond["pr_auc"]), 6),
        "balanced_accuracy": round(float(cond["balanced_accuracy"]), 6),
        "sensitivity": round(float(cond["sensitivity"]), 6),
        "specificity": round(float(cond["specificity"]), 6),
        "f1": round(float(cond["f1"]), 6),
        "delta_auc_vs_locked": round(float(cond["auc"]) - LOCKED["auc"], 6),
        "delta_pr_auc_vs_locked": round(float(cond["pr_auc"]) - LOCKED["pr_auc"], 6),
    }
    failed_df = write_failed_table(row)

    matched_table = pd.DataFrame(
        [
            {
                "candidate": "matched baseline",
                "candidate_id": base["candidate_id"],
                "AUC": float(base["auc"]),
                "PR-AUC": float(base["pr_auc"]),
                "BA": float(base["balanced_accuracy"]),
                "Sensitivity": float(base["sensitivity"]),
                "Specificity": float(base["specificity"]),
                "F1": float(base["f1"]),
                "latent_leakage_acc": float(base_leak["acc_site_latent"]),
                "manufacturer_pred_BA_from_z": float(base_pred["balanced_accuracy"]),
            },
            {
                "candidate": "decoder-only Manufacturer",
                "candidate_id": cond["candidate_id"],
                "AUC": float(cond["auc"]),
                "PR-AUC": float(cond["pr_auc"]),
                "BA": float(cond["balanced_accuracy"]),
                "Sensitivity": float(cond["sensitivity"]),
                "Specificity": float(cond["specificity"]),
                "F1": float(cond["f1"]),
                "latent_leakage_acc": float(cond_leak["acc_site_latent"]),
                "manufacturer_pred_BA_from_z": float(cond_pred["balanced_accuracy"]),
            },
        ]
    )
    delta = matched_table.iloc[1, 2:].astype(float) - matched_table.iloc[0, 2:].astype(float)

    result_md = f"""# Conditional Manufacturer Beta-VAE FULL 5x5 Result

Decision: `do_not_promote_as_primary`; retain as a secondary deconfounding audit.

## Design Integrity

- Branch: `mfrrecovered035_clfpoollocked`
- Controlled FULL 5x5 matched comparison completed.
- `035_S_6927` was restored to VAE metadata with `Manufacturer=SIEMENS`.
- `035_S_6927` was explicitly excluded from supervised classifier train/dev/test folds.
- `128_S_2002` remained excluded/unresolved.
- Classifier pool integrity: `PASS` (`n=396`, `CN=300`, `AD=96`).
- Integrity audit: `PASS`.
- Classifier features: `z_plus_age_sex`; Manufacturer was not passed to the classifier.

## Matched FULL 5x5 Comparison

{matched_table.to_markdown(index=False)}

Matched decoder-only Manufacturer deltas versus its unconditioned baseline:

- AUC: `{fmt(float(delta["AUC"]))}`
- PR-AUC: `{fmt(float(delta["PR-AUC"]))}`
- BA: `{fmt(float(delta["BA"]))}`
- F1: `{fmt(float(delta["F1"]))}`
- Latent leakage accuracy: `{fmt(float(delta["latent_leakage_acc"]))}`
- Manufacturer predictability from z, balanced accuracy: `{fmt(float(delta["manufacturer_pred_BA_from_z"]))}`

## Comparison With Final Manuscript Model

The Manufacturer-conditioned candidate improves versus its matched branch baseline, but it does not replace the locked final model:

- Candidate AUC `{fmt(float(cond["auc"]))}` vs final `{fmt(LOCKED["auc"])}`.
- Candidate PR-AUC `{fmt(float(cond["pr_auc"]))}` vs final `{fmt(LOCKED["pr_auc"])}`.
- Candidate BA `{fmt(float(cond["balanced_accuracy"]))}` vs final `{fmt(LOCKED["balanced_accuracy"])}`.
- Candidate sensitivity `{fmt(float(cond["sensitivity"]))}` vs final `{fmt(LOCKED["sensitivity"])}`.
- Candidate F1 `{fmt(float(cond["f1"]))}` vs final `{fmt(LOCKED["f1"])}`.

Interpretation: decoder-only Manufacturer conditioning is useful evidence that nuisance-aware representation learning can reduce scanner/manufacturer information in `z` while improving the matched branch's AUC/PR-AUC. However, the locked final `v5.1b [1,0,2] horizon4480/cycles56` model remains stronger on AUC, BA, sensitivity, and F1, so this conditional branch should be reported only as a secondary deconfounding audit.
"""
    (DEFENSE_DIR / "conditional_beta_vae_manufacturer_full5x5_result.md").write_text(result_md, encoding="utf-8")

    latest = """- Candidate: `conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked`
- Evaluation: FULL 5x5 secondary deconfounding audit.
- Integrity: PASS; classifier pool locked at `n=396`, `CN=300`, `AD=96`.
- Matched result: decoder-only Manufacturer improved versus its matched baseline (`AUC=0.7713` vs `0.7576`, `PR-AUC=0.5901` vs `0.5468`) and reduced latent leakage (`0.6707` vs `0.7244`).
- Decision: `do_not_promote_as_primary`; retain as secondary deconfounding audit because the locked final model has stronger AUC, BA, sensitivity, and F1."""
    replace_or_append_section(DEFENSE_DIR / "README.md", "Latest Non-Promoted Confirmation", latest)
    readme_path = DEFENSE_DIR / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_text = re.sub(
        r"^Update: .*$",
        "Update: FULL Manufacturer-conditioned beta-VAE `mfrrecovered035_clfpoollocked` is added as a secondary deconfounding audit. It improved versus its matched baseline and reduced latent leakage, but it does not replace the final v5.1b `[1,0,2]` model.",
        readme_text,
        count=1,
        flags=re.MULTILINE,
    )
    readme_path.write_text(readme_text, encoding="utf-8")

    readme_section = """The corrected `mfrrecovered035_clfpoollocked` FULL 5x5 branch completed a matched comparison between `[1,0,2]` unconditioned VAE and decoder-only Manufacturer-conditioned VAE. `035_S_6927` was restored to the VAE metadata with `Manufacturer=SIEMENS` but explicitly excluded from all supervised classifier folds; `128_S_2002` remained excluded. Classifier pool integrity passed (`n=396`, `CN=300`, `AD=96`) and the integrity audit passed.

Decoder-only Manufacturer conditioning improved over its matched unconditioned baseline (`AUC=0.7713` vs `0.7576`, `PR-AUC=0.5901` vs `0.5468`, `BA=0.7040` vs `0.7006`, `F1=0.5360` vs `0.5317`) and reduced latent scanner/manufacturer leakage (`0.6707` vs `0.7244`). It is not promoted as the primary manuscript model because the locked final v5.1b `[1,0,2]` horizon4480/cycles56 model remains stronger on AUC, BA, sensitivity, and F1. This result is retained as a secondary deconfounding audit."""
    replace_or_append_section(DEFENSE_DIR / "README.md", "Conditional Manufacturer Beta-VAE FULL 5x5", readme_section)

    final_rec_path = DEFENSE_DIR / "final_recommendation.md"
    final_text = final_rec_path.read_text(encoding="utf-8")
    final_text = final_text.replace(
        "or conditional Manufacturer beta-VAE cleanmfr FAST 3x3. None satisfy",
        "conditional Manufacturer beta-VAE cleanmfr FAST 3x3, or conditional Manufacturer beta-VAE FULL 5x5 as primary. None satisfy",
    )
    final_rec_path.write_text(final_text, encoding="utf-8")
    full_section = """The corrected `mfrrecovered035_clfpoollocked` FULL 5x5 Manufacturer-conditioned beta-VAE branch completed with classifier-pool integrity PASS and integrity audit PASS. `035_S_6927` was restored to the VAE metadata with `Manufacturer=SIEMENS` but excluded from supervised classifier folds; `128_S_2002` remained excluded. The supervised classifier pool stayed locked at `n=396`, `CN=300`, `AD=96`.

Decoder-only Manufacturer conditioning improved over its matched unconditioned baseline: AUC `0.7713` vs `0.7576`, PR-AUC `0.5901` vs `0.5468`, BA `0.7040` vs `0.7006`, and F1 `0.5360` vs `0.5317`. It also reduced latent scanner/manufacturer leakage (`0.6707` vs `0.7244`) and Manufacturer predictability from `z`.

Decision: `do_not_promote_as_primary`; retain as a secondary deconfounding audit. The locked final v5.1b horizon4480/cycles56 FULL `[1,0,2]` model remains the manuscript model because it has stronger AUC (`0.782951`), balanced accuracy (`0.735417`), sensitivity (`0.770833`), and F1 (`0.569231`) than the Manufacturer-conditioned candidate."""
    replace_or_append_section(final_rec_path, "Conditional Manufacturer Beta-VAE FULL 5x5", full_section)
    cleanmfr_final_section = """The cleanmfr FAST 3x3 screen remains a non-promoted exploratory precursor. It showed that Manufacturer decoder conditioning could reduce latent scanner/manufacturer leakage, but the FAST result alone was not promoted as a manuscript-model change.

This FAST-stage result is now superseded for decision-making by the corrected `mfrrecovered035_clfpoollocked` FULL 5x5 audit below, which keeps the supervised classifier pool locked while restoring `035_S_6927` only for VAE conditioning metadata."""
    replace_or_append_section(final_rec_path, "Conditional Manufacturer Beta-VAE cleanmfr FAST 3x3", cleanmfr_final_section)

    # Refresh the failed-optimization table embedded in reviewer response and replace the stale cleanmfr-only section.
    reviewer_path = DEFENSE_DIR / "reviewer_response_ready_text.md"
    reviewer_text = reviewer_path.read_text(encoding="utf-8")
    failed_md = (DEFENSE_DIR / "failed_optimization_table.md").read_text(encoding="utf-8")
    table_only = failed_md.split("\n\n", 1)[1]
    failed_section = """## Failed Optimization Summary

The table below includes controlled negative/non-promoted checks. `ch1_only_offdiag_channelmean` is retained as a secondary simplified/channel-ablation model. The Manufacturer-conditioned FULL 5x5 branch is retained as a secondary deconfounding audit because it improved versus its matched branch baseline and reduced latent leakage, but it did not replace the locked final model.

""" + table_only.rstrip() + "\n\n"
    reviewer_text = re.sub(
        r"^## Failed Optimization Summary\n.*?(?=^## Conditional Age/Sex Beta-VAE Exploratory Audit)",
        failed_section,
        reviewer_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    reviewer_path.write_text(reviewer_text, encoding="utf-8")
    manufacturer_reviewer_section = """We completed the corrected `mfrrecovered035_clfpoollocked` FULL 5x5 Manufacturer-conditioned beta-VAE comparison. This branch restored `035_S_6927` to the VAE metadata with `Manufacturer=SIEMENS` while explicitly excluding it from all supervised classifier folds; `128_S_2002` remained excluded. Classifier pool integrity passed with `n=396`, `CN=300`, and `AD=96`, and the run-level integrity audit passed.

Compared with its matched unconditioned baseline, decoder-only Manufacturer conditioning improved AUC (`0.7713` vs `0.7576`), PR-AUC (`0.5901` vs `0.5468`), BA (`0.7040` vs `0.7006`), and F1 (`0.5360` vs `0.5317`), while reducing latent scanner/manufacturer leakage (`0.6707` vs `0.7244`). This supports the value of scanner-aware deconfounding as an audit.

We did not promote it as the primary manuscript model because the locked final `v5.1b [1,0,2] horizon4480/cycles56` model remains stronger on the primary clinical readout tradeoff: AUC `0.7830`, BA `0.7354`, sensitivity `0.7708`, and F1 `0.5692`. We therefore retain Manufacturer conditioning as a secondary deconfounding analysis rather than changing the manuscript model."""
    replace_or_append_section(
        reviewer_path,
        "Reviewer Response: Manufacturer-Conditioned Beta-VAE FULL 5x5",
        manufacturer_reviewer_section,
    )
    cleanmfr_reviewer_section = """We first evaluated Manufacturer conditioning in a clean-Manufacturer FAST 3x3 screen. This screen showed that decoder-only Manufacturer conditioning can reduce scanner/manufacturer information in the latent representation, but the FAST-stage evidence alone was not promoted as a manuscript-model change.

Because the strongest valid diagnostic signal was in the `[1,0,2]` Manufacturer-conditioned branch, we then ran the corrected FULL 5x5 `mfrrecovered035_clfpoollocked` audit described below. The cleanmfr FAST result is therefore retained as screening context rather than as the final decision point."""
    replace_or_append_section(
        reviewer_path,
        "Reviewer Response: Manufacturer-Conditioned Beta-VAE (cleanmfr FAST 3x3)",
        cleanmfr_reviewer_section,
    )

    log_path = DEFENSE_DIR / "command_log.json"
    command_log = json.loads(log_path.read_text(encoding="utf-8"))
    command_log["last_updated_utc"] = utc_now()
    command_log["latest_update_script"] = str(Path(__file__).resolve())
    action = "add_conditional_manufacturer_beta_vae_full5x5_secondary_deconfounding_audit"
    command_log["updates"] = [
        item for item in command_log.get("updates", []) if item.get("action") != action
    ]
    command_log.setdefault("updates", []).append(
        {
            "action": action,
            "source_dir": str(SOURCE_DIR),
            "candidate_recorded": {
                **row,
                "integrity_audit": "PASS",
                "branch": "mfrrecovered035_clfpoollocked",
                "classifier_pool_integrity": {
                    "pass": pool_ok,
                    "n": 396,
                    "cn": 300,
                    "ad": 96,
                    "035_S_6927_excluded_from_supervised_folds": True,
                    "128_S_2002_excluded": True,
                },
                "matched_baseline": {
                    "auc": round(float(base["auc"]), 6),
                    "pr_auc": round(float(base["pr_auc"]), 6),
                    "balanced_accuracy": round(float(base["balanced_accuracy"]), 6),
                    "f1": round(float(base["f1"]), 6),
                    "latent_leakage": round(float(base_leak["acc_site_latent"]), 6),
                },
                "decoder_only_manufacturer": {
                    "auc": round(float(cond["auc"]), 6),
                    "pr_auc": round(float(cond["pr_auc"]), 6),
                    "balanced_accuracy": round(float(cond["balanced_accuracy"]), 6),
                    "f1": round(float(cond["f1"]), 6),
                    "latent_leakage": round(float(cond_leak["acc_site_latent"]), 6),
                },
            },
            "decision": "do_not_promote_as_primary_secondary_deconfounding_audit",
            "final_model_reference": LOCKED,
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "configs_modified": False,
            "existing_model_outputs_modified": False,
            "updated_utc": utc_now(),
        }
    )
    log_path.write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "updated",
                "failed_table_rows": int(len(failed_df)),
                "candidate_auc": row["auc"],
                "candidate_pr_auc": row["pr_auc"],
                "decision": "do_not_promote_as_primary_secondary_deconfounding_audit",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
