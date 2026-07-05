#!/usr/bin/env python3
"""Update manuscript defense package with objective-v2 offdiag negative confirmation.

This is a lightweight read-only-with-respect-to-data update: it writes only
manuscript defense summaries under results/. It does not train, edit tensors,
change metadata, update ledgers, alter configs, or modify model-output folders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"

FINAL_REF = {
    "run_id": "v5_1b_horizon4480_cycles56",
    "label": "v5.1b horizon4480/cycles56 FULL [1,0,2]",
    "auc": 0.782951,
    "pr_auc": 0.559873,
    "balanced_accuracy": 0.735417,
    "sensitivity": 0.770833,
    "specificity": 0.700000,
    "f1": 0.569231,
}

CURRENT_LOSS = {
    "auc": 0.774433,
    "pr_auc": 0.558842,
    "balanced_accuracy": 0.712801,
    "sensitivity": 0.742268,
    "specificity": 0.683333,
    "f1": 0.545455,
}

OBJECTIVE_V2 = {
    "candidate": "v5_1c_objective_v2_offdiag_channelmean_horizon10000_cycles125",
    "category": "objective-v2 loss-scaling sensitivity",
    "evaluation_stage": "FULL 5x5",
    "decision": (
        "not promoted: integrity PASS but offdiag_channelmean_sum did not beat "
        "v5.1b horizon4480 on AUC/PR-AUC"
    ),
    "auc": 0.773196,
    "pr_auc": 0.516672,
    "balanced_accuracy": 0.731289,
    "sensitivity": 0.752577,
    "specificity": 0.710000,
    "f1": 0.568093,
}

STAGE_A = {
    "classifier": "logreg",
    "auc_final": 0.7952,
    "pr_auc": 0.5974,
    "sensitivity": 0.1532,
}


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def update_failed_optimization_table() -> None:
    path = OUT_DIR / "failed_optimization_table.csv"
    df = pd.read_csv(path)
    df = df[df["candidate"] != OBJECTIVE_V2["candidate"]].copy()
    df = pd.concat([df, pd.DataFrame([OBJECTIVE_V2])], ignore_index=True)
    df["delta_auc_vs_locked"] = df["auc"].astype(float) - FINAL_REF["auc"]
    df["delta_pr_auc_vs_locked"] = df["pr_auc"].astype(float) - FINAL_REF["pr_auc"]
    df.to_csv(path, index=False, float_format="%.6f")
    write_markdown_table(df, OUT_DIR / "failed_optimization_table.md")


def write_postmortem() -> None:
    text = f"""# v5.1c Objective-v2 Offdiag Horizon10000/Cycles125 Negative Confirmation

Primary comparison reference: `{FINAL_REF['label']}`.

Integrity audit: **PASS**.

## Controlled Change

The candidate changed the reconstruction objective to `offdiag_channelmean_sum`, while keeping the v5.1c horizon10000/cycles125 settings otherwise fixed. The goal was to remove diagonal reconstruction contribution and avoid linear reconstruction-loss scaling with channel count.

## Stage A

- Canonical Stage A logreg `auc_final`: `{STAGE_A['auc_final']:.4f}`
- Canonical Stage A logreg `PR-AUC`: `{STAGE_A['pr_auc']:.4f}`
- Canonical Stage A logreg sensitivity: `{STAGE_A['sensitivity']:.4f}`

The Stage A ranking metrics were numerically high, but sensitivity was only `0.1532`, so Stage A is not the manuscript operating point and cannot justify promotion.

## Stage B Primary Readout

Primary readout: classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.

| run | AUC | PR-AUC | BA | Sens | Spec | F1 |
|---|---:|---:|---:|---:|---:|---:|
| v5.1b horizon4480/cycles56 | {FINAL_REF['auc']:.6f} | {FINAL_REF['pr_auc']:.6f} | {FINAL_REF['balanced_accuracy']:.6f} | {FINAL_REF['sensitivity']:.6f} | {FINAL_REF['specificity']:.6f} | {FINAL_REF['f1']:.6f} |
| v5.1c horizon10000/cycles125 current loss | {CURRENT_LOSS['auc']:.6f} | {CURRENT_LOSS['pr_auc']:.6f} | {CURRENT_LOSS['balanced_accuracy']:.6f} | {CURRENT_LOSS['sensitivity']:.6f} | {CURRENT_LOSS['specificity']:.6f} | {CURRENT_LOSS['f1']:.6f} |
| v5.1c objective-v2 offdiag horizon10000/cycles125 | {OBJECTIVE_V2['auc']:.6f} | {OBJECTIVE_V2['pr_auc']:.6f} | {OBJECTIVE_V2['balanced_accuracy']:.6f} | {OBJECTIVE_V2['sensitivity']:.6f} | {OBJECTIVE_V2['specificity']:.6f} | {OBJECTIVE_V2['f1']:.6f} |

Delta versus v5.1b horizon4480:

- ROC-AUC: `{OBJECTIVE_V2['auc'] - FINAL_REF['auc']:+.6f}`
- PR-AUC: `{OBJECTIVE_V2['pr_auc'] - FINAL_REF['pr_auc']:+.6f}`
- Balanced accuracy: `{OBJECTIVE_V2['balanced_accuracy'] - FINAL_REF['balanced_accuracy']:+.6f}`
- F1: `{OBJECTIVE_V2['f1'] - FINAL_REF['f1']:+.6f}`

## Decision

Do not promote `v5.1c objective-v2 offdiag_channelmean horizon10000/cycles125`. Although BA/F1 were close to the v5.1b reference and better than the v5.1c current-loss horizon10000 run, the candidate did not beat the final v5.1b horizon4480 model on ROC-AUC or PR-AUC. The final model remains `v5.1b horizon4480/cycles56`.
"""
    (OUT_DIR / "v5_1c_objective_v2_offdiag_channelmean_horizon10000_cycles125_postmortem.md").write_text(
        text, encoding="utf-8"
    )


def update_final_recommendation() -> None:
    text = f"""# Final Recommendation

Keep `v5.1b horizon4480/cycles56` FULL tanh `[1,0,2]` as the manuscript model unless a later integrity audit invalidates it.

Current final reference metrics for classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`:

- ROC-AUC: `{FINAL_REF['auc']:.6f}`
- PR-AUC: `{FINAL_REF['pr_auc']:.6f}`
- Balanced accuracy: `{FINAL_REF['balanced_accuracy']:.6f}`
- Sensitivity: `{FINAL_REF['sensitivity']:.6f}`
- Specificity: `{FINAL_REF['specificity']:.6f}`
- F1: `{FINAL_REF['f1']:.6f}`

Do not promote fc0, dropout010, dropout020, beta65, no_decoder_dropout, block_order=norm_act, final_activation=none, manufacturer-balanced VAE sampling, channel dropout, batch_size=32, ultra-regularized logreg readout, frozen-latent classifier-selection variants, `v5.1c horizon10000/cycles125`, or `v5.1c objective-v2 offdiag_channelmean horizon10000/cycles125`. None satisfy the core promotion rule of improving ROC-AUC and PR-AUC without subgroup or operating-point tradeoffs.

The `v5.1c horizon10000/cycles125` confirmation is non-promoted. Stage A canonical logreg reached `auc_raw=0.8016` and `auc_final=0.7996`, but sensitivity was only `0.1537`, so that Stage A readout is not a clinically usable operating point and is not the manuscript readout. The Stage B primary readout produced ROC-AUC `0.774433`, PR-AUC `0.558842`, BA `0.712801`, sensitivity `0.742268`, specificity `0.683333`, and F1 `0.545455`, which does not beat the v5.1b horizon4480 final reference (`AUC=0.782951`, `PR-AUC=0.559873`, `BA=0.735417`, `F1=0.569231`).

The `v5.1c objective-v2 offdiag_channelmean horizon10000/cycles125` confirmation is also non-promoted. Integrity audit passed. Stage A canonical logreg reached `auc_final=0.7952` and `PR-AUC=0.5974`, but sensitivity was only `0.1532`, so Stage A is not a usable manuscript operating point. The Stage B primary readout produced ROC-AUC `0.773196`, PR-AUC `0.516672`, BA `0.731289`, sensitivity `0.752577`, specificity `0.710000`, and F1 `0.568093`. This improved BA/F1 relative to v5.1c current-loss horizon10000, but it did not beat the v5.1b horizon4480 reference on ROC-AUC or PR-AUC.

The dropout010 FULL 5x5 confirmation changed only `dropout_rate_vae` from `0.15` to `0.10`. It did not improve ranking metrics: ROC-AUC decreased from `0.778785` to `0.743542`, and PR-AUC decreased from `0.551832` to `0.492376`. Threshold metrics were essentially unchanged or slightly lower, so this capacity-relaxation variant should not replace the locked model.

The dropout020 FULL 5x5 confirmation changed only `dropout_rate_vae` from `0.15` to `0.20`. It also did not improve ranking metrics: ROC-AUC decreased from `0.778785` to `0.772014`, and PR-AUC decreased from `0.551832` to `0.535853`. This closes the one-step dropout sensitivity check on both sides of the locked `0.15` setting.

The no_decoder_dropout FULL 5x5 confirmation changed only `vae_dropout_scope` from the default/effective `legacy_all` to `no_decoder_dropout`, keeping `dropout_rate_vae=0.15` and all other locked model settings unchanged. Removing decoder dropout made training faster and preserved threshold-level metrics, but worsened threshold-independent ranking: ROC-AUC decreased from `0.778785` to `0.761285`, and PR-AUC decreased from `0.551832` to `0.510264`. Therefore the decoder-dropout hygiene variant should not replace the locked model.

The beta65 FULL 5x5 confirmation changed only `beta_vae` from `2.5` to `6.5`. This increased the information-bottleneck pressure under the same reconstruction-loss scaling, but did not improve threshold-independent ranking metrics: ROC-AUC decreased from `0.778785` to `0.772465`, and PR-AUC decreased from `0.551832` to `0.535039`. BA and F1 also decreased (`0.712917` to `0.695833`, and `0.544747` to `0.525097`). Therefore beta65 should not replace the locked current FULL tanh `[1,0,2]` model.

The ultra-regularized `logreg_l2` readout audit tested the concern that the locked classifier selected `C=0.001`, the lower bound of the original grid, in all folds. The extended lower-C grid selected interior stronger-regularization values (`3e-4`, `3e-4`, `1e-5`, `1e-3`, `1e-4`), confirming that the original lower-bound signal was real. However, stronger regularization hurt test ranking substantially: ROC-AUC decreased to `0.723889`, and PR-AUC decreased to `0.447743`. The locked readout remains the reference.

The block-order norm_act FULL 5x5 confirmation changed only `vae_block_order` from the default/effective `legacy_act_norm` to `norm_act`, keeping the locked objective, beta, dropout, channels, scheduler, split, and Stage B readout unchanged. It did not improve ranking metrics: ROC-AUC decreased from `0.778785` to `0.752187`, and PR-AUC decreased from `0.551832` to `0.492421`. Although BA/F1 were slightly higher at the selected threshold, the pre-specified promotion rule required improvement in both AUC and PR-AUC. Therefore block_order=norm_act should not replace the locked model.

The manuscript should present the final v5.1b horizon4480/cycles56 model with conservative claims, threshold-independent metrics, the leakage-safe threshold rule, manufacturer and SiteCode audits, calibration/Brier evidence where available, and the negative optimization table as evidence against cherry-picking.
"""
    (OUT_DIR / "final_recommendation.md").write_text(text, encoding="utf-8")


def update_readme() -> None:
    text = f"""# Locked Current FULL [1,0,2] Manuscript Defense Package

This package is read-only with respect to model/data artifacts. It consolidates locked predictions, bootstrap curves, calibration, subgroup/site audits, scanner leakage summaries, and failed optimization evidence.

Update: `v5.1c objective-v2 offdiag_channelmean horizon10000/cycles125` is added as a negative/non-promoted confirmation. Integrity audit passed, but the Stage B primary readout did not beat the v5.1b horizon4480/cycles56 final reference on AUC/PR-AUC.

## Current Final Model Reference

- Model: `v5.1b horizon4480/cycles56 FULL [1,0,2]`
- Primary readout: classifier-only `logreg_l2`
- Primary threshold: `inner_oof_target_sens_ge_0p70_max_spec`
- ROC-AUC: `{FINAL_REF['auc']:.4f}`
- PR-AUC: `{FINAL_REF['pr_auc']:.4f}`
- Balanced accuracy: `{FINAL_REF['balanced_accuracy']:.4f}`
- Sensitivity: `{FINAL_REF['sensitivity']:.4f}`
- Specificity: `{FINAL_REF['specificity']:.4f}`
- F1: `{FINAL_REF['f1']:.4f}`

## Latest Non-Promoted Confirmation

- Candidate: `v5.1c objective-v2 offdiag_channelmean horizon10000/cycles125`
- Integrity audit: PASS
- Stage A logreg: `auc_final=0.7952`, `PR-AUC=0.5974`, `sensitivity=0.1532`
- Stage B primary: `AUC=0.773196`, `PR-AUC=0.516672`, `BA=0.731289`, `Sens=0.752577`, `Spec=0.710000`, `F1=0.568093`
- Decision: do not promote.

## Decision

Keep the v5.1b horizon4480/cycles56 FULL tanh `[1,0,2]` model as the manuscript model. Stop AUC micro-optimization on this cohort.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def update_command_log() -> None:
    path = OUT_DIR / "command_log.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updates = data.setdefault("updates", [])
    record = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "action": "add_v5_1c_objective_v2_offdiag_channelmean_horizon10000_negative_confirmation",
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "configs_modified": False,
        "existing_model_outputs_modified": False,
        "integrity_audit": "PASS",
        "decision": "do_not_promote",
        "final_model_reference": FINAL_REF,
        "candidate_recorded": {
            **OBJECTIVE_V2,
            "stage_a_logreg": STAGE_A,
            "primary_readout": "classifier-only logreg_l2",
            "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
            "decision": "do_not_promote",
        },
    }
    updates = [u for u in updates if u.get("action") != record["action"]]
    updates.append(record)
    data["updates"] = updates
    data["last_updated_utc"] = record["updated_utc"]
    data["latest_update_script"] = record["script"]
    data["training_launched"] = False
    data["tensor_modified"] = False
    data["metadata_modified"] = False
    data["ledger_modified"] = False
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"Missing output directory: {OUT_DIR}")
    update_failed_optimization_table()
    write_postmortem()
    update_final_recommendation()
    update_readme()
    update_command_log()
    print(f"Updated manuscript defense package: {OUT_DIR}")


if __name__ == "__main__":
    main()
