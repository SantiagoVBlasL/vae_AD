#!/usr/bin/env python3
"""Update manuscript defense package with ch1-only offdiag confirmation.

This writes only manuscript-defense summaries under results/. It does not train,
edit tensors, change metadata, update ledgers, alter configs, or modify
model-output folders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFENSE_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
POSTMORTEM_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1b_ch1_only_offdiag_channelmean_postmortem"

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

CH1 = {
    "candidate": "ch1_only_offdiag_channelmean",
    "category": "scale-corrected channel ablation",
    "evaluation_stage": "FULL 5x5",
    "decision": (
        "secondary simplified/channel-ablation model: integrity PASS; AUROC and "
        "specificity improved, but PR-AUC, sensitivity, BA, and F1 worsened"
    ),
    "auc": 0.789375,
    "pr_auc": 0.542605,
    "balanced_accuracy": 0.724375,
    "sensitivity": 0.718750,
    "specificity": 0.730000,
    "f1": 0.560976,
}


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.6f}")
    path.write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def replace_or_append_section(path: Path, heading: str, body: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = f"\n## {heading}\n"
    section = f"\n\n## {heading}\n\n{body.strip()}\n"
    if marker in old:
        before, rest = old.split(marker, 1)
        next_idx = rest.find("\n## ")
        if next_idx >= 0:
            after = rest[next_idx:]
        else:
            after = ""
        new = before.rstrip() + section + after
    else:
        new = old.rstrip() + section
    path.write_text(new.strip() + "\n", encoding="utf-8")


def update_failed_optimization_table() -> None:
    path = DEFENSE_DIR / "failed_optimization_table.csv"
    df = pd.read_csv(path)
    df = df[df["candidate"] != CH1["candidate"]].copy()
    df = pd.concat([df, pd.DataFrame([CH1])], ignore_index=True)
    df["delta_auc_vs_locked"] = df["auc"].astype(float) - FINAL_REF["auc"]
    df["delta_pr_auc_vs_locked"] = df["pr_auc"].astype(float) - FINAL_REF["pr_auc"]
    df.to_csv(path, index=False, float_format="%.6f")
    write_markdown_table(df, DEFENSE_DIR / "failed_optimization_table.md")


def write_ch1_postmortem_note() -> None:
    text = f"""# Ch1-Only Offdiag-Channelmean FULL Confirmation

Integrity audit: **PASS**.

Primary comparison reference: `{FINAL_REF['label']}`.

## Controlled Change

The confirmation used a scale-corrected single-channel model:

- `channels_to_use`: `[1,0,2]` -> `[1]`
- `recon_loss_mode`: `mse_sum_batchmean_current` -> `offdiag_channelmean_sum`

All interpretation uses the Stage B classifier-only `logreg_l2` readout with the pre-specified `inner_oof_target_sens_ge_0p70_max_spec` threshold rule.

## Result

| run | AUC | PR-AUC | BA | Sens | Spec | F1 |
|---|---:|---:|---:|---:|---:|---:|
| v5.1b `[1,0,2]` final | {FINAL_REF['auc']:.6f} | {FINAL_REF['pr_auc']:.6f} | {FINAL_REF['balanced_accuracy']:.6f} | {FINAL_REF['sensitivity']:.6f} | {FINAL_REF['specificity']:.6f} | {FINAL_REF['f1']:.6f} |
| ch1-only offdiag_channelmean | {CH1['auc']:.6f} | {CH1['pr_auc']:.6f} | {CH1['balanced_accuracy']:.6f} | {CH1['sensitivity']:.6f} | {CH1['specificity']:.6f} | {CH1['f1']:.6f} |

Delta ch1-only minus final `[1,0,2]`:

- ROC-AUC: `{CH1['auc'] - FINAL_REF['auc']:+.6f}`
- PR-AUC: `{CH1['pr_auc'] - FINAL_REF['pr_auc']:+.6f}`
- Balanced accuracy: `{CH1['balanced_accuracy'] - FINAL_REF['balanced_accuracy']:+.6f}`
- Sensitivity: `{CH1['sensitivity'] - FINAL_REF['sensitivity']:+.6f}`
- Specificity: `{CH1['specificity'] - FINAL_REF['specificity']:+.6f}`
- F1: `{CH1['f1'] - FINAL_REF['f1']:+.6f}`

## Decision

Do not promote ch1-only as the main model. It is reportable as a secondary simplified/channel-ablation model because it improves AUROC and specificity, confirming that Pearson Full is the dominant channel. The final main manuscript model remains v5.1b `[1,0,2]` because it retains better PR-AUC, sensitivity, balanced accuracy, and F1.
"""
    (DEFENSE_DIR / "ch1_only_offdiag_channelmean_confirmation.md").write_text(text, encoding="utf-8")


def update_final_recommendation() -> None:
    body = f"""The FULL ch1-only offdiag-channelmean confirmation is integrity-audit PASS and improves AUROC versus the final `[1,0,2]` reference (`{CH1['auc']:.6f}` vs `{FINAL_REF['auc']:.6f}`), with higher specificity (`{CH1['specificity']:.6f}` vs `{FINAL_REF['specificity']:.6f}`). It should not replace the main model because it worsens PR-AUC (`{CH1['pr_auc']:.6f}` vs `{FINAL_REF['pr_auc']:.6f}`), sensitivity (`{CH1['sensitivity']:.6f}` vs `{FINAL_REF['sensitivity']:.6f}`), balanced accuracy (`{CH1['balanced_accuracy']:.6f}` vs `{FINAL_REF['balanced_accuracy']:.6f}`), and F1 (`{CH1['f1']:.6f}` vs `{FINAL_REF['f1']:.6f}`).

Interpretation: Pearson Full is the dominant individual channel and the single-channel model is a useful parsimonious secondary ablation. The primary manuscript model remains `v5.1b horizon4480/cycles56 FULL [1,0,2]` because it better preserves precision-recall behavior and sensitivity under the pre-specified leakage-safe operating point."""
    replace_or_append_section(DEFENSE_DIR / "final_recommendation.md", "Ch1-Only Offdiag-Channelmean Confirmation", body)


def update_readme() -> None:
    body = f"""- Candidate: `ch1_only_offdiag_channelmean`
- Integrity audit: PASS
- Stage B primary: `AUC={CH1['auc']:.6f}`, `PR-AUC={CH1['pr_auc']:.6f}`, `BA={CH1['balanced_accuracy']:.6f}`, `Sens={CH1['sensitivity']:.6f}`, `Spec={CH1['specificity']:.6f}`, `F1={CH1['f1']:.6f}`
- Decision: report as secondary simplified/channel-ablation model; do not promote as main model.
- Rationale: AUROC and specificity improved, but PR-AUC, sensitivity, BA, and F1 worsened relative to v5.1b `[1,0,2]`."""
    replace_or_append_section(DEFENSE_DIR / "README.md", "Ch1-Only Channel-Ablation Confirmation", body)


def update_reviewer_text() -> None:
    failed = pd.read_csv(DEFENSE_DIR / "failed_optimization_table.csv")
    failed_summary = (
        "The table below includes controlled negative/non-promoted checks. "
        "`ch1_only_offdiag_channelmean` is not a failed model in the same sense as the optimization variants; "
        "it is retained as a secondary simplified/channel-ablation model but not promoted as the main manuscript model.\n\n"
        + failed.to_markdown(index=False)
    )
    replace_or_append_section(DEFENSE_DIR / "reviewer_response_ready_text.md", "Failed Optimization Summary", failed_summary)

    body = f"""We added a scale-corrected FULL 5x5 single-channel confirmation using only Pearson Full connectivity (`[1]`) with the off-diagonal channel-mean reconstruction objective. This model improved AUROC compared with the final multichannel `[1,0,2]` model (`{CH1['auc']:.4f}` vs `{FINAL_REF['auc']:.4f}`) and improved specificity (`{CH1['specificity']:.4f}` vs `{FINAL_REF['specificity']:.4f}`). This supports the interpretation that Pearson Full is the dominant channel.

However, we did not select the single-channel model as the primary readout because it worsened PR-AUC (`{CH1['pr_auc']:.4f}` vs `{FINAL_REF['pr_auc']:.4f}`), sensitivity (`{CH1['sensitivity']:.4f}` vs `{FINAL_REF['sensitivity']:.4f}`), balanced accuracy (`{CH1['balanced_accuracy']:.4f}` vs `{FINAL_REF['balanced_accuracy']:.4f}`), and F1 (`{CH1['f1']:.4f}` vs `{FINAL_REF['f1']:.4f}`) at the pre-specified leakage-safe sensitivity-constrained threshold. We therefore report `[1]` as a secondary simplified/channel-ablation model and retain `[1,0,2]` as the final manuscript model."""
    replace_or_append_section(DEFENSE_DIR / "reviewer_response_ready_text.md", "Reviewer Q2: Channel Selection and Pearson-Full Ablation", body)
    (DEFENSE_DIR / "reviewer_q2_response_text.md").write_text(
        "# Reviewer Q2 Response Text\n\n" + body + "\n", encoding="utf-8"
    )


def write_manuscript_paragraph() -> None:
    text = f"""# Manuscript Channel-Ablation Paragraph

As a scale-corrected confirmation of channel relevance, we trained a FULL 5x5 model using only the Pearson Full Fisher-z channel (`[1]`) with the off-diagonal channel-mean reconstruction objective. This parsimonious model achieved slightly higher ROC-AUC than the multichannel `[1,0,2]` model ({CH1['auc']:.4f} vs {FINAL_REF['auc']:.4f}) and higher specificity ({CH1['specificity']:.4f} vs {FINAL_REF['specificity']:.4f}), confirming that Pearson Full connectivity carries the dominant single-channel AD/CN rank signal. However, the multichannel model retained better PR-AUC ({FINAL_REF['pr_auc']:.4f} vs {CH1['pr_auc']:.4f}), sensitivity ({FINAL_REF['sensitivity']:.4f} vs {CH1['sensitivity']:.4f}), balanced accuracy ({FINAL_REF['balanced_accuracy']:.4f} vs {CH1['balanced_accuracy']:.4f}), and F1 ({FINAL_REF['f1']:.4f} vs {CH1['f1']:.4f}) at the pre-specified sensitivity-constrained operating point. We therefore retained `[1,0,2]` as the primary manuscript model and report `[1]` as a secondary simplified channel-ablation model rather than an AUROC-only replacement.
"""
    (DEFENSE_DIR / "manuscript_channel_ablation_paragraph.md").write_text(text, encoding="utf-8")


def update_command_log() -> None:
    path = DEFENSE_DIR / "command_log.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updates = data.setdefault("updates", [])
    record = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "action": "add_ch1_only_offdiag_channelmean_confirmation",
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "configs_modified": False,
        "existing_model_outputs_modified": False,
        "integrity_audit": "PASS",
        "decision": "secondary_simplified_model_not_main",
        "final_model_reference": FINAL_REF,
        "candidate_recorded": {
            **CH1,
            "integrity_audit": "PASS",
            "primary_readout": "classifier-only logreg_l2",
            "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
        },
        "source_postmortem_dir": str(POSTMORTEM_DIR),
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
    data["configs_modified"] = False
    data["existing_results_modified"] = False
    data["vae_retrained"] = False
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if not DEFENSE_DIR.exists():
        raise FileNotFoundError(f"Missing defense package directory: {DEFENSE_DIR}")
    update_failed_optimization_table()
    write_ch1_postmortem_note()
    update_final_recommendation()
    update_readme()
    update_reviewer_text()
    write_manuscript_paragraph()
    update_command_log()
    print(f"Updated manuscript defense package: {DEFENSE_DIR}")


if __name__ == "__main__":
    main()
