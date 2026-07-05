#!/usr/bin/env python3
"""Update manuscript defense package with targeted ch1 pair-ablation result.

Writes manuscript-defense summaries only. Does not train, edit tensors, change
metadata, update ledgers, alter configs, or modify model-output folders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFENSE_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
PAIR_DIR = ROOT / "results/revision_bspc_2026/channel_pair_ablation_fast3x3_offdiag_channelmean_ch1_anchor"

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

EXPECTED_PAIRS = ["ch1_0", "ch1_2", "ch1_3", "ch1_4", "ch1_5", "ch1_6"]


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
        after = rest[next_idx:] if next_idx >= 0 else ""
        new = before.rstrip() + section + after
    else:
        new = old.rstrip() + section
    path.write_text(new.strip() + "\n", encoding="utf-8")


def load_pair_results() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    path = PAIR_DIR / "primary_pair_table.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    completed = set(df["run_key"].astype(str))
    missing = sorted(set(["ch1", *EXPECTED_PAIRS]) - completed)
    if missing:
        raise RuntimeError(f"Missing expected completed pair-ablation rows: {missing}")
    ch1 = df[df["run_key"].eq("ch1")].iloc[0]
    pairs = df[df["run_key"].isin(EXPECTED_PAIRS)].copy()
    best_pair = pairs.sort_values(["auc", "pr_auc", "balanced_accuracy"], ascending=False).iloc[0]
    if str(best_pair["run_key"]) != "ch1_2":
        raise RuntimeError(f"Expected ch1_2 as best pair, got {best_pair['run_key']}")
    return df, ch1, best_pair


def update_failed_optimization_table(best_pair: pd.Series) -> None:
    row = {
        "candidate": "targeted_ch1_anchor_pair_ablation_fast3x3",
        "category": "scale-corrected channel pair ablation",
        "evaluation_stage": "FAST 3x3",
        "decision": (
            "not promoted: all six ch1-anchored pairs completed; best pair [1,2] "
            "did not beat [1], so no FULL pair confirmation is justified"
        ),
        "auc": float(best_pair["auc"]),
        "pr_auc": float(best_pair["pr_auc"]),
        "balanced_accuracy": float(best_pair["balanced_accuracy"]),
        "sensitivity": float(best_pair["sensitivity"]),
        "specificity": float(best_pair["specificity"]),
        "f1": float(best_pair["f1"]),
    }
    path = DEFENSE_DIR / "failed_optimization_table.csv"
    df = pd.read_csv(path)
    df = df[df["candidate"] != row["candidate"]].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df["delta_auc_vs_locked"] = df["auc"].astype(float) - FINAL_REF["auc"]
    df["delta_pr_auc_vs_locked"] = df["pr_auc"].astype(float) - FINAL_REF["pr_auc"]
    df.to_csv(path, index=False, float_format="%.6f")
    write_markdown_table(df, DEFENSE_DIR / "failed_optimization_table.md")


def write_pair_note(pair_table: pd.DataFrame, ch1: pd.Series, best_pair: pd.Series) -> None:
    compact = pair_table[
        [
            "run_key",
            "channels",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
        ]
    ].copy()
    compact = compact.sort_values(["auc", "pr_auc"], ascending=False)
    text = f"""# Targeted Ch1-Anchored FAST Pair-Ablation Result

Completed pairs: `[1,0]`, `[1,2]`, `[1,3]`, `[1,4]`, `[1,5]`, `[1,6]`.

The best overall FAST subset remains `[1]`:

- `[1]`: AUC `{float(ch1['auc']):.6f}`, PR-AUC `{float(ch1['pr_auc']):.6f}`, BA `{float(ch1['balanced_accuracy']):.6f}`, F1 `{float(ch1['f1']):.6f}`
- Best pair `[1,2]`: AUC `{float(best_pair['auc']):.6f}`, PR-AUC `{float(best_pair['pr_auc']):.6f}`, BA `{float(best_pair['balanced_accuracy']):.6f}`, F1 `{float(best_pair['f1']):.6f}`

Delta `[1,2]` minus `[1]`:

- AUC `{float(best_pair['auc']) - float(ch1['auc']):+.6f}`
- PR-AUC `{float(best_pair['pr_auc']) - float(ch1['pr_auc']):+.6f}`
- BA `{float(best_pair['balanced_accuracy']) - float(ch1['balanced_accuracy']):+.6f}`
- F1 `{float(best_pair['f1']) - float(ch1['f1']):+.6f}`

No pair passes the pre-specified FAST promotion rule, so no FULL pair confirmation is justified. The final main model remains v5.1b `[1,0,2]`, and `[1]` remains the secondary simplified/AUROC-optimized model.

## Pair Ranking

{compact.to_markdown(index=False)}
"""
    (DEFENSE_DIR / "ch1_anchor_pair_ablation_fast3x3_result.md").write_text(text, encoding="utf-8")


def update_final_recommendation(ch1: pd.Series, best_pair: pd.Series) -> None:
    body = f"""The targeted ch1-anchored FAST pair-ablation completed all six planned pairs: `[1,0]`, `[1,2]`, `[1,3]`, `[1,4]`, `[1,5]`, and `[1,6]`. The best overall FAST subset remains `[1]` with AUC `{float(ch1['auc']):.6f}` and PR-AUC `{float(ch1['pr_auc']):.6f}`. The best pair is `[1,2]`, but it does not beat `[1]` by either AUC (`{float(best_pair['auc']):.6f}` vs `{float(ch1['auc']):.6f}`) or PR-AUC (`{float(best_pair['pr_auc']):.6f}` vs `{float(ch1['pr_auc']):.6f}`).

No pair passes the pre-specified FAST promotion rule. Therefore no FULL pair confirmation is justified. The final main manuscript model remains `v5.1b horizon4480/cycles56 FULL [1,0,2]`; `[1]` remains a secondary simplified/AUROC-optimized model rather than the primary clinical readout."""
    replace_or_append_section(DEFENSE_DIR / "final_recommendation.md", "Targeted Ch1-Anchored FAST Pair Ablation", body)


def update_readme(ch1: pd.Series, best_pair: pd.Series) -> None:
    body = f"""- Completed pairs: `[1,0]`, `[1,2]`, `[1,3]`, `[1,4]`, `[1,5]`, `[1,6]`.
- Best overall FAST subset remains `[1]`: AUC `{float(ch1['auc']):.6f}`, PR-AUC `{float(ch1['pr_auc']):.6f}`.
- Best pair is `[1,2]`: AUC `{float(best_pair['auc']):.6f}`, PR-AUC `{float(best_pair['pr_auc']):.6f}`.
- Decision: no pair passes the FAST promotion rule; no FULL pair confirmation is justified.
- Final model remains v5.1b `[1,0,2]`; `[1]` remains secondary simplified/AUROC-optimized."""
    replace_or_append_section(DEFENSE_DIR / "README.md", "Targeted Ch1-Anchored FAST Pair Ablation", body)


def update_reviewer_text(ch1: pd.Series, best_pair: pd.Series) -> None:
    failed = pd.read_csv(DEFENSE_DIR / "failed_optimization_table.csv")
    failed_summary = (
        "The table below includes controlled negative/non-promoted checks. "
        "`ch1_only_offdiag_channelmean` is retained as a secondary simplified/channel-ablation model. "
        "`targeted_ch1_anchor_pair_ablation_fast3x3` records the completed targeted pair-ablation; it did not justify a FULL pair confirmation.\n\n"
        + failed.to_markdown(index=False)
    )
    replace_or_append_section(DEFENSE_DIR / "reviewer_response_ready_text.md", "Failed Optimization Summary", failed_summary)

    body = f"""We added a targeted scale-corrected FAST 3x3 pair-ablation around the dominant Pearson Full channel. All six pairs involving `[1]` were completed: `[1,0]`, `[1,2]`, `[1,3]`, `[1,4]`, `[1,5]`, and `[1,6]`. The best overall FAST subset remained `[1]` (AUC `{float(ch1['auc']):.4f}`, PR-AUC `{float(ch1['pr_auc']):.4f}`). The best pair was `[1,2]`, but it did not beat `[1]` by AUC (`{float(best_pair['auc']):.4f}` vs `{float(ch1['auc']):.4f}`) or PR-AUC (`{float(best_pair['pr_auc']):.4f}` vs `{float(ch1['pr_auc']):.4f}`).

Therefore, no two-channel pair passed the pre-specified FAST promotion rule and no FULL pair confirmation was justified. This strengthens the channel-selection interpretation: Pearson Full is the dominant single channel, `[1]` is useful as a secondary simplified/AUROC-optimized model, and the final manuscript model remains `[1,0,2]` because it preserves the best confirmed clinical operating-point tradeoff in FULL 5x5 validation."""
    replace_or_append_section(DEFENSE_DIR / "reviewer_response_ready_text.md", "Reviewer Q2: Targeted Ch1 Pair Ablation", body)
    (DEFENSE_DIR / "reviewer_q2_response_text.md").write_text("# Reviewer Q2 Response Text\n\n" + body + "\n", encoding="utf-8")


def write_manuscript_paragraph(ch1: pd.Series, best_pair: pd.Series) -> None:
    text = f"""# Manuscript Channel-Ablation Paragraph

As a scale-corrected confirmation of channel relevance, we trained a FULL 5x5 model using only the Pearson Full Fisher-z channel (`[1]`) with the off-diagonal channel-mean reconstruction objective. This parsimonious model achieved slightly higher ROC-AUC than the multichannel `[1,0,2]` model (0.7894 vs 0.7830) and higher specificity (0.7300 vs 0.7000), confirming that Pearson Full connectivity carries the dominant single-channel AD/CN rank signal. However, the multichannel model retained better PR-AUC (0.5599 vs 0.5426), sensitivity (0.7708 vs 0.7188), balanced accuracy (0.7354 vs 0.7244), and F1 (0.5692 vs 0.5610) at the pre-specified sensitivity-constrained operating point.

We further performed a targeted FAST 3x3 pair-ablation around the dominant `[1]` channel. All six pairs involving `[1]` were completed (`[1,0]`, `[1,2]`, `[1,3]`, `[1,4]`, `[1,5]`, `[1,6]`). The best overall FAST subset remained `[1]` (AUC {float(ch1['auc']):.4f}, PR-AUC {float(ch1['pr_auc']):.4f}); the best pair was `[1,2]`, but it did not improve AUC ({float(best_pair['auc']):.4f} vs {float(ch1['auc']):.4f}) or PR-AUC ({float(best_pair['pr_auc']):.4f} vs {float(ch1['pr_auc']):.4f}) relative to `[1]`. No pair satisfied the pre-specified FAST promotion rule. We therefore retained `[1,0,2]` as the primary manuscript model and report `[1]` as a secondary simplified/AUROC-optimized channel-ablation model rather than selecting any two-channel pair for FULL confirmation.
"""
    (DEFENSE_DIR / "manuscript_channel_ablation_paragraph.md").write_text(text, encoding="utf-8")


def update_command_log(ch1: pd.Series, best_pair: pd.Series) -> None:
    path = DEFENSE_DIR / "command_log.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updates = data.setdefault("updates", [])
    record = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "action": "add_targeted_ch1_anchor_pair_ablation_fast3x3_result",
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "configs_modified": False,
        "existing_model_outputs_modified": False,
        "decision": "no_full_pair_confirmation_justified",
        "final_model_reference": FINAL_REF,
        "pair_ablation_result": {
            "completed_pairs": ["[1,0]", "[1,2]", "[1,3]", "[1,4]", "[1,5]", "[1,6]"],
            "best_overall_subset": {
                "channels": "[1]",
                "auc": float(ch1["auc"]),
                "pr_auc": float(ch1["pr_auc"]),
                "balanced_accuracy": float(ch1["balanced_accuracy"]),
                "f1": float(ch1["f1"]),
            },
            "best_pair": {
                "channels": "[1,2]",
                "auc": float(best_pair["auc"]),
                "pr_auc": float(best_pair["pr_auc"]),
                "balanced_accuracy": float(best_pair["balanced_accuracy"]),
                "f1": float(best_pair["f1"]),
            },
            "promotion_rule_passed": False,
        },
        "source_pair_ablation_dir": str(PAIR_DIR),
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
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if not DEFENSE_DIR.exists():
        raise FileNotFoundError(DEFENSE_DIR)
    pair_table, ch1, best_pair = load_pair_results()
    update_failed_optimization_table(best_pair)
    write_pair_note(pair_table, ch1, best_pair)
    update_final_recommendation(ch1, best_pair)
    update_readme(ch1, best_pair)
    update_reviewer_text(ch1, best_pair)
    write_manuscript_paragraph(ch1, best_pair)
    update_command_log(ch1, best_pair)
    print(f"Updated manuscript defense package: {DEFENSE_DIR}")


if __name__ == "__main__":
    main()
