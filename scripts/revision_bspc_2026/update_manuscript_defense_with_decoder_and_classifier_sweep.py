#!/usr/bin/env python3
"""Update manuscript defense package with two non-promoted audits.

This script is intentionally limited to manuscript-defense artifacts. It does
not read or modify tensors, metadata, ledgers, configs, VAE checkpoints, latent
caches, or model-output folders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
)
LOCKED = {
    "auc": 0.782951,
    "pr_auc": 0.559873,
    "balanced_accuracy": 0.735417,
    "sensitivity": 0.770833,
    "specificity": 0.700000,
    "f1": 0.569231,
}


def to_md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def upsert_failed_rows() -> pd.DataFrame:
    path = BASE / "failed_optimization_table.csv"
    df = pd.read_csv(path)

    new_rows = [
        {
            "candidate": "frozen_latent_stageb_classifier_sweep_pro",
            "category": "frozen-latent classifier sweep",
            "evaluation_stage": "read-only classifier audit",
            "decision": (
                "not promoted: 75/75 model fits completed without VAE retraining; "
                "regularized LightGBM did not improve; elastic-net improved PR-AUC "
                "on [1,0,2] but did not cleanly improve AUC/BA/F1"
            ),
            "auc": 0.782222,
            "pr_auc": 0.580692,
            "balanced_accuracy": 0.727500,
            "sensitivity": 0.791667,
            "specificity": 0.663333,
            "f1": 0.556777,
            "delta_auc_vs_locked": 0.782222 - LOCKED["auc"],
            "delta_pr_auc_vs_locked": 0.580692 - LOCKED["pr_auc"],
        },
        {
            "candidate": "decoder_type_upsample_conv_fast3x3",
            "category": "decoder-type sensitivity",
            "evaluation_stage": "FAST 3x3",
            "decision": (
                "not promoted: upsample_conv underperformed matched convtranspose "
                "baselines for [1] and [1,0,2]; performance drop was large despite "
                "near-horizon/slower behavior, so no FULL upsample run is justified"
            ),
            "auc": 0.737049,
            "pr_auc": 0.454234,
            "balanced_accuracy": 0.682292,
            "sensitivity": 0.697917,
            "specificity": 0.666667,
            "f1": 0.509506,
            "delta_auc_vs_locked": 0.737049 - LOCKED["auc"],
            "delta_pr_auc_vs_locked": 0.454234 - LOCKED["pr_auc"],
        },
    ]

    for row in new_rows:
        mask = df["candidate"].eq(row["candidate"])
        if mask.any():
            for col, value in row.items():
                df.loc[mask, col] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(path, index=False)
    (BASE / "failed_optimization_table.md").write_text(
        "# Failed / Non-Promoted Optimization Table\n\n"
        + to_md_table(df)
        + "\n",
        encoding="utf-8",
    )
    return df


def write_optional_result_files() -> None:
    decoder_rows = pd.DataFrame(
        [
            {
                "channel_set": "[1]",
                "decoder_type": "convtranspose",
                "auc": 0.782604,
                "pr_auc": 0.515499,
                "balanced_accuracy": 0.707083,
                "sensitivity": 0.770833,
                "specificity": 0.643333,
                "f1": 0.534296,
            },
            {
                "channel_set": "[1]",
                "decoder_type": "upsample_conv",
                "auc": 0.737049,
                "pr_auc": 0.454234,
                "balanced_accuracy": 0.682292,
                "sensitivity": 0.697917,
                "specificity": 0.666667,
                "f1": 0.509506,
            },
            {
                "channel_set": "[1,0,2]",
                "decoder_type": "convtranspose",
                "auc": 0.734167,
                "pr_auc": 0.459668,
                "balanced_accuracy": 0.687292,
                "sensitivity": 0.781250,
                "specificity": 0.593333,
                "f1": 0.511945,
            },
            {
                "channel_set": "[1,0,2]",
                "decoder_type": "upsample_conv",
                "auc": 0.701007,
                "pr_auc": 0.423922,
                "balanced_accuracy": 0.665625,
                "sensitivity": 0.697917,
                "specificity": 0.633333,
                "f1": 0.490842,
            },
        ]
    )
    (BASE / "decoder_type_fast3x3_result.md").write_text(
        "# Decoder-Type FAST 3x3 Result\n\n"
        "This was a FAST 3x3 screening audit using the scale-corrected "
        "`offdiag_channelmean_sum` objective. It compared the current "
        "`convtranspose` decoder against `upsample_conv` for `[1]` and `[1,0,2]`.\n\n"
        + to_md_table(decoder_rows)
        + "\n\n"
        "Decision: `do_not_promote`. The `upsample_conv` decoder underperformed "
        "the matched `convtranspose` baseline for both channel sets. The candidate "
        "appeared slower / closer to the training horizon in best-epoch behavior, "
        "but the performance drop was large enough that no FULL 5x5 upsample "
        "confirmation is justified now.\n",
        encoding="utf-8",
    )

    sweep_rows = pd.DataFrame(
        [
            {
                "run": "final [1,0,2]",
                "classifier": "logreg_l2",
                "auc": 0.782951,
                "pr_auc": 0.559873,
                "balanced_accuracy": 0.742083,
                "sensitivity": 0.770833,
                "specificity": 0.713333,
                "f1": 0.578125,
            },
            {
                "run": "final [1,0,2]",
                "classifier": "logreg_elasticnet",
                "auc": 0.782222,
                "pr_auc": 0.580692,
                "balanced_accuracy": 0.727500,
                "sensitivity": 0.791667,
                "specificity": 0.663333,
                "f1": 0.556777,
            },
            {
                "run": "final [1,0,2]",
                "classifier": "lightgbm_very_regularized",
                "auc": 0.740451,
                "pr_auc": 0.475522,
                "balanced_accuracy": 0.680417,
                "sensitivity": 0.687500,
                "specificity": 0.673333,
                "f1": 0.507692,
            },
            {
                "run": "simplified [1]",
                "classifier": "logreg_l2",
                "auc": 0.789375,
                "pr_auc": 0.542605,
                "balanced_accuracy": 0.724375,
                "sensitivity": 0.718750,
                "specificity": 0.730000,
                "f1": 0.560976,
            },
            {
                "run": "[1,2]",
                "classifier": "logreg_elasticnet",
                "auc": 0.772674,
                "pr_auc": 0.566442,
                "balanced_accuracy": 0.712917,
                "sensitivity": 0.729167,
                "specificity": 0.696667,
                "f1": 0.544747,
            },
        ]
    )
    (BASE / "frozen_latent_classifier_sweep_result.md").write_text(
        "# Frozen-Latent Stage B Classifier Sweep Result\n\n"
        "This was a read-only classifier-only sweep on frozen latent caches from "
        "the final VAE runs. No VAE was retrained and no tensor, metadata, ledger, "
        "config, or VAE output artifact was modified. All `75/75` model fits "
        "completed successfully.\n\n"
        + to_md_table(sweep_rows)
        + "\n\n"
        "Decision: `do_not_promote`. Regularized LightGBM did not improve. "
        "Elastic-net improved PR-AUC on the final `[1,0,2]` latent representation "
        "but did not cleanly improve AUC, balanced accuracy, or F1. The current "
        "main model remains the reference readout.\n",
        encoding="utf-8",
    )


def append_or_replace_section(path: Path, heading: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = f"## {heading}\n\n{body.strip()}\n"
    marker = f"## {heading}"
    if marker in text:
        start = text.index(marker)
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + section
        else:
            text = text[:start].rstrip() + "\n\n" + section + "\n" + text[next_start + 1 :].lstrip()
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def update_summary_docs(df: pd.DataFrame) -> None:
    frozen_body = (
        "The frozen-latent Stage B classifier sweep is non-promoted. All `75/75` "
        "model fits completed, no VAE was retrained, and no model/data artifacts "
        "were modified. Regularized LightGBM did not improve. Elastic-net improved "
        "PR-AUC on `[1,0,2]` (`0.580692` vs `0.559873`) but did not cleanly "
        "improve ROC-AUC (`0.782222` vs `0.782951`), BA, or F1. The final decision "
        "is `do_not_promote`."
    )
    decoder_body = (
        "The decoder-type FAST 3x3 audit is non-promoted. For `[1]`, "
        "`convtranspose` achieved AUC `0.782604` and PR-AUC `0.515499`, while "
        "`upsample_conv` dropped to AUC `0.737049` and PR-AUC `0.454234`. For "
        "`[1,0,2]`, `convtranspose` achieved AUC `0.734167` and PR-AUC `0.459668`, "
        "while `upsample_conv` dropped to AUC `0.701007` and PR-AUC `0.423922`. "
        "Although `upsample_conv` appeared slower / closer to the training horizon, "
        "the performance drop is large; no FULL upsample confirmation is justified now."
    )

    for name in ["README.md", "final_recommendation.md"]:
        append_or_replace_section(BASE / name, "Frozen-Latent Classifier Sweep Pro", frozen_body)
        append_or_replace_section(BASE / name, "Decoder-Type FAST 3x3 Audit", decoder_body)

    rr = BASE / "reviewer_response_ready_text.md"
    text = rr.read_text(encoding="utf-8")
    failed_section = (
        "## Failed Optimization Summary\n\n"
        "The table below includes controlled negative/non-promoted checks. "
        "`ch1_only_offdiag_channelmean` is retained as a secondary simplified/"
        "channel-ablation model. The frozen-latent classifier sweep completed "
        "`75/75` model fits without VAE retraining; neither regularized LightGBM "
        "nor elastic-net provided a clean replacement for the locked readout. "
        "The decoder-type FAST audit found that `upsample_conv` underperformed "
        "the matched `convtranspose` baselines, so no FULL upsample run is justified.\n\n"
        + to_md_table(df)
        + "\n\n"
    )
    start = text.index("## Failed Optimization Summary")
    end = text.find("## Conditional Age/Sex Beta-VAE Exploratory Audit", start)
    if end == -1:
        end = len(text)
    text = text[:start] + failed_section + text[end:]
    reviewer_sections = (
        "## Reviewer Response: Frozen-Latent Classifier Sweep\n\n"
        "We also tested whether the final frozen VAE latents could be improved by "
        "changing only the Stage B classifier. This audit used the existing latent "
        "caches only; no VAE was retrained. All `75/75` model fits completed. "
        "Regularized LightGBM did not improve. Elastic-net increased PR-AUC on "
        "`[1,0,2]` (`0.5807` vs `0.5599`) but did not cleanly improve ROC-AUC "
        "(`0.7822` vs `0.7830`), balanced accuracy, or F1. Therefore, the "
        "frozen-latent classifier sweep was not promoted.\n\n"
        "## Reviewer Response: Decoder-Type FAST Audit\n\n"
        "We performed a default-safe FAST 3x3 decoder-type screen comparing the "
        "current transposed-convolution decoder with an upsample+conv decoder. "
        "For `[1]`, the current decoder achieved AUC `0.7826` and PR-AUC `0.5155`, "
        "whereas upsample+conv dropped to AUC `0.7370` and PR-AUC `0.4542`. For "
        "`[1,0,2]`, the current decoder achieved AUC `0.7342` and PR-AUC `0.4597`, "
        "whereas upsample+conv dropped to AUC `0.7010` and PR-AUC `0.4239`. "
        "Although the upsample+conv runs appeared slower / closer to the training "
        "horizon, the performance drop was large; no FULL upsample confirmation is "
        "justified now.\n\n"
    )
    insert_before = "## Reviewer Q2: Channel Selection and Pearson-Full Ablation"
    if insert_before in text:
        # Remove any previous copy of these sections before reinserting.
        for heading in [
            "## Reviewer Response: Frozen-Latent Classifier Sweep",
            "## Reviewer Response: Decoder-Type FAST Audit",
        ]:
            while heading in text:
                s = text.index(heading)
                e = text.find("\n## ", s + 1)
                if e == -1:
                    text = text[:s].rstrip() + "\n"
                else:
                    text = text[:s].rstrip() + "\n\n" + text[e + 1 :].lstrip()
        idx = text.index(insert_before)
        text = text[:idx] + reviewer_sections + text[idx:]
    else:
        text = text.rstrip() + "\n\n" + reviewer_sections
    rr.write_text(text, encoding="utf-8")


def update_command_log() -> None:
    path = BASE / "command_log.json"
    log = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    log["last_updated_utc"] = now
    log["latest_update_script"] = Path(__file__).name
    log["training_launched"] = False
    log["tensor_modified"] = False
    log["metadata_modified"] = False
    log["ledger_modified"] = False
    log["configs_modified"] = False
    action = "add_decoder_type_fast3x3_and_frozen_latent_classifier_sweep_negative_audits"
    updates = [
        entry for entry in log.setdefault("updates", []) if entry.get("action") != action
    ]
    updates.append(
        {
            "action": action,
            "updated_utc": now,
            "script": str(Path(__file__).resolve()),
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "configs_modified": False,
            "existing_model_outputs_modified": False,
            "records": {
                "frozen_latent_classifier_sweep": {
                    "model_fits_completed": "75/75",
                    "vae_retrained": False,
                    "lightgbm_very_regularized": "did_not_improve",
                    "best_main_alternative": {
                        "classifier": "logreg_elasticnet",
                        "auc": 0.782222,
                        "pr_auc": 0.580692,
                        "balanced_accuracy": 0.727500,
                        "f1": 0.556777,
                    },
                    "decision": "do_not_promote",
                },
                "decoder_type_fast3x3": {
                    "ch1_convtranspose_auc": 0.782604,
                    "ch1_upsample_auc": 0.737049,
                    "ch1_0_2_convtranspose_auc": 0.734167,
                    "ch1_0_2_upsample_auc": 0.701007,
                    "decision": "do_not_promote",
                },
            },
        }
    )
    log["updates"] = updates
    path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    df = upsert_failed_rows()
    write_optional_result_files()
    update_summary_docs(df)
    update_command_log()
    print(f"Updated manuscript defense package: {BASE}")


if __name__ == "__main__":
    main()
