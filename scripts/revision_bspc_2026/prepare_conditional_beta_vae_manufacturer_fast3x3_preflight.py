#!/usr/bin/env python3
"""Prepare Manufacturer-conditioned beta-VAE FAST 3x3 preflight.

This script writes an experiment matrix and command dry-run report only. It
does not launch training. Real training is intentionally refused.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_manufacturer_fast3x3_preflight"
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
TENSOR = "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
METADATA = "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Explicit no-training mode. This is the only supported behavior.")
    parser.add_argument("--confirm-training", action="store_true", help="This preflight refuses real training; included as a safety check.")
    return parser.parse_args()


def planned_rows() -> List[Dict[str, Any]]:
    channel_sets = [
        ("ch1", [1], "Pearson_Full_FisherZ_Signed"),
        ("ch1_0_2", [1, 0, 2], "Pearson_Full_FisherZ_Signed | Pearson_OMST_GCE_Signed_Weighted | MI_KNN_Symmetric"),
    ]
    conditions = [
        ("baseline_unconditioned", "none", "none"),
        ("decoder_only_manufacturer", "decoder_only", "manufacturer"),
    ]
    rows: List[Dict[str, Any]] = []
    for channel_key, channels, channel_names in channel_sets:
        for condition_name, mode, vars_ in conditions:
            rows.append(
                {
                    "candidate_id": f"{channel_key}_{condition_name}",
                    "channels": channels,
                    "channel_names": channel_names,
                    "vae_conditioning_mode": mode,
                    "vae_conditioning_vars": vars_,
                    "readout_feature_set": "z_plus_age_sex",
                    "manufacturer_passed_to_classifier": False,
                }
            )
    return rows


def stage_a_command(row: Dict[str, Any], run_dir: Path) -> List[str]:
    return [
        PYTHON,
        str(TRAIN_SCRIPT),
        "--global_tensor_path", TENSOR,
        "--metadata_path", METADATA,
        "--output_dir", str(run_dir),
        "--channels_to_use", *[str(x) for x in row["channels"]],
        "--classifier_types", "logreg",
        "--classifier_stratify_cols", "Manufacturer",
        "--vae_stratify_cols", "Manufacturer",
        "--classifier_calibrate",
        "--classifier_use_class_weight",
        "--latent_features_type", "mu",
        "--gridsearch_scoring", "roc_auc",
        "--outer_folds", "3",
        "--inner_folds", "3",
        "--repeated_outer_folds_n_repeats", "1",
        "--num_conv_layers_encoder", "4",
        "--decoder_type", "convtranspose",
        "--epochs_vae", "960",
        "--vae_val_split_ratio", "0.2",
        "--early_stopping_patience_vae", "240",
        "--cyclical_beta_n_cycles", "12",
        "--cyclical_beta_ratio_increase", "0.4",
        "--beta_vae", "2.5",
        "--dropout_rate_vae", "0.15",
        "--vae_dropout_scope", "legacy_all",
        "--vae_block_order", "legacy_act_norm",
        "--latent_dim", "256",
        "--batch_size", "64",
        "--lr_vae", "0.0001",
        "--lr_scheduler_type", "cosine_warm",
        "--lr_scheduler_T0", "80",
        "--weight_decay_vae", "5e-7",
        "--vae_final_activation", "tanh",
        "--intermediate_fc_dim_vae", "quarter",
        "--metadata_features", "Age", "Sex",
        "--norm_mode", "zscore_offdiag",
        "--recon_loss_mode", "offdiag_channelmean_sum",
        "--vae_conditioning_mode", row["vae_conditioning_mode"],
        "--vae_conditioning_vars", row["vae_conditioning_vars"],
        "--vae_latent_covariate_corr_lambda", "0.0",
        "--seed", "42",
        "--num_workers", "4",
        "--save_fold_artefacts",
        "--save_vae_training_history",
        "--qc_analyze_distributions",
        "--qc_check_scanner_leakage",
        "--qc_rate_distortion",
        "--qc_latent_information",
        "--qc_nuisance_cols", "Manufacturer", "SiteCode", "Sex", "Age_Group",
        "--n_iter_logreg", "1",
        "--dry-run",
    ]


def stage_b_command(row: Dict[str, Any], run_dir: Path) -> List[str]:
    readout_dir = run_dir / "classifier_only_readout_z_plus_age_sex"
    return [
        PYTHON,
        str(STAGE_B_SCRIPT),
        "--run-dir", str(run_dir),
        "--output-dir", str(readout_dir),
        "--models", "logreg_l2",
        "--readout-feature-sets", "z_plus_age_sex",
        "--outer-folds", "3",
        "--inner-folds", "3",
        "--reuse-latent-cache",
    ]


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.confirm_training:
        raise SystemExit("This preflight script never launches real training. Remove --confirm-training.")
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    configs_dir = outdir / "configs"
    configs_dir.mkdir(exist_ok=True)

    rows = planned_rows()
    matrix_rows = []
    dry_lines = [
        "# Manufacturer-Conditioned beta-VAE FAST 3x3 Dry-Run Report",
        "",
        "No real training was launched. Stage A commands include `--dry-run`; Stage B commands are command previews only.",
        "",
    ]
    for row in rows:
        run_dir = outdir / "runs" / row["candidate_id"]
        stage_a = stage_a_command(row, run_dir)
        stage_b = stage_b_command(row, run_dir)
        cfg_payload = {
            **row,
            "dataset": "v5.1b no_pybandpass",
            "outer_folds": 3,
            "inner_folds": 3,
            "epochs_vae": 960,
            "cyclical_beta_n_cycles": 12,
            "cycle_len": 80,
            "lr_scheduler_T0": 80,
            "beta_vae": 2.5,
            "recon_loss_mode": "offdiag_channelmean_sum",
            "vae_final_activation": "tanh",
            "decoder_type": "convtranspose",
            "stage_a_command": stage_a,
            "stage_b_command": stage_b,
        }
        (configs_dir / f"{row['candidate_id']}.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")
        matrix_rows.append({k: v for k, v in cfg_payload.items() if not k.endswith("_command")})
        dry_lines.extend(
            [
                f"## {row['candidate_id']}",
                "",
                "Stage A:",
                "",
                "```bash",
                shlex.join(stage_a),
                "```",
                "",
                "Stage B:",
                "",
                "```bash",
                shlex.join(stage_b),
                "```",
                "",
            ]
        )

    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(outdir / "experiment_matrix.csv", index=False)
    write_markdown_table(outdir / "experiment_matrix.md", matrix)
    (outdir / "dry_run_report.md").write_text("\n".join(dry_lines), encoding="utf-8")
    (outdir / "README.md").write_text(
        "# Manufacturer-Conditioned beta-VAE FAST 3x3 Preflight\n\n"
        "This package prepares a default-safe FAST 3x3 screen for optional decoder-only "
        "Manufacturer conditioning. It is a preflight only; no real training was launched.\n\n"
        "## Design\n\n"
        "- Dataset: v5.1b final no-pybandpass branch.\n"
        "- Channel sets: `[1]` and `[1,0,2]`.\n"
        "- Candidates per channel set: unconditioned baseline and `decoder_only Manufacturer`.\n"
        "- Encoder remains imaging-only: `z = Encoder(x)`.\n"
        "- Decoder receives `[z, manufacturer_onehot]` only when conditioning is explicitly enabled.\n"
        "- Manufacturer is not passed to the classifier; Stage B remains `z_plus_age_sex`.\n"
        "- Default `vae_conditioning_mode=none` and `vae_conditioning_vars=none` preserves existing behavior.\n\n"
        "## Manufacturer Preprocessing\n\n"
        "- Manufacturer categories are fit on VAE actual train rows only.\n"
        "- Known variants are normalized to `GE`, `PHILIPS`, and `SIEMENS`.\n"
        "- Missing or unseen val/test manufacturers fail clearly because no unknown category is enabled.\n"
        "- Fold-level QC is saved by the training pipeline when conditioning is enabled.\n\n"
        "## Required QC After Real Training\n\n"
        "- AD/CN ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, and F1.\n"
        "- Manufacturer predictability from latent `z`.\n"
        "- Site/manufacturer leakage raw versus latent.\n"
        "- Rate-distortion, best epoch, beta*KLD/R, active units, total correlation.\n"
        "- Manufacturer subgroup performance.\n",
        encoding="utf-8",
    )
    (outdir / "architecture_audit.md").write_text(
        "# Architecture Audit\n\n"
        "The implemented design is decoder-only conditioning. The encoder remains imaging-only and therefore cannot directly use Manufacturer. "
        "For conditioned runs, the decoder input is `concat(z, manufacturer_onehot)` before the decoder FC stack. "
        "No conditional prior and no convolutional conditioning are introduced.\n\n"
        "Default behavior remains unchanged: when `vae_conditioning_mode=none`, `conditioning_dim=0`, the decoder receives only `z`, and the state-dict keys match the historical model.\n\n"
        "This preflight intentionally does not pass Manufacturer to Stage B classification. Stage B uses latent `mu + Age + Sex` only, preserving the existing readout contract while testing whether decoder conditioning removes nuisance information from `z`.\n",
        encoding="utf-8",
    )
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "output_dir": str(outdir),
        "n_planned_candidates": len(rows),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
