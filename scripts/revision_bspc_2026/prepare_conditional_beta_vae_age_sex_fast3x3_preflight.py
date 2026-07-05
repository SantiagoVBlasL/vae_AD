#!/usr/bin/env python3
"""Prepare conditional beta-VAE Age/Sex FAST 3x3 preflight.

This script writes an experiment matrix and command dry-run report only. It
does not launch training. Real training is intentionally refused unless a future
launcher is written explicitly.
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
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_age_sex_fast3x3_preflight"
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
TENSOR = "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
METADATA = "/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Explicit no-training mode. This is the only supported behavior for this preflight.")
    parser.add_argument("--confirm-training", action="store_true", help="This preflight still refuses real training; included as an explicit safety check.")
    return parser.parse_args()


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
        "--vae_latent_covariate_corr_lambda", str(row["corr_lambda"]),
        "--seed", "42",
        "--num_workers", "4",
        "--save_fold_artefacts",
        "--save_vae_training_history",
        "--qc_analyze_distributions",
        "--qc_check_scanner_leakage",
        "--qc_rate_distortion",
        "--qc_latent_information",
        "--n_iter_logreg", "1",
        "--dry-run",
    ]


def stage_b_command(row: Dict[str, Any], run_dir: Path) -> List[str]:
    readout_dir = run_dir / f"classifier_only_readout_{row['readout_feature_set']}"
    return [
        PYTHON,
        str(STAGE_B_SCRIPT),
        "--run-dir", str(run_dir),
        "--output-dir", str(readout_dir),
        "--models", "logreg_l2",
        "--readout-feature-sets", row["readout_feature_set"],
        "--outer-folds", "3",
        "--inner-folds", "3",
        "--reuse-latent-cache",
    ]


def planned_rows() -> List[Dict[str, Any]]:
    conditions = [
        ("A", "baseline_current_z_plus_age_sex", "none", "none", 0.0, "z_plus_age_sex"),
        ("B", "baseline_current_z_only", "none", "none", 0.0, "z_only"),
        ("C", "decoder_only_sex_lambda0_z_only", "decoder_only", "sex", 0.0, "z_only"),
        ("D", "decoder_only_sex_lambda001_z_only", "decoder_only", "sex", 0.01, "z_only"),
        ("E", "decoder_only_age_sex_lambda0_z_only", "decoder_only", "age_sex", 0.0, "z_only"),
        ("F", "decoder_only_age_sex_lambda001_z_only", "decoder_only", "age_sex", 0.01, "z_only"),
        ("G", "decoder_only_age_sex_lambda001_z_plus_age_sex", "decoder_only", "age_sex", 0.01, "z_plus_age_sex"),
    ]
    channel_sets = [
        ("ch1", [1], "Pearson_Full_FisherZ_Signed"),
        ("ch1_0_2", [1, 0, 2], "Pearson_Full_FisherZ_Signed | Pearson_OMST_GCE_Signed_Weighted | MI_KNN_Symmetric"),
    ]
    rows = []
    for channel_key, channels, names in channel_sets:
        for condition_id, name, mode, vars_, lam, readout in conditions:
            rows.append({
                "candidate_id": f"{channel_key}_{name}",
                "condition_id": condition_id,
                "channels": channels,
                "channel_names": names,
                "vae_conditioning_mode": mode,
                "vae_conditioning_vars": vars_,
                "corr_lambda": lam,
                "readout_feature_set": readout,
            })
    return rows


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
        "# Conditional beta-VAE Age/Sex FAST 3x3 Dry-Run Report",
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
            "stage_a_command": stage_a,
            "stage_b_command": stage_b,
        }
        (configs_dir / f"{row['candidate_id']}.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")
        matrix_rows.append({k: v for k, v in cfg_payload.items() if not k.endswith("_command")})
        dry_lines.extend([
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
        ])
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(outdir / "experiment_matrix.csv", index=False)
    write_markdown_table(outdir / "experiment_matrix.md", matrix)
    (outdir / "dry_run_report.md").write_text("\n".join(dry_lines), encoding="utf-8")
    (outdir / "README.md").write_text(
        "# Conditional beta-VAE Age/Sex FAST 3x3 Preflight\n\n"
        "This package prepares a default-safe FAST 3x3 preflight for optional decoder-only Age/Sex conditioning.\n\n"
        "- Default `vae_conditioning_mode=none`, `vae_conditioning_vars=none`, `lambda=0` preserves current behavior.\n"
        "- Encoder remains imaging-only.\n"
        "- Decoder receives `[z,c]` only when `decoder_only` is explicitly enabled.\n"
        "- Age/Sex transforms are fold-local and fit on VAE training rows only.\n"
        "- Stage A commands are dummy-canonical-logreg dry-runs and must not be used for ranking.\n"
        "- Stage B command previews use classifier-only `logreg_l2` with true inner-CV OOF thresholding.\n"
        "- No real training was launched.\n\n"
        "## FAST Matrix\n\n"
        "- Channel sets: `[1]`, `[1,0,2]`.\n"
        "- Conditions: A-G, covering baseline z+Age/Sex, baseline z-only, decoder-only Sex, decoder-only Age+Sex, and optional correlation penalty.\n"
        "- Readouts: `z_only` and `z_plus_age_sex` where pre-registered.\n\n"
        "## QC Required After Real Training\n\n"
        "- AD/CN ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, and F1.\n"
        "- MI(z;Age), MI(z;Sex), and fold-safe prediction of Age/Sex from latent `z`.\n"
        "- Scanner/manufacturer leakage from raw input versus latent representation.\n"
        "- Active units, total correlation, KLD/R, and beta*KLD/R.\n"
        "- Foldwise metrics and subgroup diagnostics.\n",
        encoding="utf-8",
    )
    (outdir / "architecture_audit.md").write_text(
        "# Architecture Audit\n\n"
        "Implemented conditional mode is minimal decoder conditioning: `z = Encoder(x)`, then `Decoder([z,c])`.\n"
        "The encoder, convolutional stacks, latent heads, dropout scope, block order, final activation, loss mode, and default constructor behavior are unchanged when conditioning is `none`.\n"
        "Conditional prior and convolutional conditioning are intentionally not implemented.\n\n"
        "## Covariate Handling\n\n"
        "- Age is standardized from VAE training-fold rows only.\n"
        "- Sex is encoded fold-safely from VAE training-fold categories.\n"
        "- Missing or unsupported Age/Sex values fail clearly instead of imputing from outer-test data.\n"
        "- Encoder remains imaging-only, so latent extraction for conditional models requires the saved fold-local conditioning transformer.\n\n"
        "## Loss\n\n"
        "The historical VAE loss remains unchanged unless `vae_latent_covariate_corr_lambda > 0`.\n"
        "When enabled, the added term is the mean squared correlation between latent `mu` dimensions and conditioning variables inside the training batch.\n",
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
