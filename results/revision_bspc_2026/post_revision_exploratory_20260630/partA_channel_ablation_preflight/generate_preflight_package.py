#!/usr/bin/env python3
"""
Part A — FAST++ channel-ablation preflight generator.

Writes all preflight deliverables (configs, planned_runs.csv, launch commands,
resource estimate, channel mapping verification, preflight_status.md,
command_log.json) without launching any training.

Guardrails:
- Does NOT execute training.
- Does NOT modify any existing result, config, tensor, or metadata file.
- All new files are written only inside partA_channel_ablation_preflight/.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[4]
OUT_DIR    = Path(__file__).parent
CONFIG_DIR = OUT_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

PYTHON     = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAIN_SCRIPT = str(ROOT / "scripts/run_vae_clf_ad_inference.py")
TENSOR_PATH  = (
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
META_PATH    = str(
    ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight"
    / "patched_metadata_candidate.csv"
)
RESULTS_BASE = str(ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630"
                   / "partA_channel_ablation_preflight/runs")

T0 = datetime.now(timezone.utc).isoformat()

# ── channel master map ────────────────────────────────────────────────────────
CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}

# ── ablation runs spec ────────────────────────────────────────────────────────
# (tag, channels_to_use, description, optional)
RUNS = [
    ("ch1_pearsonFull",
     [1],
     "Single-channel ablation: Full Pearson Fisher-Z only. Isolates static functional "
     "connectivity baseline without graph-thresholding or MI-kNN.",
     False),

    ("ch1_0_pearsonFull_OMST",
     [1, 0],
     "Two-channel: Full Pearson + OMST-Pearson. Tests whether OMST graph-thresholding "
     "adds information beyond dense correlation.",
     False),

    ("ch1_0_2_pearsonFull_OMST_MI",
     [1, 0, 2],
     "Three-channel positive control — matches promoted model channel set. "
     "Expected outcome: AUC ≈ promoted (0.795). Any substantial deviation "
     "flags a hyperparameter or loss-function sensitivity.",
     False),

    ("ch4_0_1_dfcStd_OMST_pearsonFull",
     [4, 0, 1],
     "Dynamic FC substitution: replace MI-kNN (ch2) with dFC_StdDev (ch4). "
     "Tests whether temporal variability adds information beyond static FC.",
     False),

    ("ch3_0_1_dfcMean_OMST_pearsonFull",
     [3, 0, 1],
     "Dynamic FC substitution: replace MI-kNN (ch2) with dFC_AbsDiffMean (ch3). "
     "Captures mean absolute temporal change in connectivity.",
     False),

    ("ch5_0_1_distCorr_OMST_pearsonFull",
     [5, 0, 1],
     "Distance correlation substitution: replace MI-kNN (ch2) with DistanceCorr (ch5). "
     "Tests non-linear statistical dependence beyond MI-kNN.",
     False),

    ("ch3_4_0_1_dfcBoth_OMST_pearsonFull",
     [3, 4, 0, 1],
     "Optional 4-channel run: dFC_AbsDiffMean + dFC_StdDev + OMST + Full Pearson. "
     "Higher memory footprint — run only if GPU memory confirmed safe.",
     True),   # optional
]

# ── shared FAST++ parameters (locked, do not change) ─────────────────────────
FAST_PARAMS = {
    # schedule
    "epochs_vae": 800,
    "lr_scheduler_T0": 80,
    "cyclical_beta_n_cycles": 10,          # 10 × 80 = 800 epochs exactly
    "cyclical_beta_ratio_increase": 0.4,
    "early_stopping_patience_vae": 200,   # 25% of 800; ~2.5 cycles
    # architecture — matches promoted model
    "beta_vae": 3.75,
    "latent_dim": 384,
    "num_conv_layers_encoder": 4,
    "decoder_type": "convtranspose",
    "vae_block_order": "legacy_act_norm",
    "vae_dropout_scope": "legacy_all",
    "dropout_rate_vae": 0.15,
    "vae_final_activation": "tanh",
    "intermediate_fc_dim_vae": "quarter",
    "use_layernorm_vae_fc": False,
    # loss — required for cross-channel comparability
    "recon_loss_mode": "offdiag_channelmean_sum",
    # normalisation
    "norm_mode": "zscore_offdiag",
    # CV — FAST screening with 3 folds
    "outer_folds": 3,
    "inner_folds": 3,
    "repeated_outer_folds_n_repeats": 1,
    # classifier — LogReg L2 only
    "classifier_types": ["logreg"],
    "classifier_calibrate": False,         # skip calibration for screening speed
    "classifier_use_class_weight": True,
    "classifier_stratify_cols": ["Manufacturer"],
    "vae_stratify_cols": ["Manufacturer"],
    "latent_features_type": "mu",
    "gridsearch_scoring": "roc_auc",
    "n_iter_logreg": 100,
    # optimiser
    "lr_vae": 1e-4,
    "lr_scheduler_type": "cosine_warm",
    "lr_scheduler_eta_min": 5e-7,
    "lr_scheduler_patience_vae": 15,
    "weight_decay_vae": 5e-7,
    # training
    "batch_size": 64,
    "vae_val_split_ratio": 0.2,
    "vae_train_sampler_strategy": "none",
    "seed": 42,
    "num_workers": 4,
    "log_interval_epochs_vae": 10,
    # metadata
    "metadata_features": ["Age", "Sex"],
    # QC
    "save_fold_artefacts": True,
    "save_vae_training_history": True,
    "qc_analyze_distributions": True,
    "qc_check_scanner_leakage": True,
    "qc_rate_distortion": True,
    "qc_latent_information": True,
    "qc_mi_n_neighbors": 3,
    "qc_mi_top_k": 10,
    "qc_rd_log_base": 2.0,
    "qc_tc_ridge": 1e-6,
    "qc_var_eps_active": 1e-4,
    "use_optuna_pruner": False,
    "use_smote": False,
    "tune_sampler_params": False,
    "mlp_classifier_hidden_layers": "64,16",
    "n_iter_svm": 1,
}

# ── gate criteria for promotion to FULL 5×5 ──────────────────────────────────
GATE_CRITERIA = {
    "primary": "OOF ROC-AUC (logreg_l2, 3-fold) ≥ 0.760 (threshold = promoted baseline − 0.035)",
    "secondary": "OOF PR-AUC ≥ 0.500",
    "stability": "Fold-wise AUC std ≤ 0.07",
    "leakage": "Manufacturer scanner-leakage balanced-accuracy ≤ 0.45 (chance=0.33)",
    "active_units": "Active latent units (var > 1e-4) ≥ 50 across all folds",
    "positive_control": (
        "ch1_0_2 control run AUC must be within 0.03 of promoted baseline "
        "(0.795 ± 0.03). If control fails, other results are unreliable."
    ),
    "note": (
        "Gate criteria are necessary but not sufficient for promotion. "
        "FULL 5×5 promotion requires additional sign-off and a replication run. "
        "offdiag_channelmean_sum loss is mandatory for cross-channel comparisons."
    ),
}

# ── GPU memory estimate ────────────────────────────────────────────────────────
# RTX 3060 12 GB, latent_dim=384, batch=64
def mem_estimate_gb(n_channels: int) -> dict:
    roi = 131
    elem_bytes = 4
    batch = 64
    # input tensor: (batch, C, roi, roi)
    input_mb  = batch * n_channels * roi * roi * elem_bytes / 1e6
    # model params rough: ~6M base + 1.5M per extra channel
    params_mb = (6_000_000 + 1_500_000 * (n_channels - 1)) * elem_bytes / 1e6
    # activations (rough: 4× input per layer, 4 layers)
    act_mb = input_mb * 16
    total_mb = input_mb + params_mb + act_mb
    safe = total_mb < 9000  # leave 3 GB headroom
    return {
        "n_channels": n_channels,
        "input_mb": round(input_mb, 1),
        "params_mb": round(params_mb, 1),
        "activations_mb": round(act_mb, 1),
        "total_est_mb": round(total_mb, 1),
        "gpu_safe_12gb": safe,
    }

# ── epoch timing estimate ──────────────────────────────────────────────────────
def time_estimate(n_channels: int, n_folds: int = 3, epochs: int = 800) -> dict:
    # Empirical: ~3-4 min/fold for 3 channels at 800 epochs on RTX 3060 with N=397-647
    # Scale roughly linearly with channels
    min_per_fold = 3.5 * (n_channels / 3) * (epochs / 800)
    total_min = min_per_fold * n_folds
    return {
        "est_min_per_fold": round(min_per_fold, 1),
        "est_total_min": round(total_min, 1),
        "est_total_hours": round(total_min / 60, 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Write per-run configs and accumulate plan rows
# ══════════════════════════════════════════════════════════════════════════════
planned_rows = []
cmd_entries = []

print(f"Writing {len(RUNS)} channel-ablation configs …")
for tag, channels, desc, optional in RUNS:
    run_name = f"ablation_{tag}_fast3fold_800ep"
    out_dir  = str(Path(RESULTS_BASE) / run_name)
    selected_names = [CHANNEL_NAMES[c] for c in channels]

    cfg = {
        "run_name": run_name,
        "description": desc,
        "optional": optional,
        "python_executable": PYTHON,
        "channel_names_master_in_tensor_order": list(CHANNEL_NAMES.values()),
        "selected_channel_names": selected_names,
        "ablation_note": (
            "FAST 3-fold screening run. "
            "recon_loss_mode=offdiag_channelmean_sum is MANDATORY for cross-channel "
            "comparability (normalises reconstruction loss by number of channels). "
            "Do not compare to runs using mse_sum_batchmean_current."
        ),
        "gate_criteria": GATE_CRITERIA,
        "paths": {
            "training_script": TRAIN_SCRIPT,
            "global_tensor_path": TENSOR_PATH,
            "metadata_path": META_PATH,
            "output_dir": out_dir,
            "big_disk_output_dir": out_dir,  # same machine
        },
        "parameters": {
            "channels_to_use": channels,
            **FAST_PARAMS,
        },
    }

    cfg_path = CONFIG_DIR / f"{run_name}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # ── CLI command string ─────────────────────────────────────────────────────
    ch_str = " ".join(str(c) for c in channels)
    meta_feat_str = " ".join(FAST_PARAMS["metadata_features"])
    clf_types_str = " ".join(FAST_PARAMS["classifier_types"])
    clf_strat_str = " ".join(FAST_PARAMS["classifier_stratify_cols"])
    vae_strat_str = " ".join(FAST_PARAMS["vae_stratify_cols"])

    cmd = (
        f"{PYTHON} {TRAIN_SCRIPT} "
        f"--global_tensor_path {TENSOR_PATH} "
        f"--metadata_path {META_PATH} "
        f"--output_dir {out_dir} "
        f"--channels_to_use {ch_str} "
        f"--classifier_types {clf_types_str} "
        f"--classifier_stratify_cols {clf_strat_str} "
        f"--vae_stratify_cols {vae_strat_str} "
        f"{'--classifier_calibrate ' if FAST_PARAMS['classifier_calibrate'] else ''}"
        f"--classifier_use_class_weight "
        f"--latent_features_type {FAST_PARAMS['latent_features_type']} "
        f"--gridsearch_scoring {FAST_PARAMS['gridsearch_scoring']} "
        f"--outer_folds {FAST_PARAMS['outer_folds']} "
        f"--inner_folds {FAST_PARAMS['inner_folds']} "
        f"--repeated_outer_folds_n_repeats {FAST_PARAMS['repeated_outer_folds_n_repeats']} "
        f"--num_conv_layers_encoder {FAST_PARAMS['num_conv_layers_encoder']} "
        f"--decoder_type {FAST_PARAMS['decoder_type']} "
        f"--epochs_vae {FAST_PARAMS['epochs_vae']} "
        f"--vae_val_split_ratio {FAST_PARAMS['vae_val_split_ratio']} "
        f"--early_stopping_patience_vae {FAST_PARAMS['early_stopping_patience_vae']} "
        f"--cyclical_beta_n_cycles {FAST_PARAMS['cyclical_beta_n_cycles']} "
        f"--cyclical_beta_ratio_increase {FAST_PARAMS['cyclical_beta_ratio_increase']} "
        f"--beta_vae {FAST_PARAMS['beta_vae']} "
        f"--dropout_rate_vae {FAST_PARAMS['dropout_rate_vae']} "
        f"--vae_dropout_scope {FAST_PARAMS['vae_dropout_scope']} "
        f"--vae_block_order {FAST_PARAMS['vae_block_order']} "
        f"--latent_dim {FAST_PARAMS['latent_dim']} "
        f"--batch_size {FAST_PARAMS['batch_size']} "
        f"--lr_vae {FAST_PARAMS['lr_vae']} "
        f"--lr_scheduler_type {FAST_PARAMS['lr_scheduler_type']} "
        f"--lr_scheduler_T0 {FAST_PARAMS['lr_scheduler_T0']} "
        f"--lr_scheduler_eta_min {FAST_PARAMS['lr_scheduler_eta_min']} "
        f"--lr_scheduler_patience_vae {FAST_PARAMS['lr_scheduler_patience_vae']} "
        f"--weight_decay_vae {FAST_PARAMS['weight_decay_vae']} "
        f"--vae_final_activation {FAST_PARAMS['vae_final_activation']} "
        f"--intermediate_fc_dim_vae {FAST_PARAMS['intermediate_fc_dim_vae']} "
        f"--n_jobs_gridsearch 8 "
        f"--metadata_features {meta_feat_str} "
        f"--norm_mode {FAST_PARAMS['norm_mode']} "
        f"--recon_loss_mode {FAST_PARAMS['recon_loss_mode']} "
        f"--seed {FAST_PARAMS['seed']} "
        f"--num_workers {FAST_PARAMS['num_workers']} "
        f"--log_interval_epochs_vae {FAST_PARAMS['log_interval_epochs_vae']} "
        f"--save_fold_artefacts --save_vae_training_history "
        f"--qc_analyze_distributions --qc_check_scanner_leakage "
        f"--qc_rate_distortion --qc_latent_information "
        f"--qc_mi_n_neighbors {FAST_PARAMS['qc_mi_n_neighbors']} "
        f"--qc_mi_top_k {FAST_PARAMS['qc_mi_top_k']} "
        f"--qc_rd_log_base {FAST_PARAMS['qc_rd_log_base']} "
        f"--qc_tc_ridge {FAST_PARAMS['qc_tc_ridge']} "
        f"--qc_var_eps_active {FAST_PARAMS['qc_var_eps_active']} "
        f"--vae_train_sampler_strategy {FAST_PARAMS['vae_train_sampler_strategy']} "
        f"--mlp_classifier_hidden_layers {FAST_PARAMS['mlp_classifier_hidden_layers']} "
        f"--n_iter_logreg {FAST_PARAMS['n_iter_logreg']} "
        f"--vae_required_metadata_cols ResearchGroup_Mapped Manufacturer Age Sex "
        f"--vae_abort_if_val_split_fails"
    ).rstrip()

    mem = mem_estimate_gb(len(channels))
    tim = time_estimate(len(channels))

    planned_rows.append({
        "run_id": len(planned_rows) + 1,
        "run_name": run_name,
        "channels": str(channels),
        "selected_channel_names": "; ".join(selected_names),
        "optional": optional,
        "epochs_vae": FAST_PARAMS["epochs_vae"],
        "n_cycles": FAST_PARAMS["cyclical_beta_n_cycles"],
        "outer_folds": FAST_PARAMS["outer_folds"],
        "beta_vae": FAST_PARAMS["beta_vae"],
        "latent_dim": FAST_PARAMS["latent_dim"],
        "recon_loss_mode": FAST_PARAMS["recon_loss_mode"],
        "est_gpu_mb": mem["total_est_mb"],
        "gpu_safe_12gb": mem["gpu_safe_12gb"],
        "est_time_h": tim["est_total_hours"],
        "config_path": str(cfg_path),
        "output_dir": out_dir,
    })

    cmd_entries.append({
        "run_name": run_name,
        "optional": optional,
        "channels": channels,
        "command": cmd,
    })

    status = "OPTIONAL" if optional else "PLANNED"
    print(f"  [{status}] {run_name}")
    print(f"           channels={channels}, est_gpu={mem['total_est_mb']:.0f} MB, "
          f"est_time={tim['est_total_hours']:.1f} h")

# ── Write planned_runs.csv ─────────────────────────────────────────────────────
import csv
fieldnames = list(planned_rows[0].keys())
with open(OUT_DIR / "planned_runs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(planned_rows)
print(f"\nWrote planned_runs.csv ({len(planned_rows)} runs)")

# ── Write channel_mapping_verification.csv ────────────────────────────────────
mapping_rows = []
for idx, name in CHANNEL_NAMES.items():
    used_in = [r["run_name"] for r in planned_rows if idx in eval(r["channels"])]
    mapping_rows.append({
        "tensor_channel_index": idx,
        "channel_name": name,
        "type": (
            "static_FC" if idx in [0, 1, 2, 5] else
            "dynamic_FC" if idx in [3, 4] else
            "effective_connectivity"
        ),
        "in_promoted_model": idx in [1, 0, 2],
        "runs_using_this_channel": "; ".join(used_in) if used_in else "none",
        "note": (
            "ch0 is OMST-thresholded Pearson; ch1 is dense Fisher-Z Pearson; "
            "ordering in channels_to_use determines VAE input channel order"
            if idx in [0, 1] else ""
        ),
    })
with open(OUT_DIR / "channel_mapping_verification.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
    w.writeheader()
    w.writerows(mapping_rows)
print("Wrote channel_mapping_verification.csv")

# ── Write launch_commands.sh ───────────────────────────────────────────────────
sh_lines = [
    "#!/usr/bin/env bash",
    "# FAST++ channel-ablation screening launch commands",
    "# Generated: " + T0[:19],
    "# DO NOT EXECUTE without review. These are planned commands only.",
    "# Confirm GPU memory, disk space, and gate criteria before launching.",
    "# Run ONE at a time; wait for completion before starting the next.",
    "",
    f"# Total planned runs: {sum(1 for r in planned_rows if not r['optional'])} required + "
    f"{sum(1 for r in planned_rows if r['optional'])} optional",
    "",
]
for entry in cmd_entries:
    lbl = "OPTIONAL" if entry["optional"] else "REQUIRED"
    sh_lines += [
        f"# ── [{lbl}] {entry['run_name']} ──",
        f"# channels: {entry['channels']}",
        f"# CONFIG: configs/{entry['run_name']}.json",
        entry["command"],
        "",
    ]
(OUT_DIR / "launch_commands.sh").write_text("\n".join(sh_lines), encoding="utf-8")
(OUT_DIR / "launch_commands.sh").chmod(0o755)
print("Wrote launch_commands.sh")

# ── Write resource_estimate.md ─────────────────────────────────────────────────
total_h_req  = sum(r["est_time_h"] for r in planned_rows if not r["optional"])
total_h_opt  = sum(r["est_time_h"] for r in planned_rows if r["optional"])
max_gpu      = max(r["est_gpu_mb"] for r in planned_rows)

resource_md = f"""# Resource Estimate — FAST++ Channel Ablation

**Generated:** {T0[:19]}
**GPU:** NVIDIA GeForce RTX 3060, 12 GB VRAM
**RAM:** 31 GB
**Disk:** Required output dir: {RESULTS_BASE}

## Per-run estimates

| run_name | channels | est_gpu_MB | est_time_h | optional |
|---|---|---:|---:|---|
{chr(10).join(f"| {r['run_name']} | {r['channels']} | {r['est_gpu_mb']:.0f} | {r['est_time_h']:.1f} | {'YES' if r['optional'] else 'no'} |" for r in planned_rows)}

## Summary

- **Required runs:** {sum(1 for r in planned_rows if not r['optional'])} × ~{total_h_req/sum(1 for r in planned_rows if not r['optional']):.1f} h each = **{total_h_req:.1f} h total**
- **Optional run (4-channel):** +{total_h_opt:.1f} h
- **Peak GPU estimate:** {max_gpu:.0f} MB — well within 12 GB (>3 GB headroom)
- **Disk per run:** ~500 MB–1.5 GB (fold artifacts + training histories)

## Notes

- Timing estimates are approximate (empirical: ~3.5 min/fold/800ep for 3 channels).
- Runs should be executed sequentially (one at a time) to avoid GPU contention.
- 4-channel optional run: estimated GPU ~{mem_estimate_gb(4)['total_est_mb']:.0f} MB — safe on 12 GB.
- Early stopping (patience=200) may terminate runs before 800 epochs; timing
  estimates are for full 800 epochs.
- If VAE val-split fails for any fold (small class count), the run aborts
  safely with --vae_abort_if_val_split_fails.
"""
(OUT_DIR / "resource_estimate.md").write_text(resource_md, encoding="utf-8")
print("Wrote resource_estimate.md")

# ── Write preflight_status.md ─────────────────────────────────────────────────
# Check actual file existence
tensor_ok  = Path(TENSOR_PATH).exists()
meta_ok    = Path(META_PATH).exists()
script_ok  = Path(TRAIN_SCRIPT).exists()
python_ok  = Path(PYTHON).exists()

preflight_md = f"""# Preflight Status — FAST++ Channel Ablation

**Generated:** {T0[:19]}
**Status:** {'READY' if all([tensor_ok, meta_ok, script_ok, python_ok]) else 'BLOCKED — see checks below'}

---

## File checks

| Item | Path | Status |
|---|---|---|
| Global tensor NPZ | {TENSOR_PATH} | {'✓ FOUND' if tensor_ok else '✗ MISSING'} |
| Metadata CSV | {META_PATH} | {'✓ FOUND' if meta_ok else '✗ MISSING'} |
| Training script | {TRAIN_SCRIPT} | {'✓ FOUND' if script_ok else '✗ MISSING'} |
| Python executable | {PYTHON} | {'✓ FOUND' if python_ok else '✗ MISSING'} |
| Output base dir | {RESULTS_BASE} | will be created at launch |

---

## Configuration check

| Parameter | Value | Rationale |
|---|---|---|
| epochs_vae | {FAST_PARAMS['epochs_vae']} | FAST screen; 10 full cosine cycles |
| lr_scheduler_T0 | {FAST_PARAMS['lr_scheduler_T0']} | Matches promoted model cycle length |
| cyclical_beta_n_cycles | {FAST_PARAMS['cyclical_beta_n_cycles']} | 10 × 80 = 800 epochs exactly |
| early_stopping_patience | {FAST_PARAMS['early_stopping_patience_vae']} | 25% of 800; ~2.5 cycles |
| beta_vae | {FAST_PARAMS['beta_vae']} | Matches promoted model |
| latent_dim | {FAST_PARAMS['latent_dim']} | Matches promoted model |
| recon_loss_mode | {FAST_PARAMS['recon_loss_mode']} | MANDATORY for cross-channel fairness |
| outer_folds | {FAST_PARAMS['outer_folds']} | FAST: 3 folds (vs 5 for FULL) |
| inner_folds | {FAST_PARAMS['inner_folds']} | 3-fold inner CV |
| classifier | logreg L2 only | Fastest screening readout |
| n_iter_logreg | {FAST_PARAMS['n_iter_logreg']} | Coarse but sufficient HP search |
| classifier_calibrate | {FAST_PARAMS['classifier_calibrate']} | Skip for screening speed |

---

## Loss mode warning

The promoted model was trained with `recon_loss_mode=mse_sum_batchmean_current`.
This ablation uses `offdiag_channelmean_sum`, which normalises reconstruction
loss by the number of off-diagonal elements AND the number of channels.
**This makes AUC comparable across different channel counts but means the
ablation AUC cannot be directly compared to the promoted model's Stage A
AUC.** Compare only within this ablation series.

---

## Planned runs ({len(planned_rows)} total; {sum(1 for r in planned_rows if r['optional'])} optional)

{chr(10).join(f"- `{r['run_name']}` — channels {r['channels']} — {r['est_time_h']:.1f} h {'(OPTIONAL)' if r['optional'] else ''}" for r in planned_rows)}

---

## Gate criteria for promotion to FULL 5×5

| Criterion | Threshold |
|---|---|
| Primary: OOF ROC-AUC | ≥ 0.760 (promoted baseline 0.795 − 0.035) |
| Secondary: OOF PR-AUC | ≥ 0.500 |
| Stability: fold AUC std | ≤ 0.070 |
| Manufacturer leakage BA | ≤ 0.450 (chance ≈ 0.333) |
| Active latent units | ≥ 50 across all folds |
| Positive control (ch1_0_2) | AUC within 0.03 of promoted baseline |

**Note:** Any candidate that passes gate criteria advances to FULL 5×5 training
with `recon_loss_mode=offdiag_channelmean_sum`, `outer_folds=5`, `inner_folds=5`,
`epochs_vae=10000`, `cyclical_beta_n_cycles=125`, and full Optuna HP search
(`n_iter_logreg=300+`).

---

## Pre-launch checklist

- [ ] GPU memory confirmed available (target: ≤ 9 GB / 12 GB)
- [ ] Disk space confirmed (> 10 GB free for all 6 required runs)
- [ ] Positive control (ch1_0_2) run FIRST — abort if control fails gate
- [ ] Stage B classifier-only readout script ready after each training run
- [ ] No concurrent GPU jobs running
"""

(OUT_DIR / "preflight_status.md").write_text(preflight_md, encoding="utf-8")
print("Wrote preflight_status.md")

# ── Write command_log.json ─────────────────────────────────────────────────────
T1 = datetime.now(timezone.utc).isoformat()
cmd_log = {
    "script": str(Path(__file__).resolve()),
    "generated_utc": T0,
    "finished_utc": T1,
    "mode": "preflight_generation_only",
    "training_NOT_launched": True,
    "n_runs_planned": len(planned_rows),
    "n_runs_required": sum(1 for r in planned_rows if not r["optional"]),
    "n_runs_optional": sum(1 for r in planned_rows if r["optional"]),
    "files_written": [
        str(CONFIG_DIR / f"{r['run_name']}.json") for r in planned_rows
    ] + [
        str(OUT_DIR / "planned_runs.csv"),
        str(OUT_DIR / "channel_mapping_verification.csv"),
        str(OUT_DIR / "launch_commands.sh"),
        str(OUT_DIR / "resource_estimate.md"),
        str(OUT_DIR / "preflight_status.md"),
    ],
    "launch_commands": [
        {"run_name": e["run_name"], "optional": e["optional"], "command": e["command"]}
        for e in cmd_entries
    ],
    "guardrails": {
        "training_not_executed": True,
        "no_config_overwrite": True,
        "no_tensor_edit": True,
        "no_metadata_edit": True,
        "no_manuscript_edit": True,
    }
}
(OUT_DIR / "command_log.json").write_text(json.dumps(cmd_log, indent=2), encoding="utf-8")
print("Wrote command_log.json")
print(f"\n✓ Part A preflight package complete. Output: {OUT_DIR}")
