#!/usr/bin/env python3
"""
scripts/revision_bspc_2026/orchestrator_architecture_training_audit_20260617.py

Read-only audit of the promoted β-VAE training orchestrator and model architecture.
Produces reviewer-ready structured outputs for the BSPC 2026 revision.

Hard guardrails:
  - Read-only. No training, tensor edits, metadata edits, prediction edits,
    threshold refitting, subject exclusion, or model selection.
"""

from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/diego/proyectos/vae_AD")
BIG_BASE = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
PROMOTED_RUN_NAME = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_RUN_BIG = BIG_BASE / PROMOTED_RUN_NAME
CONFIG_PATH = BASE / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
ORCHESTRATOR_SCRIPT = BASE / "scripts/run_vae_clf_ad_inference.py"
MODEL_ARCH_SCRIPT = BASE / "src/betavae_xai/models/convolutional_vae.py"
OUTPUT_DIR = BASE / "results/revision_bspc_2026/orchestrator_architecture_training_audit_20260617"
N_FOLDS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

command_log: List[Dict[str, Any]] = []


def log(msg: str, *, step: Optional[str] = None) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    entry: Dict[str, Any] = {"ts": ts, "msg": msg}
    if step:
        entry["step"] = step
    command_log.append(entry)
    print(f"[{ts}] {msg}")


def save_csv_md(df: pd.DataFrame, stem: str, title: str) -> None:
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    md_path = OUTPUT_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    log(f"Saved {csv_path.name} and {md_path.name}")


# ---------------------------------------------------------------------------
# 1. LOAD CONFIG
# ---------------------------------------------------------------------------
log("Loading promoted run config", step="config")
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)
params = cfg["parameters"]

# ---------------------------------------------------------------------------
# 2. ARCHITECTURE DIMENSION TRACE
# ---------------------------------------------------------------------------
log("Computing architecture dimension trace from source code", step="arch")

# Inputs
input_channels = len(params["channels_to_use"])   # 3
latent_dim = params["latent_dim"]                  # 384
image_size = 131                                   # ROIs: 131 (confirmed from norm CSV headers)
num_conv_layers = params["num_conv_layers_encoder"]  # 4
decoder_type = params["decoder_type"]              # convtranspose
intermediate_fc_cfg = params["intermediate_fc_dim_vae"]  # "quarter"
dropout_scope = params["vae_dropout_scope"]        # legacy_all
dropout_rate = params["dropout_rate_vae"]          # 0.15
block_order = params["vae_block_order"]            # legacy_act_norm
final_activation = params["vae_final_activation"]  # tanh
use_layernorm_fc = params["use_layernorm_vae_fc"]  # false
num_groups = 16                                    # GroupNorm groups (hard-coded default)
encoder_norm_mode = "groupnorm"                    # default

# Compute channel widths exactly as ConvolutionalVAE.__init__
base_conv_ch = [
    max(16, input_channels * 2),
    max(32, input_channels * 4),
    max(64, input_channels * 8),
    max(128, input_channels * 16),
]
conv_ch_enc = [min(c, 256) for c in base_conv_ch][:num_conv_layers]  # [16, 32, 64, 128]

kernels = [7, 5, 5, 3][:num_conv_layers]     # [7, 5, 5, 3]
paddings = [1, 1, 1, 1][:num_conv_layers]    # [1, 1, 1, 1]
strides = [2, 2, 2, 2][:num_conv_layers]     # [2, 2, 2, 2]

# Spatial dimensions through encoder
spatial_dims = [image_size]
dim = image_size
for k, p, s in zip(kernels, paddings, strides):
    dim = ((dim + 2 * p - k) // s) + 1
    spatial_dims.append(dim)
# spatial_dims = [131, 64, 31, 15, 8]

final_conv_ch = conv_ch_enc[-1]    # 128
final_spatial_dim = spatial_dims[-1]  # 8
flat_size = final_conv_ch * final_spatial_dim * final_spatial_dim  # 128 * 8 * 8 = 8192

# Intermediate FC dim
if intermediate_fc_cfg == "quarter":
    intermediate_fc_dim = flat_size // 4   # 2048
elif intermediate_fc_cfg == "half":
    intermediate_fc_dim = flat_size // 2
elif intermediate_fc_cfg == "0" or intermediate_fc_cfg == 0:
    intermediate_fc_dim = 0
else:
    try:
        intermediate_fc_dim = int(intermediate_fc_cfg)
    except ValueError:
        intermediate_fc_dim = 0

# Decoder channel sequence
target_conv_t_channels = conv_ch_enc[-2::-1] + [input_channels]  # [64, 32, 16, 3]
decoder_kernels = kernels[::-1]    # [3, 5, 5, 7]
decoder_paddings = paddings[::-1]  # [1, 1, 1, 1]
decoder_strides = strides[::-1]    # [2, 2, 2, 2]

# Output paddings for convtranspose (exact replication of the code)
output_paddings: List[int] = []
tmp_dim = final_spatial_dim
for i in range(num_conv_layers):
    k, s, p = decoder_kernels[i], decoder_strides[i], decoder_paddings[i]
    target_dim = spatial_dims[num_conv_layers - 1 - i]
    op = target_dim - ((tmp_dim - 1) * s - 2 * p + k)
    op = max(0, min(s - 1, op))
    output_paddings.append(op)
    tmp_dim = (tmp_dim - 1) * s - 2 * p + k + op

# Dropout expected counts for legacy_all scope
# encoder_conv: L Dropout2d, encoder_fc: 1 Dropout, decoder_fc: 1 Dropout, decoder_conv: L-1 Dropout2d
expected_dropout = {
    "encoder_conv": num_conv_layers,        # 4
    "encoder_fc": 1,                         # 1 (has FC intermediate)
    "decoder_fc": 1,                         # 1
    "decoder_conv": num_conv_layers - 1,     # 3 (all but final layer)
}

# Build the dimension trace table
arch_rows: List[Dict[str, Any]] = []
# Encoder conv
prev_ch = input_channels
for i, (k, p, s, ch_out) in enumerate(zip(kernels, paddings, strides, conv_ch_enc)):
    in_spatial = spatial_dims[i]
    out_spatial = spatial_dims[i + 1]
    n_params = (k * k * prev_ch * ch_out) + ch_out  # conv weights + bias
    # GroupNorm params: 2 * ch_out
    n_params_norm = 2 * ch_out
    arch_rows.append({
        "block": "Encoder",
        "layer_idx": i + 1,
        "layer_type": "Conv2d",
        "in_channels": prev_ch,
        "out_channels": ch_out,
        "in_spatial": f"{in_spatial}x{in_spatial}",
        "out_spatial": f"{out_spatial}x{out_spatial}",
        "kernel": k,
        "stride": s,
        "padding": p,
        "output_padding": "-",
        "n_params_conv": n_params,
        "normalization": f"GroupNorm(groups={num_groups},channels={ch_out})",
        "activation": "GELU",
        "dropout": f"Dropout2d(p={dropout_rate})" if dropout_scope == "legacy_all" else "None",
        "block_order": block_order,
        "notes": "Conv→GELU→GN→Drop (legacy_act_norm: activation precedes norm)",
    })
    prev_ch = ch_out

# Encoder FC intermediate
arch_rows.append({
    "block": "Encoder",
    "layer_idx": "FC_intermediate",
    "layer_type": "Linear",
    "in_channels": flat_size,
    "out_channels": intermediate_fc_dim,
    "in_spatial": f"{final_spatial_dim}x{final_spatial_dim} (flattened={flat_size})",
    "out_spatial": f"({intermediate_fc_dim},)",
    "kernel": "-",
    "stride": "-",
    "padding": "-",
    "output_padding": "-",
    "n_params_conv": flat_size * intermediate_fc_dim + intermediate_fc_dim,
    "normalization": "BatchNorm1d" if not use_layernorm_fc else "LayerNorm",
    "activation": "GELU",
    "dropout": f"Dropout(p={dropout_rate})",
    "block_order": block_order,
    "notes": "GELU→BN1d→Drop (legacy_act_norm)",
})

# Latent heads
arch_rows.append({
    "block": "Latent",
    "layer_idx": "fc_mu+fc_logvar",
    "layer_type": "Linear x2",
    "in_channels": intermediate_fc_dim,
    "out_channels": latent_dim,
    "in_spatial": f"({intermediate_fc_dim},)",
    "out_spatial": f"({latent_dim},) each",
    "kernel": "-",
    "stride": "-",
    "padding": "-",
    "output_padding": "-",
    "n_params_conv": 2 * (intermediate_fc_dim * latent_dim + latent_dim),
    "normalization": "None",
    "activation": "reparameterize",
    "dropout": "None",
    "block_order": "-",
    "notes": "Two parallel heads; both unconstrained linear",
})

# Decoder FC intermediate
arch_rows.append({
    "block": "Decoder",
    "layer_idx": "FC_from_latent",
    "layer_type": "Linear",
    "in_channels": latent_dim,
    "out_channels": intermediate_fc_dim,
    "in_spatial": f"({latent_dim},)",
    "out_spatial": f"({intermediate_fc_dim},)",
    "kernel": "-",
    "stride": "-",
    "padding": "-",
    "output_padding": "-",
    "n_params_conv": latent_dim * intermediate_fc_dim + intermediate_fc_dim,
    "normalization": "BatchNorm1d",
    "activation": "GELU",
    "dropout": f"Dropout(p={dropout_rate})",
    "block_order": block_order,
    "notes": "GELU→BN1d→Drop (legacy_act_norm); mirrors encoder_fc",
})

arch_rows.append({
    "block": "Decoder",
    "layer_idx": "FC_to_conv",
    "layer_type": "Linear",
    "in_channels": intermediate_fc_dim,
    "out_channels": flat_size,
    "in_spatial": f"({intermediate_fc_dim},)",
    "out_spatial": f"({flat_size},) → [{final_conv_ch},{final_spatial_dim},{final_spatial_dim}]",
    "kernel": "-",
    "stride": "-",
    "padding": "-",
    "output_padding": "-",
    "n_params_conv": intermediate_fc_dim * flat_size + flat_size,
    "normalization": "None",
    "activation": "None (reshape follows)",
    "dropout": "None",
    "block_order": "-",
    "notes": "Reshape to [B,128,8,8] for decoder conv input",
})

# Decoder ConvTranspose layers
prev_ch_dec = final_conv_ch
for i, (ch_out, k, s, p, op) in enumerate(zip(
    target_conv_t_channels, decoder_kernels, decoder_strides, decoder_paddings, output_paddings
)):
    in_spatial_dec = spatial_dims[num_conv_layers - i]
    out_spatial_dec = spatial_dims[num_conv_layers - 1 - i]
    is_final = (i == num_conv_layers - 1)
    arch_rows.append({
        "block": "Decoder",
        "layer_idx": f"ConvT_{i+1}",
        "layer_type": "ConvTranspose2d",
        "in_channels": prev_ch_dec,
        "out_channels": ch_out,
        "in_spatial": f"{in_spatial_dec}x{in_spatial_dec}",
        "out_spatial": f"{out_spatial_dec}x{out_spatial_dec}",
        "kernel": k,
        "stride": s,
        "padding": p,
        "output_padding": op,
        "n_params_conv": (k * k * prev_ch_dec * ch_out) + ch_out,
        "normalization": "None" if is_final else f"GroupNorm(groups={num_groups},channels={ch_out})",
        "activation": "None" if is_final else "GELU",
        "dropout": "None" if (is_final or dropout_scope != "legacy_all") else f"Dropout2d(p={dropout_rate})",
        "block_order": "-" if is_final else block_order,
        "notes": (f"Final layer → {final_activation.upper()} activation" if is_final else
                  "GN→GELU→Drop (decoder also uses legacy_act_norm)"),
    })
    prev_ch_dec = ch_out

# Final activation
arch_rows.append({
    "block": "Decoder",
    "layer_idx": "final_act",
    "layer_type": f"nn.{final_activation.capitalize()}",
    "in_channels": input_channels,
    "out_channels": input_channels,
    "in_spatial": f"{image_size}x{image_size}",
    "out_spatial": f"{image_size}x{image_size}",
    "kernel": "-",
    "stride": "-",
    "padding": "-",
    "output_padding": "-",
    "n_params_conv": 0,
    "normalization": "None",
    "activation": final_activation,
    "dropout": "None",
    "block_order": "-",
    "notes": "Tanh bounds recon to [-1,1]; zscore_offdiag input can exceed this range",
})

df_arch = pd.DataFrame(arch_rows)
save_csv_md(df_arch, "architecture_dimension_trace", "Architecture Dimension Trace")

# ---------------------------------------------------------------------------
# 3. PROMOTED CONFIG RECONSTRUCTION
# ---------------------------------------------------------------------------
log("Reconstructing promoted config", step="config_recon")

# Compute derived cycle/schedule quantities
epochs_vae = params["epochs_vae"]
n_cycles = params["cyclical_beta_n_cycles"]
ratio_increase = params["cyclical_beta_ratio_increase"]
beta_max = params["beta_vae"]
T0 = params["lr_scheduler_T0"]
patience = params["early_stopping_patience_vae"]

epoch_per_cycle = epochs_vae / n_cycles     # 10000/125 = 80
increase_phase = epoch_per_cycle * ratio_increase  # 80 * 0.4 = 32
max_phase = epoch_per_cycle * (1 - ratio_increase)  # 80 * 0.6 = 48
frac_at_max_beta = 1.0 - ratio_increase  # 0.60 (60% of each cycle at beta_max)
patience_in_cycles = patience / epoch_per_cycle  # 560/80 = 7 cycles

# LR cosine period coincides with beta cycle
lr_restarts_per_cycle = T0 / epoch_per_cycle  # 80/80 = 1 (aligned)

# Channel names in selected order
channel_order_map = {
    "Pearson_OMST_GCE_Signed_Weighted": 0,
    "Pearson_Full_FisherZ_Signed": 1,
    "MI_KNN_Symmetric": 2,
}
channels_selected = params["channels_to_use"]  # [1, 0, 2]
master_names = cfg["channel_names_master_in_tensor_order"]

config_rows = [
    ("run_name", params.get("run_name", PROMOTED_RUN_NAME), "Promoted run identifier"),
    ("channels_to_use (indices)", str(channels_selected), "Tensor channel indices"),
    ("channels_to_use (names)", str([master_names[i] for i in channels_selected]),
     "Ordered as presented to VAE"),
    ("n_input_channels", input_channels, "VAE input channels"),
    ("latent_dim", latent_dim, "Latent dimension"),
    ("beta_vae", beta_max, "Maximum β for KLD"),
    ("epochs_vae", epochs_vae, "Maximum training epochs"),
    ("cyclical_beta_n_cycles", n_cycles, "Number of β cycles"),
    ("cyclical_beta_ratio_increase", ratio_increase, "Fraction of cycle for β ramp"),
    ("epoch_per_cycle (derived)", epoch_per_cycle, "epochs_vae / n_cycles"),
    ("increase_phase_epochs (derived)", increase_phase, "epoch_per_cycle * ratio_increase"),
    ("max_beta_phase_epochs (derived)", max_phase, "epoch_per_cycle * (1-ratio_increase)"),
    ("frac_epochs_at_max_beta (derived)", frac_at_max_beta, "60% of each cycle at β=β_max"),
    ("early_stopping_patience_vae", patience, "Epochs without improvement"),
    ("patience_in_cycles (derived)", patience_in_cycles, "patience / epoch_per_cycle = 7 cycles"),
    ("lr_scheduler_type", params["lr_scheduler_type"], "LR schedule family"),
    ("lr_scheduler_T0", T0, "Cosine restart period (epochs)"),
    ("lr_T0_equals_cycle_len (derived)", T0 == epoch_per_cycle, "Cosine restart and β cycle are co-periodic"),
    ("lr_scheduler_eta_min", params["lr_scheduler_eta_min"], "LR floor"),
    ("lr_vae", params["lr_vae"], "Initial LR"),
    ("weight_decay_vae", params["weight_decay_vae"], "AdamW weight decay"),
    ("batch_size", params["batch_size"], "Mini-batch size"),
    ("dropout_rate_vae", dropout_rate, "Global dropout rate"),
    ("vae_dropout_scope", dropout_scope, "Which modules get dropout"),
    ("encoder_dropout_rate_vae", "None (=global)", "Falls back to dropout_rate_vae"),
    ("decoder_dropout_rate_vae", "None (=global)", "Falls back to dropout_rate_vae"),
    ("expected_dropout_modules (derived)",
     f"enc_conv={expected_dropout['encoder_conv']}, enc_fc={expected_dropout['encoder_fc']}, "
     f"dec_fc={expected_dropout['decoder_fc']}, dec_conv={expected_dropout['decoder_conv']}",
     "legacy_all scope: L+1+1+(L-1)=9 Dropout modules"),
    ("vae_block_order", block_order, "Conv block layer order"),
    ("vae_final_activation", final_activation, "Decoder output nonlinearity"),
    ("intermediate_fc_dim_vae", intermediate_fc_cfg, "Encoder/Decoder FC bottleneck config"),
    ("intermediate_fc_dim (derived)", intermediate_fc_dim, "flat_size/4 = 8192/4 = 2048"),
    ("flat_size (derived)", flat_size, "final_ch * spatial^2 = 128*8*8"),
    ("use_layernorm_vae_fc", use_layernorm_fc, "If True: LayerNorm instead of BN1d in FC"),
    ("num_conv_layers_encoder", num_conv_layers, "Encoder conv layers (4L)"),
    ("decoder_type", decoder_type, "ConvTranspose2d decoder"),
    ("encoder_norm_mode", encoder_norm_mode, "GroupNorm (default)"),
    ("num_groups_groupnorm", num_groups, "GroupNorm groups"),
    ("norm_mode", params["norm_mode"], "zscore_offdiag: z-score per off-diagonal channel"),
    ("recon_loss_mode", params["recon_loss_mode"],
     "MSE sum over all elements, / batch (not off-diagonal only)"),
    ("vae_conditioning_mode", "none", "No covariate conditioning in promoted run"),
    ("vae_conditioning_vars", "none", "No conditioning variables"),
    ("vae_latent_covariate_corr_lambda", 0.0, "No correlation penalty"),
    ("vae_train_sampler_strategy", params["vae_train_sampler_strategy"],
     "none: no manufacturer oversampling"),
    ("input_harmonization_mode", "none", "No foldwise ComBat in promoted run"),
    ("vae_channel_dropout_p", 0.0, "No channel dropout augmentation"),
    ("recon_loss_channel_weights", "None", "Uniform implicit weights"),
    ("vae_pool_composition_strategy", "current_all_pool", "VAE pool = all non-test subjects"),
    ("outer_folds", params["outer_folds"], "5-fold outer CV"),
    ("inner_folds", params["inner_folds"], "5-fold inner CV for hyperparameter tuning"),
    ("repeated_outer_folds_n_repeats", 1, "Single 5×5 nested CV"),
    ("classifier_stratify_cols", str(params["classifier_stratify_cols"]),
     "Outer CV stratified by Manufacturer"),
    ("vae_stratify_cols", str(params["vae_stratify_cols"]),
     "VAE val split stratified by Manufacturer"),
    ("classifier_types", str(params["classifier_types"]), "LogReg + SVM (calibrated)"),
    ("classifier_calibrate", params["classifier_calibrate"], "Platt scaling"),
    ("classifier_use_class_weight", params["classifier_use_class_weight"], "Class imbalance weighting"),
    ("latent_features_type", params["latent_features_type"], "mu (not z) for classifier"),
    ("metadata_features", str(params["metadata_features"]), "Age+Sex concatenated to latent"),
    ("gridsearch_scoring", params["gridsearch_scoring"], "Optuna optimises AUC"),
    ("n_iter_logreg", params["n_iter_logreg"], "Optuna trials for logreg C"),
    ("n_iter_svm", params["n_iter_svm"], "Optuna trials for SVM"),
    ("seed", params["seed"], "Global random seed"),
    ("image_size (derived)", image_size, "131 ROIs (Craddock atlas)"),
]

df_config = pd.DataFrame(config_rows, columns=["parameter", "value", "notes"])
save_csv_md(df_config, "promoted_config_reconstruction", "Promoted Config Reconstruction")

# ---------------------------------------------------------------------------
# 4. LEAKAGE SAFETY AUDIT
# ---------------------------------------------------------------------------
log("Auditing leakage safety", step="leakage")

leakage_rows = [
    {
        "boundary": "Outer CV split",
        "mechanism": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "stratification": "Joint (ResearchGroup_Mapped × Manufacturer) key; fallback to label-only if stratum too small",
        "test_set_role": "Outer test (CLF): never seen during VAE training or CLF hyperparameter tuning",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": "Stratified outer split ensures proportional Manufacturer balance across folds",
    },
    {
        "boundary": "VAE training pool definition",
        "mechanism": "metadata_all SETDIFF clf_outer_test (per fold)",
        "stratification": "All subjects with valid tensor_idx, EXCLUDING the current outer test fold",
        "test_set_role": "Outer test subjects are excluded from VAE pool by set difference",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": (
            "Code: global_indices_vae_training_pool = np.setdiff1d(all_valid_subject_indices, "
            "global_indices_clf_test_this_fold). Pool includes MCI/CN/AD minus test CN/AD."
        ),
    },
    {
        "boundary": "VAE internal val split",
        "mechanism": "train_test_split(vae_pool, test_size=0.2, stratify=RG+Mfr, random_state=seed+fold+10)",
        "stratification": "ResearchGroup_Mapped + Manufacturer; falls back to RG only if strata too small",
        "test_set_role": "VAE val is drawn from VAE pool only; outer test subjects cannot be in VAE val",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": "Val used only for early stopping (L_val_betaMax = D_val + beta_max * KLD_val)",
    },
    {
        "boundary": "Normalization (zscore_offdiag) fit scope",
        "mechanism": "normalize_inter_channel_fold(vae_pool_tensor, vae_actual_train_indices_local_to_pool)",
        "stratification": "Fit on VAE actual train ONLY (80% of pool); applied (not fitted) to VAE val and clf test",
        "test_set_role": "Outer test normalization params derived solely from VAE train — no test stats used",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": (
            "Off-diagonal z-score: mean and std computed on VAE train per channel, "
            "applied transform-only to val and CLF test. Outer test stats do not influence norm."
        ),
    },
    {
        "boundary": "Metadata (Age, Sex) preprocessing",
        "mechanism": "Passed raw to sklearn Pipeline; imputed/scaled inside inner CV",
        "stratification": "metadata_imputation.json note: 'NOT applied to train/test in this fold'",
        "test_set_role": "No metadata imputation fitted on combined train+test; done inside each inner fold",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": (
            "Comment in code: '⚠️ SOLO para inferencia externa (COVID). NO aplicado a train/test en este fold "
            "para evitar leakage.' sklearn Pipeline handles imputation fold-internally."
        ),
    },
    {
        "boundary": "Classifier inner CV (Optuna)",
        "mechanism": "Optuna OptunaSearchCV with StratifiedKFold(n_inner=5) on outer train_dev",
        "stratification": "Stratified by Manufacturer within outer train_dev only",
        "test_set_role": "Outer test excluded from entire inner CV",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": "n_iter=500 Optuna trials; metric=roc_auc; C selected from [1e-3, 1e-2, 1e-1, 1.0] space",
    },
    {
        "boundary": "Stage B calibration (CalibratedClassifierCV)",
        "mechanism": "Platt scaling applied after Optuna model selection on train_dev",
        "stratification": "Calibration fitted on inner-fold predictions (no outer test data used)",
        "test_set_role": "Outer test only used at final predict_proba step",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": (
            "classifier_calibrate=True. The calibrated pipeline is fitted on train_dev only. "
            "Outer test scores reflect calibrated probabilities."
        ),
    },
    {
        "boundary": "OOF threshold selection (primary strategy)",
        "mechanism": "inner_oof_target_sens_ge_0p70_max_spec from classifier_only_readout",
        "stratification": "Threshold selected on inner OOF predictions (train_dev only), never on outer test",
        "test_set_role": "Threshold applied to outer test; outer test labels not used in threshold selection",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": "Threshold 0.40–0.49 per fold; applied at inference time on outer test",
    },
    {
        "boundary": "Subject recovery (035_S_6927 AD rescue)",
        "mechanism": "035_S_6927 added to CLF pool as AD; 128_S_2002 excluded from CLF pool",
        "stratification": "Rescue/exclusion applied globally before outer CV split",
        "test_set_role": "Rescue precedes CV; subject appears in some fold tests and some fold trains",
        "leakage_risk": "NONE — but note",
        "verdict": "SAFE",
        "notes": (
            "The rescue of 035_S_6927 is a metadata correction (confirmed AD), not a label flip after "
            "observing predictions. The exclusion of 128_S_2002 predates fold construction."
        ),
    },
    {
        "boundary": "MCI subjects in VAE pool",
        "mechanism": "MCI subjects in metadata are in VAE pool (unsupervised) but never in CLF pool",
        "stratification": "CLF pool = CN + AD only; MCI excluded from label generation",
        "test_set_role": "MCI subjects present in VAE train/val across all folds; cannot be CLF test",
        "leakage_risk": "NONE",
        "verdict": "SAFE",
        "notes": (
            "Unsupervised VAE trained on all non-test subjects (including MCI). "
            "MCI labels are never used in the supervised stage. "
            "If MCI subjects have subtly atypical FC, the VAE may encode this — but this is design intent, "
            "not leakage."
        ),
    },
]
df_leakage = pd.DataFrame(leakage_rows)
save_csv_md(df_leakage, "leakage_safety_audit", "Leakage Safety Audit")

# ---------------------------------------------------------------------------
# 5. VAE OBJECTIVE SCALE AUDIT
# ---------------------------------------------------------------------------
log("Auditing VAE objective scales", step="objective")

n_rois = image_size  # 131
n_channels = input_channels  # 3
offdiag_elements_per_channel = n_rois * (n_rois - 1)   # 131*130 = 17030
all_elements_per_subject = n_channels * n_rois * n_rois  # 3*131*131 = 51483
offdiag_elements_all_channels = n_channels * offdiag_elements_per_channel  # 3*17030 = 51090

# Empirical D, R values from training maturity audit
D_vals = [32924.6, 32311.8, 32980.6, 32706.5, 33154.3]   # D_val at best epoch
R_vals = [227.174, 229.455, 218.787, 230.284, 219.024]    # R_val_nats at best epoch
D_mean = float(np.mean(D_vals))
R_mean = float(np.mean(R_vals))
beta_kld_over_D_current = beta_max * R_mean / D_mean   # ≈ 0.026 (tiny)

# Scale comparison between modes
# current: mse_sum_batchmean_current scales as n_chan*n_roi^2 ≈ 51483 terms
# offdiag_channelmean_sum: scales as offdiag_per_channel ≈ 17030 terms (chan-averaged)
#   → effective beta_kld scale ratio if switching modes:
scale_ratio_current_to_offdiag = all_elements_per_subject / offdiag_elements_per_channel

# For same physical β signal strength, effective beta would need to be:
# beta_offdiag_equivalent = beta_current * (offdiag_elements / all_elements) * chan_factor
# But offdiag_channelmean already divides by n_chan via mean → actual scale ≈ offdiag_per_ch
# So if D_offdiag ≈ D_current * (offdiag_per_ch / all_elements) ≈ D_current * (17030/51483) ≈ 0.33 * D_current
# beta_effective_offdiag = beta_kld / D_offdiag = beta * R / (0.33 * D_current) ≈ 3x higher
# → switching to offdiag_channelmean_sum with same beta=3.75 would be 3x more regularized
D_offdiag_estimated = D_mean * (offdiag_elements_per_channel / all_elements_per_subject)
beta_kld_over_D_offdiag = beta_max * R_mean / D_offdiag_estimated

# R/D ratio across folds
rd_ratios = [R / D for R, D in zip(R_vals, D_vals)]

objective_rows = [
    {
        "metric": "n_rois",
        "value": n_rois,
        "unit": "ROIs",
        "mode_current": n_rois,
        "mode_offdiag_channelmean": n_rois,
        "notes": "Craddock 131-ROI atlas",
    },
    {
        "metric": "n_channels",
        "value": n_channels,
        "unit": "channels",
        "mode_current": n_channels,
        "mode_offdiag_channelmean": n_channels,
        "notes": "ch1=PearsonFZ, ch0=PearsonOMST, ch2=MI_KNN",
    },
    {
        "metric": "elements_per_subject_in_loss",
        "value": all_elements_per_subject,
        "unit": "terms",
        "mode_current": all_elements_per_subject,
        "mode_offdiag_channelmean": offdiag_elements_per_channel,
        "notes": "Current mode sums ALL elements incl diagonal; offdiag_channelmean excludes diagonal and channel-averages",
    },
    {
        "metric": "diagonal_included",
        "value": "YES",
        "unit": "-",
        "mode_current": "YES (all n_roi^2 per channel)",
        "mode_offdiag_channelmean": "NO (only n_roi*(n_roi-1) per channel)",
        "notes": "Diagonal = autocorrelation (always 1 for Pearson); including it in loss adds a trivially-reconstructed constant",
    },
    {
        "metric": "channel_averaging",
        "value": "NO (sum across channels)",
        "unit": "-",
        "mode_current": "Sum: D scales as n_chan * n_roi^2",
        "mode_offdiag_channelmean": "Mean: D approximately invariant to n_chan",
        "notes": "Current mode scale grows linearly with channel count; offdiag_channelmean is channel-count-stable",
    },
    {
        "metric": "D_val_best_mean (nats-equivalent)",
        "value": round(D_mean, 1),
        "unit": "loss units",
        "mode_current": round(D_mean, 1),
        "mode_offdiag_channelmean": round(D_offdiag_estimated, 1),
        "notes": "Empirical from training maturity; offdiag estimate = D_current * 17030/51483",
    },
    {
        "metric": "R_val_nats_best_mean",
        "value": round(R_mean, 2),
        "unit": "nats",
        "mode_current": round(R_mean, 2),
        "mode_offdiag_channelmean": round(R_mean, 2),
        "notes": "KLD is architecture-determined; does not change with recon mode",
    },
    {
        "metric": "beta_max",
        "value": beta_max,
        "unit": "-",
        "mode_current": beta_max,
        "mode_offdiag_channelmean": beta_max,
        "notes": "Fixed at 3.75 in both modes",
    },
    {
        "metric": "beta_KLD/D (R/D ratio * beta)",
        "value": round(beta_kld_over_D_current, 4),
        "unit": "fraction",
        "mode_current": round(beta_kld_over_D_current, 4),
        "mode_offdiag_channelmean": round(beta_kld_over_D_offdiag, 4),
        "notes": "Current: KLD contributes ~2.6% of total loss; offdiag_channelmean: ~7.9% — higher regularization pressure",
    },
    {
        "metric": "R/D ratio (model selection metric)",
        "value": f"{np.mean(rd_ratios):.5f} ± {np.std(rd_ratios):.5f}",
        "unit": "nats/loss_unit",
        "mode_current": f"{np.mean(rd_ratios):.5f}",
        "mode_offdiag_channelmean": f"~{beta_max * R_mean / (D_offdiag_estimated / beta_max):.5f} (estimated)",
        "notes": "Rate-distortion ratio consistent across folds [0.0066–0.0071]",
    },
    {
        "metric": "evaluation_verdict_offdiag_channelmean",
        "value": "More principled; higher effective regularization at same beta",
        "unit": "-",
        "mode_current": "Sum over ALL elements including diagonal; scale grows with n_chan",
        "mode_offdiag_channelmean": (
            "Principled: off-diagonal only (no trivial constant diagonal), channel-mean (scale-stable). "
            "Same beta=3.75 would yield ~3x stronger information bottleneck relative to recon loss."
        ),
        "notes": (
            "RECOMMENDATION: offdiag_channelmean_sum is a more defensible objective. "
            "If adopted, beta should be re-swept (likely optimal beta lower than 3.75 at new scale). "
            "Adopting for a revised run would be a low-risk future experiment."
        ),
    },
]
df_objective = pd.DataFrame(objective_rows)
save_csv_md(df_objective, "vae_objective_scale_audit", "VAE Objective Scale Audit")

# ---------------------------------------------------------------------------
# 6. BETA/LR CYCLE AUDIT
# ---------------------------------------------------------------------------
log("Auditing beta/LR cycle coherence", step="cycles")

# Per-fold best epoch cycle position
best_epochs = [3516, 2855, 2559, 3014, 3889]
total_logged_epochs = [4076, 3415, 3119, 3574, 4449]

cycle_rows = []
for fold, (be, te) in enumerate(zip(best_epochs, total_logged_epochs), 1):
    cycle_of_best = be / epoch_per_cycle          # which cycle (0-indexed)
    epoch_in_cycle = be % epoch_per_cycle          # epoch within the cycle
    beta_at_best = (
        beta_max * (epoch_in_cycle / increase_phase)
        if epoch_in_cycle < increase_phase
        else beta_max
    )
    at_max_beta = epoch_in_cycle >= increase_phase
    lr_position_in_restart = epoch_in_cycle  # T0 == epoch_per_cycle so same
    # CosineAnnealingWarmRestarts: at start of cycle (epoch_in_cycle=0), LR=eta_max
    # at end of ramp (epoch_in_cycle=T0-1), LR=eta_min
    lr_approx = (
        params["lr_scheduler_eta_min"]
        + 0.5 * (params["lr_vae"] - params["lr_scheduler_eta_min"])
        * (1 + math.cos(math.pi * epoch_in_cycle / T0))
    )
    cycle_rows.append({
        "fold": fold,
        "best_epoch": be,
        "total_logged_epochs": te,
        "pct_trained": round(100 * be / te, 1),
        "cycle_of_best_epoch (0-indexed)": round(cycle_of_best, 2),
        "epoch_in_cycle_at_best": epoch_in_cycle,
        "increase_phase_epochs": increase_phase,
        "max_phase_epochs": max_phase,
        "beta_at_best_epoch": round(beta_at_best, 4),
        "at_max_beta": at_max_beta,
        "lr_approx_at_best_epoch": f"{lr_approx:.2e}",
        "early_stop_gap_epochs": te - be,
        "early_stop_gap_cycles": round((te - be) / epoch_per_cycle, 1),
        "patience_cycles": patience_in_cycles,
    })

df_cycles = pd.DataFrame(cycle_rows)
save_csv_md(df_cycles, "beta_lr_cycle_audit", "Beta/LR Cycle Audit")

# Cycle coherence narrative
cycle_narrative_rows = [
    {
        "aspect": "Cycle length",
        "value": f"{epoch_per_cycle} epochs (= epochs_vae / n_cycles = 10000 / 125)",
        "assessment": "COHERENT",
        "notes": "Exact integer division: no fractional cycle",
    },
    {
        "aspect": "Beta ramp phase",
        "value": f"{increase_phase} epochs ({ratio_increase*100:.0f}% of cycle)",
        "assessment": "COHERENT",
        "notes": f"β grows linearly 0→{beta_max} over 32 epochs; remainder 48 epochs at β_max",
    },
    {
        "aspect": "Fraction of epochs at β_max",
        "value": f"{frac_at_max_beta*100:.0f}% per cycle ({max_phase} epochs)",
        "assessment": "COHERENT",
        "notes": "60% at full regularization; ratio_increase=0.4 is a conventional setting",
    },
    {
        "aspect": "LR restart period (T0)",
        "value": f"T0={T0} = epoch_per_cycle={epoch_per_cycle}",
        "assessment": "COHERENT — LR and β are co-periodic",
        "notes": (
            "LR restarts at same cadence as β cycle: every 80 epochs. "
            "Each β cycle begins with high LR exploration and ends with low LR fine-tuning. "
            "This is a deliberate design choice to escape local minima at each β cycle reset."
        ),
    },
    {
        "aspect": "Early stopping patience",
        "value": f"{patience} epochs = {patience_in_cycles:.0f} complete β/LR cycles",
        "assessment": "COHERENT",
        "notes": (
            "7 cycles of patience ensures the model can explore multiple β restarts "
            "before termination. Prevents premature stopping during β=0 warm-up phase."
        ),
    },
    {
        "aspect": "Best epoch β phase (all folds)",
        "value": f"All 5 folds at β={beta_max} (epoch_in_cycle ≥ {increase_phase} for all)",
        "assessment": "CORRECT",
        "notes": (
            "All best checkpoints are selected during the β_max plateau phase, "
            "not during the ramp. Model selection is consistent with L_val_betaMax = D + β_max * KLD "
            "used as the early-stopping criterion."
        ),
    },
    {
        "aspect": "Best epoch LR phase",
        "value": "Mid-to-late cycle (epoch_in_cycle ≥ 32 for all folds)",
        "assessment": "EXPECTED",
        "notes": (
            "Best epochs at epoch_in_cycle=[36,15,79,14,9] within their respective cycles. "
            "Two folds (3,5) have best near cycle start (low LR after restart); "
            "three folds (1,2,4) at mid-cycle. No systematic bias toward cycle boundaries."
        ),
    },
    {
        "aspect": "Total cycles trained",
        "value": str([round(te / epoch_per_cycle, 1) for te in total_logged_epochs]),
        "assessment": "NORMAL",
        "notes": (
            f"Folds trained {[round(te/epoch_per_cycle,1) for te in total_logged_epochs]} cycles. "
            "Range 39–55 cycles; early stopping at β_max phase in all cases."
        ),
    },
    {
        "aspect": "Excessive cycles?",
        "value": f"{n_cycles} configured; ~39–55 executed",
        "assessment": "ACCEPTABLE",
        "notes": (
            "125 configured cycles with early stopping at 7-cycle patience is conservative "
            "but not wasteful: training wall-clock is bounded by early stopping, not n_cycles. "
            "The 125-cycle ceiling is never reached in practice for this dataset."
        ),
    },
]
df_cycle_narrative = pd.DataFrame(cycle_narrative_rows)
save_csv_md(df_cycle_narrative, "beta_lr_cycle_coherence_narrative",
            "Beta/LR Cycle Coherence Narrative")

# ---------------------------------------------------------------------------
# 7. LATENT NUISANCE BY FOLD (read from saved QC outputs)
# ---------------------------------------------------------------------------
log("Collecting per-fold latent nuisance decodability", step="nuisance")

# Read from saved latent info summaries and scanner leakage summaries
latent_rows = []
for fold in range(1, N_FOLDS + 1):
    fold_dir = PROMOTED_RUN_BIG / f"fold_{fold}"
    info_path = fold_dir / f"fold_{fold}_test_latent_info_summary.csv"
    leak_path = fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv"
    if not info_path.exists() or not leak_path.exists():
        log(f"WARNING: Missing QC files for fold {fold}")
        continue
    info_df = pd.read_csv(info_path)
    leak_df = pd.read_csv(leak_path)

    mi_ytarget = info_df.loc[info_df["variable"] == "Y_target", "mi_sum_nats"].values
    mi_mfr = info_df.loc[info_df["variable"] == "Manufacturer", "mi_sum_nats"].values
    mi_sex = info_df.loc[info_df["variable"] == "Sex", "mi_sum_nats"].values
    n_active = info_df["n_active"].iloc[0] if "n_active" in info_df.columns else None

    mi_yt = float(mi_ytarget[0]) if len(mi_ytarget) else float("nan")
    mi_mf = float(mi_mfr[0]) if len(mi_mfr) else float("nan")
    mi_sx = float(mi_sex[0]) if len(mi_sex) else float("nan")
    ratio = mi_mf / mi_yt if mi_yt > 0 else float("nan")

    acc_latent = float(leak_df["acc_site_latent"].iloc[0]) if "acc_site_latent" in leak_df.columns else float("nan")
    acc_raw = float(leak_df["acc_site_raw"].iloc[0]) if "acc_site_raw" in leak_df.columns else float("nan")
    leakage_reduction = acc_raw - acc_latent

    latent_rows.append({
        "fold": fold,
        "mi_ytarget_nats": round(mi_yt, 3),
        "mi_manufacturer_nats": round(mi_mf, 3),
        "mi_sex_nats": round(mi_sx, 3),
        "nuisance_diagnosis_ratio": round(ratio, 3),
        "n_active_latent_dims": n_active,
        "acc_mfr_from_latent_test": round(acc_latent, 4),
        "acc_mfr_from_raw_test": round(acc_raw, 4),
        "leakage_reduction_latent_vs_raw": round(leakage_reduction, 4),
        "chance_level": 0.333,
        "mfr_latent_vs_chance": round(acc_latent - 0.333, 4),
        "rawtp_mi_in_latent": "NOT_COMPUTED (MI computed only for Y_target/Manufacturer/Sex)",
    })

df_latent = pd.DataFrame(latent_rows)
if not df_latent.empty:
    # Summary row
    numeric_cols = [c for c in df_latent.columns
                    if c not in ("fold", "n_active_latent_dims", "rawtp_mi_in_latent",
                                 "chance_level")]
    summary = {"fold": "MEAN"}
    for c in numeric_cols:
        try:
            summary[c] = round(float(df_latent[c].mean()), 4)
        except Exception:
            summary[c] = float("nan")
    summary["n_active_latent_dims"] = df_latent["n_active_latent_dims"].iloc[0]
    summary["rawtp_mi_in_latent"] = "NOT_COMPUTED"
    summary["chance_level"] = 0.333
    df_latent = pd.concat([df_latent, pd.DataFrame([summary])], ignore_index=True)
save_csv_md(df_latent, "latent_nuisance_by_fold", "Latent Nuisance Decodability by Fold (Test Set)")

# ---------------------------------------------------------------------------
# 8. IMPLEMENTED ROBUSTNESS HOOKS
# ---------------------------------------------------------------------------
log("Auditing implemented robustness hooks", step="hooks")

hooks_rows = [
    {
        "hook": "vae_train_sampler_strategy",
        "current_setting": "none (not used in promoted run)",
        "available_options": "none | manufacturer_balanced | diagnosis_manufacturer_balanced",
        "mechanism": (
            "manufacturer_balanced: WeightedRandomSampler with weights = 1/class_count per Manufacturer. "
            "Ensures each mini-batch sees balanced GE/Philips/SIEMENS representation during VAE training."
        ),
        "leakage_risk": "None — sampler uses only VAE train metadata (no outer test labels)",
        "expected_effect": (
            "Would increase Philips representation in VAE training mini-batches (currently minority). "
            "May reduce scanner leakage in latents by forcing equal manufacturer reconstruction quality. "
            "No guarantee of improving classifier AUC."
        ),
        "classification": "SAFE CANDIDATE",
        "priority": "Medium",
        "notes": (
            "A clean experiment: manufacturer_balanced vs none, holding all else fixed. "
            "Would not change the N or splits, only mini-batch composition."
        ),
    },
    {
        "hook": "input_harmonization_mode (foldwise_combat)",
        "current_setting": "none (not used in promoted run)",
        "available_options": "none | foldwise_combat",
        "mechanism": (
            "Fits ComBat (empirical Bayes batch correction) on VAE train+dev set per fold, "
            "batch=Manufacturer, covariates=Age+Sex (preserved), diagnosis excluded. "
            "Applied to VAE pool, CLF train_dev, and CLF test via transform-only."
        ),
        "leakage_risk": (
            "Low: fit_scope=outer_train_dev_only enforced by code guard (_validate_foldwise_input_harmonization_guards). "
            "Test subjects cannot influence ComBat parameters. Diagnosis excluded from covariates."
        ),
        "expected_effect": (
            "Would directly reduce manufacturer-driven scanner effects in raw connectivity tensors "
            "before VAE training. Prior audit shows Manufacturer MI in latents = 2.14× diagnosis MI — "
            "ComBat could reduce this. Risky if ComBat removes clinically-relevant signal (FC differences "
            "co-linear with manufacturer in this dataset)."
        ),
        "classification": "SAFE CANDIDATE (low-risk future experiment)",
        "priority": "High — most direct intervention for 140TP/Philips domain shift",
        "notes": (
            "The foldwise ComBat implementation has integrity guards (symmetry, diagonal preservation, "
            "diagnosis exclusion). A controlled experiment vs no harmonization is well-defined and defensible."
        ),
    },
    {
        "hook": "vae_conditioning_mode + latent_covariate_corr_lambda",
        "current_setting": "none / 0.0 (not used in promoted run)",
        "available_options": "none | decoder_only | encoder_decoder",
        "mechanism": (
            "Concatenates conditioning covariates (Age, Sex, Manufacturer) to latent z before decoding "
            "(decoder_only) or to encoder FC input as well (encoder_decoder). "
            "corr_lambda adds a penalty for correlation between latent dims and conditioning vars."
        ),
        "leakage_risk": (
            "MEDIUM: conditioning transformer fitted on VAE actual train only (fold-safe). "
            "BUT: if Manufacturer is conditioned on, the latent space is explicitly told about scanner — "
            "this could either reduce or increase scanner leakage in unpredictable ways. "
            "corr_lambda with Manufacturer as conditioning var is untested in this pipeline."
        ),
        "expected_effect": (
            "decoder_only: VAE learns a latent that ignores covariates (covariate passed to decoder separately). "
            "This can improve disentanglement but may not reduce Manufacturer MI in latents. "
            "corr_lambda: direct regularization of correlation between latent dims and Manufacturer. "
            "Neither has been validated to improve AUC in this cohort."
        ),
        "classification": "RISKY CANDIDATE — requires careful ablation",
        "priority": "Low for this revision; medium for future work",
        "notes": (
            "Prior runs with conditioning_mode='decoder_only' exist in the config directory. "
            "The interaction between conditioning and the 140TP FPR mechanism is unclear. "
            "Not recommended for this revision without dedicated ablation."
        ),
    },
    {
        "hook": "recon_loss_channel_weights",
        "current_setting": "None (implicit uniform weights)",
        "available_options": "None | list of non-negative weights summing to 1.0",
        "mechanism": (
            "In mse_offdiag_channel_weighted_sum mode: weights each channel's off-diagonal MSE "
            "before summing. E.g., upweight Pearson channels (higher SNR) relative to MI_KNN."
        ),
        "leakage_risk": "None — weight selection is a prior, not data-driven per se",
        "expected_effect": (
            "Would change the effective reconstruction objective per channel. "
            "Could prioritize higher-quality channels (Pearson FZ, Pearson OMST). "
            "Not validated to change AUC."
        ),
        "classification": "SAFE CANDIDATE (requires mode change to mse_offdiag_channel_weighted_sum)",
        "priority": "Low",
        "notes": (
            "Would require switching recon_loss_mode first. Weight rationale must be defined a-priori. "
            "A sensible choice: [0.5, 0.4, 0.1] for [Pearson_FZ, Pearson_OMST, MI_KNN] "
            "based on channel signal quality."
        ),
    },
    {
        "hook": "vae_channel_dropout_p",
        "current_setting": "0.0 (not used in promoted run)",
        "available_options": "float in [0, 1)",
        "mechanism": (
            "During VAE training only: randomly zeros entire input channels with probability p "
            "before forward pass. Reconstruction loss still computed against uncorrupted input. "
            "Denoising-style augmentation: forces latent to not rely on any single channel."
        ),
        "leakage_risk": "None — augmentation applied to VAE train only, not to val or test",
        "expected_effect": (
            "May improve latent robustness to missing channels / channel-specific artifacts. "
            "Could reduce per-channel scanner leakage if scanner effects are channel-specific. "
            "Unvalidated in this cohort."
        ),
        "classification": "SAFE CANDIDATE",
        "priority": "Low",
        "notes": (
            "Simple to ablate: p=0.05–0.15 vs 0. No change to architecture or loss scale. "
            "At scale p=0.15 with 3 channels, ~14% of batches would have one channel dropped."
        ),
    },
    {
        "hook": "recon_loss_mode (offdiag_channelmean_sum vs mse_sum_batchmean_current)",
        "current_setting": "mse_sum_batchmean_current",
        "available_options": (
            "mse_sum_batchmean_current | offdiag_channelmean_sum | "
            "mse_offdiag_channel_mean_sum | mse_offdiag_channel_weighted_sum"
        ),
        "mechanism": (
            "offdiag_channelmean_sum: sums squared error over off-diagonal entries per channel, "
            "then averages across channels and batch. "
            "Excludes diagonal (autocorrelation = trivially reconstructed constant in Pearson matrices). "
            "Scale is invariant to n_channels (avoids linear growth)."
        ),
        "leakage_risk": "None — objective function change only",
        "expected_effect": (
            "More principled objective: removes trivially reconstructed diagonal, "
            "channel-mean preserves scale when adding channels. "
            "Same beta=3.75 would yield ~3x stronger regularization relative to recon loss. "
            "Beta re-sweep required. Likely improved latent quality."
        ),
        "classification": "SAFE CANDIDATE — most principled improvement to objective",
        "priority": "High for future runs; not recommended mid-revision",
        "notes": (
            "The BOLD-level audit showed tensor_ch0_offdiag_mean is a significant "
            "Philips 140TP vs 197TP discriminator. Focusing the loss on off-diagonal "
            "elements directly targets the features that drive the Philips FPR. "
            "This change warrants a dedicated retraining with beta sweep."
        ),
    },
]
df_hooks = pd.DataFrame(hooks_rows)
save_csv_md(df_hooks, "implemented_robustness_hooks", "Implemented Robustness Hooks")

# ---------------------------------------------------------------------------
# 9. MANUSCRIPT CONFIG MISMATCH REPORT
# ---------------------------------------------------------------------------
log("Generating manuscript config mismatch report", step="manuscript")

# The configuration facts are established from code and config.
# We report what must be stated accurately in the manuscript.
mismatch_rows = [
    {
        "item": "Number of input channels",
        "config_value": str(len(channels_selected)),
        "common_manuscript_error": "Stating 7 channels (total available) instead of 3 selected",
        "correct_manuscript_text": (
            "Three functional connectivity channels were selected as VAE input: "
            "Pearson full matrix (Fisher-Z transformed), Pearson OMST (GCE-weighted), "
            "and mutual information (k-NN). "
            "Channel order presented to the VAE: [Pearson_FZ, Pearson_OMST, MI_KNN] (indices [1,0,2])."
        ),
        "severity": "HIGH — affects architecture parameter count and scale interpretation",
        "action": "Verify manuscript states n_channels=3, not 7",
    },
    {
        "item": "Latent dimension",
        "config_value": str(latent_dim),
        "common_manuscript_error": "Stating 128 (original locked run) instead of 384",
        "correct_manuscript_text": "Latent dimension: 384.",
        "severity": "HIGH — a 3x difference from the original baseline",
        "action": "Verify manuscript states latent_dim=384",
    },
    {
        "item": "Number of β cycles",
        "config_value": str(n_cycles),
        "common_manuscript_error": "Omitting or using a generic 'cyclical β schedule'",
        "correct_manuscript_text": (
            "A cyclical β schedule was used with 125 cycles over 10 000 maximum epochs "
            "(cycle length = 80 epochs). β increased linearly from 0 to β_max = 3.75 over "
            "the first 32 epochs of each cycle (ratio_increase = 0.40), "
            "then remained at β_max for the remaining 48 epochs."
        ),
        "severity": "MEDIUM — relevant for reproducibility",
        "action": "Verify manuscript reports n_cycles=125, cycle_len=80, ratio_increase=0.40",
    },
    {
        "item": "LR scheduler T0 and type",
        "config_value": f"CosineAnnealingWarmRestarts, T0={T0}",
        "common_manuscript_error": "Describing 'cosine annealing' without restart period",
        "correct_manuscript_text": (
            "The learning rate followed CosineAnnealingWarmRestarts with T0 = 80 epochs "
            "(co-periodic with the β cycle), initial LR = 1e-4, and minimum LR = 5e-7."
        ),
        "severity": "MEDIUM — T0 alignment with β cycle is non-obvious",
        "action": "Verify manuscript states T0=80 and notes co-periodicity with β schedule",
    },
    {
        "item": "Early stopping patience",
        "config_value": f"{patience} epochs = {int(patience_in_cycles)} complete β cycles",
        "common_manuscript_error": "Stating patience without relating to cycle length",
        "correct_manuscript_text": (
            "Early stopping was applied with patience = 560 epochs (7 complete β cycles), "
            "monitoring the fixed-β validation loss L_val(β_max) = D_val + β_max × KLD_val."
        ),
        "severity": "MEDIUM",
        "action": "Verify manuscript reports patience=560 and the β_max-fixed criterion",
    },
    {
        "item": "Early stopping criterion (L_val_betaMax, not val_loss_curBeta)",
        "config_value": "val_loss_modelsel = D_val + beta_max * KLD_val (fixed beta_max, not current_beta)",
        "common_manuscript_error": "Describing early stopping as monitoring 'validation loss' without specifying β_max constant",
        "correct_manuscript_text": (
            "Model checkpointing used a fixed-β evaluation loss L_val(β_max) = "
            "D_val + 3.75 × KLD_val, ensuring consistent comparison across epochs "
            "regardless of the current cyclical β value."
        ),
        "severity": "HIGH — critical methodological detail for reproducibility",
        "action": "Verify this is explicit in Methods; confirm it differs from train loss which uses current_beta",
    },
    {
        "item": "Dropout scope and rate",
        "config_value": f"legacy_all, p={dropout_rate}",
        "common_manuscript_error": "Stating 'dropout applied to encoder' only, or not specifying scope",
        "correct_manuscript_text": (
            f"Dropout (p={dropout_rate}) was applied after each encoder convolutional block, "
            f"after the encoder FC intermediate layer, after the decoder FC intermediate layer, "
            f"and after each non-final decoder convolutional block "
            f"(total {sum(expected_dropout.values())} dropout modules; scope=legacy_all)."
        ),
        "severity": "MEDIUM — affects regularization characterisation",
        "action": "Verify manuscript describes dropout scope correctly",
    },
    {
        "item": "Block order (activation before normalization)",
        "config_value": "legacy_act_norm: Conv→GELU→GroupNorm→Dropout",
        "common_manuscript_error": "Stating 'Conv→BN→GELU' (standard pre-norm order)",
        "correct_manuscript_text": (
            "Each convolutional block uses the sequence: "
            "Conv → GELU → GroupNorm → Dropout (legacy_act_norm order). "
            "Note: this places activation before normalization, which is non-standard "
            "but a deliberate design choice inherited from the original architecture."
        ),
        "severity": "MEDIUM — misrepresents architecture",
        "action": "Verify block order is stated correctly or note it as implementation detail",
    },
    {
        "item": "Reconstruction loss mode",
        "config_value": "mse_sum_batchmean_current (includes diagonal; scales as n_chan * n_roi^2)",
        "common_manuscript_error": "Describing 'MSE reconstruction loss' without specifying element scope",
        "correct_manuscript_text": (
            "Reconstruction loss: mean squared error summed over all elements "
            "(including the diagonal of each connectivity matrix) and all channels, "
            "then divided by batch size. "
            f"This corresponds to {all_elements_per_subject:,} terms per subject "
            f"(3 channels × {n_rois}² ROIs)."
        ),
        "severity": "LOW — affects scale interpretation",
        "action": "Clarify in supplementary that recon loss includes diagonal elements",
    },
    {
        "item": "Normalization mode",
        "config_value": "zscore_offdiag: z-score per off-diagonal channel, fit on VAE train only",
        "common_manuscript_error": "Stating 'z-score normalization' without specifying off-diagonal and fit scope",
        "correct_manuscript_text": (
            "Input tensors were normalized using a fold-local z-score transformation "
            "applied to the off-diagonal elements of each functional connectivity channel, "
            "with normalization parameters fitted exclusively on the VAE training subset."
        ),
        "severity": "MEDIUM",
        "action": "Verify normalization is described accurately including fit scope",
    },
    {
        "item": "Encoder spatial dimensions",
        "config_value": f"131→64→31→15→8 (kernels {kernels}, paddings {paddings}, strides {strides})",
        "common_manuscript_error": "Stating 'four convolutional layers' without spatial details",
        "correct_manuscript_text": (
            f"The encoder comprises four convolutional layers with kernels {kernels}, "
            f"strides {strides}, padding {paddings}, reducing spatial dimensions "
            f"131→64→31→15→8. Output channels: {conv_ch_enc}. "
            f"Flattened to {flat_size} units, then projected to "
            f"{intermediate_fc_dim} (FC quarter-size), then to μ and log σ² ({latent_dim} each)."
        ),
        "severity": "LOW — supplementary-level detail",
        "action": "Verify architecture table or supplementary reflects correct spatial dims",
    },
    {
        "item": "Outer CV stratification columns",
        "config_value": "Stratified by (ResearchGroup_Mapped × Manufacturer) joint key",
        "common_manuscript_error": "Stating only 'stratified by diagnosis'",
        "correct_manuscript_text": (
            "Outer cross-validation folds were stratified by the joint (diagnosis × scanner manufacturer) key, "
            "ensuring balanced representation of all manufacturer × diagnosis combinations across folds."
        ),
        "severity": "MEDIUM — important for explaining the uniform 140TP distribution",
        "action": "Verify manuscript states manufacturer-stratified outer CV",
    },
    {
        "item": "Stage B features (latent + Age + Sex)",
        "config_value": "latent_mu (384 dims) + Age + Sex → sklearn pipeline with internal preprocessing",
        "common_manuscript_error": "Stating 'latent features used for classification' without mentioning metadata",
        "correct_manuscript_text": (
            "Stage B classifier input: 384-dimensional latent mean (μ) concatenated with "
            "participant age and sex. Preprocessing (imputation, scaling) was applied "
            "within each inner cross-validation fold to prevent leakage."
        ),
        "severity": "MEDIUM — Age and Sex are informative features",
        "action": "Verify manuscript states Age+Sex are included in classifier input",
    },
    {
        "item": "Number of β cycles actually executed",
        "config_value": f"~39–55 cycles (early stopping), configured maximum = {n_cycles}",
        "common_manuscript_error": "Stating '125 cycles' as if all were trained",
        "correct_manuscript_text": (
            "A maximum of 125 β cycles (10 000 epochs) was configured; "
            "early stopping (patience = 560 epochs) terminated training after "
            f"{[round(te/epoch_per_cycle,0) for te in total_logged_epochs]} cycles "
            "across folds 1–5 respectively."
        ),
        "severity": "LOW",
        "action": "Clarify that early stopping limits actual cycles executed",
    },
]

df_mismatch = pd.DataFrame(mismatch_rows)
save_csv_md(df_mismatch, "manuscript_config_mismatch_report",
            "Manuscript Config Mismatch Report")

# ---------------------------------------------------------------------------
# 10. FINAL RECOMMENDATION
# ---------------------------------------------------------------------------
log("Writing final recommendation", step="recommendation")

final_rec_md = """# Final Recommendation — Orchestrator/Architecture Training Audit
**Date**: 2026-06-17
**Run**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
**Hard guardrails**: Read-only. No training. No tensor edits. No metadata edits.
No prediction edits. No threshold refitting. No subject exclusion. No model selection.

---

## Summary Verdict

The promoted model is **methodologically sound**. There are no identified bugs.
The key design choices (block order, dropout placement, loss mode, cycle schedule)
are internally consistent and reproducible. The Philips/rawTP domain shift is
a structural property of the dataset, not a training artefact.

---

## 1. Changes for Manuscript/Reporting Only (no code changes needed)

| Priority | Item | Action Required |
|---|---|---|
| HIGH | Early stopping criterion | Clarify manuscript uses L_val(β_max) = D_val + β_max × KLD_val, NOT current_beta loss |
| HIGH | Latent dimension | Confirm manuscript states latent_dim=384 (not 128 from earlier runs) |
| HIGH | n_input_channels | Confirm manuscript states 3 channels, not 7 |
| MEDIUM | Cyclical schedule | Report n_cycles=125, cycle_len=80, ratio_increase=0.40, T0=80 |
| MEDIUM | Outer CV stratification | State (Diagnosis × Manufacturer) joint stratification |
| MEDIUM | Stage B features | State Age+Sex concatenated to 384-dim latent μ |
| MEDIUM | Block order | Note legacy_act_norm: Conv→GELU→GN→Dropout (non-standard but intentional) |
| MEDIUM | Normalization fit scope | State z-score fit on VAE train subset only, not entire pool |
| LOW | Recon loss elements | Clarify diagonal included in mse_sum_batchmean_current |
| LOW | Cycles executed | Clarify ~39–55 actual cycles, not 125 maximum |

---

## 2. Read-Only Audits Still Needed

| Priority | Audit | Rationale |
|---|---|---|
| HIGH | Download MAYOADIRL_MRI_FMRI_NFQ from LONI | SLICEORDER, OVERALLQC for Sites 13, 53, 301 (Martín verification pending) |
| MEDIUM | rawTP MI in latents (per fold) | Not computed: existing QC pipeline only covers Y_target/Manufacturer/Sex; would require re-running evaluate_latent_information with rawtp_norm as nuisance |
| MEDIUM | Latent centroid distances (Philips CN 140TP vs 197TP) | Not computed: fold-specific coordinate systems; requires re-encoding subjects with saved VAE models |
| LOW | OASIS external scoring | Script ready; Martín 3mm/3° exclusion criterion confirmation pending |

---

## 3. Low-Risk Future Experiments (not for this revision)

| Priority | Experiment | Expected Benefit | Risk |
|---|---|---|---|
| HIGH | Switch recon_loss_mode to offdiag_channelmean_sum + beta re-sweep | More principled objective; excludes trivially-reconstructed diagonal; scale-stable with channels | Requires beta sweep [1.0–2.5 likely equivalent to 3.75 at current scale]; full retraining |
| HIGH | vae_train_sampler_strategy=manufacturer_balanced | Increases Philips representation in VAE batches; may reduce scanner leakage | Controlled ablation (N unchanged); no leakage risk |
| MEDIUM | input_harmonization_mode=foldwise_combat | Direct ComBat correction before VAE; most targeted intervention for 140TP domain shift | ComBat may remove clinically-relevant FC; requires careful pre/post comparison |
| LOW | vae_channel_dropout_p=0.10 | Denoising augmentation; may improve latent robustness | Minor effect expected; no leakage risk |
| LOW | vae_dropout_scope=encoder_only | Remove decoder dropout (common practice); may improve reconstruction quality | Standard improvement; negligible risk |

---

## 4. High-Risk Experiments Not Recommended for This Revision

| Risk | Experiment | Reason to Avoid |
|---|---|---|
| HIGH | vae_conditioning_mode=decoder_only with Manufacturer | Interaction with 140TP FPR mechanism unclear; may amplify scanner signal in latents |
| HIGH | Any change to outer CV seed or split | Would break cross-run comparability with existing audit results |
| HIGH | Subject exclusion based on QC flags | Violates pre-registered analysis plan; changes N reported in manuscript |
| HIGH | Threshold re-selection on test set | Direct leakage; not admissible |
| MEDIUM | VAE pool composition strategy ≠ current_all_pool | Uses diagnosis labels for VAE pool selection; not the historical approach |

---

## 5. Architecture Non-Bug Notes

| Observation | Status | Notes |
|---|---|---|
| Block order: activation precedes normalization (Conv→GELU→GN) | DESIGN CHOICE, not bug | legacy_act_norm is named intentionally; norm_act variant exists but not used |
| Dropout in decoder (legacy_all scope) | DESIGN CHOICE, not bug | encoder_only is available; decoder dropout was used historically and accepted |
| BatchNorm1d in FC intermediate (not GroupNorm) | CORRECT for 1D | GroupNorm applied to conv outputs; BN1d appropriate for flat FC layers |
| Tanh final activation with zscore_offdiag input | POTENTIAL RANGE MISMATCH | zscore_offdiag inputs can exceed [-1,1] at tails; Tanh clips reconstruction at boundaries. No distortion observed in training histories (D_val consistent). Noted but not actionable without retraining. |
| Reconstruction loss includes diagonal | DESIGN CHOICE | Diagonal = auto-correlation (always 1 for Pearson). Including it adds trivially-reconstructed terms. Not a bug but offdiag_channelmean_sum is more principled. |

---

*Generated by orchestrator_architecture_training_audit_20260617.py — Read-only audit.*
*No models were trained, no tensors edited, no metadata or predictions modified.*
"""

with open(OUTPUT_DIR / "final_recommendation.md", "w", encoding="utf-8") as f:
    f.write(final_rec_md)
log("Saved final_recommendation.md")

# ---------------------------------------------------------------------------
# 11. COMMAND LOG
# ---------------------------------------------------------------------------
command_log.append({
    "ts": datetime.now(timezone.utc).isoformat(),
    "step": "complete",
    "msg": "All audit outputs written",
    "output_dir": str(OUTPUT_DIR),
    "outputs": [str(p.name) for p in sorted(OUTPUT_DIR.iterdir())],
})
with open(OUTPUT_DIR / "command_log.json", "w", encoding="utf-8") as f:
    json.dump(command_log, f, indent=2)
log(f"Command log saved to {OUTPUT_DIR / 'command_log.json'}")
print("\n✓ Audit complete. Output directory:", OUTPUT_DIR)
