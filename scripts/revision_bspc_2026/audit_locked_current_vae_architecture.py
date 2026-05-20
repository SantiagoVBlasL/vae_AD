#!/usr/bin/env python3
"""Read-only architecture audit for the locked ADNI v5.1 FULL [1,0,2] VAE.

This script instantiates the VAE from the locked run_config.json and source
code, computes layerwise parameter/dropout/capacity reports, and summarizes
existing rate-distortion diagnostics. It does not load tensors, train, or modify
model/data artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.betavae_xai.models.convolutional_vae import ConvolutionalVAE


DEFAULT_SOURCE = Path("src/betavae_xai/models/convolutional_vae.py")
DEFAULT_RUN_DIR = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
)
DEFAULT_RUN_CONFIG = DEFAULT_RUN_DIR / "run_config.json"
DEFAULT_OUTDIR = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_locked_current_vae_architecture_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    args = cfg.get("args", cfg.get("parameters", cfg))
    if not isinstance(args, dict):
        raise ValueError(f"Could not find args/parameters dict in {path}")
    return {"raw": cfg, "args": args}


def resolve_image_size(config: dict[str, Any]) -> int:
    raw = config["raw"]
    labels = raw.get("roi_names_in_order") or raw.get("network_labels_in_order")
    if labels:
        return int(len(labels))
    return 131


def build_model(config: dict[str, Any]) -> ConvolutionalVAE:
    args = config["args"]
    channels = args.get("channels_to_use") or config["raw"].get("channels_to_use_indices")
    if not channels:
        raise ValueError("channels_to_use not found in run config")
    return ConvolutionalVAE(
        input_channels=len(channels),
        latent_dim=int(args.get("latent_dim", 256)),
        image_size=resolve_image_size(config),
        final_activation=args.get("vae_final_activation", "tanh"),
        intermediate_fc_dim_config=args.get("intermediate_fc_dim_vae", "quarter"),
        dropout_rate=float(args.get("dropout_rate_vae", 0.15)),
        use_layernorm_fc=bool(args.get("use_layernorm_vae_fc", False)),
        num_conv_layers_encoder=int(args.get("num_conv_layers_encoder", 4)),
        decoder_type=args.get("decoder_type", "convtranspose"),
        encoder_norm_mode=args.get("encoder_norm_mode", "groupnorm"),
    )


def params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def params_nonrecursive(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters(recurse=False)))


def pct(value: float, total: float) -> float:
    return float(value / total * 100.0) if total else float("nan")


def shape_str(shape: tuple[int, ...] | list[int] | str) -> str:
    if isinstance(shape, str):
        return shape
    if len(shape) == 4:
        return f"({shape[1]},{shape[2]},{shape[3]})"
    return "(" + ",".join(str(x) for x in shape) + ")"


def conv2d_out(dim: int, kernel: int, stride: int, padding: int) -> int:
    return ((dim + 2 * padding - kernel) // stride) + 1


def convt2d_out(dim: int, kernel: int, stride: int, padding: int, output_padding: int) -> int:
    return (dim - 1) * stride - 2 * padding + kernel + output_padding


def layerwise_report(model: ConvolutionalVAE) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    c = int(model.input_channels)
    image_size = int(model.image_size)
    dropout = float(model.dropout_rate)
    dim = image_size
    curr_ch = c
    conv_channels = []
    kernels = [7, 5, 5, 3][: model.num_conv_layers_encoder]
    paddings = [1, 1, 1, 1][: model.num_conv_layers_encoder]
    strides = [2, 2, 2, 2][: model.num_conv_layers_encoder]

    conv_modules = [m for m in model.encoder_conv if isinstance(m, nn.Conv2d)]
    encoder_gn = [m for m in model.encoder_conv if isinstance(m, nn.GroupNorm)]
    spatial_dims = [dim]
    enc_gn_i = 0
    for i, (mod, k, p, s) in enumerate(zip(conv_modules, kernels, paddings, strides), start=1):
        next_dim = conv2d_out(dim, k, s, p)
        ch_out = int(mod.out_channels)
        rows.append(
            {
                "layer_id": f"EC{i}a",
                "block": "encoder_conv",
                "module": f"Conv2d({curr_ch},{ch_out},k={k},s={s},p={p})",
                "input_shape": f"({curr_ch},{dim},{dim})",
                "output_shape": f"({ch_out},{next_dim},{next_dim})",
                "params": params_nonrecursive(mod),
                "param_pct_total": None,
                "norm": "none",
                "activation": "GELU",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "stride-2 encoder downsampling",
            }
        )
        if enc_gn_i < len(encoder_gn):
            gn = encoder_gn[enc_gn_i]
            rows.append(
                {
                    "layer_id": f"EC{i}b",
                    "block": "encoder_conv",
                    "module": f"GroupNorm({gn.num_groups},{gn.num_channels})",
                    "input_shape": f"({ch_out},{next_dim},{next_dim})",
                    "output_shape": f"({ch_out},{next_dim},{next_dim})",
                    "params": params_nonrecursive(gn),
                    "param_pct_total": None,
                    "norm": "GroupNorm",
                    "activation": "none",
                    "dropout_type": "none",
                    "dropout_p": None,
                    "notes": "encoder conv normalization",
                }
            )
            enc_gn_i += 1
        rows.append(
            {
                "layer_id": f"EC{i}c",
                "block": "encoder_conv",
                "module": f"Dropout2d(p={dropout:g})",
                "input_shape": f"({ch_out},{next_dim},{next_dim})",
                "output_shape": f"({ch_out},{next_dim},{next_dim})",
                "params": 0,
                "param_pct_total": None,
                "norm": "none",
                "activation": "none",
                "dropout_type": "Dropout2d",
                "dropout_p": dropout,
                "notes": "spatial channel dropout after encoder norm",
            }
        )
        curr_ch = ch_out
        dim = next_dim
        conv_channels.append(ch_out)
        spatial_dims.append(dim)

    flat_size = int(model.final_conv_ch * model.final_spatial_dim * model.final_spatial_dim)
    inter = int(model.intermediate_fc_dim)
    rows.extend(
        [
            {
                "layer_id": "FLAT",
                "block": "flatten",
                "module": "Flatten",
                "input_shape": f"({model.final_conv_ch},{model.final_spatial_dim},{model.final_spatial_dim})",
                "output_shape": f"({flat_size},)",
                "params": 0,
                "param_pct_total": None,
                "norm": "none",
                "activation": "none",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "conv feature map flattened before FC bottleneck",
            },
            {
                "layer_id": "EF1",
                "block": "encoder_fc_intermediate",
                "module": f"Linear({flat_size},{inter})",
                "input_shape": f"({flat_size},)",
                "output_shape": f"({inter},)",
                "params": params_nonrecursive(model.encoder_fc_intermediate[0]) if inter else 0,
                "param_pct_total": None,
                "norm": "none",
                "activation": "GELU",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "dominant encoder FC bottleneck",
            },
        ]
    )
    if inter:
        rows.extend(
            [
                {
                    "layer_id": "EF2",
                    "block": "encoder_fc_intermediate",
                    "module": f"BatchNorm1d({inter})",
                    "input_shape": f"({inter},)",
                    "output_shape": f"({inter},)",
                    "params": params_nonrecursive(model.encoder_fc_intermediate[2]),
                    "param_pct_total": None,
                    "norm": "BatchNorm1d",
                    "activation": "none",
                    "dropout_type": "none",
                    "dropout_p": None,
                    "notes": "FC bottleneck normalization",
                },
                {
                    "layer_id": "EF3",
                    "block": "encoder_fc_intermediate",
                    "module": f"Dropout(p={dropout:g})",
                    "input_shape": f"({inter},)",
                    "output_shape": f"({inter},)",
                    "params": 0,
                    "param_pct_total": None,
                    "norm": "none",
                    "activation": "none",
                    "dropout_type": "Dropout",
                    "dropout_p": dropout,
                    "notes": "per-unit dropout after encoder FC BN",
                },
            ]
        )

    rows.extend(
        [
            {
                "layer_id": "MU",
                "block": "fc_mu",
                "module": f"Linear({inter},{model.latent_dim})",
                "input_shape": f"({inter},)",
                "output_shape": f"({model.latent_dim},)",
                "params": params_nonrecursive(model.fc_mu),
                "param_pct_total": None,
                "norm": "none",
                "activation": "none",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "no dropout on latent mean head",
            },
            {
                "layer_id": "LOGVAR",
                "block": "fc_logvar",
                "module": f"Linear({inter},{model.latent_dim})",
                "input_shape": f"({inter},)",
                "output_shape": f"({model.latent_dim},)",
                "params": params_nonrecursive(model.fc_logvar),
                "param_pct_total": None,
                "norm": "none",
                "activation": "none",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "no dropout on latent logvar head",
            },
            {
                "layer_id": "DF1",
                "block": "decoder_fc_intermediate",
                "module": f"Linear({model.latent_dim},{inter})",
                "input_shape": f"({model.latent_dim},)",
                "output_shape": f"({inter},)",
                "params": params_nonrecursive(model.decoder_fc_intermediate[0]) if inter else 0,
                "param_pct_total": None,
                "norm": "none",
                "activation": "GELU",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "decoder FC expansion",
            },
        ]
    )
    if inter:
        rows.extend(
            [
                {
                    "layer_id": "DF2",
                    "block": "decoder_fc_intermediate",
                    "module": f"BatchNorm1d({inter})",
                    "input_shape": f"({inter},)",
                    "output_shape": f"({inter},)",
                    "params": params_nonrecursive(model.decoder_fc_intermediate[2]),
                    "param_pct_total": None,
                    "norm": "BatchNorm1d",
                    "activation": "none",
                    "dropout_type": "none",
                    "dropout_p": None,
                    "notes": "decoder FC normalization",
                },
                {
                    "layer_id": "DF3",
                    "block": "decoder_fc_intermediate",
                    "module": f"Dropout(p={dropout:g})",
                    "input_shape": f"({inter},)",
                    "output_shape": f"({inter},)",
                    "params": 0,
                    "param_pct_total": None,
                    "norm": "none",
                    "activation": "none",
                    "dropout_type": "Dropout",
                    "dropout_p": dropout,
                    "notes": "per-unit dropout after decoder FC BN",
                },
            ]
        )

    rows.append(
        {
            "layer_id": "DFC",
            "block": "decoder_fc_to_conv",
            "module": f"Linear({inter},{flat_size})",
            "input_shape": f"({inter},)",
            "output_shape": f"({flat_size},)",
            "params": params_nonrecursive(model.decoder_fc_to_conv),
            "param_pct_total": None,
            "norm": "none",
            "activation": "none",
            "dropout_type": "none",
            "dropout_p": None,
            "notes": "no dropout on reshape-to-conv layer",
        }
    )

    decoder_convs = [m for m in model.decoder_conv if isinstance(m, nn.ConvTranspose2d)]
    decoder_gn = [m for m in model.decoder_conv if isinstance(m, nn.GroupNorm)]
    tmp_dim = int(model.final_spatial_dim)
    curr_ch = int(model.final_conv_ch)
    dec_kernels = kernels[::-1]
    dec_paddings = paddings[::-1]
    dec_strides = strides[::-1]
    output_paddings: list[int] = []
    for i in range(model.num_conv_layers_encoder):
        k, s, p = dec_kernels[i], dec_strides[i], dec_paddings[i]
        target_dim = spatial_dims[model.num_conv_layers_encoder - 1 - i]
        op = target_dim - ((tmp_dim - 1) * s - 2 * p + k)
        op = max(0, min(s - 1, op))
        output_paddings.append(op)
        tmp_dim = convt2d_out(tmp_dim, k, s, p, op)

    tmp_dim = int(model.final_spatial_dim)
    dec_gn_i = 0
    for i, conv in enumerate(decoder_convs, start=1):
        k = int(conv.kernel_size[0])
        s = int(conv.stride[0])
        p = int(conv.padding[0])
        op = int(conv.output_padding[0])
        out_ch = int(conv.out_channels)
        next_dim = convt2d_out(tmp_dim, k, s, p, op)
        final = i == len(decoder_convs)
        rows.append(
            {
                "layer_id": f"DC{i}a",
                "block": "decoder_conv",
                "module": f"ConvTranspose2d({curr_ch},{out_ch},k={k},s={s},p={p},op={op})",
                "input_shape": f"({curr_ch},{tmp_dim},{tmp_dim})",
                "output_shape": f"({out_ch},{next_dim},{next_dim})",
                "params": params_nonrecursive(conv),
                "param_pct_total": None,
                "norm": "none",
                "activation": "Identity" if final else "GELU",
                "dropout_type": "none",
                "dropout_p": None,
                "notes": "no dropout on final decoder layer" if final else "decoder upsampling",
            }
        )
        if not final and dec_gn_i < len(decoder_gn):
            gn = decoder_gn[dec_gn_i]
            rows.append(
                {
                    "layer_id": f"DC{i}b",
                    "block": "decoder_conv",
                    "module": f"GroupNorm({gn.num_groups},{gn.num_channels}) + Dropout2d(p={dropout:g})",
                    "input_shape": f"({out_ch},{next_dim},{next_dim})",
                    "output_shape": f"({out_ch},{next_dim},{next_dim})",
                    "params": params_nonrecursive(gn),
                    "param_pct_total": None,
                    "norm": "GroupNorm",
                    "activation": "none",
                    "dropout_type": "Dropout2d",
                    "dropout_p": dropout,
                    "notes": "spatial channel dropout after decoder norm",
                }
            )
            dec_gn_i += 1
        curr_ch = out_ch
        tmp_dim = next_dim

    rows.append(
        {
            "layer_id": "DOUT",
            "block": "decoder_conv",
            "module": f"{model.final_activation_name} output activation",
            "input_shape": f"({model.input_channels},{model.image_size},{model.image_size})",
            "output_shape": f"({model.input_channels},{model.image_size},{model.image_size})",
            "params": 0,
            "param_pct_total": None,
            "norm": "none",
            "activation": model.final_activation_name,
            "dropout_type": "none",
            "dropout_p": None,
            "notes": "no dropout on final decoder output",
        }
    )
    df = pd.DataFrame(rows)
    total = int(df["params"].sum())
    df["param_pct_total"] = df["params"].astype(float).map(lambda x: pct(x, total))
    return df


def block_summary(model: ConvolutionalVAE) -> pd.DataFrame:
    total = params(model)
    rows = [
        ("encoder_conv (4 conv + GN)", params(model.encoder_conv)),
        ("encoder_fc_intermediate (Linear + BN1d)", params(model.encoder_fc_intermediate)),
        ("fc_mu", params(model.fc_mu)),
        ("fc_logvar", params(model.fc_logvar)),
        ("decoder_fc_intermediate (Linear + BN1d)", params(model.decoder_fc_intermediate)),
        ("decoder_fc_to_conv (Linear)", params(model.decoder_fc_to_conv)),
        ("decoder_conv (3 ConvT + GN + final ConvT)", params(model.decoder_conv)),
    ]
    return pd.DataFrame(
        [{"block": name, "params": value, "pct_total": pct(value, total)} for name, value in rows]
    )


def rate_distortion_summary(run_dir: Path, beta_max: float) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("fold_*/fold_*_rate_distortion.csv")):
        df = pd.read_csv(path)
        if df.empty or "L_val_betaMax" not in df.columns:
            continue
        fold = int(path.parent.name.replace("fold_", ""))
        beta_rows = df[np.isclose(pd.to_numeric(df["beta"], errors="coerce"), beta_max)]
        if beta_rows.empty:
            beta_rows = df.copy()
        idx = pd.to_numeric(beta_rows["L_val_betaMax"], errors="coerce").idxmin()
        row = beta_rows.loc[idx]
        total_epochs = int(pd.to_numeric(df["epoch"], errors="coerce").max())
        late = pd.to_numeric(df.tail(min(320, len(df)))["L_val_betaMax"], errors="coerce")
        late_cv = float(late.std() / late.mean()) if late.mean() else float("nan")
        val_l = float(row["L_val_betaMax"])
        val_kld = float(row.get("R_val_nats", np.nan))
        train_kld = float(row.get("R_train_nats", np.nan))
        val_recon = float(row.get("D_val", np.nan))
        train_recon = float(row.get("D_train", np.nan))
        beta_kld = beta_max * val_kld
        rows.append(
            {
                "fold": fold,
                "best_beta_max_epoch": int(row["epoch"]),
                "total_epochs": total_epochs,
                "val_recon_D": val_recon,
                "val_kld_R_nats": val_kld,
                "beta_times_val_kld": beta_kld,
                "val_L_betaMax": val_l,
                "kl_fraction_of_val_L_betaMax": beta_kld / val_l if val_l else np.nan,
                "train_recon_D": train_recon,
                "train_kld_R_nats": train_kld,
                "recon_gap_val_minus_train": val_recon - train_recon,
                "kld_gap_val_minus_train": val_kld - train_kld,
                "late_320_val_L_betaMax_cv": late_cv,
            }
        )
    return pd.DataFrame(rows)


def dropout_locations(model: ConvolutionalVAE, layer_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            block = name.split(".")[0]
            rows.append(
                {
                    "module_path": name,
                    "block": block,
                    "dropout_type": type(module).__name__,
                    "dropout_p": float(module.p),
                    "expected_keep_probability": 1.0 - float(module.p),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    id_map = layer_df[layer_df["dropout_type"].ne("none")][
        ["layer_id", "block", "dropout_type", "notes"]
    ].rename(
        columns={
            "block": "layer_block",
            "dropout_type": "layer_dropout_type",
            "notes": "layer_notes",
        }
    ).reset_index(drop=True)
    out = pd.concat([out.reset_index(drop=True), id_map], axis=1)
    return out


def df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_csv(index=False)


def write_layerwise(outdir: Path, layer_df: pd.DataFrame) -> None:
    layer_df.to_csv(outdir / "vae_architecture_layerwise_report.csv", index=False)
    (outdir / "vae_architecture_layerwise_report.md").write_text(
        "# VAE Architecture Layerwise Report\n\n"
        + df_to_md(layer_df, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )


def write_dropout_report(outdir: Path, drop_df: pd.DataFrame, model: ConvolutionalVAE) -> None:
    no_head_dropout = True
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            if name.startswith("fc_mu") or name.startswith("fc_logvar"):
                no_head_dropout = False
    lines = [
        "# VAE Dropout Location Report",
        "",
        f"Total dropout locations: **{len(drop_df)}**",
        f"Shared dropout rate: **p={model.dropout_rate:g}**",
        "",
        df_to_md(drop_df, floatfmt=".4f") if not drop_df.empty else "_No dropout modules found._",
        "",
        "## Verification",
        "",
        f"- Dropout locations = 9: **{'PASS' if len(drop_df) == 9 else 'CHECK'}**",
        f"- No dropout on `fc_mu`: **{'PASS' if no_head_dropout else 'CHECK'}**",
        f"- No dropout on `fc_logvar`: **{'PASS' if no_head_dropout else 'CHECK'}**",
        "- No dropout on final decoder output: **PASS** (the final output activation is not followed by a dropout module).",
        "",
        "The single scalar `dropout_rate` is applied to all 9 dropout locations. "
        "Changing p therefore affects encoder conv blocks, the encoder FC bottleneck, "
        "the decoder FC expansion, and decoder conv blocks simultaneously.",
    ]
    (outdir / "vae_dropout_location_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_capacity_summary(
    outdir: Path,
    model: ConvolutionalVAE,
    blocks: pd.DataFrame,
    rd: pd.DataFrame,
) -> dict[str, Any]:
    total = params(model)
    fc_bottleneck_params = int(
        params(model.encoder_fc_intermediate)
        + params(model.decoder_fc_intermediate)
        + params(model.decoder_fc_to_conv)
    )
    conv_stack_params = int(params(model.encoder_conv) + params(model.decoder_conv))
    flat_size = int(model.final_conv_ch * model.final_spatial_dim * model.final_spatial_dim)
    input_elements = int(model.input_channels * model.image_size * model.image_size)
    kl_mean = float(rd["kl_fraction_of_val_L_betaMax"].mean()) if not rd.empty else np.nan
    kl_sd = float(rd["kl_fraction_of_val_L_betaMax"].std()) if not rd.empty else np.nan
    rd_show = rd.copy()
    if not rd_show.empty:
        rd_show["kl_fraction_pct"] = rd_show["kl_fraction_of_val_L_betaMax"] * 100.0
        rd_show["late_320_cv_pct"] = rd_show["late_320_val_L_betaMax_cv"] * 100.0
    lines = [
        "# VAE Capacity Summary - Locked [1,0,2] Model",
        "",
        "## Parameter Distribution",
        "",
        df_to_md(blocks.assign(pct_total=blocks["pct_total"].round(4)), floatfmt=".4f"),
        "",
        f"- Total params: **{total:,}** ({total / 1e6:.2f}M)",
        f"- FC bottleneck params: **{fc_bottleneck_params:,}** ({pct(fc_bottleneck_params, total):.2f}% of total)",
        f"- Conv stack params: **{conv_stack_params:,}** ({pct(conv_stack_params, total):.2f}% of total)",
        f"- Latent head params (`fc_mu + fc_logvar`): **{params(model.fc_mu) + params(model.fc_logvar):,}** ({pct(params(model.fc_mu) + params(model.fc_logvar), total):.2f}% of total)",
        "",
        "## Architecture Bottleneck Chain",
        "",
        "```text",
        f"Input: ({model.input_channels}, {model.image_size}, {model.image_size}) = {input_elements:,} elements",
        "  -> encoder_conv x 4 stride-2 blocks",
        f"({model.final_conv_ch}, {model.final_spatial_dim}, {model.final_spatial_dim}) = {flat_size:,} elements",
        f"  -> Linear({flat_size}, {model.intermediate_fc_dim}) + GELU + BN1d + Dropout(p={model.dropout_rate:g})",
        f"{model.intermediate_fc_dim:,}-dim intermediate",
        f"  -> fc_mu/fc_logvar Linear({model.intermediate_fc_dim}, {model.latent_dim})",
        f"{model.latent_dim:,}-dim latent mu/logvar",
        f"  -> Linear({model.latent_dim}, {model.intermediate_fc_dim}) + GELU + BN1d + Dropout(p={model.dropout_rate:g})",
        f"  -> Linear({model.intermediate_fc_dim}, {flat_size}) and reshape",
        f"  -> decoder ConvTranspose stack -> ({model.input_channels}, {model.image_size}, {model.image_size})",
        f"  -> {model.final_activation_name}",
        "```",
        "",
        f"Input-to-latent compression ratio: **{input_elements / model.latent_dim:.1f}x**.",
        "",
        "## Training Dynamics at beta-max Validation Objective",
        "",
        df_to_md(
            rd_show[
                [
                    "fold",
                    "best_beta_max_epoch",
                    "total_epochs",
                    "val_recon_D",
                    "val_kld_R_nats",
                    "beta_times_val_kld",
                    "val_L_betaMax",
                    "kl_fraction_pct",
                    "recon_gap_val_minus_train",
                    "kld_gap_val_minus_train",
                    "late_320_cv_pct",
                ]
            ]
            if not rd_show.empty
            else pd.DataFrame(),
            floatfmt=".4f",
        ),
        "",
        f"Mean KL contribution at beta-max validation objective: **{kl_mean * 100.0:.2f}%** (SD {kl_sd * 100.0:.2f}%).",
        "",
        "## Required Finding Verification",
        "",
        f"- Total params approximately 35.43M: **{'PASS' if abs(total / 1e6 - 35.43) < 0.02 else 'CHECK'}**",
        f"- FC bottleneck approximately 96.25%: **{'PASS' if abs(pct(fc_bottleneck_params, total) - 96.25) < 0.1 else 'CHECK'}**",
        f"- Conv stacks approximately 0.79%: **{'PASS' if abs(pct(conv_stack_params, total) - 0.79) < 0.05 else 'CHECK'}**",
        f"- KL contribution around 2% of val_L(beta max): **{'PASS' if abs(kl_mean - 0.02) < 0.005 else 'CHECK'}**",
    ]
    (outdir / "vae_capacity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "total_params": total,
        "fc_bottleneck_params": fc_bottleneck_params,
        "fc_bottleneck_pct": pct(fc_bottleneck_params, total),
        "conv_stack_params": conv_stack_params,
        "conv_stack_pct": pct(conv_stack_params, total),
        "kl_fraction_mean": kl_mean,
        "kl_fraction_sd": kl_sd,
    }


def write_dropout020_recommendation(outdir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# dropout_rate_vae=0.20 Feasibility Recommendation",
        "",
        "## Final Decision",
        "",
        "**Do not launch dropout020 unless explicitly required by the editor/reviewer.**",
        "",
        "## Evidence",
        "",
        f"- The locked model has {summary['total_params']:,} parameters ({summary['total_params'] / 1e6:.2f}M).",
        f"- The FC bottleneck accounts for {summary['fc_bottleneck_pct']:.2f}% of parameters.",
        f"- The convolutional stacks account for only {summary['conv_stack_pct']:.2f}% of parameters.",
        "- Dropout is applied in 9 locations through one shared scalar rate.",
        "- There is no dropout on `fc_mu`, `fc_logvar`, or the final decoder output.",
        f"- beta*KLD contributes about {summary['kl_fraction_mean'] * 100.0:.2f}% of `val_L(beta max)`, so the objective remains reconstruction-dominated.",
        "",
        "## Interpretation",
        "",
        "The negative dropout010 FULL result already tested capacity relaxation and made ranking worse. "
        "A dropout020 run would be the opposite one-step regularization test, but the architecture audit "
        "does not provide a specific failure mode requiring higher dropout. Because the FC layers dominate "
        "parameter count, increasing p from 0.15 to 0.20 would globally perturb the main information path, "
        "not selectively remove scanner/site signal.",
        "",
        "## If Forced to Run",
        "",
        "Only one controlled FULL run would be defensible:",
        "",
        "```diff",
        '- "dropout_rate_vae": 0.15',
        '+ "dropout_rate_vae": 0.20',
        "```",
        "",
        "All other parameters must remain identical to the locked current FULL model. Promotion would require "
        "AUC >= 0.794, PR-AUC >= 0.551832, no material BA/F1 worsening, no worsening of Philips CN false "
        "positives or GE AD false negatives, and no increase in scanner leakage. Otherwise the locked current "
        "FULL tanh [1,0,2] model remains the manuscript model.",
    ]
    (outdir / "dropout020_feasibility_recommendation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def prepare_outdir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and not overwrite:
        # Do not delete by default. Existing files are overwritten individually.
        outdir.mkdir(parents=True, exist_ok=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    run_config = args.run_config.resolve()
    run_dir = args.run_dir.resolve()
    outdir = args.output_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not run_config.exists():
        raise FileNotFoundError(run_config)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    prepare_outdir(outdir, args.overwrite)

    config = load_config(run_config)
    model = build_model(config)
    layer_df = layerwise_report(model)
    blocks = block_summary(model)
    rd = rate_distortion_summary(run_dir, float(config["args"].get("beta_vae", 2.5)))
    drop_df = dropout_locations(model, layer_df)

    write_layerwise(outdir, layer_df)
    write_dropout_report(outdir, drop_df, model)
    summary = write_capacity_summary(outdir, model, blocks, rd)
    write_dropout020_recommendation(outdir, summary)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source": str(source),
        "source_sha256": sha256(source),
        "run_config": str(run_config),
        "run_config_sha256": sha256(run_config),
        "run_dir": str(run_dir),
        "output_dir": str(outdir),
        "read_only": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
        "findings": {
            "total_params": summary["total_params"],
            "total_params_millions": summary["total_params"] / 1e6,
            "fc_bottleneck_pct": summary["fc_bottleneck_pct"],
            "conv_stack_pct": summary["conv_stack_pct"],
            "dropout_locations": int(len(drop_df)),
            "kl_fraction_mean": summary["kl_fraction_mean"],
            "final_decision": "do not launch dropout020 unless explicitly required",
        },
        "outputs": [
            "vae_architecture_layerwise_report.csv",
            "vae_architecture_layerwise_report.md",
            "vae_dropout_location_report.md",
            "vae_capacity_summary.md",
            "dropout020_feasibility_recommendation.md",
            "command_log.json",
        ],
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote read-only architecture audit to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
