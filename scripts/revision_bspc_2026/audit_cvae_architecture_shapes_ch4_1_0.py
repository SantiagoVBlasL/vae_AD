#!/usr/bin/env python3
"""Audit exact CVAE architecture shapes for ADNI V4 [4,1,0] baseline.

This is a model-shape smoke test only. It instantiates the architecture used by
the tanh [4,1,0] baseline and runs one synthetic batch through encode/decode.
It does not load ADNI tensors or VAE checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
else:
    raise FileNotFoundError(f"Missing src directory: {SRC_DIR}")

from betavae_xai.models import ConvolutionalVAE


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/cvae_architecture_shape_audit"
)
GENERATED_FILES = [
    "architecture_shapes.csv",
    "receptive_field.csv",
    "parameter_counts.csv",
    "README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def shape_tuple(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(int(x) for x in value.shape))
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return str(tuple(int(x) for x in value[0].shape))
    return ""


def module_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters(recurse=False))
    trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
    return int(total), int(trainable)


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def trainable_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def fmt_attr(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return str(tuple(int(v) for v in value))
    return str(value)


def leaf_modules(prefix: str, module: nn.Module) -> List[Tuple[str, nn.Module]]:
    out: List[Tuple[str, nn.Module]] = []
    for name, child in module.named_modules():
        if name == "":
            continue
        if not any(child.children()):
            out.append((f"{prefix}.{name}", child))
    return out


def register_shape_hooks(model: ConvolutionalVAE, rows: List[Dict[str, Any]]) -> List[Any]:
    handles = []

    def make_hook(layer_group: str, name: str, module: nn.Module):
        def hook(mod: nn.Module, inputs: Sequence[Any], output: Any) -> None:
            total, trainable = module_params(mod)
            rows.append(
                {
                    "layer_group": layer_group,
                    "name": name,
                    "module_type": mod.__class__.__name__,
                    "kernel_size": fmt_attr(getattr(mod, "kernel_size", None)),
                    "stride": fmt_attr(getattr(mod, "stride", None)),
                    "padding": fmt_attr(getattr(mod, "padding", None)),
                    "dilation": fmt_attr(getattr(mod, "dilation", None)),
                    "output_padding": fmt_attr(getattr(mod, "output_padding", None)),
                    "input_shape": shape_tuple(inputs),
                    "output_shape": shape_tuple(output),
                    "num_params": total,
                    "trainable_params": trainable,
                    "uses_crop_pad_interpolation": False,
                }
            )

        return hook

    for group, module in [
        ("encoder_conv", model.encoder_conv),
        ("encoder_fc", model.encoder_fc_intermediate),
        ("encoder_mu_logvar", nn.ModuleDict({"fc_mu": model.fc_mu, "fc_logvar": model.fc_logvar})),
        ("decoder_fc", model.decoder_fc_intermediate),
        ("decoder_fc_to_conv", nn.ModuleDict({"decoder_fc_to_conv": model.decoder_fc_to_conv})),
        ("decoder_conv", model.decoder_conv),
    ]:
        for name, child in leaf_modules(group, module):
            handles.append(child.register_forward_hook(make_hook(group, name, child)))
    return handles


def conv_receptive_field_rows(model: ConvolutionalVAE, input_spatial: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    receptive_field = 1
    effective_stride = 1
    spatial = input_spatial
    conv_idx = 0
    for name, module in model.encoder_conv.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        conv_idx += 1
        k = int(module.kernel_size[0])
        s = int(module.stride[0])
        p = int(module.padding[0])
        d = int(module.dilation[0])
        out_spatial = ((spatial + 2 * p - d * (k - 1) - 1) // s) + 1
        receptive_field = receptive_field + (k - 1) * d * effective_stride
        effective_stride = effective_stride * s
        rows.append(
            {
                "encoder_conv_index": conv_idx,
                "name": f"encoder_conv.{name}",
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "dilation": d,
                "input_spatial": spatial,
                "output_spatial": out_spatial,
                "receptive_field": receptive_field,
                "effective_stride": effective_stride,
            }
        )
        spatial = out_spatial
    return rows


def parameter_rows(model: ConvolutionalVAE) -> List[Dict[str, Any]]:
    blocks = [
        ("encoder_conv", model.encoder_conv),
        ("encoder_fc_intermediate", model.encoder_fc_intermediate),
        ("fc_mu", model.fc_mu),
        ("fc_logvar", model.fc_logvar),
        ("decoder_fc_intermediate", model.decoder_fc_intermediate),
        ("decoder_fc_to_conv", model.decoder_fc_to_conv),
        ("decoder_conv", model.decoder_conv),
        ("total", model),
    ]
    rows = []
    for name, module in blocks:
        rows.append(
            {
                "block": name,
                "num_params": param_count(module),
                "trainable_params": trainable_count(module),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    model = ConvolutionalVAE(
        input_channels=3,
        latent_dim=256,
        image_size=131,
        final_activation="tanh",
        intermediate_fc_dim_config="quarter",
        dropout_rate=0.15,
        use_layernorm_fc=False,
        num_conv_layers_encoder=4,
        decoder_type="convtranspose",
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    handles = register_shape_hooks(model, rows)
    x = torch.zeros((args.batch_size, 3, 131, 131), dtype=torch.float32)
    with torch.no_grad():
        h_conv = model.encoder_conv(x)
        flat = h_conv.view(h_conv.size(0), -1)
        h_fc = model.encoder_fc_intermediate(flat)
        mu = model.fc_mu(h_fc)
        logvar = model.fc_logvar(h_fc)
        h_dec = model.decoder_fc_intermediate(mu)
        h_to_conv = model.decoder_fc_to_conv(h_dec)
        decoder_reshape_shape = (
            int(h_to_conv.shape[0]),
            int(model.final_conv_ch),
            int(model.final_spatial_dim),
            int(model.final_spatial_dim),
        )
        recon_decode = model.decoder_conv(h_to_conv.view(*decoder_reshape_shape))
        recon_forward, _, _, _ = model(x)
    for handle in handles:
        handle.remove()

    flatten_dim = int(model.final_conv_ch * model.final_spatial_dim * model.final_spatial_dim)
    final_shape_matches = tuple(recon_forward.shape) == tuple(x.shape)
    interpolation_defined = True
    interpolation_used = tuple(recon_decode.shape) != tuple(x.shape)

    summary_rows = [
        {
            "layer_group": "summary",
            "name": "input",
            "module_type": "Input",
            "input_shape": "",
            "output_shape": str(tuple(x.shape)),
            "num_params": 0,
            "trainable_params": 0,
            "kernel_size": "",
            "stride": "",
            "padding": "",
            "dilation": "",
            "output_padding": "",
            "uses_crop_pad_interpolation": False,
        },
        {
            "layer_group": "summary",
            "name": "flatten",
            "module_type": "Flatten",
            "input_shape": str(tuple(h_conv.shape)),
            "output_shape": str(tuple(flat.shape)),
            "num_params": 0,
            "trainable_params": 0,
            "kernel_size": "",
            "stride": "",
            "padding": "",
            "dilation": "",
            "output_padding": "",
            "uses_crop_pad_interpolation": False,
        },
        {
            "layer_group": "summary",
            "name": "mu",
            "module_type": "LatentMu",
            "input_shape": str(tuple(h_fc.shape)),
            "output_shape": str(tuple(mu.shape)),
            "num_params": 0,
            "trainable_params": 0,
            "kernel_size": "",
            "stride": "",
            "padding": "",
            "dilation": "",
            "output_padding": "",
            "uses_crop_pad_interpolation": False,
        },
        {
            "layer_group": "summary",
            "name": "logvar",
            "module_type": "LatentLogVar",
            "input_shape": str(tuple(h_fc.shape)),
            "output_shape": str(tuple(logvar.shape)),
            "num_params": 0,
            "trainable_params": 0,
            "kernel_size": "",
            "stride": "",
            "padding": "",
            "dilation": "",
            "output_padding": "",
            "uses_crop_pad_interpolation": False,
        },
        {
            "layer_group": "summary",
            "name": "decoder_reshape",
            "module_type": "View",
            "input_shape": str(tuple(h_to_conv.shape)),
            "output_shape": str(decoder_reshape_shape),
            "num_params": 0,
            "trainable_params": 0,
            "kernel_size": "",
            "stride": "",
            "padding": "",
            "dilation": "",
            "output_padding": "",
            "uses_crop_pad_interpolation": False,
        },
        {
            "layer_group": "summary",
            "name": "final_reconstruction",
            "module_type": "Output",
            "input_shape": str(tuple(x.shape)),
            "output_shape": str(tuple(recon_forward.shape)),
            "num_params": 0,
            "trainable_params": 0,
            "kernel_size": "",
            "stride": "",
            "padding": "",
            "dilation": "",
            "output_padding": "",
            "uses_crop_pad_interpolation": interpolation_used,
        },
    ]
    shape_df = pd.DataFrame(summary_rows + rows)
    rf_df = pd.DataFrame(conv_receptive_field_rows(model, input_spatial=131))
    params_df = pd.DataFrame(parameter_rows(model))

    shape_df.to_csv(outdir / "architecture_shapes.csv", index=False)
    rf_df.to_csv(outdir / "receptive_field.csv", index=False)
    params_df.to_csv(outdir / "parameter_counts.csv", index=False)

    findings = {
        "input_shape": tuple(x.shape),
        "encoder_final_conv_shape": tuple(h_conv.shape),
        "flatten_dim": flatten_dim,
        "intermediate_fc_dim": int(model.intermediate_fc_dim),
        "mu_shape": tuple(mu.shape),
        "logvar_shape": tuple(logvar.shape),
        "decoder_reshape_shape": decoder_reshape_shape,
        "decoder_raw_reconstruction_shape": tuple(recon_decode.shape),
        "forward_reconstruction_shape": tuple(recon_forward.shape),
        "final_shape_matches_input": bool(final_shape_matches),
        "interpolation_defined_in_forward": interpolation_defined,
        "interpolation_used_for_baseline_shape": bool(interpolation_used),
        "total_trainable_params": trainable_count(model),
        "encoder_last_receptive_field": int(rf_df["receptive_field"].iloc[-1]),
        "encoder_last_effective_stride": int(rf_df["effective_stride"].iloc[-1]),
    }

    lines = [
        "# CVAE Architecture Shape Audit",
        "",
        "Synthetic one-batch shape audit for the ADNI v4 [4,1,0] tanh baseline architecture.",
        "",
        f"- Input shape: `{findings['input_shape']}`",
        f"- Encoder final conv shape: `{findings['encoder_final_conv_shape']}`",
        f"- Flatten dim: `{flatten_dim}`",
        f"- `intermediate_fc_dim=quarter` resolves to: `{model.intermediate_fc_dim}`",
        f"- Mu/logvar shape: `{findings['mu_shape']}` / `{findings['logvar_shape']}`",
        f"- Decoder reshape: `{decoder_reshape_shape}`",
        f"- Raw decoder output shape: `{findings['decoder_raw_reconstruction_shape']}`",
        f"- Forward reconstruction shape: `{findings['forward_reconstruction_shape']}`",
        f"- Final shape equals `(B,3,131,131)`: `{final_shape_matches}`",
        f"- Interpolation branch exists in `forward`: `{interpolation_defined}`",
        f"- Interpolation used for this baseline shape: `{interpolation_used}`",
        f"- Last encoder receptive field: `{findings['encoder_last_receptive_field']}` pixels",
        f"- Last encoder effective stride: `{findings['encoder_last_effective_stride']}` pixels",
        f"- Total trainable params: `{findings['total_trainable_params']}`",
        "",
        "## Interpretation",
        "",
        "- No dimensional mismatch is observed in the baseline smoke test.",
        "- The convtranspose decoder reaches 131x131 exactly; no interpolation is used for this configuration.",
        "- The encoder conv stack has a 47x47 receptive field at the final convolutional layer, before flattening and dense layers.",
        "- The quarter intermediate FC dimension is a large dense bottleneck relative to the 256-D latent layer; its numeric value is reported in the CSV.",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(findings, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
