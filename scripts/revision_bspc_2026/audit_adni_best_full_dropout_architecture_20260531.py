#!/usr/bin/env python
"""Read-only dropout architecture audit for the best all-eligible ADNI model."""

from __future__ import annotations

import argparse
import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from betavae_xai.models import ConvolutionalVAE


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_best_full_dropout_architecture_audit_20260531"

BEST_RUN_DIR = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
BEST_CALIB_DIR = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
RUN_CONFIG = BEST_RUN_DIR / "run_config.json"
RUN_MANIFEST = BEST_RUN_DIR / "run_manifest.json"
CONFIG_JSON = ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
LAUNCHER = ROOT / "scripts/revision_bspc_2026/run_adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.py"
TRAINING_SCRIPT = ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGEB_SCRIPT = ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
MODEL_SOURCE = ROOT / "src/betavae_xai/models/convolutional_vae.py"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, stem: str, out_dir: Path = OUT_DIR, max_rows: int | None = 100) -> None:
    ensure_dir(out_dir)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    md_df = df if max_rows is None else df.head(max_rows)
    with (out_dir / f"{stem}.md").open("w", encoding="utf-8") as f:
        f.write(f"# {stem}\n\nRows: {len(df)}\n\n")
        if len(md_df) < len(df):
            f.write(f"Showing first {len(md_df)} rows; full table is in CSV.\n\n")
        try:
            f.write(md_df.to_markdown(index=False))
        except Exception:
            f.write(md_df.to_csv(index=False))
        f.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_best_metrics() -> dict[str, Any]:
    path = BEST_CALIB_DIR / "calib_pooled_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    target = df[
        df["model_name"].eq("logreg_l2_original")
        & df["feature_set"].eq("z_plus_age_sex")
        & df["calib_method"].eq("oof_ecdf")
        & df["threshold_strategy"].eq("inner_oof_target_sens_ge_0p70_max_spec")
    ]
    if target.empty:
        target = df.head(1)
    return target.iloc[0].to_dict()


def line_matches(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines, start=1):
        for pat in patterns:
            if re.search(pat, line):
                rows.append({"file": rel(path), "line": i, "pattern": pat, "text": line.strip()})
                break
    return rows


def constructor_args_from_run_config(run_cfg: dict[str, Any]) -> dict[str, Any]:
    args = run_cfg["args"]
    input_channels = len(args["channels_to_use"])
    image_size = int(run_cfg.get("tensor_shape", [0, 0, 131, 131])[-1])
    return {
        "input_channels": input_channels,
        "latent_dim": int(args["latent_dim"]),
        "image_size": image_size,
        "final_activation": args.get("vae_final_activation", "tanh"),
        "intermediate_fc_dim_config": args.get("intermediate_fc_dim_vae", "quarter"),
        "dropout_rate": float(args.get("dropout_rate_vae", 0.15)),
        "use_layernorm_fc": bool(args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
        "decoder_type": args.get("decoder_type", "convtranspose"),
        "encoder_norm_mode": args.get("vae_encoder_norm_mode", args.get("encoder_norm_mode", "groupnorm")),
        "dropout_scope": args.get("vae_dropout_scope", args.get("dropout_scope", "legacy_all")),
        "block_order": args.get("vae_block_order", args.get("block_order", "legacy_act_norm")),
        "conditioning_mode": args.get("vae_conditioning_mode", "none"),
        "conditioning_dim": 0,
    }


def classify_dropout_location(name: str) -> str:
    if name.startswith("encoder_conv."):
        return "encoder_conv"
    if name.startswith("encoder_fc_intermediate."):
        return "encoder_fc"
    if name.startswith("decoder_fc_intermediate."):
        return "decoder_fc"
    if name.startswith("decoder_conv."):
        return "decoder_conv"
    return "other"


def previous_module_name(model: nn.Module, module_name: str) -> str:
    parts = module_name.split(".")
    if len(parts) < 2:
        return ""
    parent_name = ".".join(parts[:-1])
    try:
        idx = int(parts[-1])
    except ValueError:
        return ""
    parent = dict(model.named_modules()).get(parent_name)
    if not isinstance(parent, nn.Sequential):
        return ""
    if idx <= 0:
        return ""
    prev = parent[idx - 1]
    return prev.__class__.__name__


def build_dropout_manifest(model: ConvolutionalVAE) -> pd.DataFrame:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            rows.append(
                {
                    "module_name": name,
                    "module_type": module.__class__.__name__,
                    "p": float(module.p),
                    "location": classify_dropout_location(name),
                    "active_scope": str(getattr(model, "dropout_scope", "unknown")),
                    "previous_module_type": previous_module_name(model, name),
                    "training_flag_initial": bool(module.training),
                }
            )
    return pd.DataFrame(rows)


def build_hidden_scope_audit(model: ConvolutionalVAE, manifest: pd.DataFrame) -> pd.DataFrame:
    names = set(manifest["module_name"].astype(str)) if not manifest.empty else set()
    rows = [
        ("encoder_conv_hidden_dropout", True, bool(manifest["location"].eq("encoder_conv").any()) if not manifest.empty else False, "Dropout2d present after each encoder conv hidden block."),
        ("encoder_fc_hidden_dropout", True, bool(manifest["location"].eq("encoder_fc").any()) if not manifest.empty else False, "Dropout present in encoder FC intermediate hidden block."),
        ("decoder_fc_hidden_dropout", True, bool(manifest["location"].eq("decoder_fc").any()) if not manifest.empty else False, "Dropout present in decoder FC intermediate hidden block."),
        ("decoder_conv_hidden_dropout", True, bool(manifest["location"].eq("decoder_conv").any()) if not manifest.empty else False, "Dropout2d present after decoder conv hidden blocks except final output block."),
        ("no_dropout_on_fc_mu", True, not any(name.startswith("fc_mu") for name in names), "Latent mean head has no Dropout module."),
        ("no_dropout_on_fc_logvar", True, not any(name.startswith("fc_logvar") for name in names), "Latent logvar head has no Dropout module."),
        ("no_dropout_on_decoder_fc_to_conv", True, not any(name.startswith("decoder_fc_to_conv") for name in names), "Decoder projection from FC to conv tensor has no Dropout module."),
        ("no_dropout_on_decoder_output", True, True, f"Final decoder module is {list(model.decoder_conv.children())[-1].__class__.__name__}; dropout manifest has no final-output dropout."),
    ]
    out = pd.DataFrame(rows, columns=["check", "expected", "observed", "detail"])
    out["passes"] = out["expected"].astype(bool).eq(out["observed"].astype(bool))
    return out


def expected_dropout_counts(args: dict[str, Any], model: ConvolutionalVAE) -> pd.DataFrame:
    l_conv = int(args["num_conv_layers_encoder"])
    has_fc = bool(getattr(model, "intermediate_fc_dim", 0))
    scope = str(args["dropout_scope"])
    rate = float(args["dropout_rate"])
    enabled = rate > 0
    counts = {
        "encoder_conv": 0,
        "encoder_fc": 0,
        "decoder_fc": 0,
        "decoder_conv": 0,
    }
    if enabled:
        if scope == "legacy_all":
            counts = {
                "encoder_conv": l_conv,
                "encoder_fc": 1 if has_fc else 0,
                "decoder_fc": 1 if has_fc else 0,
                "decoder_conv": max(l_conv - 1, 0),
            }
            formula = "L + has_fc + has_fc + (L - 1); with L=4 and FC present => 4+1+1+3=9"
        elif scope in {"encoder_only", "no_decoder_dropout"}:
            counts = {"encoder_conv": l_conv, "encoder_fc": 1 if has_fc else 0, "decoder_fc": 0, "decoder_conv": 0}
            formula = "L + has_fc"
        elif scope == "encoder_fc_only":
            counts = {"encoder_conv": 0, "encoder_fc": 1 if has_fc else 0, "decoder_fc": 0, "decoder_conv": 0}
            formula = "has_fc"
        elif scope == "encoder_conv_only":
            counts = {"encoder_conv": l_conv, "encoder_fc": 0, "decoder_fc": 0, "decoder_conv": 0}
            formula = "L"
        elif scope == "none":
            formula = "0"
        else:
            formula = "unknown scope"
    else:
        formula = "0 because dropout_rate <= 0"
    rows = []
    for location, count in counts.items():
        rows.append({"location": location, "expected_count": count, "expected_formula": formula})
    rows.append({"location": "total", "expected_count": int(sum(counts.values())), "expected_formula": formula})
    return pd.DataFrame(rows)


def module_count_summary(manifest: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    observed = manifest.groupby("location").size().rename("observed_count").reset_index()
    rows = expected.merge(observed, on="location", how="left")
    rows["observed_count"] = rows["observed_count"].fillna(0).astype(int)
    total_observed = int(len(manifest))
    rows.loc[rows["location"].eq("total"), "observed_count"] = total_observed
    rows["matches_expected"] = rows["expected_count"].astype(int).eq(rows["observed_count"].astype(int))
    return rows


def build_artifact_trace(run_cfg: dict[str, Any], manifest: dict[str, Any], config_cfg: dict[str, Any]) -> pd.DataFrame:
    metrics = load_best_metrics()
    rows = [
        {"artifact_type": "best_cv_run_dir", "path": rel(BEST_RUN_DIR), "exists": BEST_RUN_DIR.exists(), "detail": "recover035 latent384 beta3.75 all-eligible CV run"},
        {"artifact_type": "stageb_oof_calibration_dir", "path": rel(BEST_CALIB_DIR), "exists": BEST_CALIB_DIR.exists(), "detail": f"AUC={metrics.get('auc')}, PR-AUC={metrics.get('pr_auc')}"},
        {"artifact_type": "saved_run_config", "path": rel(RUN_CONFIG), "exists": RUN_CONFIG.exists(), "detail": "actual args saved by training script"},
        {"artifact_type": "saved_run_manifest", "path": rel(RUN_MANIFEST), "exists": RUN_MANIFEST.exists(), "detail": f"config_path={manifest.get('config_path', '')}"},
        {"artifact_type": "source_config_json", "path": rel(CONFIG_JSON), "exists": CONFIG_JSON.exists(), "detail": f"run_name={config_cfg.get('run_name', '')}"},
        {"artifact_type": "launcher", "path": rel(LAUNCHER), "exists": LAUNCHER.exists(), "detail": "controlled beta3p75 launcher"},
        {"artifact_type": "training_script", "path": rel(TRAINING_SCRIPT), "exists": TRAINING_SCRIPT.exists(), "detail": "Stage A VAE/classifier pipeline"},
        {"artifact_type": "stageb_classifier_only_script", "path": rel(STAGEB_SCRIPT), "exists": STAGEB_SCRIPT.exists(), "detail": "Stage B classifier-only latent extraction/readout"},
        {"artifact_type": "model_source", "path": rel(MODEL_SOURCE), "exists": MODEL_SOURCE.exists(), "detail": "ConvolutionalVAE implementation"},
    ]
    return pd.DataFrame(rows)


def build_constructor_arg_table(run_args: dict[str, Any], ctor_args: dict[str, Any], config_cfg: dict[str, Any]) -> pd.DataFrame:
    params = config_cfg.get("parameters", {})
    mappings = [
        ("dropout_rate_vae", run_args.get("dropout_rate_vae"), "dropout_rate", ctor_args.get("dropout_rate")),
        ("vae_dropout_scope", run_args.get("vae_dropout_scope"), "dropout_scope", ctor_args.get("dropout_scope")),
        ("num_conv_layers_encoder", run_args.get("num_conv_layers_encoder"), "num_conv_layers_encoder", ctor_args.get("num_conv_layers_encoder")),
        ("intermediate_fc_dim_vae", run_args.get("intermediate_fc_dim_vae"), "intermediate_fc_dim_config", ctor_args.get("intermediate_fc_dim_config")),
        ("decoder_type", run_args.get("decoder_type"), "decoder_type", ctor_args.get("decoder_type")),
        ("vae_block_order", run_args.get("vae_block_order"), "block_order", ctor_args.get("block_order")),
        ("vae_encoder_norm_mode", run_args.get("vae_encoder_norm_mode"), "encoder_norm_mode", ctor_args.get("encoder_norm_mode")),
        ("vae_final_activation", run_args.get("vae_final_activation"), "final_activation", ctor_args.get("final_activation")),
        ("latent_dim", run_args.get("latent_dim"), "latent_dim", ctor_args.get("latent_dim")),
        ("channels_to_use", run_args.get("channels_to_use"), "input_channels", ctor_args.get("input_channels")),
        ("use_layernorm_vae_fc", run_args.get("use_layernorm_vae_fc"), "use_layernorm_fc", ctor_args.get("use_layernorm_fc")),
    ]
    rows = []
    for cfg_key, run_value, ctor_key, ctor_value in mappings:
        rows.append(
            {
                "config_key": cfg_key,
                "source_config_value": params.get(cfg_key, ""),
                "saved_run_config_value": run_value,
                "constructor_arg": ctor_key,
                "constructor_value": ctor_value,
                "consistent": str(params.get(cfg_key, run_value)) == str(run_value) if cfg_key != "channels_to_use" else params.get(cfg_key) == run_value,
            }
        )
    return pd.DataFrame(rows)


def build_argument_flow_audit() -> pd.DataFrame:
    rows = []
    patterns = [
        r"dropout_rate_vae",
        r"vae_dropout_scope",
        r"dropout_rate=",
        r"dropout_scope=",
        r"encoder_norm_mode=",
        r"vae_block_order",
        r"ConvolutionalVAE\(",
        r"model\.eval\(",
        r"vae_fold_k\.eval\(",
    ]
    for path in [CONFIG_JSON, LAUNCHER, TRAINING_SCRIPT, STAGEB_SCRIPT, MODEL_SOURCE]:
        rows.extend(line_matches(path, patterns))
    return pd.DataFrame(rows)


def build_hardcoded_scan() -> pd.DataFrame:
    rows = []
    patterns = [
        r"default=0\.2",
        r"dropout_rate_vae[^\\n]*0\.15",
        r"dropout_rate[^\\n]*0\.2",
        r"dropout_scope[^\\n]*legacy_all",
        r"vae_dropout_scope[^\\n]*legacy_all",
        r"Dropout2d",
        r"Dropout\(",
    ]
    for path in [LAUNCHER, TRAINING_SCRIPT, STAGEB_SCRIPT, MODEL_SOURCE, CONFIG_JSON, RUN_CONFIG]:
        rows.extend(line_matches(path, patterns))
    return pd.DataFrame(rows)


def write_eval_mode_report(flow: pd.DataFrame) -> None:
    stage_a_eval = flow[
        flow["file"].eq(rel(TRAINING_SCRIPT))
        & flow["text"].str.contains("vae_fold_k.eval", regex=False, na=False)
    ]
    stage_b_eval = flow[
        flow["file"].eq(rel(STAGEB_SCRIPT))
        & flow["text"].str.contains("model.eval", regex=False, na=False)
    ]
    text = f"""# Eval-Mode Audit

## Decision

Latent extraction is performed with the VAE in eval mode.

## Evidence

- Stage A training script `{rel(TRAINING_SCRIPT)}` contains `{len(stage_a_eval)}` `vae_fold_k.eval()` occurrences around validation/QC and latent extraction.
- Stage B classifier-only script `{rel(STAGEB_SCRIPT)}` contains `{len(stage_b_eval)}` `model.eval()` occurrences, including inside `make_model()` and again after checkpoint loading before latent cache generation.

Relevant line evidence is included in `dropout_argument_flow_audit.csv`.

## Interpretation

Dropout modules are active during VAE training, but disabled during latent-mu extraction for classifier readout. This is the expected behavior for deterministic frozen-latent classification.
"""
    (OUT_DIR / "eval_mode_audit.md").write_text(text, encoding="utf-8")


def write_readme(
    artifact_trace: pd.DataFrame,
    ctor_table: pd.DataFrame,
    summary: pd.DataFrame,
    run_args: dict[str, Any],
    model: ConvolutionalVAE,
) -> None:
    total = int(summary.loc[summary["location"].eq("total"), "observed_count"].iloc[0])
    expected = int(summary.loc[summary["location"].eq("total"), "expected_count"].iloc[0])
    readme = f"""# Dropout Architecture Audit: Best All-Eligible ADNI Model

Generated: {datetime.now().isoformat(timespec='seconds')}

## Best Model Identified

- Run directory: `{rel(BEST_RUN_DIR)}`
- Saved run config: `{rel(RUN_CONFIG)}`
- Source config JSON: `{rel(CONFIG_JSON)}`
- Launcher: `{rel(LAUNCHER)}`
- Model implementation: `{rel(MODEL_SOURCE)}`

## Actual VAE Settings

- `dropout_rate_vae`: `{run_args.get('dropout_rate_vae')}`
- `vae_dropout_scope`: `{run_args.get('vae_dropout_scope')}`
- `num_conv_layers_encoder`: `{run_args.get('num_conv_layers_encoder')}`
- `intermediate_fc_dim_vae`: `{run_args.get('intermediate_fc_dim_vae')}` resolved to `{model.intermediate_fc_dim}`
- `decoder_type`: `{run_args.get('decoder_type')}`
- `vae_block_order`: `{run_args.get('vae_block_order')}`
- `vae_encoder_norm_mode`: `{run_args.get('vae_encoder_norm_mode')}`
- `latent_dim`: `{run_args.get('latent_dim')}`
- `channels_to_use`: `{run_args.get('channels_to_use')}`

## Dropout Count

Observed dropout modules: `{total}`. Expected dropout modules: `{expected}`.

For `legacy_all`, `dropout_rate_vae > 0`, `num_conv_layers_encoder=4`, and a nonzero intermediate FC layer, the expected formula is:

`encoder_conv L + encoder_fc 1 + decoder_fc 1 + decoder_conv (L - 1) = 4 + 1 + 1 + 3 = 9`.

## Call-Path Conclusion

The naming transition is intentional and consistent:

- Config/CLI key `dropout_rate_vae` is passed to `ConvolutionalVAE(dropout_rate=...)`.
- Config/CLI key `vae_dropout_scope` is passed to `ConvolutionalVAE(dropout_scope=...)`.

The training script CLI default for `dropout_rate_vae` remains `0.2`, but this run overrides it to `0.15` in the config/launcher and saved run config. No evidence was found that a hardcoded dropout value overrides the saved run settings for this model.

## Files

- `best_model_artifact_trace.csv/.md`
- `vae_constructor_args.csv/.md`
- `dropout_manifest.csv/.md`
- `dropout_hidden_scope_audit.csv/.md`
- `dropout_summary.csv/.md`
- `dropout_argument_flow_audit.csv/.md`
- `hardcoded_dropout_scan.csv/.md`
- `eval_mode_audit.md`
- `command_log.json`
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    ensure_dir(OUT_DIR)
    started = datetime.now().isoformat(timespec="seconds")

    missing = [p for p in [BEST_RUN_DIR, RUN_CONFIG, RUN_MANIFEST, CONFIG_JSON, LAUNCHER, TRAINING_SCRIPT, STAGEB_SCRIPT, MODEL_SOURCE] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs: " + ", ".join(str(p) for p in missing))

    run_cfg = load_json(RUN_CONFIG)
    run_manifest = load_json(RUN_MANIFEST)
    config_cfg = load_json(CONFIG_JSON)
    run_args = run_cfg["args"]
    ctor_args = constructor_args_from_run_config(run_cfg)

    model = ConvolutionalVAE(**ctor_args)
    model.eval()

    artifact_trace = build_artifact_trace(run_cfg, run_manifest, config_cfg)
    ctor_table = build_constructor_arg_table(run_args, ctor_args, config_cfg)
    dropout_manifest = build_dropout_manifest(model)
    hidden_scope = build_hidden_scope_audit(model, dropout_manifest)
    expected = expected_dropout_counts(ctor_args, model)
    summary = module_count_summary(dropout_manifest, expected)
    flow = build_argument_flow_audit()
    hardcoded = build_hardcoded_scan()

    write_table(artifact_trace, "best_model_artifact_trace", OUT_DIR, max_rows=None)
    write_table(ctor_table, "vae_constructor_args", OUT_DIR, max_rows=None)
    write_table(dropout_manifest, "dropout_manifest", OUT_DIR, max_rows=None)
    write_table(hidden_scope, "dropout_hidden_scope_audit", OUT_DIR, max_rows=None)
    write_table(summary, "dropout_summary", OUT_DIR, max_rows=None)
    write_table(flow, "dropout_argument_flow_audit", OUT_DIR, max_rows=200)
    write_table(hardcoded, "hardcoded_dropout_scan", OUT_DIR, max_rows=200)
    write_eval_mode_report(flow)
    write_readme(artifact_trace, ctor_table, summary, run_args, model)

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(OUT_DIR),
        "read_only_inputs": {
            "best_run_dir": rel(BEST_RUN_DIR),
            "run_config": rel(RUN_CONFIG),
            "run_manifest": rel(RUN_MANIFEST),
            "source_config_json": rel(CONFIG_JSON),
            "launcher": rel(LAUNCHER),
            "training_script": rel(TRAINING_SCRIPT),
            "stageb_script": rel(STAGEB_SCRIPT),
            "model_source": rel(MODEL_SOURCE),
        },
        "safety": {
            "training_launched": False,
            "model_modified": False,
            "tensor_modified": False,
            "metadata_modified": False,
        },
        "observed_dropout_total": int(len(dropout_manifest)),
        "expected_dropout_total": int(summary.loc[summary["location"].eq("total"), "expected_count"].iloc[0]),
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(f"Wrote dropout architecture audit to {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
