#!/usr/bin/env python3
"""Prepare controlled post-final ADNI VAE experiment configs.

This script is intentionally config-level only. It writes candidate JSON
configs, strict diff reports, dry-run validation tables, and launch commands.
It does not invoke the training script and does not read or write model
artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/controlled_model_experiment_matrix_preflight_20260617"
CONFIG_DIR = PROJECT_ROOT / "configs/runs"
RESULTS_ROOT = "results/revision_bspc_2026"
BIG_RESULTS_ROOT = "/media/diego/Datos/vae_AD_results/revision_bspc_2026"

EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def write_df(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        if df.empty:
            md_path.write_text("_No rows._\n", encoding="utf-8")
        else:
            md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten(value, child))
    elif isinstance(obj, list):
        rows[prefix] = json.dumps(obj, sort_keys=True)
    else:
        rows[prefix] = obj
    return rows


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        except Exception:
            return False
    return a == b


def append_arg(cmd: list[str], name: str, value: Any) -> None:
    flag = f"--{name}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    if isinstance(value, list):
        if not value:
            return
        cmd.append(flag)
        cmd.extend(str(x) for x in value)
        return
    cmd.extend([flag, str(value)])


def build_stage_a_dryrun_command(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    params = config["parameters"]
    cmd = [
        config.get("python_executable") or "python",
        str(Path(paths["training_script"])),
        "--global_tensor_path",
        str(Path(paths["global_tensor_path"])),
        "--metadata_path",
        str(Path(paths["metadata_path"])),
        "--output_dir",
        str(Path(paths["output_dir"])),
    ]
    for key, value in params.items():
        append_arg(cmd, key, value)
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    cmd.append("--dry-run")
    return cmd


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "E1_scheduler_dephase_T0_120",
            "run_name": "recover035_latent384_beta3p75_T120_h10000_p560_full5x5_E1_scheduler_dephase",
            "config_path": CONFIG_DIR / "adni_v5_1c_recover035_latent384_beta3p75_T120_h10000_p560_full5x5_E1_scheduler_dephase.json",
            "scientific_changes": {"parameters.lr_scheduler_T0": 120},
            "mechanistic_rationale": "Scheduler dephase sensitivity: test whether T0=120 changes VAE optimization dynamics without changing beta or objective.",
        },
        {
            "candidate_id": "E2_chmeanloss_beta1p25",
            "run_name": "recover035_latent384_beta1p25_chmeanloss_T80_h10000_p560_full5x5_E2",
            "config_path": CONFIG_DIR / "adni_v5_1c_recover035_latent384_beta1p25_chmeanloss_T80_h10000_p560_full5x5_E2.json",
            "scientific_changes": {
                "parameters.recon_loss_mode": "mse_offdiag_channel_mean_sum",
                "parameters.beta_vae": 1.25,
            },
            "mechanistic_rationale": "Channel-normalized objective with lower beta to probe effective regularization after D scaling changes.",
        },
        {
            "candidate_id": "E3_chmeanloss_beta2p0",
            "run_name": "recover035_latent384_beta2p0_chmeanloss_T80_h10000_p560_full5x5_E3",
            "config_path": CONFIG_DIR / "adni_v5_1c_recover035_latent384_beta2p0_chmeanloss_T80_h10000_p560_full5x5_E3.json",
            "scientific_changes": {
                "parameters.recon_loss_mode": "mse_offdiag_channel_mean_sum",
                "parameters.beta_vae": 2.0,
            },
            "mechanistic_rationale": "Channel-normalized objective with intermediate beta to bracket E2 while preserving promoted architecture.",
        },
        {
            "candidate_id": "E4_mfrBalancedVAE",
            "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_E4_mfrBalancedVAE",
            "config_path": CONFIG_DIR / "adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5_E4_mfrBalancedVAE.json",
            "scientific_changes": {"parameters.vae_train_sampler_strategy": "manufacturer_balanced"},
            "mechanistic_rationale": "Fold-local VAE sampling sensitivity to reduce Manufacturer imbalance in diagnosis-agnostic representation learning.",
        },
    ]


def set_by_dotted_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = obj
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def make_candidate_config(baseline: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(baseline)
    run_name = spec["run_name"]
    cfg["run_name"] = run_name
    cfg["description"] = (
        f"Controlled post-final experiment matrix candidate {spec['candidate_id']}. "
        f"Derived strictly from promoted recover035_latent384_beta3p75_T80_h10000_p560_full5x5. "
        f"Rationale: {spec['mechanistic_rationale']}"
    )
    for dotted, value in spec["scientific_changes"].items():
        set_by_dotted_path(cfg, dotted, value)
    cfg["paths"]["output_dir"] = f"{RESULTS_ROOT}/{run_name}"
    cfg["paths"]["big_disk_output_dir"] = f"{BIG_RESULTS_ROOT}/{run_name}"
    cfg["paths"]["split_preview_csv"] = f"{RESULTS_ROOT}/{run_name}_split_preview.csv"
    cfg["paths"]["split_preview_summary_csv"] = f"{RESULTS_ROOT}/{run_name}_split_preview_summary.csv"
    return cfg


def classify_diff(diff_path: str, spec: dict[str, Any]) -> tuple[bool, str]:
    scientific = set(spec["scientific_changes"])
    provenance = {
        "run_name",
        "description",
        "paths.output_dir",
        "paths.big_disk_output_dir",
        "paths.split_preview_csv",
        "paths.split_preview_summary_csv",
    }
    if diff_path in scientific:
        return True, "scientific"
    if diff_path in provenance:
        return True, "provenance"
    return False, "hidden_drift"


def diff_configs(baseline: dict[str, Any], candidate: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    base_flat = flatten(baseline)
    cand_flat = flatten(candidate)
    rows = []
    for key in sorted(set(base_flat) | set(cand_flat)):
        b = base_flat.get(key, "<MISSING>")
        c = cand_flat.get(key, "<MISSING>")
        if values_equal(b, c):
            continue
        allowed, diff_type = classify_diff(key, spec)
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "diff_path": key,
                "baseline_value": b,
                "candidate_value": c,
                "allowed": bool(allowed),
                "diff_type": diff_type,
                "status": "PASS" if allowed else "FAIL",
            }
        )
    return rows


def load_npz_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"tensor_exists": path.exists()}
    if not path.exists():
        return summary
    with np.load(path, allow_pickle=True) as npz:
        keys = set(npz.files)
        tensor_key = "data" if "data" in keys else "tensor_data" if "tensor_data" in keys else "X" if "X" in keys else None
        if tensor_key is None:
            tensor_key = next((k for k in npz.files if npz[k].ndim == 4), None)
        if tensor_key is not None:
            arr = npz[tensor_key]
            summary.update(
                {
                    "tensor_key": tensor_key,
                    "tensor_shape": "x".join(str(x) for x in arr.shape),
                    "tensor_n": int(arr.shape[0]),
                    "tensor_channels": int(arr.shape[1]) if arr.ndim >= 2 else None,
                    "tensor_roi": int(arr.shape[2]) if arr.ndim >= 3 else None,
                    "tensor_dtype": str(arr.dtype),
                }
            )
        for key in ["target_len_ts", "tr_seconds", "python_bandpass_applied"]:
            if key in keys:
                value = npz[key].tolist()
                summary[key] = value
        if "channel_names" in keys:
            summary["channel_names"] = "|".join(str(x) for x in npz["channel_names"].tolist())
        elif "channel_names_master" in keys:
            summary["channel_names"] = "|".join(str(x) for x in npz["channel_names_master"].tolist())
    return summary


def metadata_counts(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata_exists": path.exists()}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    out["metadata_rows"] = int(len(df))
    dx_col = "ResearchGroup_Mapped"
    if dx_col not in df.columns:
        out["metadata_error"] = f"missing {dx_col}"
        return out
    dx = df[dx_col].astype(str)
    for label in ["CN", "MCI", "AD"]:
        out[f"vae_pool_{label}"] = int((dx == label).sum())
    for label in ["CN", "AD"]:
        out[f"classifier_pool_{label}"] = int((dx == label).sum())
    for col in ["Age", "Sex", "Manufacturer"]:
        out[f"missing_{col}"] = int(df[col].isna().sum()) if col in df.columns else "missing_column"
    if "SubjectID" in df.columns:
        out["has_035_S_6927"] = bool((df["SubjectID"].astype(str) == "035_S_6927").any())
        out["has_128_S_2002"] = bool((df["SubjectID"].astype(str) == "128_S_2002").any())
    return out


def repository_support_check(search_text: str) -> bool:
    for root in [PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"]:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                if search_text in path.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except OSError:
                continue
    return False


def validate_candidate(
    config: dict[str, Any],
    spec: dict[str, Any],
    diff_rows: list[dict[str, Any]],
    *,
    execute_stagea_dryrun: bool,
    dryrun_log_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = config["parameters"]
    paths = config["paths"]

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add("strict_config_diff", all(r["allowed"] for r in diff_rows), "No hidden diffs" if all(r["allowed"] for r in diff_rows) else "Hidden config drift detected")
    add("json_written", Path(spec["config_path"]).exists(), rel(Path(spec["config_path"])))
    add("training_script_exists", (PROJECT_ROOT / paths["training_script"]).exists(), paths["training_script"])
    add("channels_to_use", params.get("channels_to_use") == EXPECTED_CHANNELS, str(params.get("channels_to_use")))
    add("selected_channel_names", config.get("selected_channel_names") == EXPECTED_SELECTED_CHANNEL_NAMES, str(config.get("selected_channel_names")))
    for key, expected in [
        ("latent_dim", 384),
        ("epochs_vae", 10000),
        ("early_stopping_patience_vae", 560),
        ("lr_scheduler_type", "cosine_warm"),
        ("cyclical_beta_n_cycles", 125),
        ("cyclical_beta_ratio_increase", 0.4),
        ("dropout_rate_vae", 0.15),
        ("vae_dropout_scope", "legacy_all"),
        ("vae_block_order", "legacy_act_norm"),
        ("classifier_types", ["logreg", "svm"]),
        ("metadata_features", ["Age", "Sex"]),
        ("outer_folds", 5),
        ("inner_folds", 5),
        ("seed", 42),
    ]:
        add(f"fixed_parameter_{key}", params.get(key) == expected, f"{params.get(key)!r}")
    add("loss_mode_supported", repository_support_check(str(params.get("recon_loss_mode"))), str(params.get("recon_loss_mode")))
    add("sampler_strategy_supported", repository_support_check(str(params.get("vae_train_sampler_strategy", "none"))), str(params.get("vae_train_sampler_strategy", "none")))

    tensor_summary = load_npz_summary(Path(paths["global_tensor_path"]))
    add("tensor_exists", bool(tensor_summary.get("tensor_exists")), str(paths["global_tensor_path"]))
    add("tensor_shape_roi131", tensor_summary.get("tensor_roi") == 131, str(tensor_summary))
    add("tensor_channel_count_at_least3", int(tensor_summary.get("tensor_channels") or 0) >= 3, str(tensor_summary))
    if "target_len_ts" in tensor_summary:
        add("tensor_target_len_ts_140", tensor_summary.get("target_len_ts") == 140, str(tensor_summary.get("target_len_ts")))
    if "python_bandpass_applied" in tensor_summary:
        add("python_bandpass_applied_false", tensor_summary.get("python_bandpass_applied") in [False, "False", 0], str(tensor_summary.get("python_bandpass_applied")))

    mcounts = metadata_counts(Path(paths["metadata_path"]))
    add("metadata_exists", bool(mcounts.get("metadata_exists")), str(paths["metadata_path"]))
    for label, expected in EXPECTED_VAE_COUNTS.items():
        add(f"vae_pool_{label}", mcounts.get(f"vae_pool_{label}") == expected, f"{mcounts.get(f'vae_pool_{label}')}")
    for label, expected in EXPECTED_CLF_COUNTS.items():
        add(f"classifier_pool_{label}", mcounts.get(f"classifier_pool_{label}") == expected, f"{mcounts.get(f'classifier_pool_{label}')}")
    add("metadata_required_no_missing_age", mcounts.get("missing_Age") == 0, f"{mcounts.get('missing_Age')}")
    add("metadata_required_no_missing_sex", mcounts.get("missing_Sex") == 0, f"{mcounts.get('missing_Sex')}")
    add("metadata_required_no_missing_manufacturer", mcounts.get("missing_Manufacturer") == 0, f"{mcounts.get('missing_Manufacturer')}")
    add("recover035_present", bool(mcounts.get("has_035_S_6927")), str(mcounts.get("has_035_S_6927")))
    add("128_S_2002_absent_from_metadata", not bool(mcounts.get("has_128_S_2002")), str(mcounts.get("has_128_S_2002")))

    out_path = PROJECT_ROOT / paths["output_dir"]
    stale_markers = []
    if out_path.exists():
        for marker in ["all_folds_metrics.csv", "summary_metrics.csv", "classifier_only_readout", "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:
            if (out_path / marker).exists():
                stale_markers.append(marker)
    add("no_stale_output_markers", len(stale_markers) == 0, ";".join(stale_markers) if stale_markers else rel(out_path))
    big_parent = Path(paths["big_disk_output_dir"]).parent
    add("big_disk_parent_exists", big_parent.exists(), str(big_parent))

    dry_cmd = build_stage_a_dryrun_command(config)
    add("dryrun_command_constructed", True, shlex.join(dry_cmd))
    if execute_stagea_dryrun:
        started = now_iso()
        proc = subprocess.run(
            dry_cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        dryrun_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = dryrun_log_dir / f"{spec['candidate_id']}_stdout.txt"
        stderr_path = dryrun_log_dir / f"{spec['candidate_id']}_stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        add(
            "stagea_training_script_dryrun_returncode",
            proc.returncode == 0,
            f"returncode={proc.returncode}; started={started}; stdout={rel(stdout_path)}; stderr={rel(stderr_path)}",
        )
    return rows


def experiment_matrix_rows(specs: list[dict[str, Any]], configs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        cfg = configs[spec["candidate_id"]]
        params = cfg["parameters"]
        rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "run_name": cfg["run_name"],
                "channels_to_use": json.dumps(params["channels_to_use"]),
                "latent_dim": params["latent_dim"],
                "beta_vae": params["beta_vae"],
                "recon_loss_mode": params["recon_loss_mode"],
                "lr_scheduler_T0": params["lr_scheduler_T0"],
                "cyclical_beta_n_cycles": params["cyclical_beta_n_cycles"],
                "early_stopping_patience_vae": params["early_stopping_patience_vae"],
                "dropout_rate_vae": params["dropout_rate_vae"],
                "vae_dropout_scope": params["vae_dropout_scope"],
                "vae_block_order": params["vae_block_order"],
                "vae_train_sampler_strategy": params.get("vae_train_sampler_strategy", "none"),
                "input_harmonization_mode": params.get("input_harmonization_mode", "none"),
                "mechanistic_rationale": spec["mechanistic_rationale"],
                "candidate_config_path": rel(Path(spec["config_path"])),
                "planned_output_dir": cfg["paths"]["output_dir"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-stagea-dryrun-execution",
        action="store_true",
        help="Only construct Stage A dry-run commands instead of executing them.",
    )
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = [{"timestamp": now_iso(), "event": "start", "output_dir": rel(outdir)}]

    baseline = load_json(BASELINE_CONFIG)
    specs = candidate_specs()
    configs: dict[str, dict[str, Any]] = {}
    diff_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    config_path_rows: list[dict[str, Any]] = []
    command_lines: list[str] = []

    for spec in specs:
        cfg = make_candidate_config(baseline, spec)
        configs[spec["candidate_id"]] = cfg
        write_json(Path(spec["config_path"]), cfg)
        rows = diff_configs(baseline, cfg, spec)
        diff_rows.extend(rows)
        validation_rows.extend(
            validate_candidate(
                cfg,
                spec,
                rows,
                execute_stagea_dryrun=not args.skip_stagea_dryrun_execution,
                dryrun_log_dir=outdir / "stagea_dryrun_logs",
            )
        )
        dry_cmd = build_stage_a_dryrun_command(cfg)
        command_lines.append(f"# {spec['candidate_id']} dry-run only")
        command_lines.append(shlex.join(dry_cmd))
        command_lines.append("")
        config_path_rows.append(
            {
                "candidate_id": spec["candidate_id"],
                "run_name": cfg["run_name"],
                "config_path": rel(Path(spec["config_path"])),
                "output_dir": cfg["paths"]["output_dir"],
                "big_disk_output_dir": cfg["paths"]["big_disk_output_dir"],
            }
        )
        command_log.append({"timestamp": now_iso(), "event": "candidate_config_written", "candidate_id": spec["candidate_id"], "config_path": rel(Path(spec["config_path"]))})

    matrix_df = pd.DataFrame(experiment_matrix_rows(specs, configs))
    diff_df = pd.DataFrame(diff_rows)
    config_paths_df = pd.DataFrame(config_path_rows)
    validation_df = pd.DataFrame(validation_rows)

    write_df(matrix_df, outdir / "experiment_matrix.csv", outdir / "experiment_matrix.md")
    write_df(diff_df, outdir / "config_diff_vs_promoted.csv", outdir / "config_diff_vs_promoted.md")
    write_df(config_paths_df, outdir / "candidate_config_paths.csv", outdir / "candidate_config_paths.md")
    write_df(validation_df, outdir / "dryrun_command_validation.csv", outdir / "dryrun_command_validation.md")
    (outdir / "planned_launcher_commands.txt").write_text("\n".join(command_lines).rstrip() + "\n", encoding="utf-8")

    promotion_gate = """# Promotion Gate

This package is preflight only. A candidate is eligible for later full training only if:

1. The strict config diff contains no hidden drift.
2. The dry-run validation table is all PASS.
3. The candidate has a specific mechanistic rationale.
4. The later post-run audit uses the locked ADNI OOF-ECDF/OOF-logitz protocol and reports AUC, PR-AUC, BA, sensitivity, specificity, F1.
5. The later post-run audit also reports Philips/rawTP FPR, GE/SIEMENS/Philips CN FPR, scanner leakage, rate-distortion, MI(Z;Y), MI(Z;Manufacturer), active units, TC, and OASIS only if frozen artifacts are available.

No model should be promoted on AUC alone.
"""
    (outdir / "promotion_gate.md").write_text(promotion_gate, encoding="utf-8")

    all_pass = bool(validation_df["status"].eq("PASS").all()) and bool(diff_df["allowed"].all())
    recommendation = [
        "# Final Recommendation",
        "",
        f"Preflight status: `{'PASS' if all_pass else 'FAIL'}`.",
        "",
        "Prepared four controlled candidate configs from the promoted baseline. No training was launched.",
        "",
        "- `E1_scheduler_dephase_T0_120`: scheduler sensitivity only; this intentionally dephases LR restart length from the historical T80 protocol.",
        "- `E2_chmeanloss_beta1p25`: channel-normalized reconstruction objective with low beta.",
        "- `E3_chmeanloss_beta2p0`: channel-normalized reconstruction objective with intermediate beta.",
        "- `E4_mfrBalancedVAE`: fold-local Manufacturer-balanced VAE sampling sensitivity.",
        "",
        "If launched later, run exactly one candidate at a time and audit ADNI OOF-ECDF plus Philips/rawTP/scanner leakage before any external interpretation.",
    ]
    if not all_pass:
        recommendation.extend(["", "Blocked checks are listed in `dryrun_command_validation.csv` or `config_diff_vs_promoted.csv`."])
    (outdir / "final_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")

    command_log.append(
        {
            "timestamp": now_iso(),
            "event": "complete",
            "all_pass": all_pass,
            "outputs": [
                "experiment_matrix.csv",
                "config_diff_vs_promoted.csv",
                "candidate_config_paths.csv",
                "dryrun_command_validation.csv",
                "planned_launcher_commands.txt",
                "promotion_gate.md",
                "final_recommendation.md",
            ],
        }
    )
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": rel(outdir), "preflight_pass": all_pass}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
