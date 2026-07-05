#!/usr/bin/env python3
"""Prepare/run FAST 3x3 objective-sensitivity pilot for ADNI v5.1 [1,0,2].

Default is dry-run/no training. Real training requires --confirm-training.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"

OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_objective_v2_fast_3x3_ch1_0_2"
BIG_DISK_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/adni_v5_1_batch20260514b_objective_v2_fast_3x3_ch1_0_2")

CONFIGS = [
    PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_fast3x3_control_current_loss.json",
    PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_fast3x3_objective_v2_offdiag_channelmean.json",
]

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_STRATIFY_COLS = ["Manufacturer"]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
EXPECTED_RUNS = {
    "control_current_loss": ("mse_sum_batchmean_current", "tanh"),
    "objective_v2_offdiag_channelmean": ("offdiag_channelmean_sum", "tanh"),
}

PARAM_ORDER = [
    "channels_to_use",
    "classifier_types",
    "classifier_stratify_cols",
    "vae_stratify_cols",
    "classifier_calibrate",
    "classifier_use_class_weight",
    "latent_features_type",
    "gridsearch_scoring",
    "outer_folds",
    "inner_folds",
    "repeated_outer_folds_n_repeats",
    "num_conv_layers_encoder",
    "decoder_type",
    "epochs_vae",
    "vae_val_split_ratio",
    "early_stopping_patience_vae",
    "cyclical_beta_n_cycles",
    "cyclical_beta_ratio_increase",
    "beta_vae",
    "recon_loss_mode",
    "dropout_rate_vae",
    "latent_dim",
    "batch_size",
    "lr_vae",
    "lr_scheduler_type",
    "lr_scheduler_T0",
    "lr_scheduler_eta_min",
    "lr_scheduler_patience_vae",
    "weight_decay_vae",
    "vae_final_activation",
    "intermediate_fc_dim_vae",
    "use_layernorm_vae_fc",
    "n_jobs_gridsearch",
    "metadata_features",
    "norm_mode",
    "seed",
    "num_workers",
    "log_interval_epochs_vae",
    "save_fold_artefacts",
    "save_vae_training_history",
    "qc_analyze_distributions",
    "qc_check_scanner_leakage",
    "qc_rate_distortion",
    "qc_latent_information",
    "qc_mi_n_neighbors",
    "qc_mi_top_k",
    "qc_rd_log_base",
    "qc_tc_ridge",
    "qc_var_eps_active",
    "use_optuna_pruner",
    "use_smote",
    "tune_sampler_params",
    "mlp_classifier_hidden_layers",
    "n_iter_logreg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not train.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch.")
    parser.add_argument("--resume", action="store_true", help="Skip a run if Stage B outputs are complete.")
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_arg(cmd: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    if value is None:
        return
    cmd.append(f"--{name}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_config(path: Path, cfg: Dict[str, Any]) -> None:
    params = cfg["parameters"]
    run_name = cfg["run_name"]
    expected_key = "objective_v2_offdiag_channelmean" if "objective_v2_offdiag_channelmean" in run_name else "control_current_loss"
    expected_loss, expected_activation = EXPECTED_RUNS[expected_key]

    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, f"{run_name}.channels_to_use")
    require_equal(cfg["selected_channel_names"], EXPECTED_SELECTED_NAMES, f"{run_name}.selected_channel_names")
    require_equal(params["recon_loss_mode"], expected_loss, f"{run_name}.recon_loss_mode")
    require_equal(params["vae_final_activation"], expected_activation, f"{run_name}.vae_final_activation")
    require_equal(params["outer_folds"], 3, f"{run_name}.outer_folds")
    require_equal(params["inner_folds"], 3, f"{run_name}.inner_folds")
    require_equal(params["latent_dim"], 128, f"{run_name}.latent_dim")
    require_equal(params["epochs_vae"], 960, f"{run_name}.epochs_vae")
    require_equal(params["cyclical_beta_n_cycles"], 12, f"{run_name}.cyclical_beta_n_cycles")
    require_equal(params["lr_scheduler_T0"], 80, f"{run_name}.lr_scheduler_T0")
    require_equal(params["beta_vae"], 2.5, f"{run_name}.beta_vae")
    require_equal(params["dropout_rate_vae"], 0.15, f"{run_name}.dropout_rate_vae")
    require_equal(params["norm_mode"], "zscore_offdiag", f"{run_name}.norm_mode")
    require_equal(params["classifier_types"], ["logreg"], f"{run_name}.classifier_types")
    require_equal(params["n_iter_logreg"], 1, f"{run_name}.n_iter_logreg")
    require_equal(params["classifier_stratify_cols"], EXPECTED_STRATIFY_COLS, f"{run_name}.classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], EXPECTED_STRATIFY_COLS, f"{run_name}.vae_stratify_cols")
    require_equal(params["metadata_features"], EXPECTED_METADATA_FEATURES, f"{run_name}.metadata_features")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError(f"{run_name}: Sex must remain metadata-only, not a stratifier.")
    if params["epochs_vae"] / params["cyclical_beta_n_cycles"] != 80:
        raise RuntimeError(f"{run_name}: expected 960/12 cycle length = 80")
    if params["lr_scheduler_T0"] != 80:
        raise RuntimeError(f"{run_name}: expected T0=80")
    for flag in ["qc_analyze_distributions", "qc_check_scanner_leakage", "qc_rate_distortion", "qc_latent_information"]:
        require_equal(params[flag], True, f"{run_name}.{flag}")
    readout = cfg["primary_readout"]
    require_equal(readout["model"], PRIMARY_MODEL, f"{run_name}.primary_readout.model")
    require_equal(readout["threshold_strategy"], PRIMARY_THRESHOLD, f"{run_name}.primary_readout.threshold_strategy")
    require_equal(readout["threshold_selection"], "true_inner_cv_oof", f"{run_name}.primary_readout.threshold_selection")
    unknown = sorted(set(params) - set(PARAM_ORDER))
    if unknown:
        raise RuntimeError(f"{path}: parameters not wired into Stage A command: {unknown}")


def inspect_tensor(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as zf:
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_SELECTED_NAMES, "tensor selected channel names")
    return {"shape": shape, "channel_names": channel_names, "python_bandpass_applied": False}


def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID rows.")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if cn_ad["Age"].isna().any() or cn_ad["Sex"].isna().any():
        raise RuntimeError("CN/AD rows contain missing Age/Sex.")
    return meta


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for c in cols:
        tmp[c] = tmp[c].fillna(f"{c}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        out[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
        for dx in ["AD", "CN", "MCI"]:
            out[f"{dx}_{label}"] = int((df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(mfr)).sum())
    return out


def write_split_preview(meta: pd.DataFrame, output_root: Path, params: Dict[str, Any]) -> pd.DataFrame:
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    n_splits = int(params["outer_folds"])
    seed = int(params["seed"])
    outer_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, outer_cols)
    if (y_outer.value_counts() < n_splits).any():
        raise RuntimeError("3x3 manufacturer-aware outer split is not feasible.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    rows: List[Dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_cols = ["ResearchGroup_Mapped", *params["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_cols)
        if (vae_key.value_counts() < 2).any():
            raise RuntimeError("VAE internal validation split is not feasible.")
        vae_train_idx, vae_val_idx = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + 10 + (fold - 1),
            shuffle=True,
        )
        components = [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_pool.iloc[vae_train_idx]),
            ("vae_internal_val", vae_pool.iloc[vae_val_idx]),
        ]
        for name, df in components:
            row: Dict[str, Any] = {"fold": fold, "split_component": name}
            row.update(count_fields(df))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_root / "split_preview_summary.csv", index=False)
    return out


def build_stage_a_command(cfg: Dict[str, Any], python_exe: str) -> List[str]:
    paths = cfg["paths"]
    params = cfg["parameters"]
    cmd = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        paths["global_tensor_path"],
        "--metadata_path",
        paths["metadata_path"],
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for name in PARAM_ORDER:
        append_arg(cmd, name, params.get(name))
    validate_stage_a_command(cmd, cfg)
    return cmd


def build_stage_b_command(cfg: Dict[str, Any], python_exe: str) -> List[str]:
    paths = cfg["paths"]
    params = cfg["parameters"]
    cmd = [
        python_exe,
        str(resolve(paths["classifier_only_script"])),
        "--run-dir",
        str(resolve(paths["output_dir"])),
        "--output-dir",
        str(resolve(paths["classifier_only_output_dir"])),
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--models",
        PRIMARY_MODEL,
        "--overwrite",
    ]
    validate_stage_b_command(cmd)
    return cmd


def validate_stage_a_command(cmd: Sequence[str], cfg: Dict[str, Any]) -> None:
    for idx, token in enumerate(cmd[:-1]):
        if token.startswith("--n_iter_") and str(cmd[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(values_after_flag(cmd, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(cmd, "--classifier_types"), ["logreg"], "Stage A classifier_types")
    require_equal(values_after_flag(cmd, "--n_iter_logreg"), ["1"], "Stage A n_iter_logreg")
    if "--n_iter_svm" in cmd or "svm" in values_after_flag(cmd, "--classifier_types"):
        raise RuntimeError("Stage A must not include svm.")
    for flag, expected in [
        ("--outer_folds", ["3"]),
        ("--inner_folds", ["3"]),
        ("--latent_dim", ["128"]),
        ("--epochs_vae", ["960"]),
        ("--cyclical_beta_n_cycles", ["12"]),
        ("--lr_scheduler_T0", ["80"]),
        ("--metadata_features", ["Age", "Sex"]),
        ("--classifier_stratify_cols", ["Manufacturer"]),
        ("--vae_stratify_cols", ["Manufacturer"]),
        ("--recon_loss_mode", [cfg["parameters"]["recon_loss_mode"]]),
        ("--vae_final_activation", ["tanh"]),
    ]:
        require_equal(values_after_flag(cmd, flag), expected, f"Stage A {flag}")


def validate_stage_b_command(cmd: Sequence[str]) -> None:
    require_equal(values_after_flag(cmd, "--outer-folds"), ["3"], "Stage B outer-folds")
    require_equal(values_after_flag(cmd, "--inner-folds"), ["3"], "Stage B inner-folds")
    require_equal(values_after_flag(cmd, "--models"), [PRIMARY_MODEL], "Stage B models")


def readout_complete(path: Path) -> bool:
    return all((path / name).exists() for name in ["classifier_sweep_pooled_metrics.csv", "command_log.json"])


def prepare_real_run_dir(cfg: Dict[str, Any], resume: bool) -> None:
    out = resolve(cfg["paths"]["output_dir"])
    big = Path(cfg["paths"]["big_disk_output_dir"])
    if out.exists() or out.is_symlink():
        if resume:
            return
        if any(out.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty run dir: {out}")
        return
    big.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.symlink_to(big, target_is_directory=True)


def make_planned_rows(configs: List[Tuple[Path, Dict[str, Any]]], python_exe: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for config_path, cfg in configs:
        stage_a = build_stage_a_command(cfg, python_exe)
        stage_b = build_stage_b_command(cfg, python_exe)
        params = cfg["parameters"]
        rows.append(
            {
                "run_name": cfg["run_name"],
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "channels_to_use": json.dumps(params["channels_to_use"]),
                "recon_loss_mode": params["recon_loss_mode"],
                "vae_final_activation": params["vae_final_activation"],
                "outer_folds": params["outer_folds"],
                "inner_folds": params["inner_folds"],
                "latent_dim": params["latent_dim"],
                "epochs_vae": params["epochs_vae"],
                "cyclical_beta_n_cycles": params["cyclical_beta_n_cycles"],
                "lr_scheduler_T0": params["lr_scheduler_T0"],
                "python_bandpass_applied": False,
                "stage_a_classifier": "dummy_logreg_ignored",
                "stage_b_model": PRIMARY_MODEL,
                "stage_b_threshold": PRIMARY_THRESHOLD,
                "vae_output_dir": str(resolve(cfg["paths"]["output_dir"])),
                "big_disk_output_dir": cfg["paths"]["big_disk_output_dir"],
                "readout_output_dir": str(resolve(cfg["paths"]["classifier_only_output_dir"])),
                "stage_a_command": " ".join(shlex.quote(x) for x in stage_a),
                "stage_b_command": " ".join(shlex.quote(x) for x in stage_b),
            }
        )
    return pd.DataFrame(rows)


def write_readme(output_root: Path, planned: pd.DataFrame, dry_run: bool) -> None:
    preview_cols = ["run_name", "channels_to_use", "recon_loss_mode", "vae_final_activation", "outer_folds", "inner_folds", "latent_dim", "epochs_vae"]
    preview = planned[preview_cols]
    md_lines = [
        "| " + " | ".join(preview_cols) + " |",
        "| " + " | ".join(["---"] * len(preview_cols)) + " |",
    ]
    for _, row in preview.iterrows():
        md_lines.append("| " + " | ".join(str(row[col]) for col in preview_cols) + " |")
    lines = [
        "# ADNI v5.1 batch20260514b Objective-v2 FAST 3x3 Pilot",
        "",
        "Purpose: controlled objective-sensitivity pilot for the current primary channel set `[1,0,2]`.",
        "",
        "This preparation does not modify existing paper-ready results, tensors, metadata, or ledger files.",
        "",
        "## Arms",
        "",
        "- `control_current_loss`: `recon_loss_mode=mse_sum_batchmean_current`, `vae_final_activation=tanh`.",
        "- `objective_v2_offdiag_channelmean`: `recon_loss_mode=offdiag_channelmean_sum`, `vae_final_activation=tanh`.",
        "",
        "## Shared FAST Profile",
        "",
        "- channels: `[1,0,2]`",
        "- outer/inner folds: `3x3`",
        "- latent_dim: `128`",
        "- epochs/cycles/T0: `960 / 12 / 80`",
        "- beta: `2.5`",
        "- dropout: `0.15`",
        "- norm_mode: `zscore_offdiag`",
        "- stratification: `ResearchGroup_Mapped + Manufacturer`",
        "- Sex: metadata/covariate only",
        "- Python bandpass: OFF",
        "- Stage A: dummy canonical `logreg`, `n_iter_logreg=1`, ignored for ranking",
        "- Stage B: classifier-only `logreg_l2`, true inner-CV OOF threshold selection",
        "- primary threshold: `inner_oof_target_sens_ge_0p70_max_spec`",
        "",
        "## Planned Runs",
        "",
        "\n".join(md_lines),
        "",
        "## Status",
        "",
        f"- dry_run: `{str(dry_run).lower()}`",
        "- training_launched: `false` unless wrapper is run later with `--confirm-training`",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configs = [(path, load_config(path)) for path in CONFIGS]
    for path, cfg in configs:
        validate_config(path, cfg)
    tensor_info = inspect_tensor(resolve(configs[0][1]["paths"]["global_tensor_path"]))
    meta = load_metadata(resolve(configs[0][1]["paths"]["metadata_path"]))
    split_preview = write_split_preview(meta, output_root, configs[0][1]["parameters"])
    planned = make_planned_rows(configs, args.python_executable)
    planned.to_csv(output_root / "planned_runs.csv", index=False)
    split_preview.to_csv(output_root / "split_preview_summary.csv", index=False)
    write_readme(output_root, planned, dry_run=(args.dry_run or not args.confirm_training))
    command_log = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "dry_run": bool(args.dry_run or not args.confirm_training),
        "confirm_training": bool(args.confirm_training),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "tensor_info": tensor_info,
        "planned_runs": planned[["run_name", "recon_loss_mode", "vae_final_activation"]].to_dict(orient="records"),
    }
    (output_root / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== Objective-v2 FAST 3x3 [1,0,2] pilot ===")
    print(f"Mode: {'DRY-RUN' if args.dry_run or not args.confirm_training else 'REAL RUN'}")
    print(planned[["run_name", "channels_to_use", "recon_loss_mode", "vae_final_activation", "outer_folds", "inner_folds", "latent_dim", "epochs_vae"]].to_string(index=False))
    for row in planned.itertuples(index=False):
        print(f"\n[{row.run_name}] Stage A command:")
        print(row.stage_a_command)
        print(f"[{row.run_name}] Stage B command:")
        print(row.stage_b_command)

    if args.dry_run or not args.confirm_training:
        print("\nDry-run complete. No training launched.")
        return 0

    for config_path, cfg in configs:
        readout_dir = resolve(cfg["paths"]["classifier_only_output_dir"])
        if args.resume and readout_complete(readout_dir):
            print(f"[skip] {cfg['run_name']} readout already complete")
            continue
        prepare_real_run_dir(cfg, resume=args.resume)
        stage_a = build_stage_a_command(cfg, args.python_executable)
        stage_b = build_stage_b_command(cfg, args.python_executable)
        subprocess.run(stage_a, check=True, cwd=PROJECT_ROOT)
        subprocess.run(stage_b, check=True, cwd=PROJECT_ROOT)

    print("All requested real runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
