#!/usr/bin/env python3
"""Launcher for the exploratory FULL 5x5 VAE-pool composition run.

Default mode is dry-run. Real training requires --confirm-training. The only
scientific change versus the locked v5.1b [1,0,2] horizon4480 model is:

    vae_pool_composition_strategy: current_all_pool/default -> cn_ad_plus_matched_mci_pool

This branch is exploratory because diagnosis labels are used to compose the
fold-local VAE pool. The supervised AD/CN classifier pool must remain locked at
CN=300, AD=96.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/runs/adni_v5_1b_ch1_0_2_cn_ad_plus_matched_mci_pool_horizon4480_cycles56_full_5x5.json"
REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
COMPARE_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/compare_vae_pool_ablation_cn_ad_plus_matched_mci_full5x5.py"
AUDIT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/audit_vae_pool_ablation_cn_ad_plus_matched_mci_full5x5_integrity.py"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_POOL_STRATEGY = "cn_ad_plus_matched_mci_pool"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_READOUT = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
STALE_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B execution.")
    parser.add_argument("--force-clean", action="store_true", help="Quarantine stale output contents before real training.")
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-integrity-audit", action="store_true")
    parser.add_argument("--python-executable", default=None)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.6f}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_df_pair(root: Path, stem: str, df: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def append_arg(command: List[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
        return
    command.append(f"--{name}")
    if isinstance(value, (list, tuple)):
        command.extend(str(v) for v in value)
    else:
        command.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if str(token).startswith("--"):
            break
        out.append(str(token))
    return out


def scientific_param_diffs(reference: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    ref = dict(reference["parameters"])
    tgt = dict(target["parameters"])
    ref.setdefault("vae_pool_composition_strategy", "current_all_pool")
    tgt.setdefault("vae_pool_composition_strategy", "current_all_pool")
    diffs = {}
    for key in sorted(set(ref) | set(tgt)):
        if ref.get(key) != tgt.get(key):
            diffs[key] = {"reference": ref.get(key), "candidate": tgt.get(key)}
    return diffs


def validate_config(config: Dict[str, Any], reference: Dict[str, Any]) -> None:
    params = config["parameters"]
    diffs = scientific_param_diffs(reference, config)
    require_equal(
        diffs,
        {"vae_pool_composition_strategy": {"reference": "current_all_pool", "candidate": EXPECTED_POOL_STRATEGY}},
        "effective scientific parameter diffs",
    )
    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(params["outer_folds"], 5, "outer_folds")
    require_equal(params["inner_folds"], 5, "inner_folds")
    require_equal(params["epochs_vae"], 4480, "epochs_vae")
    require_equal(params["cyclical_beta_n_cycles"], 56, "cyclical_beta_n_cycles")
    require_equal(params["lr_scheduler_T0"], 80, "lr_scheduler_T0")
    require_equal(params["beta_vae"], 2.5, "beta_vae")
    require_equal(params["latent_dim"], 256, "latent_dim")
    require_equal(params["batch_size"], 64, "batch_size")
    require_equal(params["dropout_rate_vae"], 0.15, "dropout_rate_vae")
    require_equal(params.get("vae_dropout_scope", "legacy_all"), "legacy_all", "vae_dropout_scope")
    require_equal(params.get("vae_block_order", "legacy_act_norm"), "legacy_act_norm", "vae_block_order")
    require_equal(params["decoder_type"], "convtranspose", "decoder_type")
    require_equal(params["recon_loss_mode"], "mse_sum_batchmean_current", "recon_loss_mode")
    require_equal(params["metadata_features"], ["Age", "Sex"], "metadata_features")
    require_equal(params["classifier_stratify_cols"], ["Manufacturer"], "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], ["Manufacturer"], "vae_stratify_cols")
    require_equal(params.get("vae_train_sampler_strategy", "none"), "none", "vae_train_sampler_strategy")
    require_equal(params["vae_pool_composition_strategy"], EXPECTED_POOL_STRATEGY, "vae_pool_composition_strategy")
    if config["paths"]["global_tensor_path"] != reference["paths"]["global_tensor_path"]:
        raise RuntimeError("global_tensor_path changed versus reference.")
    if config["paths"]["metadata_path"] != reference["paths"]["metadata_path"]:
        raise RuntimeError("metadata_path changed versus reference.")
    if "OASIS" in json.dumps(config) or "oasis" in json.dumps(config):
        raise RuntimeError("Config unexpectedly references OASIS.")


def normalize_dx(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    low = text.lower()
    if low in {"cn", "control", "normal"}:
        return "CN"
    if low in {"ad", "ad_dementia", "dementia"}:
        return "AD"
    if low == "mci":
        return "MCI"
    return text


def normalize_mfr(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    low = text.lower()
    if low in {"ge", "general electric", "ge medical systems"}:
        return "GE"
    if low in {"philips", "philips medical systems"}:
        return "Philips"
    if low in {"siemens", "siemens healthineers"}:
        return "SIEMENS"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tensor_idx" not in df.columns and "tensor_index" in df.columns:
        df = df.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Metadata missing columns: {missing}")
    df = df.copy()
    df["_metadata_order"] = np.arange(len(df))
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
    df["Manufacturer"] = df["Manufacturer"].map(normalize_mfr)
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["tensor_idx"] = df["tensor_idx"].astype(int)
    return df


def stratify_key(df: pd.DataFrame, stratify_cols: Sequence[str], n_splits: int) -> pd.Series:
    key = df["ResearchGroup_Mapped"].astype(str)
    for col in stratify_cols:
        if col in df.columns:
            key = key + "__" + df[col].fillna("NA").astype(str)
    if (key.value_counts() < n_splits).any():
        return df["ResearchGroup_Mapped"].astype(str)
    return key


def counts(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    out = {f"{prefix}_n": int(len(df))}
    for dx in ["CN", "AD", "MCI"]:
        out[f"{prefix}_{dx.lower()}"] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        out[f"{prefix}_mfr_{mfr}"] = int(df["Manufacturer"].eq(mfr).sum())
    return out


def apply_cn_ad_plus_matched_mci(vae_pool: pd.DataFrame, seed: int, fold: int) -> pd.DataFrame:
    groups = vae_pool["ResearchGroup_Mapped"].fillna("UNKNOWN").astype(str)
    cn_ad = vae_pool[groups.isin(["CN", "AD"])].copy()
    mci = vae_pool[groups.eq("MCI")].copy()
    ad_n = int(groups.eq("AD").sum())
    if ad_n <= 0 or len(mci) <= 0:
        raise RuntimeError(f"Fold {fold}: cannot sample MCI matched to AD count.")
    selected_mci = mci.sample(n=min(ad_n, len(mci)), replace=False, random_state=seed + fold * 1009)
    return pd.concat([cn_ad, selected_mci], ignore_index=False).sort_values("_metadata_order").reset_index(drop=True)


def split_and_pool_preflight(meta: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    params = config["parameters"]
    seed = int(params["seed"])
    n_splits = int(params["outer_folds"])
    classifier = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    pool_counts = classifier["ResearchGroup_Mapped"].value_counts()
    require_equal(int(len(classifier)), 396, "classifier pool n")
    require_equal(int(pool_counts.get("CN", 0)), 300, "classifier pool CN")
    require_equal(int(pool_counts.get("AD", 0)), 96, "classifier pool AD")
    key = stratify_key(classifier, ["Manufacturer"], n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_tensor_idx = set(meta["tensor_idx"].astype(int).tolist())
    rows: List[Dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(classifier)), key), start=1):
        train_dev = classifier.iloc[train_dev_idx].copy()
        test = classifier.iloc[test_idx].copy()
        test_idx_set = set(test["tensor_idx"].astype(int).tolist())
        vae_base = meta[meta["tensor_idx"].astype(int).isin(sorted(all_tensor_idx - test_idx_set))].copy()
        selected = apply_cn_ad_plus_matched_mci(vae_base, seed=seed, fold=fold)
        overlap = set(selected["tensor_idx"].astype(int).tolist()).intersection(test_idx_set)
        row: Dict[str, Any] = {
            "fold": fold,
            "vae_pool_composition_strategy": EXPECTED_POOL_STRATEGY,
            "outer_test_overlap_n": int(len(overlap)),
            "classifier_pool_n": int(len(classifier)),
            "classifier_pool_cn": int(pool_counts.get("CN", 0)),
            "classifier_pool_ad": int(pool_counts.get("AD", 0)),
            "exploratory_uses_diagnosis_for_pool_composition": True,
        }
        row.update(counts(train_dev, "classifier_train_dev"))
        row.update(counts(test, "classifier_test"))
        row.update(counts(vae_base, "vae_pool_before"))
        row.update(counts(selected, "vae_pool_after"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df["outer_test_overlap_n"].eq(0).all():
        raise RuntimeError("VAE pool selection overlaps outer classifier test subjects.")
    return df


def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    params = config["parameters"]
    paths = config["paths"]
    cmd = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        str(resolve(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve(paths["metadata_path"])),
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for key, value in params.items():
        append_arg(cmd, key, value)
    require_equal(values_after_flag(cmd, "--vae_pool_composition_strategy"), [EXPECTED_POOL_STRATEGY], "Stage A pool strategy")
    require_equal(values_after_flag(cmd, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(cmd, "--outer_folds"), ["5"], "Stage A outer folds")
    require_equal(values_after_flag(cmd, "--inner_folds"), ["5"], "Stage A inner folds")
    return cmd


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    params = config["parameters"]
    return [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(outdir),
        "--output-dir",
        str(outdir / "classifier_only_readout"),
        "--models",
        PRIMARY_MODEL,
        "--readout-feature-sets",
        PRIMARY_READOUT,
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--reuse-latent-cache",
    ]


def build_compare_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [python_exe, str(COMPARE_SCRIPT), "--candidate-run-dir", str(outdir)]


def build_audit_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [python_exe, str(AUDIT_SCRIPT), "--candidate-run-dir", str(outdir)]


def stale_markers(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    markers = []
    for child in output_dir.iterdir():
        if child.name in STALE_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    return sorted(markers)


def quarantine_output(output_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = output_dir.parent / f"{output_dir.name}_quarantine_{stamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = output_dir.parent / f"{output_dir.name}_quarantine_{stamp}_{suffix}"
        suffix += 1
    quarantine.mkdir(parents=True)
    for child in list(output_dir.iterdir()):
        shutil.move(str(child), str(quarantine / child.name))
    return quarantine


def prepare_output(output_dir: Path, force_clean: bool) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale = stale_markers(output_dir)
    if stale and not force_clean:
        raise RuntimeError(
            "Refusing to start real training with existing run artifacts. Use --force-clean to quarantine them.\n"
            + "\n".join(str(p) for p in stale[:20])
        )
    if stale and force_clean:
        return quarantine_output(output_dir)
    return None


def verify_fresh_checkpoints(config: Dict[str, Any], run_start: float) -> pd.DataFrame:
    outdir = resolve(config["paths"]["output_dir"])
    rows = []
    for fold in range(1, 6):
        ckpt = outdir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        mtime = ckpt.stat().st_mtime if ckpt.exists() else np.nan
        rows.append(
            {
                "fold": fold,
                "checkpoint_path": str(ckpt),
                "exists": ckpt.exists(),
                "mtime_epoch": mtime,
                "fresh_after_launcher_start": bool(ckpt.exists() and mtime > run_start),
            }
        )
    df = pd.DataFrame(rows)
    if not df["fresh_after_launcher_start"].all():
        raise RuntimeError("Not all fold checkpoints are fresh after launcher start:\n" + df.to_string(index=False))
    return df


def run_command(cmd: Sequence[str]) -> int:
    print(shlex.join(cmd))
    return subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), check=False).returncode


def write_package_docs(outdir: Path, config: Dict[str, Any], pool_df: pd.DataFrame, commands: Mapping[str, Sequence[str]], dry_run: bool) -> None:
    write_df_pair(outdir, "pool_composition_preflight", pool_df)
    manifest = {
        "created_utc": now(),
        "run_name": config["run_name"],
        "exploratory": True,
        "controlled_scientific_diff": "vae_pool_composition_strategy current_all_pool/default -> cn_ad_plus_matched_mci_pool",
        "classifier_pool_guard": "n=396, CN=300, AD=96",
        "outer_test_overlap_guard": "VAE pool selection is fold-local and excludes outer classifier test subjects.",
        "ranking_source": f"Stage B classifier-only {PRIMARY_MODEL} on {PRIMARY_READOUT}",
        "threshold_strategy": PRIMARY_THRESHOLD,
        "stage_a_command": list(commands["stage_a"]),
        "stage_a_command_shell": shlex.join(commands["stage_a"]),
        "stage_b_command": list(commands["stage_b"]),
        "stage_b_command_shell": shlex.join(commands["stage_b"]),
        "comparison_command": list(commands["compare"]),
        "comparison_command_shell": shlex.join(commands["compare"]),
        "integrity_audit_command": list(commands["audit"]),
        "integrity_audit_command_shell": shlex.join(commands["audit"]),
        "dry_run": dry_run,
    }
    write_json(outdir / "run_manifest_preflight.json", manifest)
    readme = f"""# VAE Pool-Composition FULL 5x5 Exploratory Package

This package prepares a controlled FULL 5x5 confirmation of
`cn_ad_plus_matched_mci_pool` for the locked ADNI v5.1b 140TR `[1,0,2]` model.

The only scientific change versus the locked v5.1b horizon4480/cycles56 model
is `vae_pool_composition_strategy=cn_ad_plus_matched_mci_pool`.

This branch is exploratory because diagnosis labels are used to alter the VAE
pool composition within each outer-training fold. The supervised classifier
pool remains locked at `n=396`, `CN=300`, `AD=96`.

Primary ranking after a real run must use Stage B classifier-only `logreg_l2`
with `z_plus_age_sex` and `{PRIMARY_THRESHOLD}`.

No real training was launched by package preparation.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def write_real_run_manifest(outdir: Path, config: Dict[str, Any], commands: Mapping[str, Sequence[str]], quarantine: Path | None) -> None:
    write_json(
        outdir / "run_manifest.json",
        {
            "created_utc": now(),
            "run_name": config["run_name"],
            "exploratory": True,
            "controlled_scientific_diff": "vae_pool_composition_strategy current_all_pool/default -> cn_ad_plus_matched_mci_pool",
            "classifier_pool_guard": "n=396, CN=300, AD=96",
            "stage_a_command": list(commands["stage_a"]),
            "stage_a_command_shell": shlex.join(commands["stage_a"]),
            "stage_b_command": list(commands["stage_b"]),
            "stage_b_command_shell": shlex.join(commands["stage_b"]),
            "comparison_command": list(commands["compare"]),
            "comparison_command_shell": shlex.join(commands["compare"]),
            "integrity_audit_command": list(commands["audit"]),
            "integrity_audit_command_shell": shlex.join(commands["audit"]),
            "quarantine_dir": str(quarantine) if quarantine else "",
            "training_launched": True,
        },
    )


def main() -> int:
    args = parse_args()
    if args.dry_run and args.confirm_training:
        raise SystemExit("Use --dry-run or --confirm-training, not both.")
    dry_run = bool(args.dry_run or not args.confirm_training)
    config = load_json(resolve(args.config))
    reference = load_json(resolve(args.reference_config))
    validate_config(config, reference)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    outdir = resolve(config["paths"]["output_dir"])
    tensor_path = resolve(config["paths"]["global_tensor_path"])
    metadata_path = resolve(config["paths"]["metadata_path"])
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    meta = load_metadata(metadata_path)
    pool_df = split_and_pool_preflight(meta, config)
    stage_a = build_stage_a_command(config, python_exe)
    stage_b = build_stage_b_command(config, python_exe)
    compare = build_compare_command(config, python_exe)
    audit = build_audit_command(config, python_exe)
    commands = {"stage_a": stage_a, "stage_b": stage_b, "compare": compare, "audit": audit}
    outdir.mkdir(parents=True, exist_ok=True)
    write_package_docs(outdir, config, pool_df, commands, dry_run=dry_run)
    write_json(
        outdir / "command_log.json",
        {
            "timestamp": now(),
            "dry_run": dry_run,
            "confirm_training": bool(args.confirm_training),
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "locked_model_outputs_modified": False,
        },
    )

    print(f"Config: {resolve(args.config)}")
    print(f"Output: {outdir}")
    print("Mode:", "DRY-RUN" if dry_run else "REAL TRAINING")
    print("Controlled diff: vae_pool_composition_strategy -> cn_ad_plus_matched_mci_pool")
    print(pool_df[["fold", "classifier_pool_n", "classifier_pool_cn", "classifier_pool_ad", "vae_pool_after_n", "vae_pool_after_cn", "vae_pool_after_mci", "vae_pool_after_ad", "outer_test_overlap_n"]].to_string(index=False))
    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B command:")
    print(shlex.join(stage_b))
    print("\nComparison command:")
    print(shlex.join(compare))
    print("\nIntegrity audit command:")
    print(shlex.join(audit))
    stale = stale_markers(outdir)
    if stale:
        print(f"\nExisting run-artifact markers detected: {len(stale)}")
        for path in stale[:20]:
            print(f"  - {path}")
    if dry_run:
        print("\nDry-run complete. No training launched.")
        return 0

    quarantine = prepare_output(outdir, force_clean=args.force_clean)
    if quarantine:
        print(f"Quarantined existing output contents: {quarantine}")
    write_real_run_manifest(outdir, config, commands, quarantine)
    run_start = time.time()
    rc = run_command(stage_a)
    if rc != 0:
        return rc
    fresh = verify_fresh_checkpoints(config, run_start)
    write_df_pair(outdir, "fresh_checkpoint_validation", fresh)
    if args.skip_classifier_readout:
        return 0
    rc = run_command(stage_b)
    if rc != 0:
        return rc
    if not args.skip_integrity_audit:
        rc = run_command(audit)
        if rc != 0:
            return rc
    else:
        print("Integrity audit skipped. Run:", shlex.join(audit))
    if not args.skip_comparison:
        rc = run_command(compare)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
