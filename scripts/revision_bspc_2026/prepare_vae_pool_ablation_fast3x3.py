#!/usr/bin/env python
"""Prepare a dry-run VAE training-pool ablation package.

This script is intentionally read-only with respect to tensors, metadata,
ledgers, and existing model outputs. It writes only the requested preflight
artifacts under results/revision_bspc_2026/vae_pool_ablation_fast3x3/.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/vae_pool_ablation_fast3x3"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGEB_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"

CANDIDATES: Tuple[str, ...] = (
    "current_all_pool",
    "cn_ad_only_pool",
    "balanced_cn_ad_mci_pool",
    "cn_ad_plus_matched_mci_pool",
)

GROUP_ORDER: Tuple[str, ...] = ("CN", "MCI", "AD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--candidate", choices=[*CANDIDATES, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Prepare artifacts and validate Stage A dry-run commands.")
    parser.add_argument(
        "--skip-stagea-command-validation",
        action="store_true",
        help="Do not execute the training script --dry-run argument validation commands.",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Accepted only to make the safety contract explicit. This preflight script does not launch training.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows) + "\n"


def write_df_pair(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_text(md_path, df_to_md(df))


def normalize_dx(value: Any) -> str:
    s = str(value).strip()
    low = s.lower()
    if low in {"cn", "control", "normal"}:
        return "CN"
    if low in {"ad", "ad_dementia", "dementia"}:
        return "AD"
    if low == "mci":
        return "MCI"
    return s


def normalize_mfr(value: Any) -> str:
    s = str(value).strip()
    low = s.lower()
    if low in {"ge", "general electric", "ge medical systems"}:
        return "GE"
    if low in {"philips", "philips medical systems"}:
        return "Philips"
    if low in {"siemens", "siemens healthineers"}:
        return "SIEMENS"
    return s


def classifier_stratify_key(df: pd.DataFrame, stratify_cols: Sequence[str]) -> pd.Series:
    key = df["ResearchGroup_Mapped"].astype(str)
    for col in stratify_cols:
        if col in df.columns:
            key = key + "__" + df[col].fillna("NA").astype(str)
    counts = key.value_counts()
    too_small = sorted(counts[counts < 3].index.tolist())
    if too_small:
        # Match the pipeline's conservative intent: use diagnosis when an extra
        # stratum cannot support 3-fold splitting.
        return df["ResearchGroup_Mapped"].astype(str)
    return key


def counts_by_group(df: pd.DataFrame) -> Dict[str, int]:
    vc = df["ResearchGroup_Mapped"].value_counts()
    return {f"n_{g.lower()}": int(vc.get(g, 0)) for g in GROUP_ORDER}


def sample_group(df: pd.DataFrame, group: str, n: int, random_state: int) -> pd.DataFrame:
    sub = df[df["ResearchGroup_Mapped"].eq(group)]
    if n >= len(sub):
        return sub.copy()
    return sub.sample(n=n, random_state=random_state, replace=False)


def apply_pool_strategy(df: pd.DataFrame, strategy: str, seed: int, fold: int) -> Tuple[pd.DataFrame, str]:
    if strategy == "current_all_pool":
        return df.copy(), "Current behavior: all fold-local CN/MCI/AD subjects except the outer classifier test subjects."
    if strategy == "cn_ad_only_pool":
        selected = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
        return selected, "Exploratory: VAE pool restricted to fold-local CN and AD only."
    if strategy == "balanced_cn_ad_mci_pool":
        counts = df["ResearchGroup_Mapped"].value_counts()
        missing = [g for g in GROUP_ORDER if int(counts.get(g, 0)) == 0]
        if missing:
            raise RuntimeError(f"Cannot balance CN/AD/MCI in fold {fold}: missing {missing}")
        target_n = min(int(counts.get(g, 0)) for g in GROUP_ORDER)
        parts = [
            sample_group(df, group, target_n, seed + fold * 1009 + idx * 17)
            for idx, group in enumerate(GROUP_ORDER)
        ]
        selected = pd.concat(parts, axis=0).sort_values("_metadata_order").copy()
        return selected, f"Exploratory: balanced CN/MCI/AD with n={target_n} per group, sampled fold-locally."
    if strategy == "cn_ad_plus_matched_mci_pool":
        counts = df["ResearchGroup_Mapped"].value_counts()
        ad_n = int(counts.get("AD", 0))
        mci_n = int(counts.get("MCI", 0))
        target_mci = min(ad_n, mci_n)
        cn_ad = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])]
        mci = sample_group(df, "MCI", target_mci, seed + fold * 1009 + 71)
        selected = pd.concat([cn_ad, mci], axis=0).sort_values("_metadata_order").copy()
        return selected, f"Exploratory: all fold-local CN/AD plus MCI sampled to n={target_mci}."
    raise ValueError(f"Unknown strategy: {strategy}")


def shlex_join(parts: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def add_flag(parts: List[str], flag: str, value: Any | None = None) -> None:
    if value is True:
        parts.append(flag)
    elif value is False or value is None:
        return
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        parts.append(flag)
        parts.extend(str(x) for x in value)
    else:
        parts.extend([flag, str(value)])


def build_stagea_command(
    python_exe: str,
    cfg: Mapping[str, Any],
    params: Mapping[str, Any],
    run_dir: Path,
    strategy: str,
    dry_run: bool,
) -> str:
    paths = cfg["paths"]
    parts: List[str] = [
        python_exe,
        str(TRAINING_SCRIPT.relative_to(PROJECT_ROOT)),
        "--global_tensor_path",
        paths["global_tensor_path"],
        "--metadata_path",
        paths["metadata_path"],
        "--output_dir",
        str(run_dir),
    ]
    add_flag(parts, "--channels_to_use", [1, 0, 2])
    add_flag(parts, "--outer_folds", 3)
    add_flag(parts, "--inner_folds", 3)
    add_flag(parts, "--repeated_outer_folds_n_repeats", 1)
    add_flag(parts, "--classifier_stratify_cols", ["Manufacturer"])
    add_flag(parts, "--vae_stratify_cols", ["Manufacturer"])
    add_flag(parts, "--num_conv_layers_encoder", params.get("num_conv_layers_encoder", 4))
    add_flag(parts, "--decoder_type", params.get("decoder_type", "convtranspose"))
    add_flag(parts, "--latent_dim", 256)
    add_flag(parts, "--lr_vae", params.get("lr_vae", 1e-4))
    add_flag(parts, "--epochs_vae", 960)
    add_flag(parts, "--batch_size", params.get("batch_size", 64))
    add_flag(parts, "--beta_vae", params.get("beta_vae", 2.5))
    add_flag(parts, "--recon_loss_mode", params.get("recon_loss_mode", "mse_sum_batchmean_current"))
    add_flag(parts, "--cyclical_beta_n_cycles", 12)
    add_flag(parts, "--cyclical_beta_ratio_increase", params.get("cyclical_beta_ratio_increase", 0.4))
    add_flag(parts, "--weight_decay_vae", params.get("weight_decay_vae", 5e-7))
    add_flag(parts, "--vae_final_activation", params.get("vae_final_activation", "tanh"))
    add_flag(parts, "--intermediate_fc_dim_vae", params.get("intermediate_fc_dim_vae", "quarter"))
    add_flag(parts, "--dropout_rate_vae", params.get("dropout_rate_vae", 0.15))
    add_flag(parts, "--vae_dropout_scope", params.get("vae_dropout_scope", "legacy_all"))
    add_flag(parts, "--vae_block_order", params.get("vae_block_order", "legacy_act_norm"))
    add_flag(parts, "--vae_encoder_norm_mode", params.get("vae_encoder_norm_mode", "groupnorm"))
    add_flag(parts, "--vae_train_sampler_strategy", params.get("vae_train_sampler_strategy", "none"))
    add_flag(parts, "--vae_pool_composition_strategy", strategy)
    add_flag(parts, "--vae_val_split_ratio", params.get("vae_val_split_ratio", 0.2))
    add_flag(parts, "--early_stopping_patience_vae", params.get("early_stopping_patience_vae", 320))
    add_flag(parts, "--lr_scheduler_patience_vae", params.get("lr_scheduler_patience_vae", 15))
    add_flag(parts, "--lr_scheduler_type", params.get("lr_scheduler_type", "cosine_warm"))
    add_flag(parts, "--lr_scheduler_T0", 80)
    add_flag(parts, "--lr_scheduler_eta_min", params.get("lr_scheduler_eta_min", 5e-7))
    add_flag(parts, "--classifier_types", ["logreg"])
    add_flag(parts, "--latent_features_type", params.get("latent_features_type", "mu"))
    add_flag(parts, "--gridsearch_scoring", params.get("gridsearch_scoring", "roc_auc"))
    add_flag(parts, "--classifier_use_class_weight", params.get("classifier_use_class_weight", True))
    add_flag(parts, "--classifier_calibrate", params.get("classifier_calibrate", True))
    add_flag(parts, "--metadata_features", ["Age", "Sex"])
    add_flag(parts, "--n_iter_logreg", 1)
    add_flag(parts, "--norm_mode", params.get("norm_mode", "zscore_offdiag"))
    add_flag(parts, "--seed", params.get("seed", 42))
    add_flag(parts, "--num_workers", params.get("num_workers", 4))
    add_flag(parts, "--n_jobs_gridsearch", params.get("n_jobs_gridsearch", 8))
    add_flag(parts, "--log_interval_epochs_vae", params.get("log_interval_epochs_vae", 10))
    add_flag(parts, "--save_fold_artefacts", True)
    add_flag(parts, "--save_vae_training_history", True)
    add_flag(parts, "--qc_analyze_distributions", params.get("qc_analyze_distributions", True))
    add_flag(parts, "--qc_check_scanner_leakage", params.get("qc_check_scanner_leakage", True))
    add_flag(parts, "--qc_rate_distortion", params.get("qc_rate_distortion", True))
    add_flag(parts, "--qc_latent_information", params.get("qc_latent_information", True))
    add_flag(parts, "--qc_mi_n_neighbors", params.get("qc_mi_n_neighbors", 3))
    add_flag(parts, "--qc_mi_top_k", params.get("qc_mi_top_k", 10))
    add_flag(parts, "--qc_rd_log_base", params.get("qc_rd_log_base", 2.0))
    add_flag(parts, "--qc_tc_ridge", params.get("qc_tc_ridge", 1e-6))
    add_flag(parts, "--qc_var_eps_active", params.get("qc_var_eps_active", 1e-4))
    if dry_run:
        parts.append("--dry-run")
    return shlex_join(parts)


def build_stageb_command(python_exe: str, run_dir: Path) -> str:
    parts = [
        python_exe,
        str(STAGEB_SCRIPT.relative_to(PROJECT_ROOT)),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(run_dir / "classifier_only_readout"),
        "--models",
        "logreg_l2",
        "--readout-feature-sets",
        "z_plus_age_sex",
        "--outer-folds",
        "3",
        "--inner-folds",
        "3",
        "--reuse-latent-cache",
    ]
    return shlex_join(parts)


def config_for_candidate(
    cfg: Mapping[str, Any],
    strategy: str,
    run_dir: Path,
    stagea_command: str,
    stageb_command: str,
) -> Dict[str, Any]:
    new_cfg = json.loads(json.dumps(cfg))
    new_cfg["run_name"] = f"adni_v5_1_batch20260514b_vae_pool_ablation_fast3x3_{strategy}"
    new_cfg["description"] = (
        "FAST 3x3 dry-run VAE pool-composition ablation for locked ADNI 140TR [1,0,2]. "
        "Non-current VAE pool strategies are exploratory because diagnosis labels influence "
        "the unsupervised VAE pool composition within each outer-training fold."
    )
    new_cfg["paths"]["output_dir"] = str(run_dir)
    new_cfg["parameters"].update(
        {
            "channels_to_use": [1, 0, 2],
            "outer_folds": 3,
            "inner_folds": 3,
            "epochs_vae": 960,
            "cyclical_beta_n_cycles": 12,
            "lr_scheduler_T0": 80,
            "classifier_types": ["logreg"],
            "n_iter_logreg": 1,
            "metadata_features": ["Age", "Sex"],
            "vae_pool_composition_strategy": strategy,
            "stageb_classifier_only_model": "logreg_l2",
            "stageb_readout_feature_set": "z_plus_age_sex",
            "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
        }
    )
    new_cfg["planned_commands"] = {
        "stage_a_train_vae_dummy_logreg": stagea_command,
        "stage_b_classifier_only_logreg_l2": stageb_command,
    }
    return new_cfg


def load_metadata(cfg: Mapping[str, Any]) -> pd.DataFrame:
    metadata_path = Path(cfg["paths"]["metadata_path"])
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    df = pd.read_csv(metadata_path)
    if "SubjectID" not in df.columns:
        raise RuntimeError("Metadata missing SubjectID")
    if "ResearchGroup_Mapped" not in df.columns:
        raise RuntimeError("Metadata missing ResearchGroup_Mapped")
    if "Manufacturer" not in df.columns:
        raise RuntimeError("Metadata missing Manufacturer")
    df = df.copy()
    df["_metadata_order"] = np.arange(len(df))
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].map(normalize_dx)
    df["Manufacturer"] = df["Manufacturer"].map(normalize_mfr)
    if "tensor_idx" not in df.columns:
        df["tensor_idx"] = np.arange(len(df))
    return df


def simulate_pool_composition(cfg: Mapping[str, Any], candidates: Sequence[str]) -> pd.DataFrame:
    params = cfg["parameters"]
    seed = int(params.get("seed", 42))
    df = load_metadata(cfg)
    classifier_df = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    classifier_df = classifier_df.dropna(subset=["SubjectID", "ResearchGroup_Mapped", "Manufacturer"])
    y = classifier_df["ResearchGroup_Mapped"].to_numpy()
    stratify_key = classifier_stratify_key(classifier_df, ["Manufacturer"]).to_numpy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows: List[Dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(cv.split(classifier_df, stratify_key), start=1):
        train_dev_df = classifier_df.iloc[train_dev_idx].copy()
        test_df = classifier_df.iloc[test_idx].copy()
        test_tensor_idx = set(test_df["tensor_idx"].astype(int).tolist())
        vae_base = df[~df["tensor_idx"].astype(int).isin(test_tensor_idx)].copy()
        before_counts = counts_by_group(vae_base)
        for strategy in candidates:
            selected, note = apply_pool_strategy(vae_base, strategy, seed=seed, fold=fold)
            after_counts = counts_by_group(selected)
            train_dev_counts = counts_by_group(train_dev_df)
            test_counts = counts_by_group(test_df)
            selected_tensor_idx = set(selected["tensor_idx"].astype(int).tolist())
            rows.append(
                {
                    "candidate": strategy,
                    "fold": fold,
                    "classifier_train_dev_n": int(len(train_dev_df)),
                    "classifier_train_dev_cn": train_dev_counts["n_cn"],
                    "classifier_train_dev_ad": train_dev_counts["n_ad"],
                    "classifier_test_n": int(len(test_df)),
                    "classifier_test_cn": test_counts["n_cn"],
                    "classifier_test_ad": test_counts["n_ad"],
                    "vae_pool_before_n": int(len(vae_base)),
                    "vae_pool_before_cn": before_counts["n_cn"],
                    "vae_pool_before_mci": before_counts["n_mci"],
                    "vae_pool_before_ad": before_counts["n_ad"],
                    "vae_pool_after_n": int(len(selected)),
                    "vae_pool_after_cn": after_counts["n_cn"],
                    "vae_pool_after_mci": after_counts["n_mci"],
                    "vae_pool_after_ad": after_counts["n_ad"],
                    "outer_test_overlap_n": int(len(selected_tensor_idx.intersection(test_tensor_idx))),
                    "exploratory_uses_diagnosis_for_pool_composition": strategy != "current_all_pool",
                    "selection_note": note,
                }
            )
    return pd.DataFrame(rows)


def validate_stagea_commands(commands: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for command in commands:
        started = datetime.now().isoformat(timespec="seconds")
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "command": command,
                "started_at": started,
                "returncode": int(proc.returncode),
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
        )
        if proc.returncode != 0:
            break
    return rows


def write_readme(outdir: Path, cfg: Mapping[str, Any], manifest: pd.DataFrame) -> None:
    paths = cfg["paths"]
    lines = [
        "# VAE Pool Ablation FAST 3x3 Preflight",
        "",
        "This package prepares a read-only/dry-run ablation of the fold-local VAE training pool for the locked ADNI 140TR [1,0,2] tensor.",
        "",
        "No VAE training, classifier training, tensor rebuild, metadata edit, ledger edit, or existing model-output modification was performed by this preflight.",
        "",
        "## Locked Inputs",
        "",
        f"- Tensor: `{paths['global_tensor_path']}`",
        f"- Metadata: `{paths['metadata_path']}`",
        "- Channels: `[1,0,2]`",
        "- ROI order: locked ADNI 131 ROI order from the existing tensor branch",
        "- FAST split: outer_folds=3, inner_folds=3, seed=42",
        "- VAE horizon: epochs_vae=960, cyclical_beta_n_cycles=12, lr_scheduler_T0=80",
        "- Stage A: VAE training with dummy canonical logreg only, n_iter_logreg=1; Stage A classifier metrics are not for ranking",
        "- Stage B: classifier-only logreg_l2 on latent mu + Age + Sex with true inner-CV OOF thresholding",
        "",
        "## Candidates",
        "",
        "- `current_all_pool`: CN+MCI+AD, current behavior.",
        "- `cn_ad_only_pool`: exploratory, VAE pool restricted to CN+AD in each outer training fold.",
        "- `balanced_cn_ad_mci_pool`: exploratory, equal CN/MCI/AD counts sampled within each outer training fold.",
        "- `cn_ad_plus_matched_mci_pool`: exploratory, all CN+AD plus MCI sampled to match AD count within each outer training fold.",
        "",
        "Non-current candidates use diagnosis labels to compose the otherwise diagnosis-agnostic VAE pool. They are therefore exploratory ablations, not default pipeline changes.",
        "",
        "## Generated Files",
        "",
        "- `run_manifest.csv/.md`",
        "- `pool_composition_by_fold.csv/.md`",
        "- `dry_run_report.md`",
        "- `primary_results.csv/.md` placeholder",
        "- `final_recommendation.md`",
        "- `command_log.json`",
        "- `configs/*.json` documentation configs for each planned candidate",
        "",
        "## Planned Runs",
        "",
        df_to_md(manifest[["candidate", "run_dir", "config_path", "stage_a_status", "stage_b_status"]]),
    ]
    write_text(outdir / "README.md", "\n".join(lines))


def write_final_recommendation(outdir: Path) -> None:
    text = """
# Final Recommendation

This is a dry-run package only. No ranking decision can be made until the FAST
3x3 Stage A/Stage B runs are explicitly launched and aggregated.

The current default behavior remains `current_all_pool`, using all eligible
CN/MCI/AD subjects in the fold-local VAE training pool after excluding the
outer classifier test subjects. The three alternative pool-composition
strategies are exploratory because they use diagnosis labels to alter the
unsupervised VAE pool composition.

If this package is later executed, ranking should use only Stage B
classifier-only `logreg_l2` metrics at
`inner_oof_target_sens_ge_0p70_max_spec`. Stage A dummy logreg metrics should
remain ignored for ranking.

No candidate should be considered for FULL 5x5 unless it improves FAST AUC and
PR-AUC versus `current_all_pool`, preserves BA/F1 and sensitivity, and does not
increase scanner/manufacturer leakage.
"""
    write_text(outdir / "final_recommendation.md", text)


def main() -> int:
    args = parse_args()
    if args.confirm_training:
        raise SystemExit(
            "This is a read-only/dry-run preparation script. It does not launch real training. "
            "Use the generated commands only after explicit confirmation in a separate run."
        )

    outdir = resolve(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(resolve(args.base_config))
    params = cfg["parameters"]
    paths = cfg["paths"]

    tensor_path = Path(paths["global_tensor_path"])
    metadata_path = Path(paths["metadata_path"])
    selected_candidates = list(CANDIDATES if args.candidate == "all" else [args.candidate])
    python_exe = cfg.get("python_executable", "/home/diego/anaconda3/envs/vae_ad/bin/python")

    validations: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(resolve(args.base_config)),
        "tensor_path_exists": tensor_path.exists(),
        "metadata_path_exists": metadata_path.exists(),
        "training_script_exists": TRAINING_SCRIPT.exists(),
        "stageb_script_exists": STAGEB_SCRIPT.exists(),
        "dry_run_requested": bool(args.dry_run),
        "real_training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_outputs_modified": False,
    }
    if not validations["tensor_path_exists"]:
        raise FileNotFoundError(tensor_path)
    if not validations["metadata_path_exists"]:
        raise FileNotFoundError(metadata_path)
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(TRAINING_SCRIPT)
    if not STAGEB_SCRIPT.exists():
        raise FileNotFoundError(STAGEB_SCRIPT)

    manifest_rows: List[Dict[str, Any]] = []
    stagea_dry_run_commands: List[str] = []
    for strategy in selected_candidates:
        run_dir = outdir / "runs" / strategy
        stagea_command = build_stagea_command(python_exe, cfg, params, run_dir, strategy, dry_run=False)
        stagea_dry_run_command = build_stagea_command(python_exe, cfg, params, run_dir, strategy, dry_run=True)
        stageb_command = build_stageb_command(python_exe, run_dir)
        config_path = outdir / "configs" / f"{strategy}.json"
        write_json(config_path, config_for_candidate(cfg, strategy, run_dir, stagea_command, stageb_command))
        manifest_rows.append(
            {
                "candidate": strategy,
                "channels_to_use": "[1,0,2]",
                "vae_pool_composition_strategy": strategy,
                "exploratory_uses_diagnosis_for_pool_composition": strategy != "current_all_pool",
                "recon_loss_mode": params.get("recon_loss_mode", "mse_sum_batchmean_current"),
                "epochs_vae": 960,
                "cyclical_beta_n_cycles": 12,
                "lr_scheduler_T0": 80,
                "outer_folds": 3,
                "inner_folds": 3,
                "stage_a_status": "planned_not_run",
                "stage_b_status": "planned_after_stage_a",
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "stage_a_command": stagea_command,
                "stage_a_dry_run_command": stagea_dry_run_command,
                "stage_b_command": stageb_command,
            }
        )
        stagea_dry_run_commands.append(stagea_dry_run_command)

    manifest = pd.DataFrame(manifest_rows)
    pool_comp = simulate_pool_composition(cfg, selected_candidates)
    placeholder = pd.DataFrame(
        [
            {
                "candidate": c,
                "model_name": "logreg_l2",
                "readout_feature_set": "z_plus_age_sex",
                "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
                "auc": np.nan,
                "pr_auc": np.nan,
                "balanced_accuracy": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "f1": np.nan,
                "status": "placeholder_not_run",
            }
            for c in selected_candidates
        ]
    )

    command_validation_rows: List[Dict[str, Any]] = []
    if args.dry_run and not args.skip_stagea_command_validation:
        command_validation_rows = validate_stagea_commands(stagea_dry_run_commands)
    validations["stagea_dry_run_commands_validated"] = bool(command_validation_rows)
    validations["stagea_dry_run_all_ok"] = all(r["returncode"] == 0 for r in command_validation_rows) if command_validation_rows else None

    write_df_pair(manifest, outdir / "run_manifest.csv", outdir / "run_manifest.md")
    write_df_pair(pool_comp, outdir / "pool_composition_by_fold.csv", outdir / "pool_composition_by_fold.md")
    write_df_pair(placeholder, outdir / "primary_results.csv", outdir / "primary_results.md")
    write_readme(outdir, cfg, manifest)
    write_final_recommendation(outdir)

    dry_run_lines = [
        "# Dry-Run Report",
        "",
        f"- Timestamp: `{validations['timestamp']}`",
        f"- Base config: `{validations['base_config']}`",
        f"- Tensor exists: `{validations['tensor_path_exists']}`",
        f"- Metadata exists: `{validations['metadata_path_exists']}`",
        f"- Training script exists: `{validations['training_script_exists']}`",
        f"- Stage B script exists: `{validations['stageb_script_exists']}`",
        f"- Real training launched: `{validations['real_training_launched']}`",
        f"- Stage A dry-run command validation: `{validations['stagea_dry_run_all_ok']}`",
        "",
        "## Stage A Dry-Run Command Validation",
        "",
    ]
    if command_validation_rows:
        validation_df = pd.DataFrame(command_validation_rows)
        validation_df["command"] = validation_df["command"].str.slice(0, 240) + " ..."
        dry_run_lines.append(df_to_md(validation_df[["returncode", "started_at", "command"]]))
    else:
        dry_run_lines.append("_Skipped by request or because --dry-run was not passed._\n")
    dry_run_lines.extend(
        [
            "",
            "## Safety",
            "",
            "- No training was launched.",
            "- No tensor, metadata, ledger, or existing model-output files were modified.",
            "- Generated commands omit `--dry-run` only in the manifest's planned Stage A command; the executed validation commands include `--dry-run`.",
        ]
    )
    write_text(outdir / "dry_run_report.md", "\n".join(dry_run_lines))

    command_log = {
        "validations": validations,
        "selected_candidates": selected_candidates,
        "candidate_commands": manifest_rows,
        "stagea_dry_run_validation": command_validation_rows,
        "outputs": {
            "README": str((outdir / "README.md").relative_to(PROJECT_ROOT)),
            "run_manifest": str((outdir / "run_manifest.csv").relative_to(PROJECT_ROOT)),
            "pool_composition_by_fold": str((outdir / "pool_composition_by_fold.csv").relative_to(PROJECT_ROOT)),
            "dry_run_report": str((outdir / "dry_run_report.md").relative_to(PROJECT_ROOT)),
            "primary_results_placeholder": str((outdir / "primary_results.csv").relative_to(PROJECT_ROOT)),
            "final_recommendation": str((outdir / "final_recommendation.md").relative_to(PROJECT_ROOT)),
        },
    }
    write_json(outdir / "command_log.json", command_log)

    print(f"Prepared VAE pool ablation dry-run package: {outdir.relative_to(PROJECT_ROOT)}")
    if validations["stagea_dry_run_all_ok"] is False:
        print("WARNING: at least one Stage A dry-run validation command failed; see dry_run_report.md")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
