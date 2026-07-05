#!/usr/bin/env python3
"""Exploratory FAST 3x3 third-channel look-ahead from [1,4].

This script deliberately does not alter the strict greedy state. It reuses the
validated FAST Stage A and Stage B machinery from the greedy wrapper, but writes
separate look-ahead outputs and treats all results as exploratory screening.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"

STRICT_GREEDY_SCRIPT = (
    PROJECT_ROOT
    / "scripts/revision_bspc_2026/run_adni_v5_1_batch20260514b_greedy_fast_channel_selection.py"
)
STRICT_GREEDY_ROOT = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_v5_1_batch20260514b_greedy_fast_channel_selection_3x3"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_greedy_fast_channel_selection_3x3_triple_lookahead_1_4"
)
BIG_DISK_ROOT = (
    Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
    / "adni_v5_1_batch20260514b_greedy_fast_channel_selection_3x3_triple_lookahead_1_4"
)
FULL_CH1_0_2_READOUT = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
)

CANDIDATES: List[List[int]] = [[1, 4, 0], [1, 4, 2], [1, 4, 3], [1, 4, 5], [1, 4, 6]]
LOOKAHEAD_BASE = [1, 4]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_greedy_module() -> Any:
    spec = importlib.util.spec_from_file_location("greedy_fast_wrapper", STRICT_GREEDY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {STRICT_GREEDY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BIG_DISK_ROOT = BIG_DISK_ROOT
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Plan and validate only; no training.")
    parser.add_argument("--run", action="store_true", help="Run Stage A and Stage B for the five look-ahead triples.")
    parser.add_argument("--resume", action="store_true", help="Skip triples with complete 3x3 classifier-only readout.")
    parser.add_argument("--confirm-training", action="store_true", help="Required with --run.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=PYTHON_EXE)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_expected_fast_profile(greedy: Any) -> None:
    greedy.validate_fast_preflight()
    params = greedy.FAST_PARAMS
    expected = {
        "outer_folds": 3,
        "inner_folds": 3,
        "latent_dim": 128,
        "epochs_vae": 960,
        "cyclical_beta_n_cycles": 12,
        "lr_scheduler_T0": 80,
        "beta_vae": 2.5,
        "dropout_rate_vae": 0.15,
        "norm_mode": "zscore_offdiag",
        "metadata_features": ["Age", "Sex"],
        "classifier_types": ["logreg"],
        "n_iter_logreg": 1,
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise RuntimeError(f"FAST profile drift: {key} expected {value!r}, got {params.get(key)!r}")
    if "Manufacturer" not in params["classifier_stratify_cols"] or "Manufacturer" not in params["vae_stratify_cols"]:
        raise RuntimeError("Manufacturer must be present in classifier and VAE stratification.")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only, not a stratification column.")


def validate_plan(greedy: Any, planned: pd.DataFrame) -> None:
    expected = [json.dumps(ch, separators=(",", ":")) for ch in CANDIDATES]
    observed = planned["channels"].astype(str).tolist()
    if observed != expected:
        raise RuntimeError(f"Expected exactly {expected}, got {observed}")
    if len(planned) != 5:
        raise RuntimeError(f"Expected exactly five candidates, got {len(planned)}")
    for _, row in planned.iterrows():
        greedy.validate_planned_row_for_launch(row)
        train_tokens = shlex.split(str(row["train_command"]))
        readout_tokens = shlex.split(str(row["readout_command"]))
        if greedy.values_after_flag(train_tokens, "--outer_folds") != ["3"]:
            raise RuntimeError("Stage A command preview does not show --outer_folds 3")
        if greedy.values_after_flag(train_tokens, "--inner_folds") != ["3"]:
            raise RuntimeError("Stage A command preview does not show --inner_folds 3")
        if greedy.values_after_flag(readout_tokens, "--outer-folds") != ["3"]:
            raise RuntimeError("Stage B command preview does not show --outer-folds 3")
        if greedy.values_after_flag(readout_tokens, "--inner-folds") != ["3"]:
            raise RuntimeError("Stage B command preview does not show --inner-folds 3")


def build_plan(greedy: Any, output_root: Path, channel_names: Sequence[str], python_executable: str) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    planned = greedy.build_planned_candidates(
        output_root=output_root,
        candidates=CANDIDATES,
        step=3,
        selected_before_step=LOOKAHEAD_BASE,
        channel_names=channel_names,
        python_executable=python_executable,
    )
    validate_plan(greedy, planned)
    planned.to_csv(output_root / "planned_triple_lookahead_candidates.csv", index=False)
    return planned


def metric_mean_se(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    mean = float(numeric.mean())
    se = float(numeric.std(ddof=1) / np.sqrt(len(numeric))) if len(numeric) > 1 else 0.0
    return mean, se


def add_weak_fold_metrics(row: Dict[str, Any], readout_dir: Path) -> None:
    foldwise = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    focus = foldwise[
        foldwise["model_name"].eq(PRIMARY_MODEL)
        & foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    if focus.empty:
        return
    weak = focus.sort_values(["auc", "balanced_accuracy"], ascending=[True, True]).iloc[0]
    row["weak_fold"] = int(weak["fold"])
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        row[f"weak_fold_{metric}"] = float(weak[metric])
    fold4 = focus[focus["fold"].eq(4)]
    if not fold4.empty:
        f4 = fold4.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            row[f"fold4_{metric}"] = float(f4[metric])


def aggregate_triples(greedy: Any, output_root: Path, channel_names: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for channels in CANDIDATES:
        dirs = greedy.output_dirs(output_root, channels)
        greedy.verify_readout(
            dirs["readout_dir"],
            expected_outer_folds=int(greedy.FAST_PARAMS["outer_folds"]),
            expected_inner_folds=int(greedy.FAST_PARAMS["inner_folds"]),
        )
        row = greedy.summarize_readout(output_root, channels, channel_names)
        add_weak_fold_metrics(row, dirs["readout_dir"])
        row["analysis_role"] = "exploratory_triple_lookahead_from_[1,4]"
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(
        ["mean_outer_auc", "mean_outer_pr_auc", "mean_outer_balanced_accuracy"],
        ascending=[False, False, False],
    )
    metrics.to_csv(output_root / "triple_lookahead_metrics.csv", index=False)
    ranking = metrics.reset_index(drop=True).copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_root / "triple_lookahead_ranking.csv", index=False)
    return metrics


def strict_reference_rows() -> pd.DataFrame:
    path = STRICT_GREEDY_ROOT / "channel_set_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(path)
    keep = metrics[metrics["run_key"].isin(["ch1", "ch1_4"])].copy()
    if keep.empty:
        return keep
    keep["analysis_role"] = keep["run_key"].map(
        {
            "ch1": "strict_fast_3x3_single_reference",
            "ch1_4": "strict_fast_3x3_best_pair_reference",
        }
    )
    keep["comparison_note"] = "direct FAST 3x3 comparison"
    return keep


def full_ch1_0_2_reference_row() -> pd.DataFrame:
    path = FULL_CH1_0_2_READOUT / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    foldwise = pd.read_csv(path)
    focus = foldwise[
        foldwise["model_name"].eq(PRIMARY_MODEL)
        & foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    if focus.empty:
        return pd.DataFrame()
    row: Dict[str, Any] = {
        "run_key": "full_ch1_0_2_mfrsplit_3840",
        "channels": "[1,0,2]",
        "n_channels": 3,
        "selected_channel_names": "Pearson_Full_FisherZ_Signed | Pearson_OMST_GCE_Signed_Weighted | MI_KNN_Symmetric",
        "analysis_role": "current_full_ch1_0_2_reference_if_available",
        "comparison_note": "not directly comparable: full 3840 run/readout, not FAST 3x3 look-ahead",
    }
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        mean, se = metric_mean_se(focus[metric])
        row[f"mean_outer_{metric}"] = mean
        row[f"se_outer_{metric}"] = se
    return pd.DataFrame([row])


def write_comparison(output_root: Path, triples: pd.DataFrame) -> pd.DataFrame:
    refs = [strict_reference_rows(), full_ch1_0_2_reference_row()]
    combined = pd.concat([df for df in refs + [triples] if not df.empty], ignore_index=True, sort=False)
    if combined.empty:
        combined.to_csv(output_root / "triple_lookahead_vs_single_pair_summary.csv", index=False)
        return combined
    strict = combined.set_index("run_key", drop=False)
    single_auc = float(strict.loc["ch1", "mean_outer_auc"]) if "ch1" in strict.index else float("nan")
    pair_auc = float(strict.loc["ch1_4", "mean_outer_auc"]) if "ch1_4" in strict.index else float("nan")
    combined["delta_auc_vs_ch1_fast3x3"] = pd.to_numeric(combined["mean_outer_auc"], errors="coerce") - single_auc
    combined["delta_auc_vs_ch1_4_fast3x3"] = pd.to_numeric(combined["mean_outer_auc"], errors="coerce") - pair_auc
    preferred = [
        "analysis_role",
        "run_key",
        "channels",
        "selected_channel_names",
        "mean_outer_auc",
        "se_outer_auc",
        "mean_outer_pr_auc",
        "mean_outer_balanced_accuracy",
        "mean_outer_sensitivity",
        "mean_outer_specificity",
        "mean_outer_f1",
        "delta_auc_vs_ch1_fast3x3",
        "delta_auc_vs_ch1_4_fast3x3",
        "weak_fold",
        "weak_fold_auc",
        "weak_fold_balanced_accuracy",
        "comparison_note",
    ]
    cols = [col for col in preferred if col in combined.columns] + [col for col in combined.columns if col not in preferred]
    combined = combined[cols]
    combined.to_csv(output_root / "triple_lookahead_vs_single_pair_summary.csv", index=False)
    return combined


def write_command_log(
    output_root: Path,
    mode: str,
    planned: pd.DataFrame,
    tensor_info: Dict[str, Any],
    metadata_rows: int,
    training_launched: bool,
) -> None:
    payload = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "mode": mode,
        "exploratory_lookahead": True,
        "strict_greedy_state_modified": False,
        "lookahead_base_channels": LOOKAHEAD_BASE,
        "candidate_channels": CANDIDATES,
        "n_candidates": int(len(planned)),
        "dry_run": mode == "dry-run",
        "training_launched": bool(training_launched),
        "vae_retrained": bool(training_launched),
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "python_bandpass_applied": False,
        "tensor_path": str(greedy_tensor_path()),
        "metadata_path": str(greedy_metadata_path()),
        "tensor_shape": tensor_info.get("tensor_shape"),
        "tensor_subjects": tensor_info.get("n_subjects"),
        "metadata_rows": int(metadata_rows),
        "fast_outer_folds": 3,
        "fast_inner_folds": 3,
        "latent_dim": 128,
        "epochs_vae": 960,
        "cyclical_beta_n_cycles": 12,
        "lr_scheduler_T0": 80,
        "stage_a": "FAST VAE + dummy canonical logreg n_iter_logreg=1, ignored for ranking",
        "stage_b": "classifier-only logreg_l2 on latent mu + Age + Sex",
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof_required_for_non_0p5",
        "planned_candidates": planned[["run_key", "channels", "train_command", "readout_command"]].to_dict(orient="records"),
    }
    write_json(output_root / "command_log.json", payload)


def greedy_tensor_path() -> str:
    # Kept as a function so command_log generation remains independent from the
    # imported module object.
    return (
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
        "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
    )


def greedy_metadata_path() -> str:
    return (
        "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
        "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
        "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
    )


def write_readme(output_root: Path, mode: str, planned: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    lines = [
        "# ADNI v5.1 batch20260514b Triple Look-Ahead FAST 3x3",
        "",
        "This is an exploratory third-channel look-ahead from the best exploratory pair `[1,4]`.",
        "It does not modify the strict greedy state and is not a final performance estimate.",
        "",
        "## Safety",
        "",
        "- Python bandpass: `OFF`",
        "- Tensor modified: `NO`",
        "- Metadata modified: `NO`",
        "- Ledger modified: `NO`",
        "- Stage A canonical classifier is dummy `logreg` with `n_iter_logreg=1` and is ignored for ranking.",
        "- Stage B ranking uses classifier-only `logreg_l2` on saved latent `mu` + Age + Sex.",
        "- Non-0.5 thresholds use true inner-CV OOF selection.",
        "",
        "## FAST Profile",
        "",
        "- outer_folds: `3`",
        "- inner_folds: `3`",
        "- latent_dim: `128`",
        "- epochs_vae: `960`",
        "- cyclical_beta_n_cycles: `12`",
        "- lr_scheduler_T0: `80`",
        "- beta_vae: `2.5`",
        "- dropout: `0.15`",
        "- norm_mode: `zscore_offdiag`",
        "- split: `ResearchGroup_Mapped + Manufacturer`",
        "- Sex: metadata/covariate only",
        "",
        "## Candidates",
        "",
    ]
    for _, row in planned.iterrows():
        lines.append(f"- `{row['channels']}` -> `{row['run_key']}`")
    lines.extend(["", "## Status", "", f"- Mode: `{mode}`", f"- Planned candidates: `{len(planned)}`"])
    if metrics.empty:
        lines.extend(["- Completed triple readouts: `0`", "", "Dry-run only. No training/readout was launched."])
    else:
        top = metrics.iloc[0]
        lines.extend(
            [
                f"- Completed triple readouts: `{len(metrics)}`",
                f"- Best triple by mean outer AUC: `{top['channels']}` (`{top['selected_channel_names']}`)",
                f"- Best triple mean AUC: `{top['mean_outer_auc']:.4f}`",
                f"- Best triple PR-AUC: `{top['mean_outer_pr_auc']:.4f}`",
                f"- Best triple BA: `{top['mean_outer_balanced_accuracy']:.4f}`",
                "",
                "## Interpretation",
                "",
                "Use this only as exploratory screening. A triple should not displace the strict accepted `[1]` path unless its improvement is clearly meaningful and then validated in a full 5x5 run.",
            ]
        )
        if not comparison.empty:
            comp = comparison.set_index("run_key", drop=False)
            if top["run_key"] in comp.index:
                delta_single = comp.loc[top["run_key"], "delta_auc_vs_ch1_fast3x3"]
                delta_pair = comp.loc[top["run_key"], "delta_auc_vs_ch1_4_fast3x3"]
                lines.extend(
                    [
                        f"- Best triple delta vs `[1]` FAST 3x3: `{delta_single:.4f}`",
                        f"- Best triple delta vs `[1,4]` FAST 3x3: `{delta_pair:.4f}`",
                    ]
                )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `planned_triple_lookahead_candidates.csv`",
            "- `triple_lookahead_metrics.csv`",
            "- `triple_lookahead_ranking.csv`",
            "- `triple_lookahead_vs_single_pair_summary.csv`",
            "- `README.md`",
            "- `command_log.json`",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidates(greedy: Any, planned: pd.DataFrame, resume: bool) -> None:
    for _, row in planned.iterrows():
        greedy.run_candidate(row, resume=resume)


def main() -> int:
    args = parse_args()
    if args.run and args.dry_run:
        raise SystemExit("Use either --dry-run or --run, not both.")
    mode = "run" if args.run else "dry-run"
    if args.run and not args.confirm_training:
        raise SystemExit("Refusing real training without --confirm-training.")

    greedy = load_greedy_module()
    require_expected_fast_profile(greedy)
    tensor_info = greedy.inspect_tensor(greedy.TENSOR_PATH)
    greedy.validate_fast_preflight(tensor_info)
    metadata = greedy.load_metadata(greedy.METADATA_PATH)

    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    planned = build_plan(greedy, output_root, tensor_info["channel_names"], args.python_executable)

    print(f"Mode: {mode}")
    print(f"Output root: {output_root}")
    print("Exploratory look-ahead: from [1,4], strict greedy state is not modified.")
    print("Python bandpass: OFF")
    print("Split: ResearchGroup_Mapped + Manufacturer")
    print("Sex role: metadata/covariate only")
    print("FAST screening folds: outer=3 inner=3")
    print("Stage A: dummy canonical logreg, n_iter_logreg=1, ignored for ranking")
    print("Stage B: classifier-only logreg_l2, true inner-CV OOF thresholds")
    print(f"Planned candidates: {len(planned)}")
    print(planned[["run_key", "channels", "stage_b_primary_readout", "status"]].to_string(index=False))
    greedy.print_dry_run_command_preview(planned)

    training_launched = False
    metrics = pd.DataFrame()
    comparison = pd.DataFrame()
    if args.run:
        training_launched = True
        run_candidates(greedy, planned, resume=args.resume)
        metrics = aggregate_triples(greedy, output_root, tensor_info["channel_names"])
        ranking = pd.read_csv(output_root / "triple_lookahead_ranking.csv")
        print("Triple look-ahead ranking:")
        print(
            ranking[
                [
                    "rank",
                    "run_key",
                    "channels",
                    "mean_outer_auc",
                    "mean_outer_pr_auc",
                    "mean_outer_balanced_accuracy",
                    "mean_outer_sensitivity",
                    "mean_outer_specificity",
                    "mean_outer_f1",
                ]
            ].to_string(index=False)
        )
        comparison = write_comparison(output_root, metrics)
    else:
        pd.DataFrame().to_csv(output_root / "triple_lookahead_metrics.csv", index=False)
        pd.DataFrame().to_csv(output_root / "triple_lookahead_ranking.csv", index=False)
        pd.DataFrame().to_csv(output_root / "triple_lookahead_vs_single_pair_summary.csv", index=False)

    write_command_log(output_root, mode, planned, tensor_info, len(metadata), training_launched)
    write_readme(output_root, mode, planned, metrics, comparison)
    if mode == "dry-run":
        print("Dry-run complete. No training/readout was launched.")
    else:
        print("Run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
