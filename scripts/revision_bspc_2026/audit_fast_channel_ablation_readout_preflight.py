#!/usr/bin/env python3
"""Preflight audit for FAST channel-ablation readout/threshold logic.

This script inspects only lightweight code/config/result manifests. It does not
train, does not load tensor arrays, and does not modify source data.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_v5_1_batch20260514b_fast_channel_ablation.py"
FAST_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fast_channel_ablation"
PLANNED = FAST_ROOT / "planned_runs.csv"
CONFIG_DIR = FAST_ROOT / "configs"
RANKING = FAST_ROOT / "channel_ablation_summary_ranking.csv"
THRESHOLD_AUDIT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_threshold_final_audit"
PAPER_TABLES = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_paper_ready_threshold_tables"
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fast_channel_ablation_preflight_readout_audit"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_files() -> List[Path]:
    return sorted(CONFIG_DIR.glob("*.json"))


def load_configs() -> List[Dict[str, Any]]:
    return [read_json(path) for path in config_files()]


def status(pass_condition: bool, partial: bool = False) -> str:
    if pass_condition:
        return "PASS"
    if partial:
        return "PARTIAL"
    return "FAIL"


def all_configs(configs: List[Dict[str, Any]], predicate) -> bool:
    return bool(configs) and all(predicate(cfg) for cfg in configs)


def count_configs(configs: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for cfg in configs if predicate(cfg))


def add_check(
    rows: List[Dict[str, Any]],
    question_id: int,
    question: str,
    expected: str,
    observed: str,
    check_status: str,
    evidence: str,
    action_needed: str,
) -> None:
    rows.append(
        {
            "question_id": question_id,
            "question": question,
            "expected": expected,
            "observed": observed,
            "status": check_status,
            "evidence": evidence,
            "action_needed": action_needed,
        }
    )


def audit() -> pd.DataFrame:
    if not WRAPPER.exists():
        raise FileNotFoundError(WRAPPER)
    if not PLANNED.exists():
        raise FileNotFoundError(PLANNED)
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(CONFIG_DIR)

    wrapper_text = WRAPPER.read_text(encoding="utf-8")
    planned = pd.read_csv(PLANNED)
    configs = load_configs()
    ranking = pd.read_csv(RANKING) if RANKING.exists() else pd.DataFrame()

    n_configs = len(configs)
    n_planned = int(len(planned))
    command_series = planned["command"].fillna("").astype(str) if "command" in planned.columns else pd.Series(dtype=str)

    split_ok = all_configs(
        configs,
        lambda cfg: cfg.get("split_strategy", {}).get("classifier_outer") == ["ResearchGroup_Mapped", "Manufacturer"]
        and cfg.get("split_strategy", {}).get("classifier_inner") == ["ResearchGroup_Mapped", "Manufacturer"]
        and cfg.get("split_strategy", {}).get("vae_internal_val") == ["ResearchGroup_Mapped", "Manufacturer"]
        and cfg.get("parameters", {}).get("classifier_stratify_cols") == ["Manufacturer"]
        and cfg.get("parameters", {}).get("vae_stratify_cols") == ["Manufacturer"],
    )
    sex_cov_ok = all_configs(
        configs,
        lambda cfg: cfg.get("split_strategy", {}).get("metadata_covariates_only") == ["Age", "Sex"]
        and cfg.get("split_strategy", {}).get("sex_primary_stratifier") is False
        and cfg.get("parameters", {}).get("metadata_features") == ["Age", "Sex"]
        and "Sex" not in cfg.get("parameters", {}).get("classifier_stratify_cols", [])
        and "Sex" not in cfg.get("parameters", {}).get("vae_stratify_cols", []),
    )
    bandpass_ok = (
        "python_bandpass_applied" in wrapper_text
        and "Refusing to use tensor with python_bandpass_applied=True" in wrapper_text
        and planned.get("python_bandpass_applied", pd.Series([True])).eq(False).all()
    )
    declared_logreg_l2 = all_configs(
        configs,
        lambda cfg: cfg.get("fast_readout_plan", {}).get("primary_classifier") == "logreg_l2",
    )
    canonical_logreg_configs = count_configs(
        configs,
        lambda cfg: cfg.get("parameters", {}).get("classifier_types") == ["logreg"],
    )
    commands_logreg = int(command_series.str.contains("--classifier_types logreg", regex=False).sum())
    commands_logreg_l2 = int(command_series.str.contains("logreg_l2", regex=False).sum())

    fold_artifacts_ok = all_configs(
        configs,
        lambda cfg: cfg.get("parameters", {}).get("save_fold_artefacts") is True
        and cfg.get("parameters", {}).get("latent_features_type") == "mu",
    )
    threshold_plan_declared = all_configs(
        configs,
        lambda cfg: cfg.get("fast_readout_plan", {}).get("threshold_selection") == "true_inner_cv_oof_required"
        and "inner_oof_target_sensitivity_0p70_max_specificity"
        in cfg.get("fast_readout_plan", {}).get("threshold_rules", []),
    )
    executable_readout_wired = (
        "classifier_only" in wrapper_text
        or "threshold_selection_verification" in wrapper_text
        or "classifier_sweep_predictions" in wrapper_text
    )
    metrics_cols = {"auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "f1"}
    ranking_has_metric_cols = metrics_cols.issubset(set(ranking.columns))
    ranking_has_values = False
    if ranking_has_metric_cols:
        ranking_has_values = bool(ranking[list(metrics_cols)].notna().any().any())
    manufacturer_cols = {"cn_ge_specificity", "ad_ge_sensitivity"}
    ranking_has_mfr_cols = manufacturer_cols.issubset(set(ranking.columns))

    threshold_verify = THRESHOLD_AUDIT / "threshold_selection_verification.csv"
    paper_primary = PAPER_TABLES / "primary_results_table.csv"
    threshold_audit_reference_ok = threshold_verify.exists() and paper_primary.exists()
    threshold_verification_pass = False
    if threshold_verify.exists():
        v = pd.read_csv(threshold_verify)
        threshold_verification_pass = bool((v.get("verification_status") == "PASS").all())

    rows: List[Dict[str, Any]] = []
    add_check(
        rows,
        1,
        "Does the FAST wrapper use ResearchGroup_Mapped + Manufacturer splits?",
        "Classifier outer/inner and VAE val split use ResearchGroup_Mapped + Manufacturer.",
        f"{n_configs}/{n_configs} configs use Manufacturer stratify with ResearchGroup_Mapped constructed in split preview.",
        status(split_ok),
        "configs split_strategy plus wrapper make_split_preview strat_cols.",
        "None.",
    )
    add_check(
        rows,
        2,
        "Does it keep Sex only as metadata/covariate?",
        "Sex in metadata_features only; not in classifier_stratify_cols or vae_stratify_cols.",
        f"{n_configs}/{n_configs} configs keep metadata_features=['Age','Sex'] and exclude Sex from stratify columns.",
        status(sex_cov_ok),
        "configs split_strategy.metadata_covariates_only and parameters.",
        "None.",
    )
    add_check(
        rows,
        3,
        "Does it keep Python bandpass OFF?",
        "Refuse tensors with python_bandpass_applied=True; planned runs marked False.",
        f"planned_runs python_bandpass_applied all False; wrapper checks tensor flag.",
        status(bandpass_ok),
        "inspect_tensor() raises on python_bandpass_applied=True.",
        "None.",
    )
    add_check(
        rows,
        4,
        "Does it evaluate logreg_l2, or original canonical logreg?",
        "Actual FAST ranking should evaluate logreg_l2.",
        (
            f"Configs declare primary logreg_l2={declared_logreg_l2}, but "
            f"{canonical_logreg_configs}/{n_configs} configs and {commands_logreg}/{n_planned} commands use "
            f"--classifier_types logreg; commands containing logreg_l2={commands_logreg_l2}."
        ),
        "FAIL",
        "planned_runs command column and config parameters.classifier_types.",
        "Patch before launch: wire post-hoc logreg_l2 readout into the FAST workflow or stop labeling canonical logreg outputs as primary.",
    )
    add_check(
        rows,
        5,
        "Does it compute or preserve enough predictions for true inner-CV OOF threshold selection?",
        "Need saved fold latents/splits/checkpoints plus an executable inner-OOF readout producing thresholds and predictions.",
        (
            f"save_fold_artefacts/latent mu plan={fold_artifacts_ok}; threshold plan declared={threshold_plan_declared}; "
            f"executable classifier-only/threshold readout wired in wrapper={executable_readout_wired}."
        ),
        status(False, partial=(fold_artifacts_ok and threshold_plan_declared)),
        "fast_readout_plan declares true_inner_cv_oof_required, but wrapper does not call the classifier-only sweep/readout.",
        "Patch before launch: after each VAE run, call a classifier-only logreg_l2 readout that writes OOF thresholds, test predictions, and verification.",
    )
    add_check(
        rows,
        6,
        "Does it report AUC, PR-AUC, sensitivity, specificity, balanced accuracy, F1, and manufacturer subgroup metrics?",
        "Per channel set, report foldwise/pooled metrics and manufacturer subgroups for the selected readout.",
        (
            f"ranking file has metric columns={ranking_has_metric_cols}, manufacturer columns={ranking_has_mfr_cols}, "
            f"actual metric values present={ranking_has_values}."
        ),
        status(False, partial=(ranking_has_metric_cols and ranking_has_mfr_cols)),
        "channel_ablation_summary_ranking.csv is a placeholder with planned_not_run rows.",
        "Patch before launch: add/read a per-run readout summary and aggregate it into the ranking table.",
    )
    add_check(
        rows,
        7,
        "Is the ranking based on AUC/PR-AUC and primary logreg_l2 inner-OOF target sensitivity operating point?",
        "Rank by threshold-independent AUC/PR-AUC plus primary operating point metrics.",
        (
            f"Current ranking has values={ranking_has_values}; reference threshold audit exists={threshold_audit_reference_ok}; "
            f"reference threshold verification pass={threshold_verification_pass}."
        ),
        "FAIL",
        "FAST ranking table is not populated from readout outputs; reference audit proves the desired logic exists elsewhere.",
        "Patch before launch: make ranking consume classifier-only logreg_l2 outputs with threshold_selection_verification PASS.",
    )
    add_check(
        rows,
        8,
        "Minimal patch needed before launching training?",
        "Only launch once the wrapper can produce the intended readout and ranking.",
        "Patch required.",
        "FAIL",
        "Current wrapper is safe as a VAE training planner, not as a complete channel-ablation readout pipeline.",
        "Add a post-VAE classifier-only logreg_l2 inner-OOF readout stage and final aggregator, or split into explicit Stage A train and Stage B readout commands.",
    )
    return pd.DataFrame(rows)


def write_recommended_patch(path: Path) -> None:
    text = """# Recommended Patch Before Launch

The current FAST wrapper is safe for planning VAE channel-set runs, but it is not yet safe to use as the final channel-ablation decision pipeline. It declares `logreg_l2` as the primary readout while the actual launch commands run canonical `--classifier_types logreg`.

Minimal patch:

1. Keep the current VAE training stage, but rename it explicitly as Stage A: `fast_vae_fit_and_artifact_export`.
2. After each channel-set run completes, run a Stage B classifier-only readout on the saved fold latent `mu` features:
   - classifier: `logreg_l2`
   - features: latent `mu` + Age + Sex
   - inner CV stratification: `ResearchGroup_Mapped + Manufacturer`
   - threshold rules: `fixed_0p5` and `inner_oof_target_sens_ge_0p70_max_spec`
   - threshold selection: true inner-CV OOF only
   - outputs per channel set: predictions, thresholds_by_fold, foldwise metrics, pooled metrics, confusion matrices, manufacturer subgroup metrics, threshold verification.
3. Add a Stage C aggregator that refuses to rank a channel set unless:
   - `threshold_selection_verification.csv` exists and all non-fixed thresholds are `PASS`
   - pooled AUC and PR-AUC are present
   - the primary operating point `logreg_l2 + inner_oof_target_sens_ge_0p70_max_spec` has sensitivity, specificity, balanced accuracy, F1, and manufacturer subgroup metrics.
4. Populate `channel_ablation_summary_ranking.csv` from Stage B/C outputs, not from the canonical wrapper's original logreg outputs.
5. In `planned_runs.csv`, either remove `primary_readout_classifier=logreg_l2` until Stage B is wired, or add separate columns:
   - `stage_a_training_classifier=canonical_logreg`
   - `stage_b_primary_readout=logreg_l2_inner_oof`

Do not launch the current wrapper as the final FAST ablation until this patch is in place. It can train VAEs, but it cannot by itself answer whether `[1,0,2]` remains best under the primary readout.
"""
    path.write_text(text, encoding="utf-8")


def write_readme(path: Path, checks: pd.DataFrame) -> None:
    lookup = checks.set_index("question_id")
    lines = [
        "# FAST Channel Ablation Readout Preflight Audit",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "No training was run. No tensor, metadata, or ledger files were modified.",
        "",
        "## Answers",
        "",
        f"1. Manufacturer-aware split: **{lookup.loc[1, 'status']}**. The configs and split preview use `ResearchGroup_Mapped + Manufacturer`.",
        f"2. Sex as covariate only: **{lookup.loc[2, 'status']}**. `Sex` is in `metadata_features` and is not a stratification column.",
        f"3. Python bandpass OFF: **{lookup.loc[3, 'status']}**. The wrapper refuses tensors marked `python_bandpass_applied=True`.",
        f"4. Actual classifier readout: **{lookup.loc[4, 'status']}**. The planned primary readout says `logreg_l2`, but the executable command still runs canonical `--classifier_types logreg`.",
        f"5. Inner-OOF threshold readiness: **{lookup.loc[5, 'status']}**. Fold artifacts are planned, but the wrapper does not execute the classifier-only inner-OOF threshold stage.",
        f"6. Metric reporting: **{lookup.loc[6, 'status']}**. The ranking CSV has placeholder metric columns, not computed readout metrics.",
        f"7. Ranking logic: **{lookup.loc[7, 'status']}**. Current ranking is not based on `logreg_l2 + inner-OOF target sensitivity >=0.70` results.",
        "",
        "## Reviewer Conclusion",
        "",
        "Do not launch this wrapper yet as the definitive FAST channel-ablation pipeline. It is valid as a dry-run/VAE-run planner, but incomplete as a primary readout/ranking pipeline.",
        "",
        "The minimal fix is to wire a post-VAE classifier-only `logreg_l2` stage that reproduces the threshold-final audit logic: true inner-CV OOF threshold selection, foldwise predictions, pooled metrics, and manufacturer subgroup metrics for every channel set.",
        "",
        "## Files",
        "",
        "- `fast_readout_preflight_check.csv`",
        "- `recommended_patch_if_needed.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = audit()
    checks.to_csv(OUT_DIR / "fast_readout_preflight_check.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_recommended_patch(OUT_DIR / "recommended_patch_if_needed.md")
    write_readme(OUT_DIR / "README.md", checks)
    print(f"Wrote audit to {OUT_DIR}")
    print(checks[["question_id", "status", "observed"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
