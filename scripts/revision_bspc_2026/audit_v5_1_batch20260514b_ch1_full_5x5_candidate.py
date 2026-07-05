#!/usr/bin/env python3
"""Audit the completed ADNI v5.1 batch20260514b channel-[1] full 5x5 candidate.

This script is read-only with respect to tensors, metadata, and ledger files.
It writes paper-ready comparison tables into the existing comparison directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

CANDIDATE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
REFERENCE_READOUT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
REFERENCE_THRESHOLD_AUDIT = RESULTS / "adni_v5_1_batch20260514b_threshold_final_audit"
OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]

TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
LEDGER_PATHS = [
    RESULTS / "final_cn_ge_inventory_before_martin_request" / "ADNI_v5_1_DATA_LEDGER_CURRENT.csv",
    RESULTS / "final_cn_ge_inventory_before_martin_request" / "ADNI_v5_1_DATA_LEDGER_CURRENT_RECONCILED_20260514b.csv",
    RESULTS / "final_cn_ge_inventory_before_martin_request" / "ADNI_v5_1_DATA_LEDGER_20260514b_snapshot.csv",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def md_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{float_digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def primary_row(readout_dir: Path, run_id: str, label: str, channels: str) -> Dict[str, Any]:
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one primary pooled row for {run_id}, found {len(row)}")
    r = row.iloc[0].to_dict()
    out: Dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "channels": channels,
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", *PRIMARY_METRICS]:
        out[col] = r.get(col, np.nan)
    return out


def primary_foldwise(readout_dir: Path, run_id: str, channels: str) -> pd.DataFrame:
    df = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "channels", channels)
    keep = [
        "run_id",
        "channels",
        "fold",
        "threshold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
        "threshold_selection_context",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values(["fold"]).reset_index(drop=True)


def group_metrics_from_predictions(readout_dir: Path, run_id: str, channels: str, group_col: str) -> pd.DataFrame:
    pred = pd.read_csv(readout_dir / "classifier_sweep_predictions.csv")
    pred = pred[(pred["model_name"].eq(PRIMARY_MODEL)) & (pred["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    rows: List[Dict[str, Any]] = []
    for group_value, sub in pred.groupby(group_col, dropna=False):
        y = sub["y_true"].astype(int).to_numpy()
        score = sub["y_score"].astype(float).to_numpy()
        y_pred = sub["y_pred"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        row = {
            "run_id": run_id,
            "channels": channels,
            "grouping": group_col,
            "group_value": group_value,
            "n": int(len(sub)),
            "n_cn": int((y == 0).sum()),
            "n_ad": int((y == 1).sum()),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": (tn + tp) / len(sub) if len(sub) else np.nan,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": np.nanmean([sensitivity, specificity]),
            "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group_value", "run_id"]).reset_index(drop=True)


def side_by_side(df: pd.DataFrame, id_cols: Sequence[str], candidate_run: str, reference_run: str) -> pd.DataFrame:
    metric_cols = [c for c in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"] if c in df.columns]
    cand = df[df["run_id"].eq(candidate_run)][list(id_cols) + metric_cols].copy()
    ref = df[df["run_id"].eq(reference_run)][list(id_cols) + metric_cols].copy()
    cand = cand.rename(columns={c: f"{c}_ch1" for c in metric_cols})
    ref = ref.rename(columns={c: f"{c}_ch1_0_2" for c in metric_cols})
    merged = cand.merge(ref, on=list(id_cols), how="outer")
    for c in metric_cols:
        merged[f"delta_{c}_ch1_minus_ch1_0_2"] = merged[f"{c}_ch1"] - merged[f"{c}_ch1_0_2"]
    return merged


def validate_run_completion() -> Dict[str, Any]:
    required = [
        CANDIDATE_RUN / "run_config.json",
        CANDIDATE_RUN / "run_manifest.json",
        CANDIDATE_RUN / "all_folds_metrics_MULTI_logreg_vaeconvtranspose4l_ld256_beta2.5_normzscore_offdiag_ch1sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv",
        CANDIDATE_READOUT / "command_log.json",
        CANDIDATE_READOUT / "classifier_sweep_pooled_metrics.csv",
        CANDIDATE_READOUT / "classifier_sweep_foldwise_metrics.csv",
        CANDIDATE_READOUT / "classifier_sweep_predictions.csv",
        REFERENCE_READOUT / "classifier_sweep_pooled_metrics.csv",
        REFERENCE_READOUT / "classifier_sweep_foldwise_metrics.csv",
        REFERENCE_READOUT / "classifier_sweep_predictions.csv",
    ]
    for fold in range(1, 6):
        fold_dir = CANDIDATE_RUN / f"fold_{fold}"
        required.extend(
            [
                fold_dir / f"vae_model_fold_{fold}.pt",
                fold_dir / "vae_norm_params.joblib",
                fold_dir / "train_dev_subjects_fold.csv",
                fold_dir / "test_subjects_fold.csv",
                fold_dir / "latent_qc_metrics.csv",
                fold_dir / f"vae_train_history_fold_{fold}.joblib",
            ]
        )
    require_files(required)

    cfg = read_json(CANDIDATE_RUN / "run_config.json")
    manifest = read_json(CANDIDATE_RUN / "run_manifest.json")
    readout_log = read_json(CANDIDATE_READOUT / "command_log.json")
    assert cfg["args"]["channels_to_use"] == [1]
    assert cfg["args"]["outer_folds"] == 5 and cfg["args"]["inner_folds"] == 5
    assert cfg["args"]["latent_dim"] == 256
    assert cfg["args"]["epochs_vae"] == 3840
    assert cfg["args"]["cyclical_beta_n_cycles"] == 48
    assert cfg["args"]["lr_scheduler_T0"] == 80
    assert manifest["python_bandpass_applied"] is False
    assert readout_log["classifiers_requested"] == [PRIMARY_MODEL]
    assert readout_log["threshold_selection"].startswith("true_inner_cv_oof")
    assert readout_log["vae_retrained"] is False
    assert readout_log["tensor_modified"] is False
    assert readout_log["metadata_modified"] is False
    assert readout_log["ledger_modified"] is False

    return {
        "candidate_completed": True,
        "stage_b_logreg_l2_only": True,
        "stage_b_true_inner_cv_oof": True,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }


def data_file_status() -> pd.DataFrame:
    paths = [TENSOR_PATH, METADATA_PATH, *LEDGER_PATHS]
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation = validate_run_completion()

    main = pd.DataFrame(
        [
            primary_row(CANDIDATE_READOUT, "ch1_full_5x5", "[1] full 5x5 candidate", "[1]"),
            primary_row(REFERENCE_READOUT, "ch1_0_2_full_5x5", "[1,0,2] mfrsplit_3840 final candidate", "[1,0,2]"),
        ]
    )
    for metric in PRIMARY_METRICS:
        cand_val = float(main.loc[main["run_id"].eq("ch1_full_5x5"), metric].iloc[0])
        main[f"delta_{metric}_vs_ch1"] = main[metric].astype(float) - cand_val

    foldwise_long = pd.concat(
        [
            primary_foldwise(CANDIDATE_READOUT, "ch1_full_5x5", "[1]"),
            primary_foldwise(REFERENCE_READOUT, "ch1_0_2_full_5x5", "[1,0,2]"),
        ],
        ignore_index=True,
    )
    foldwise = side_by_side(foldwise_long, ["fold"], "ch1_full_5x5", "ch1_0_2_full_5x5")

    manufacturer_long = pd.concat(
        [
            group_metrics_from_predictions(CANDIDATE_READOUT, "ch1_full_5x5", "[1]", "Manufacturer"),
            group_metrics_from_predictions(REFERENCE_READOUT, "ch1_0_2_full_5x5", "[1,0,2]", "Manufacturer"),
        ],
        ignore_index=True,
    )
    manufacturer = side_by_side(manufacturer_long, ["grouping", "group_value"], "ch1_full_5x5", "ch1_0_2_full_5x5")

    sex_long = pd.concat(
        [
            group_metrics_from_predictions(CANDIDATE_READOUT, "ch1_full_5x5", "[1]", "Sex"),
            group_metrics_from_predictions(REFERENCE_READOUT, "ch1_0_2_full_5x5", "[1,0,2]", "Sex"),
        ],
        ignore_index=True,
    )
    sex = side_by_side(sex_long, ["grouping", "group_value"], "ch1_full_5x5", "ch1_0_2_full_5x5")

    weak = foldwise_long.sort_values("auc").groupby("run_id").head(1).reset_index(drop=True)
    data_status = data_file_status()

    main.to_csv(OUTPUT_DIR / "main_model_comparison.csv", index=False)
    foldwise.to_csv(OUTPUT_DIR / "foldwise_comparison.csv", index=False)
    manufacturer.to_csv(OUTPUT_DIR / "manufacturer_subgroup_comparison.csv", index=False)
    sex.to_csv(OUTPUT_DIR / "sex_subgroup_comparison.csv", index=False)
    weak.to_csv(OUTPUT_DIR / "weak_fold_comparison.csv", index=False)
    data_status.to_csv(OUTPUT_DIR / "source_data_file_status.csv", index=False)

    (OUTPUT_DIR / "main_model_comparison.md").write_text(md_table(main), encoding="utf-8")
    (OUTPUT_DIR / "foldwise_comparison.md").write_text(md_table(foldwise), encoding="utf-8")
    (OUTPUT_DIR / "manufacturer_subgroup_comparison.md").write_text(md_table(manufacturer), encoding="utf-8")
    (OUTPUT_DIR / "sex_subgroup_comparison.md").write_text(md_table(sex), encoding="utf-8")

    ch1 = main[main["run_id"].eq("ch1_full_5x5")].iloc[0]
    ref = main[main["run_id"].eq("ch1_0_2_full_5x5")].iloc[0]
    fold4 = foldwise[foldwise["fold"].eq(4)].iloc[0]
    lines = [
        "# Interpretation Summary",
        "",
        "## Validation",
        "",
        "- Candidate FULL 5x5 `[1]` run completed with folds 1-5, saved VAE checkpoints, QC artifacts, and classifier-only readout.",
        "- Stage A canonical classifier outputs are dummy/ignored; final readout uses classifier-only `logreg_l2` on saved latent `mu` + Age + Sex.",
        "- Non-0.5 threshold selection is true inner-CV OOF (`inner_oof_target_sens_ge_0p70_max_spec`).",
        "- Python bandpass is OFF.",
        "- Command logs report `tensor_modified=false`, `metadata_modified=false`, and `ledger_modified=false`; this audit only wrote tables in the comparison output directory.",
        "",
        "## Main Result",
        "",
        f"- `[1]`: AUC {ch1.auc:.4f}, PR-AUC {ch1.pr_auc:.4f}, BA {ch1.balanced_accuracy:.4f}, sensitivity {ch1.sensitivity:.4f}, specificity {ch1.specificity:.4f}, F1 {ch1.f1:.4f}.",
        f"- `[1,0,2]`: AUC {ref.auc:.4f}, PR-AUC {ref.pr_auc:.4f}, BA {ref.balanced_accuracy:.4f}, sensitivity {ref.sensitivity:.4f}, specificity {ref.specificity:.4f}, F1 {ref.f1:.4f}.",
        f"- `[1]` loses AUC by {float(ch1.auc - ref.auc):.4f} and PR-AUC by {float(ch1.pr_auc - ref.pr_auc):.4f}, but has slightly higher BA ({float(ch1.balanced_accuracy - ref.balanced_accuracy):+.4f}), specificity ({float(ch1.specificity - ref.specificity):+.4f}), and F1 ({float(ch1.f1 - ref.f1):+.4f}).",
        "",
        "## Fold Behavior",
        "",
        f"- Fold 4 remains the weakest fold for both models: `[1]` AUC {fold4.auc_ch1:.4f} vs `[1,0,2]` AUC {fold4.auc_ch1_0_2:.4f}.",
        f"- The `[1]` run improves Fold 4 AUC by {fold4.delta_auc_ch1_minus_ch1_0_2:+.4f}, but Fold 4 still limits the overall estimate.",
        "",
        "## Recommendation",
        "",
        "- The FULL 5x5 confirmation does not support replacing `[1,0,2]` as the primary ranking model if AUC/PR-AUC are prioritized.",
        "- `[1]` is a strong single-channel simplification/sensitivity analysis: it nearly matches the operating-point metrics and slightly improves BA/specificity/F1 under the selected threshold rule.",
        "- For manuscript reporting, keep `[1,0,2]` as the primary model and present `[1]` as a FAST-selected parsimonious confirmatory ablation.",
    ]
    (OUTPUT_DIR / "interpretation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "validation": validation,
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "outputs": [
            "main_model_comparison.csv",
            "main_model_comparison.md",
            "foldwise_comparison.csv",
            "foldwise_comparison.md",
            "manufacturer_subgroup_comparison.csv",
            "manufacturer_subgroup_comparison.md",
            "sex_subgroup_comparison.csv",
            "sex_subgroup_comparison.md",
            "weak_fold_comparison.csv",
            "source_data_file_status.csv",
            "interpretation_summary.md",
        ],
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "training_launched": False,
    }
    (OUTPUT_DIR / "audit_validation.json").write_text(json.dumps(audit_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    write_outputs()
    print(f"Wrote audit tables to: {OUTPUT_DIR}")
    print(pd.read_csv(OUTPUT_DIR / "main_model_comparison.csv")[["run_id", "channels", *PRIMARY_METRICS]].to_string(index=False))
    print("\nWeak folds:")
    print(pd.read_csv(OUTPUT_DIR / "weak_fold_comparison.csv")[["run_id", "fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
