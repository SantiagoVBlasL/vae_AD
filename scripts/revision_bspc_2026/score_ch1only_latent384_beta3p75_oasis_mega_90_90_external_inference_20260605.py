#!/usr/bin/env python
"""Read-only OASIS mega 90/90 inference for ch1-only vs promoted reference.

This reuses the frozen-ADNI mega-OASIS panel protocol from
score_oasis_mega_90_90_external_inference_model_panel_20260604.py.

Guardrails:
- no VAE training;
- no classifier fitting, threshold fitting, calibration fitting, scaler fitting,
  residualization, or ComBat fitting on OASIS;
- OASIS labels are used only for final metric evaluation;
- tensors, metadata, and model artifacts are not modified.

As in the previous mega-OASIS panel, the Stage B OOF calibration package does
not serialize external-ready classifier estimators, so each fold's Stage B
readout is deterministically reconstructed from ADNI train/dev latent caches
only, then applied to OASIS.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import score_oasis_mega_90_90_external_inference_model_panel_20260604 as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFAULT_OUTPUT = RESULTS / "ch1only_latent384_beta3p75_oasis_mega_90_90_external_inference_20260605"


CANDIDATES = [
    base.CandidateSpec(
        label="promoted_beta3p75_oof_ecdf",
        role="primary_reference",
        run_dir=RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
    ),
    base.CandidateSpec(
        label="ch1only_latent384_beta3p75_oof_ecdf",
        role="parsimonious_candidate",
        run_dir=RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
    ),
]


ADNI_PROMOTED = {
    "promoted_beta3p75_oof_ecdf": {
        "adni_auc": 0.795155,
        "adni_pr_auc": 0.573934,
        "adni_balanced_accuracy": 0.725979,
        "adni_sensitivity": 0.731959,
        "adni_specificity": 0.720000,
        "adni_f1": 0.563492,
    },
    "ch1only_latent384_beta3p75_oof_ecdf": {
        "adni_auc": 0.800378,
        "adni_pr_auc": 0.585842,
        "adni_balanced_accuracy": 0.721134,
        "adni_sensitivity": 0.742268,
        "adni_specificity": 0.700000,
        "adni_f1": 0.555985,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 100) -> None:
    base.write_table(outdir, stem, df, max_rows=max_rows)


def validate_artifacts() -> pd.DataFrame:
    original = base.CANDIDATES
    try:
        base.CANDIDATES = CANDIDATES
        return base.validate_artifacts()
    finally:
        base.CANDIDATES = original


def adni_vs_oasis(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in primary_metrics.iterrows():
        ref = ADNI_PROMOTED.get(str(row["candidate"]), {})
        out: dict[str, Any] = {
            "candidate": row["candidate"],
            "role": row["role"],
            "build_candidate": row["build_candidate"],
            **ref,
        }
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            out[f"oasis_{metric}"] = row.get(metric)
            out[f"generalization_drop_{metric}"] = row.get(metric) - ref.get(f"adni_{metric}", float("nan"))
        rows.append(out)
    return pd.DataFrame(rows)


def candidate_vs_reference(primary_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for build, g in primary_metrics.groupby("build_candidate"):
        ref = g[g["candidate"] == "promoted_beta3p75_oof_ecdf"]
        cand = g[g["candidate"] == "ch1only_latent384_beta3p75_oof_ecdf"]
        if ref.empty or cand.empty:
            continue
        r = ref.iloc[0]
        c = cand.iloc[0]
        row: dict[str, Any] = {"build_candidate": build}
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "predicted_ad_rate"]:
            row[f"promoted_{metric}"] = r.get(metric)
            row[f"ch1only_{metric}"] = c.get(metric)
            row[f"delta_ch1only_minus_promoted_{metric}"] = c.get(metric) - r.get(metric)
        for metric in ["tn", "fp", "fn", "tp"]:
            row[f"promoted_{metric}"] = r.get(metric)
            row[f"ch1only_{metric}"] = c.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation_and_decision(primary_metrics: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in comparison.iterrows():
        auc_ok = row["delta_ch1only_minus_promoted_auc"] >= 0
        pr_ok = row["delta_ch1only_minus_promoted_pr_auc"] >= 0
        ba_ok = row["delta_ch1only_minus_promoted_balanced_accuracy"] >= -0.005
        f1_ok = row["delta_ch1only_minus_promoted_f1"] >= -0.005
        if auc_ok and pr_ok and ba_ok and f1_ok:
            decision = "report_as_parsimonious_co_primary_candidate_for_this_external_build"
            rationale = "ch1-only matched or improved AUC/PR-AUC and did not materially worsen BA/F1 on this external build."
        elif auc_ok or pr_ok:
            decision = "report_as_sensitivity_only"
            rationale = "ch1-only showed some external metric improvement but not a clean across-metric advantage."
        else:
            decision = "reject_for_external_primary_role"
            rationale = "ch1-only did not improve external ranking metrics versus promoted reference."
        rows.append(
            {
                "build_candidate": row["build_candidate"],
                "decision_label": decision,
                "rationale": rationale,
                "auc_delta": row["delta_ch1only_minus_promoted_auc"],
                "pr_auc_delta": row["delta_ch1only_minus_promoted_pr_auc"],
                "ba_delta": row["delta_ch1only_minus_promoted_balanced_accuracy"],
                "f1_delta": row["delta_ch1only_minus_promoted_f1"],
            }
        )

    if rows:
        deltas = pd.DataFrame(rows)
        clean_wins = (
            (deltas["auc_delta"] >= 0)
            & (deltas["pr_auc_delta"] >= 0)
            & (deltas["ba_delta"] >= -0.005)
            & (deltas["f1_delta"] >= -0.005)
        ).sum()
        overall = "report_as_sensitivity_only"
        if clean_wins == len(deltas):
            overall = "report_as_parsimonious_co_primary"
        elif clean_wins == 0 and ((deltas["auc_delta"] < 0) & (deltas["pr_auc_delta"] < 0)).all():
            overall = "reject"
        rows.append(
            {
                "build_candidate": "overall",
                "decision_label": overall,
                "rationale": "Overall decision is conservative because OASIS external evaluation is not used for ADNI model selection.",
                "auc_delta": float(deltas["auc_delta"].mean()),
                "pr_auc_delta": float(deltas["pr_auc_delta"].mean()),
                "ba_delta": float(deltas["ba_delta"].mean()),
                "f1_delta": float(deltas["f1_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_final_recommendation(outdir: Path, primary: pd.DataFrame, comp: pd.DataFrame, decisions: pd.DataFrame) -> None:
    lines = [
        "# Final Recommendation",
        "",
        "This is a read-only mega-OASIS external inference audit for the ch1-only parsimonious branch versus the promoted [1,0,2] reference.",
        "",
        "Guardrails were preserved: no VAE training, classifier retraining on OASIS, OASIS threshold fitting, OASIS calibration, tensor modification, metadata modification, or model artifact modification.",
        "",
        "Primary convention: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
    ]
    overall = decisions[decisions["build_candidate"] == "overall"]
    if not overall.empty:
        lines.append(f"Overall decision: **{overall.iloc[0]['decision_label']}**.")
        lines.append("")
    lines.append("Build-level ch1-only minus promoted deltas:")
    for _, row in comp.iterrows():
        lines.append(
            f"- `{row['build_candidate']}`: "
            f"AUC delta={row['delta_ch1only_minus_promoted_auc']:.4f}, "
            f"PR-AUC delta={row['delta_ch1only_minus_promoted_pr_auc']:.4f}, "
            f"BA delta={row['delta_ch1only_minus_promoted_balanced_accuracy']:.4f}, "
            f"F1 delta={row['delta_ch1only_minus_promoted_f1']:.4f}."
        )
    lines.extend(
        [
            "",
            "Manuscript interpretation:",
            "- The promoted [1,0,2] model remains the primary ADNI model unless an explicitly pre-specified rule promotes the parsimonious branch.",
            "- ch1-only results should be reported as external/parsimony sensitivity unless it cleanly matches or improves the promoted reference across all focus OASIS builds.",
            "- Any OASIS advantage is external stress-test evidence, not a basis for post-hoc ADNI model selection.",
        ]
    )
    (outdir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    command_log: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "timestamp_start": now_iso(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "guardrails": [
            "no VAE training",
            "no classifier fitting on OASIS",
            "no OASIS threshold fitting",
            "no OASIS calibration fitting",
            "no tensor/metadata/model artifact modification",
        ],
    }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    artifact_df = validate_artifacts()
    tensor_df = pd.DataFrame(
        [{"build_candidate": k, "tensor_path": str(v), "exists": v.exists()} for k, v in base.TENSORS.items()]
    )
    write_table(outdir, "artifact_validation", artifact_df, max_rows=100)
    write_table(outdir, "tensor_artifact_validation", tensor_df, max_rows=100)

    if args.dry_run:
        command_log["dry_run"] = True
        command_log["timestamp_end"] = now_iso()
        write_json(outdir / "command_log.json", command_log)
        (outdir / "README.md").write_text(
            "# ch1-only OASIS mega 90/90 external inference\n\nDry-run completed. No scoring was performed.\n",
            encoding="utf-8",
        )
        return 0

    available = [spec for spec in CANDIDATES if artifact_df.loc[artifact_df["candidate"] == spec.label, "status"].iloc[0] == "available"]
    if len(available) != len(CANDIDATES):
        raise RuntimeError("Not all required candidates are available; inspect artifact_validation.csv")

    all_predictions: list[pd.DataFrame] = []
    fit_rows: list[pd.DataFrame] = []
    for build_name, tensor_path in base.TENSORS.items():
        if not tensor_path.exists():
            continue
        tensor, meta, oasis_channel_names = base.load_mega_tensor(tensor_path)
        for spec in available:
            preds, fit_meta = base.score_candidate_on_tensor(
                spec,
                build_name,
                tensor,
                meta,
                oasis_channel_names,
                device=device,
                batch_size=args.batch_size,
                n_jobs=args.n_jobs,
            )
            all_predictions.append(preds)
            fit_rows.append(fit_meta)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fit_meta = pd.concat(fit_rows, ignore_index=True)
    predictions.to_csv(outdir / "predictions.csv", index=False)
    write_table(outdir, "fold_readout_reconstruction_audit", fit_meta, max_rows=200)

    primary, foldwise, confusion, dist, scanner = base.metrics_tables(predictions)
    write_table(outdir, "primary_metrics", primary.sort_values(["build_candidate", "candidate"]), max_rows=200)
    write_table(outdir, "foldwise_metrics", foldwise.sort_values(["build_candidate", "candidate", "fold"]), max_rows=500)
    write_table(outdir, "confusion_matrices", confusion.sort_values(["build_candidate", "candidate", "prediction_level", "fold"]), max_rows=500)
    write_table(outdir, "score_distribution_by_diagnosis", dist.sort_values(["build_candidate", "candidate", "diagnosis"]), max_rows=300)
    write_table(outdir, "score_distribution_by_scanner_site_manufacturer", scanner, max_rows=500)

    comparison = adni_vs_oasis(primary)
    write_table(outdir, "adni_vs_oasis_generalization", comparison.sort_values(["candidate", "build_candidate"]), max_rows=100)
    panel_comp = candidate_vs_reference(primary)
    write_table(outdir, "ch1only_vs_promoted_oasis_comparison", panel_comp.sort_values("build_candidate"), max_rows=100)
    decisions = interpretation_and_decision(primary, panel_comp)
    write_table(outdir, "decision_labels", decisions, max_rows=100)

    fold_contrib = (
        predictions[predictions["prediction_level"] == "fold_model"]
        .groupby(["build_candidate", "candidate", "fold", "diagnosis"], dropna=False)
        .agg(
            n=("y", "size"),
            mean_score=("y_score", "mean"),
            median_score=("y_score", "median"),
            std_score=("y_score", "std"),
            predicted_ad_rate=("y_pred", "mean"),
        )
        .reset_index()
    )
    write_table(outdir, "fold_score_contribution", fold_contrib, max_rows=500)

    write_final_recommendation(outdir, primary, panel_comp, decisions)
    (outdir / "README.md").write_text(
        "\n".join(
            [
                "# ch1-only OASIS mega 90/90 external inference",
                "",
                "Scored the ch1-only parsimonious ADNI branch and promoted [1,0,2] reference on the same mega-OASIS tensors and frozen ADNI-only OOF-ECDF Stage B protocol.",
                "",
                "Focus builds: `concatenated_timeseries`, `runwise_140TR_pilot_parity`, `runwise164_pilot_parity`.",
                "",
                "Primary convention: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
                "",
                "No OASIS training, OASIS threshold fitting, calibration fitting, tensor modification, metadata modification, or model artifact modification was performed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    command_log["dry_run"] = False
    command_log["timestamp_end"] = now_iso()
    command_log["n_prediction_rows"] = int(len(predictions))
    command_log["candidates_scored"] = [s.label for s in available]
    command_log["builds_scored"] = [k for k, p in base.TENSORS.items() if p.exists()]
    write_json(outdir / "command_log.json", command_log)
    print(f"output_dir={outdir}")
    print("scoring_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
