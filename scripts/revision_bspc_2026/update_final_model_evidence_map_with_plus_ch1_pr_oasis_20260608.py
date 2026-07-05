#!/usr/bin/env python3
"""Add plus-ch1 PR-recovery frozen-readout results to final evidence tables.

This script is intentionally read-only with respect to model/tensor artifacts.
It copies the latest relevant evidence-map/manuscript-support tables into a new
package and appends only the three requested frozen readout sensitivity rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path("results/revision_bspc_2026")
BASE_EVIDENCE = ROOT / "final_full_model_evidence_map_with_ch12_beta2p75_20260607"
BASE_DECISION = ROOT / "final_model_decision_adni_oasis_manuscript_support_20260605"
PLUS_ADNI = ROOT / "plus_ch1_pr_recovery_readout_batch_20260608"
PLUS_OASIS = ROOT / "plus_ch1_pr_recovery_oasis_external_stress_test_20260608"
OUTPUT = ROOT / "final_model_evidence_map_with_plus_ch1_pr_oasis_20260608"

FOCUS_CANDIDATES = [
    "plus_ch1_meta_logreg_pr_auc_selected",
    "plus_ch1_meta_logreg_rank_features",
    "plus_ch1_meta_logreg_logit_features",
]

BUILD_TO_COLUMNS = {
    "concatenated_timeseries": ("oasis_concatenated_auc", "oasis_concatenated_pr_auc"),
    "runwise164_pilot_parity": ("oasis_runwise164_auc", "oasis_runwise164_pr_auc"),
    "runwise_140TR_pilot_parity": ("oasis_runwise140_auc", "oasis_runwise140_pr_auc"),
}

DISPLAY_NAMES = {
    "plus_ch1_meta_logreg_pr_auc_selected": "plus-ch1 PR-AUC-selected frozen readout",
    "plus_ch1_meta_logreg_rank_features": "plus-ch1 rank-feature frozen readout",
    "plus_ch1_meta_logreg_logit_features": "plus-ch1 logit-feature frozen readout",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv_md(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path = csv_path.with_suffix(".md")
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = simple_markdown_table(df)
    md_path.write_text(md + "\n", encoding="utf-8")


def simple_markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [format_value(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def oasis_lookup(oasis: pd.DataFrame, candidate: str) -> dict[str, float]:
    rows = oasis[oasis["candidate"].eq(candidate)]
    out: dict[str, float] = {}
    for build, (auc_col, pr_col) in BUILD_TO_COLUMNS.items():
        match = rows[rows["build_candidate"].eq(build)]
        if match.empty:
            out[auc_col] = float("nan")
            out[pr_col] = float("nan")
        else:
            row = match.iloc[0]
            out[auc_col] = float(row["auc"])
            out[pr_col] = float(row["pr_auc"])
    return out


def adni_row(adni: pd.DataFrame, candidate: str) -> pd.Series:
    match = adni[adni["candidate_id"].eq(candidate)]
    if match.empty:
        raise ValueError(f"Missing ADNI candidate row: {candidate}")
    return match.iloc[0]


def make_evidence_row(template_columns: Iterable[str], adni: pd.Series, oasis_vals: dict[str, float]) -> dict:
    candidate = str(adni["candidate_id"])
    row = {col: float("nan") for col in template_columns}
    row.update(
        {
            "model_id": candidate,
            "display_name": DISPLAY_NAMES[candidate],
            "decision_class": "high-AUC frozen-readout sensitivity only",
            "final_decision": "not_primary",
            "run_name": candidate,
            "run_dir": str(PLUS_ADNI),
            "channel_set_order": "[1,0,2] + [1]",
            "selected_channel_names": "frozen promoted [1,0,2] score + frozen ch1-only score + Age/Sex",
            "latent_dim": 384.0,
            "beta_vae": 3.75,
            "adni_oof_ecdf_auc": float(adni["auc"]),
            "adni_oof_ecdf_pr_auc": float(adni["pr_auc"]),
            "adni_oof_ecdf_ba": float(adni["balanced_accuracy"]),
            "adni_oof_ecdf_sens": float(adni["sensitivity"]),
            "adni_oof_ecdf_spec": float(adni["specificity"]),
            "adni_oof_ecdf_f1": float(adni["f1"]),
            "philips_cn_fpr": float(adni["philips_cn_fpr"]),
            "philips_cn_fp": int(adni["philips_cn_fp"]),
            "philips_cn_n": int(adni["philips_cn_n"]),
            "oasis_status": "external_stress_test_scored_not_for_selection",
        }
    )
    row.update(oasis_vals)
    return row


def make_decision_row(template_columns: Iterable[str], adni: pd.Series, oasis_vals: dict[str, float]) -> dict:
    candidate = str(adni["candidate_id"])
    row = {col: float("nan") for col in template_columns}
    row.update(
        {
            "model_id": candidate,
            "display_name": DISPLAY_NAMES[candidate],
            "decision_class": "high-AUC frozen-readout sensitivity only",
            "final_decision": "not_primary",
            "rationale_short": "Frozen two-source ADNI readout improves internal AUC, but OASIS stress-test AUC/PR-AUC remains below the promoted [1,0,2] reference.",
            "run_name": candidate,
            "run_dir": str(PLUS_ADNI),
            "completion_status": "complete_frozen_readout_oasis_stress_test",
            "channel_set_order": "[1,0,2] + [1]",
            "selected_channel_names": "frozen promoted [1,0,2] score + frozen ch1-only score + Age/Sex",
            "beta_vae": 3.75,
            "latent_dim": 384.0,
            "adni_metric_source": "oof_ecdf_frozen_readout",
            "adni_oof_ecdf_auc": float(adni["auc"]),
            "adni_oof_ecdf_pr_auc": float(adni["pr_auc"]),
            "adni_oof_ecdf_balanced_accuracy": float(adni["balanced_accuracy"]),
            "adni_oof_ecdf_sensitivity": float(adni["sensitivity"]),
            "adni_oof_ecdf_specificity": float(adni["specificity"]),
            "adni_oof_ecdf_f1": float(adni["f1"]),
            "philips_cn_fpr": float(adni["philips_cn_fpr"]),
            "oasis_artifact_status": "available_stress_test_only",
            "oasis_source_family": "mega90_frozen_stress_test",
            "oasis_concatenated_auc": oasis_vals["oasis_concatenated_auc"],
            "oasis_concatenated_pr_auc": oasis_vals["oasis_concatenated_pr_auc"],
            "oasis_runwise164_auc": oasis_vals["oasis_runwise164_auc"],
            "oasis_runwise164_pr_auc": oasis_vals["oasis_runwise164_pr_auc"],
            "oasis_runwise140_auc": oasis_vals["oasis_runwise140_auc"],
            "oasis_runwise140_pr_auc": oasis_vals["oasis_runwise140_pr_auc"],
            "adni_oof_ecdf_philips_cn_fpr": float(adni["philips_cn_fpr"]),
        }
    )
    return row


def make_comparison_row(template_columns: Iterable[str], adni: pd.Series, oasis_vals: dict[str, float]) -> dict:
    candidate = str(adni["candidate_id"])
    row = {col: float("nan") for col in template_columns}
    row.update(
        {
            "model_id": candidate,
            "display_name": DISPLAY_NAMES[candidate],
            "decision_class": "high-AUC frozen-readout sensitivity only",
            "final_decision": "not_primary",
            "adni_oof_ecdf_auc": float(adni["auc"]),
            "adni_oof_ecdf_pr_auc": float(adni["pr_auc"]),
            "adni_oof_ecdf_balanced_accuracy": float(adni["balanced_accuracy"]),
            "adni_oof_ecdf_sensitivity": float(adni["sensitivity"]),
            "adni_oof_ecdf_specificity": float(adni["specificity"]),
            "adni_oof_ecdf_f1": float(adni["f1"]),
            "philips_cn_fpr": float(adni["philips_cn_fpr"]),
            "oasis_concatenated_auc": oasis_vals["oasis_concatenated_auc"],
            "oasis_concatenated_pr_auc": oasis_vals["oasis_concatenated_pr_auc"],
            "oasis_runwise164_auc": oasis_vals["oasis_runwise164_auc"],
            "oasis_runwise164_pr_auc": oasis_vals["oasis_runwise164_pr_auc"],
            "oasis_runwise140_auc": oasis_vals["oasis_runwise140_auc"],
            "oasis_runwise140_pr_auc": oasis_vals["oasis_runwise140_pr_auc"],
        }
    )
    return row


def append_unique(base: pd.DataFrame, rows: list[dict], key: str = "model_id") -> pd.DataFrame:
    add = pd.DataFrame(rows)
    if key in base.columns:
        base = base[~base[key].isin(add[key])]
    return pd.concat([base, add], ignore_index=True)


def write_texts(out_dir: Path, adni: pd.DataFrame, oasis: pd.DataFrame) -> None:
    promoted = oasis[oasis["candidate"].eq("reference_promoted_ch102_latent384_beta3p75")]
    focus = oasis[oasis["candidate"].isin(FOCUS_CANDIDATES)]
    best_runwise164 = focus[focus["build_candidate"].eq("runwise164_pilot_parity")].sort_values(
        ["auc", "pr_auc"], ascending=False
    ).iloc[0]
    promoted_runwise164 = promoted[promoted["build_candidate"].eq("runwise164_pilot_parity")].iloc[0]
    best_adni = adni[adni["candidate_id"].isin(FOCUS_CANDIDATES)].sort_values(
        ["auc", "pr_auc"], ascending=False
    ).iloc[0]
    best_pr_adni = adni[adni["candidate_id"].isin(FOCUS_CANDIDATES)].sort_values(
        ["pr_auc", "auc"], ascending=False
    ).iloc[0]

    final_recommendation = f"""# Final Recommendation

Decision: `promoted_model_remains_primary`.

This update adds three frozen plus-ch1 PR-recovery readouts to the final evidence
map:

- `plus_ch1_meta_logreg_pr_auc_selected`
- `plus_ch1_meta_logreg_rank_features`
- `plus_ch1_meta_logreg_logit_features`

The plus-ch1 readouts are high-AUC frozen-readout sensitivities only. They use
existing ADNI-trained fold artifacts and do not introduce a new VAE, tensor, or
OASIS-selected model.

ADNI summary:

- Highest ADNI AUC among the three: `{best_adni['candidate_id']}` with
  AUC `{best_adni['auc']:.6f}`, PR-AUC `{best_adni['pr_auc']:.6f}`,
  BA `{best_adni['balanced_accuracy']:.6f}`, F1 `{best_adni['f1']:.6f}`,
  Philips CN FPR `{int(best_adni['philips_cn_fp'])}/{int(best_adni['philips_cn_n'])} = {best_adni['philips_cn_fpr']:.6f}`.
- Highest ADNI PR-AUC among the three: `{best_pr_adni['candidate_id']}` with
  AUC `{best_pr_adni['auc']:.6f}`, PR-AUC `{best_pr_adni['pr_auc']:.6f}`,
  BA `{best_pr_adni['balanced_accuracy']:.6f}`, F1 `{best_pr_adni['f1']:.6f}`,
  Philips CN FPR `{int(best_pr_adni['philips_cn_fp'])}/{int(best_pr_adni['philips_cn_n'])} = {best_pr_adni['philips_cn_fpr']:.6f}`.

OASIS stress-test summary:

- Best plus-ch1 focus row on runwise164 parity: `{best_runwise164['candidate']}`
  with AUC `{best_runwise164['auc']:.6f}` and PR-AUC `{best_runwise164['pr_auc']:.6f}`.
- Promoted `[1,0,2]` reference on runwise164 parity: AUC
  `{promoted_runwise164['auc']:.6f}` and PR-AUC `{promoted_runwise164['pr_auc']:.6f}`.
- The plus-ch1 PR-recovery candidates remain below the promoted reference on
  AUC and PR-AUC for concatenated, runwise164 parity, and runwise140TR parity.

Interpretation:

The plus-ch1 PR-recovery readouts improve ADNI frozen-readout geometry, but the
gain does not transfer externally as an OASIS improvement over the promoted
model. They should be reported, if at all, as high-AUC frozen-readout
sensitivities. They do not reopen primary model selection.

Guardrails: no training, no OASIS threshold/calibration fitting, no OASIS model
selection, and no tensor/metadata/model-artifact modification were performed.
"""

    results_paragraph = """The final ADNI primary model remains the recover035 [1,0,2] latent384 beta3.75 beta-VAE with the OOF-ECDF logreg_l2 readout. It achieved ADNI OOF AUC=0.795155 and PR-AUC=0.573934 with balanced accuracy=0.725979 and F1=0.563492. Frozen plus-ch1 two-source readouts improved internal ADNI AUC in post-final sensitivity analysis, including plus_ch1_meta_logreg_pr_auc_selected (AUC=0.812027, PR-AUC=0.581864), plus_ch1_meta_logreg_rank_features (AUC=0.810893, PR-AUC=0.583065), and plus_ch1_meta_logreg_logit_features (AUC=0.800636, PR-AUC=0.592010). However, these readouts did not improve external OASIS stress-test ranking over the promoted model: on runwise164_pilot_parity, the best plus-ch1 focus row reached AUC=0.629630 and PR-AUC=0.641218, compared with promoted [1,0,2] AUC=0.647778 and PR-AUC=0.667838. The plus-ch1 rows are therefore retained as high-AUC frozen-readout sensitivities rather than primary models."""

    limitations_paragraph = """The plus-ch1 PR-recovery analysis was a frozen-readout sensitivity using existing ADNI-trained fold artifacts, not a newly trained VAE model. It was evaluated on OASIS only as an external stress test, with no OASIS threshold fitting, calibration fitting, or model selection. Although the plus-ch1 readouts improved internal ADNI AUC and, for the logit-feature variant, PR-AUC, the external OASIS gain did not transfer relative to the promoted [1,0,2] model. This supports treating the result as ADNI-specific score-geometry evidence and reinforces that OASIS remains a stress-test dataset rather than a tuning set."""

    (out_dir / "final_recommendation.md").write_text(final_recommendation, encoding="utf-8")
    (out_dir / "manuscript_results_paragraph.md").write_text(results_paragraph + "\n", encoding="utf-8")
    (out_dir / "manuscript_limitations_paragraph.md").write_text(limitations_paragraph + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    evidence = read_csv(BASE_EVIDENCE / "full_model_evidence_map.csv")
    decision = read_csv(BASE_DECISION / "final_model_decision_table.csv")
    comparison = read_csv(BASE_DECISION / "adni_oasis_final_comparison.csv")
    adni = read_csv(PLUS_ADNI / "candidate_metrics.csv")
    oasis = read_csv(PLUS_OASIS / "primary_metrics.csv")

    oasis = oasis[oasis["candidate"].isin(FOCUS_CANDIDATES + [
        "reference_promoted_ch102_latent384_beta3p75",
        "reference_ch1only_latent384_beta3p75",
        "reference_previous_plus_ch1_auc_selected",
    ])].copy()
    oasis = oasis[oasis["prediction_level"].eq("ensemble_mean_score_majority_vote")]

    missing_adni = sorted(set(FOCUS_CANDIDATES) - set(adni["candidate_id"]))
    missing_oasis = sorted(set(FOCUS_CANDIDATES) - set(oasis["candidate"]))
    if missing_adni or missing_oasis:
        raise RuntimeError(f"Missing ADNI={missing_adni}, OASIS={missing_oasis}")

    evidence_rows = []
    decision_rows = []
    comparison_rows = []
    for candidate in FOCUS_CANDIDATES:
        arow = adni_row(adni, candidate)
        ovals = oasis_lookup(oasis, candidate)
        evidence_rows.append(make_evidence_row(evidence.columns, arow, ovals))
        decision_rows.append(make_decision_row(decision.columns, arow, ovals))
        comparison_rows.append(make_comparison_row(comparison.columns, arow, ovals))

    evidence_out = append_unique(evidence, evidence_rows)
    decision_out = append_unique(decision, decision_rows)
    comparison_out = append_unique(comparison, comparison_rows)

    if args.dry_run:
        print(json.dumps({
            "output_dir": str(out_dir),
            "base_evidence_rows": int(evidence.shape[0]),
            "base_decision_rows": int(decision.shape[0]),
            "base_comparison_rows": int(comparison.shape[0]),
            "added_candidates": FOCUS_CANDIDATES,
            "evidence_rows_after": int(evidence_out.shape[0]),
            "decision_rows_after": int(decision_out.shape[0]),
            "comparison_rows_after": int(comparison_out.shape[0]),
        }, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_md(evidence_out, out_dir / "full_model_evidence_map.csv")
    write_csv_md(decision_out, out_dir / "final_model_decision_table.csv")
    write_csv_md(comparison_out, out_dir / "adni_oasis_final_comparison.csv")
    write_texts(out_dir, adni, oasis)

    readme = f"""# Final Model Evidence Map With Plus-Ch1 PR-Recovery OASIS Stress Test

This package appends the three requested frozen plus-ch1 PR-recovery readouts to
the final model evidence map and manuscript-support tables.

Inputs:

- `{PLUS_ADNI}`
- `{PLUS_OASIS}`
- `{BASE_EVIDENCE}`
- `{BASE_DECISION}`

Guardrails:

- no training
- no OASIS calibration or threshold fitting
- no tensor/metadata/model artifact modification
- no model selection based on OASIS
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    command_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "inputs": {
            "base_evidence": str(BASE_EVIDENCE),
            "base_decision": str(BASE_DECISION),
            "plus_ch1_adni": str(PLUS_ADNI),
            "plus_ch1_oasis": str(PLUS_OASIS),
        },
        "output": str(out_dir),
        "added_candidates": FOCUS_CANDIDATES,
        "guardrails": [
            "no training",
            "no OASIS calibration fitting",
            "no OASIS threshold fitting",
            "no tensor modification",
            "no metadata modification",
            "no model artifact modification",
            "no model selection based on OASIS",
        ],
    }
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "full_model_evidence_map_rows": int(evidence_out.shape[0]),
        "final_model_decision_table_rows": int(decision_out.shape[0]),
        "adni_oasis_final_comparison_rows": int(comparison_out.shape[0]),
        "added_candidates": FOCUS_CANDIDATES,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
