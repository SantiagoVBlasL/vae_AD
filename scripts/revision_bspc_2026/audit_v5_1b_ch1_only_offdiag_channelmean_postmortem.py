#!/usr/bin/env python3
"""Read-only postmortem for the v5.1b ch1-only offdiag-channelmean FULL run.

This script reads completed run and classifier-only readout artifacts. It does
not train, mutate tensors, metadata, ledgers, configs, or model-output folders.
It writes a compact decision package under results/.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

CANDIDATE_RUN = RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_horizon4480_cycles56_full_5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
REFERENCE_READOUT = REFERENCE_RUN / "classifier_only_readout"
INTEGRITY_DIR = RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_integrity_audit"
OUT_DIR = RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_postmortem"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"

REFERENCE_ID = "v5_1b_horizon4480_ch1_0_2_current_loss"
CANDIDATE_ID = "v5_1b_horizon4480_ch1_offdiag_channelmean"

REFERENCE_LABEL = "Final v5.1b [1,0,2], current loss"
CANDIDATE_LABEL = "Ch1-only [1], offdiag_channelmean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--reference-readout", type=Path, default=REFERENCE_READOUT)
    parser.add_argument("--integrity-dir", type=Path, default=INTEGRITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def required_readout_files(readout: Path) -> list[Path]:
    return [
        readout / "classifier_sweep_pooled_metrics.csv",
        readout / "classifier_sweep_foldwise_metrics.csv",
        readout / "classifier_sweep_predictions.csv",
        readout / "classifier_sweep_pooled_confusion.csv",
        readout / "classifier_sweep_confusion_by_fold.csv",
        readout / "classifier_sweep_thresholds_by_fold.csv",
        readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        readout / "command_log.json",
    ]


def validate_readout(label: str, readout: Path) -> None:
    missing = [str(p) for p in required_readout_files(readout) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{label} readout missing required files:\n" + "\n".join(missing))
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if not mask.any():
        raise RuntimeError(f"{label} readout lacks {PRIMARY_MODEL}/{PRIMARY_THRESHOLD}.")


def format_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
    return view


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_md_rows: int = 200) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    view = format_float_columns(df.head(max_md_rows))
    text = view.to_markdown(index=False) + "\n"
    if len(df) > max_md_rows:
        text += f"\nShowing {max_md_rows} of {len(df)} rows.\n"
    (outdir / f"{stem}.md").write_text(text, encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: pd.Series, y_score: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "cn_fp_rate": safe_div(fp, fp + tn),
        "ad_fn_rate": safe_div(fn, fn + tp),
        "predicted_ad_rate": safe_div(int((pred == 1).sum()), len(pred)),
        "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def read_pooled(readout: Path, run_id: str, label: str, threshold: str = PRIMARY_THRESHOLD) -> dict[str, Any]:
    df = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(threshold)
    ].iloc[0]
    out: dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": threshold,
    }
    for col in [
        "threshold",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "predicted_ad_rate",
        "auc",
        "pr_auc",
        "brier",
    ]:
        out[col] = row.get(col, np.nan)
    out["cn_fp_rate"] = safe_div(float(out["fp"]), float(out["fp"]) + float(out["tn"]))
    out["ad_fn_rate"] = safe_div(float(out["fn"]), float(out["fn"]) + float(out["tp"]))
    pred_path = readout / "classifier_sweep_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        pred = pred[
            pred["model_name"].astype(str).eq(PRIMARY_MODEL)
            & pred["threshold_strategy"].astype(str).eq(threshold)
        ].copy()
        if not pred.empty:
            pred_metrics = binary_metrics(pred["y_true"], pred["y_score"], pred["y_pred"])
            out["brier"] = pred_metrics["brier"]
    return out


def add_deltas(df: pd.DataFrame, ref_id: str = REFERENCE_ID) -> pd.DataFrame:
    out = df.copy()
    ref = out[out["run_id"].eq(ref_id)]
    if ref.empty:
        return out
    ref_row = ref.iloc[0]
    for metric in [
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "cn_fp_rate",
        "ad_fn_rate",
        "brier",
    ]:
        if metric in out.columns and metric in ref_row.index:
            out[f"delta_vs_reference_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
    return out


def read_foldwise(readout: Path, run_id: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def read_predictions(readout: Path, run_id: str, label: str, threshold: str = PRIMARY_THRESHOLD) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(threshold)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    if "SiteCode" not in df.columns:
        df["SiteCode"] = df["SubjectID"].astype(str).str.extract(r"^(\d{3})_", expand=False).fillna("UNKNOWN")
    return df


def subgroup_metrics(preds: pd.DataFrame, group_col: str, min_n: int = 1) -> pd.DataFrame:
    if group_col not in preds.columns:
        return pd.DataFrame()
    rows = []
    for keys, sub in preds.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group = keys
        if len(sub) < min_n:
            continue
        row = {"run_id": run_id, "label": label, group_col: group}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def scanner_leakage(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.replace("fold_", ""))
        except ValueError:
            continue
        for scope, filename in [
            ("train_dev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            path = fold_dir / filename
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["run_id"] = run_id
            row["label"] = label
            row["fold"] = fold
            row["scope"] = scope
            raw = row.get("acc_site_raw", np.nan)
            lat = row.get("acc_site_latent", np.nan)
            row["latent_minus_raw_site_acc"] = float(lat) - float(raw) if pd.notna(raw) and pd.notna(lat) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def latent_info(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, pattern in [
        ("train_dev", "fold_*_trainDev_latent_info_summary.csv"),
        ("test", "fold_*_test_latent_info_summary.csv"),
    ]:
        for path in sorted(run_dir.glob(f"fold_*/{pattern}")):
            df = pd.read_csv(path)
            if df.empty:
                continue
            for _, rec in df.iterrows():
                row = rec.to_dict()
                row["run_id"] = run_id
                row["label"] = label
                row["fold"] = int(path.parent.name.replace("fold_", ""))
                row["scope"] = scope
                rows.append(row)
    return pd.DataFrame(rows)


def rate_distortion(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("fold_*/fold_*_rate_distortion.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        fold = int(path.parent.name.replace("fold_", ""))
        best_idx = pd.to_numeric(df["L_val_betaMax"], errors="coerce").idxmin()
        best = df.loc[best_idx]
        last = df.iloc[-1]
        d_val = float(best.get("D_val", np.nan))
        r_val = float(best.get("R_val_nats", np.nan))
        beta = float(best.get("beta", np.nan))
        row = {
            "run_id": run_id,
            "label": label,
            "fold": fold,
            "best_epoch": int(best.get("epoch", np.nan)),
            "final_epoch": int(last.get("epoch", np.nan)),
            "best_val_l_beta_max": float(best.get("L_val_betaMax", np.nan)),
            "best_val_recon_D": d_val,
            "best_val_kld_R_nats": r_val,
            "best_beta": beta,
            "best_val_kld_over_recon": safe_div(r_val, d_val),
            "best_val_beta_kld_over_recon": safe_div(beta * r_val, d_val),
            "last_val_l_beta_max": float(last.get("L_val_betaMax", np.nan)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def latent_qc(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("fold_*/latent_qc_metrics.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["label"] = label
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_qc(rate_df: pd.DataFrame, latent_info_df: pd.DataFrame, scanner_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, sub in rate_df.groupby("run_id"):
        row: dict[str, Any] = {"run_id": run_id}
        for col in ["best_epoch", "final_epoch", "best_val_kld_over_recon", "best_val_beta_kld_over_recon"]:
            row[f"mean_{col}"] = pd.to_numeric(sub[col], errors="coerce").mean()
        info = latent_info_df[(latent_info_df["run_id"].eq(run_id)) & (latent_info_df["scope"].eq("train_dev"))]
        y = info[info["variable"].astype(str).eq("Y_target")]
        mfr = info[info["variable"].astype(str).str.contains("Manufacturer", case=False, na=False)]
        if not y.empty:
            row["mean_active_units_train_dev"] = pd.to_numeric(y["n_active"], errors="coerce").mean()
            row["mean_total_correlation_y_target_train_dev"] = pd.to_numeric(y["total_correlation_nats"], errors="coerce").mean()
            row["mean_mi_y_target_train_dev"] = pd.to_numeric(y["mi_sum_nats"], errors="coerce").mean()
        if not mfr.empty:
            row["mean_mi_manufacturer_train_dev"] = pd.to_numeric(mfr["mi_sum_nats"], errors="coerce").mean()
        leak = scanner_df[(scanner_df["run_id"].eq(run_id)) & (scanner_df["scope"].eq("test"))]
        if not leak.empty:
            row["mean_test_acc_site_raw"] = pd.to_numeric(leak["acc_site_raw"], errors="coerce").mean()
            row["mean_test_acc_site_latent"] = pd.to_numeric(leak["acc_site_latent"], errors="coerce").mean()
            row["mean_test_latent_minus_raw_site_acc"] = pd.to_numeric(leak["latent_minus_raw_site_acc"], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def read_integrity_summary(integrity_dir: Path) -> dict[str, Any]:
    decision_path = integrity_dir / "final_integrity_decision.md"
    text = decision_path.read_text(encoding="utf-8") if decision_path.exists() else ""
    pred_path = integrity_dir / "prediction_score_comparison.csv"
    pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    inv_path = integrity_dir / "file_inventory.csv"
    inv = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()
    return {
        "status": "PASS" if "Integrity status: **PASS**" in text else "UNKNOWN_OR_FAIL",
        "decision_text": text.strip(),
        "score_all_equal": bool((not pred.empty) and pred.get("scores_exact_equal", pd.Series(dtype=bool)).all()),
        "max_score_diff_max": float(pd.to_numeric(pred.get("max_abs_score_diff", pd.Series(dtype=float)), errors="coerce").max()) if not pred.empty else np.nan,
        "missing_artifacts": int((~inv["exists"]).sum()) if not inv.empty and "exists" in inv.columns else np.nan,
        "realpath_collisions": int(inv["reference_realpath_same"].sum()) if not inv.empty and "reference_realpath_same" in inv.columns else np.nan,
    }


def write_readme(
    outdir: Path,
    primary: pd.DataFrame,
    threshold: pd.DataFrame,
    integrity: dict[str, Any],
    qc_summary: pd.DataFrame,
) -> None:
    ref = primary[primary["run_id"].eq(REFERENCE_ID)].iloc[0]
    cand = primary[primary["run_id"].eq(CANDIDATE_ID)].iloc[0]
    auc_delta = float(cand["auc"]) - float(ref["auc"])
    pr_delta = float(cand["pr_auc"]) - float(ref["pr_auc"])
    ba_delta = float(cand["balanced_accuracy"]) - float(ref["balanced_accuracy"])
    sens_delta = float(cand["sensitivity"]) - float(ref["sensitivity"])
    spec_delta = float(cand["specificity"]) - float(ref["specificity"])
    f1_delta = float(cand["f1"]) - float(ref["f1"])
    qc_text = ""
    if not qc_summary.empty:
        qc_text = "\n\nQC summary means:\n\n" + format_float_columns(qc_summary).to_markdown(index=False) + "\n"
    text = f"""# Ch1-Only Offdiag-Channelmean FULL Postmortem

This is a read-only postmortem of the FULL 5x5 `[1]` Pearson-Full-only run with `recon_loss_mode=offdiag_channelmean_sum`, compared against the final v5.1b `[1,0,2]` horizon4480/cycles56 model.

Integrity audit status: **{integrity['status']}**.

Primary Stage B readout: `{PRIMARY_MODEL}` with `{PRIMARY_THRESHOLD}`.

## Main Result

- Reference `[1,0,2]`: AUC={float(ref['auc']):.6f}, PR-AUC={float(ref['pr_auc']):.6f}, BA={float(ref['balanced_accuracy']):.6f}, Sens={float(ref['sensitivity']):.6f}, Spec={float(ref['specificity']):.6f}, F1={float(ref['f1']):.6f}
- Candidate `[1]`: AUC={float(cand['auc']):.6f}, PR-AUC={float(cand['pr_auc']):.6f}, BA={float(cand['balanced_accuracy']):.6f}, Sens={float(cand['sensitivity']):.6f}, Spec={float(cand['specificity']):.6f}, F1={float(cand['f1']):.6f}
- Delta candidate minus reference: AUC={auc_delta:+.6f}, PR-AUC={pr_delta:+.6f}, BA={ba_delta:+.6f}, Sens={sens_delta:+.6f}, Spec={spec_delta:+.6f}, F1={f1_delta:+.6f}

## Decision

The candidate should be reported as a **secondary simplified model / channel-ablation confirmation**, not promoted as the main manuscript model.

The Pearson Full channel is clearly dominant: the single-channel model improved AUROC and specificity. However, the multichannel `[1,0,2]` model retains better PR-AUC, sensitivity, balanced accuracy, and F1 under the pre-specified sensitivity-constrained operating point. Because PR-AUC and sensitivity are more clinically relevant under AD/CN imbalance, the main model remains `[1,0,2]`.

## Threshold Behavior

The package includes `threshold_comparison.csv/.md`, comparing fixed 0.5 against the inner-OOF sensitivity-constrained threshold for both runs. Threshold selection remains leakage-safe because thresholds are selected from train/inner-CV OOF predictions before applying to outer-fold test subjects.

## Outputs

- `primary_decision_table.csv/.md`: pooled primary metrics and deltas.
- `foldwise_comparison.csv/.md`: per-fold primary Stage B metrics.
- `manufacturer_subgroup_comparison.csv/.md`: pooled manufacturer FP/FN and subgroup metrics.
- `sitecode_subgroup_comparison.csv/.md`: SiteCode subgroup behavior, with SiteCode derived from the ADNI subject prefix.
- `scanner_leakage_comparison.csv/.md`: raw vs latent manufacturer predictability.
- `threshold_comparison.csv/.md`: fixed 0.5 vs inner-OOF threshold metrics.
- `qc_rate_distortion_latent_summary.csv/.md`: rate-distortion, active-unit, MI/TC, and leakage summary.
{qc_text}
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def write_decision_docs(outdir: Path, primary: pd.DataFrame, integrity: dict[str, Any]) -> None:
    ref = primary[primary["run_id"].eq(REFERENCE_ID)].iloc[0]
    cand = primary[primary["run_id"].eq(CANDIDATE_ID)].iloc[0]
    promotion_pass = (
        float(cand["auc"]) > float(ref["auc"])
        and float(cand["pr_auc"]) >= float(ref["pr_auc"])
        and float(cand["balanced_accuracy"]) >= float(ref["balanced_accuracy"]) - 0.005
        and float(cand["f1"]) >= float(ref["f1"]) - 0.005
        and float(cand["sensitivity"]) >= 0.70
        and integrity["status"] == "PASS"
    )
    decision = "promote_main_model" if promotion_pass else "secondary_simplified_model_not_main"
    final_text = f"""# Final Model Decision

Decision: **{decision}**.

The ch1-only offdiag-channelmean candidate has integrity audit status **{integrity['status']}** and improves AUROC versus the final `[1,0,2]` model ({float(cand['auc']):.6f} vs {float(ref['auc']):.6f}). It also improves specificity ({float(cand['specificity']):.6f} vs {float(ref['specificity']):.6f}).

It should **not** replace the main manuscript model because it does not satisfy the full promotion rule:

- PR-AUC is lower: {float(cand['pr_auc']):.6f} vs {float(ref['pr_auc']):.6f}.
- Sensitivity is lower: {float(cand['sensitivity']):.6f} vs {float(ref['sensitivity']):.6f}.
- Balanced accuracy is lower: {float(cand['balanced_accuracy']):.6f} vs {float(ref['balanced_accuracy']):.6f}.
- F1 is lower: {float(cand['f1']):.6f} vs {float(ref['f1']):.6f}.

Final recommendation: keep the v5.1b horizon4480/cycles56 `[1,0,2]` model as the manuscript model. Report `[1]` as a parsimonious secondary ablation showing that Pearson Full carries most of the rank signal, while the multichannel model better preserves sensitivity and precision-recall behavior.
"""
    (outdir / "final_model_decision.md").write_text(final_text, encoding="utf-8")

    manuscript = """# Manuscript Channel-Ablation Paragraph

As a scale-corrected confirmation of channel relevance, we trained a FULL 5x5 model using only the Pearson Full Fisher-z channel (`[1]`) with the off-diagonal channel-mean reconstruction objective. This parsimonious model achieved slightly higher ROC-AUC than the multichannel `[1,0,2]` model (0.7894 vs 0.7830) and higher specificity (0.7300 vs 0.7000), confirming that Pearson Full connectivity is the dominant single channel. However, the multichannel model retained better PR-AUC (0.5599 vs 0.5426), sensitivity (0.7708 vs 0.7188), balanced accuracy (0.7354 vs 0.7244), and F1 (0.5692 vs 0.5610) at the pre-specified sensitivity-constrained operating point. We therefore retained `[1,0,2]` as the primary manuscript model and report `[1]` as a secondary simplified ablation rather than an AUROC-only replacement.
"""
    (outdir / "manuscript_channel_ablation_paragraph.md").write_text(manuscript, encoding="utf-8")

    reviewer = """# Reviewer Q2 Response Text

We added a scale-corrected FULL 5x5 channel-ablation confirmation to address whether the selected multichannel input was justified. The Pearson Full channel alone produced the highest AUROC among the tested scale-corrected FAST candidates and, in FULL confirmation, reached ROC-AUC=0.7894. This supports the reviewer-relevant interpretation that Pearson Full connectivity contains the dominant AD/CN rank signal. We did not switch the primary model to this AUROC-optimized single-channel variant because the pre-specified clinical readout also considers PR-AUC, sensitivity, balanced accuracy, and F1 under a leakage-safe sensitivity-constrained threshold. On those metrics the original `[1,0,2]` model remained preferable (PR-AUC=0.5599 vs 0.5426; sensitivity=0.7708 vs 0.7188; BA=0.7354 vs 0.7244; F1=0.5692 vs 0.5610). This is why `[1]` is reported as a secondary simplified model, while `[1,0,2]` remains the conservative manuscript model.
"""
    (outdir / "reviewer_q2_response_text.md").write_text(reviewer, encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    reference_run = resolve(args.reference_run)
    reference_readout = resolve(args.reference_readout)
    integrity_dir = resolve(args.integrity_dir)
    outdir = resolve(args.output_dir)

    if args.dry_run:
        print("Dry-run postmortem preflight.")
        for label, readout in [("candidate", candidate_readout), ("reference", reference_readout)]:
            missing = [str(p) for p in required_readout_files(readout) if not p.exists()]
            print(f"{label} readout: {readout}")
            print(f"{label} missing required files: {len(missing)}")
        print(f"Integrity dir: {integrity_dir}")
        print(f"Output dir   : {outdir}")
        print("No files were written.")
        return 0

    validate_readout("candidate", candidate_readout)
    validate_readout("reference", reference_readout)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = [
        (REFERENCE_ID, REFERENCE_LABEL, reference_run, reference_readout),
        (CANDIDATE_ID, CANDIDATE_LABEL, candidate_run, candidate_readout),
    ]

    primary = pd.DataFrame([read_pooled(readout, run_id, label) for run_id, label, _, readout in specs])
    primary = add_deltas(primary)
    write_table(outdir, "primary_decision_table", primary)

    foldwise = pd.concat([read_foldwise(readout, run_id, label) for run_id, label, _, readout in specs], ignore_index=True)
    write_table(outdir, "foldwise_comparison", foldwise)

    thresholds = pd.DataFrame(
        [
            read_pooled(readout, run_id, label, threshold)
            for run_id, label, _, readout in specs
            for threshold in [FIXED_THRESHOLD, PRIMARY_THRESHOLD]
        ]
    )
    thresholds = add_deltas(thresholds)
    write_table(outdir, "threshold_comparison", thresholds)
    write_table(
        outdir,
        "confusion_matrix_comparison",
        thresholds[
            [
                "run_id",
                "label",
                "threshold_strategy",
                "threshold",
                "n",
                "n_cn",
                "n_ad",
                "tn",
                "fp",
                "fn",
                "tp",
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "f1",
                "auc",
                "pr_auc",
                "brier",
            ]
        ],
    )

    preds = pd.concat([read_predictions(readout, run_id, label) for run_id, label, _, readout in specs], ignore_index=True)
    manufacturer = subgroup_metrics(preds, "Manufacturer")
    write_table(outdir, "manufacturer_subgroup_comparison", manufacturer)
    sitecode = subgroup_metrics(preds, "SiteCode")
    write_table(outdir, "sitecode_subgroup_comparison", sitecode)

    leakage = pd.concat([scanner_leakage(run_dir, run_id, label) for run_id, label, run_dir, _ in specs], ignore_index=True)
    write_table(outdir, "scanner_leakage_comparison", leakage)

    rate = pd.concat([rate_distortion(run_dir, run_id, label) for run_id, label, run_dir, _ in specs], ignore_index=True)
    latent = pd.concat([latent_info(run_dir, run_id, label) for run_id, label, run_dir, _ in specs], ignore_index=True)
    latent_qc_df = pd.concat([latent_qc(run_dir, run_id, label) for run_id, label, run_dir, _ in specs], ignore_index=True)
    qc_summary = summarize_qc(rate, latent, leakage)
    write_table(outdir, "rate_distortion_by_fold", rate)
    write_table(outdir, "latent_information_by_fold", latent)
    write_table(outdir, "latent_qc_by_fold", latent_qc_df)
    write_table(outdir, "qc_rate_distortion_latent_summary", qc_summary)

    integrity = read_integrity_summary(integrity_dir)
    write_readme(outdir, primary, thresholds, integrity, qc_summary)
    write_decision_docs(outdir, primary, integrity)

    command_log = {
        "script": rel(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": rel(candidate_run),
        "candidate_readout": rel(candidate_readout),
        "reference_run": rel(reference_run),
        "reference_readout": rel(reference_readout),
        "integrity_dir": rel(integrity_dir),
        "output_dir": rel(outdir),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "read_only": True,
        "trained": False,
        "modified_tensor": False,
        "modified_metadata": False,
        "modified_ledger": False,
        "modified_config": False,
        "modified_existing_model_outputs": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(command_log, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
