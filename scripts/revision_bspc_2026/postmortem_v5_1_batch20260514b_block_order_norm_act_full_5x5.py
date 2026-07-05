#!/usr/bin/env python3
"""Read-only postmortem for the block_order=norm_act FULL 5x5 candidate.

The script reads completed run artifacts only. It does not train, rewrite
model-output folders, tensors, metadata, or ledgers. It writes a compact
postmortem package and updates the manuscript defense failed-optimization table
with the non-promoted norm_act confirmation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
LOCKED_READOUT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
NORM_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_block_order_norm_act_full_5x5"
NORM_READOUT = NORM_RUN / "classifier_only_readout"

METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

OUTDIR = RESULTS / "adni_v5_1_batch20260514b_block_order_norm_act_full_5x5_postmortem"
DEFENSE_DIR = RESULTS / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
FAILED_TABLE = DEFENSE_DIR / "failed_optimization_table.csv"
FAILED_TABLE_MD = DEFENSE_DIR / "failed_optimization_table.md"
FINAL_RECOMMENDATION = DEFENSE_DIR / "final_recommendation.md"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
LOCKED_METRICS = {
    "auc": 0.778785,
    "pr_auc": 0.551832,
    "balanced_accuracy": 0.712917,
    "sensitivity": 0.729167,
    "specificity": 0.696667,
    "f1": 0.544747,
}
NORM_METRICS = {
    "auc": 0.752187,
    "pr_auc": 0.492421,
    "balanced_accuracy": 0.717708,
    "sensitivity": 0.718750,
    "specificity": 0.716667,
    "f1": 0.552000,
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.6f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(stem: str, df: pd.DataFrame, max_rows: int = 80, outdir: Path = OUTDIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "cn_fp_rate": safe_div(fp, fp + tn),
        "ad_fn_rate": safe_div(fn, fn + tp),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def normalize_mfr(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def sitecode_from_subject(subject_id: Any) -> str:
    text = "" if pd.isna(subject_id) else str(subject_id)
    return text[:3] if len(text) >= 3 else "UNKNOWN"


def read_pooled(readout: Path, run_id: str, label: str) -> Dict[str, Any]:
    df = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out = {"run_id": run_id, "label": label, "model_name": PRIMARY_MODEL, "threshold_strategy": PRIMARY_THRESHOLD}
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        out[col] = row.get(col, np.nan)
    return out


def main_comparison() -> pd.DataFrame:
    rows = [
        read_pooled(LOCKED_READOUT, "locked_current", "locked current FULL legacy_act_norm [1,0,2]"),
        read_pooled(NORM_READOUT, "block_order_norm_act", "block_order=norm_act FULL [1,0,2]"),
    ]
    df = pd.DataFrame(rows)
    ref = df[df["run_id"].eq("locked_current")].iloc[0]
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        df[f"delta_vs_locked_{metric}"] = pd.to_numeric(df[metric], errors="coerce") - float(ref[metric])
    return df


def load_vae_history(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    cfg = read_json(run_dir / "run_config.json")
    args = cfg.get("args", {})
    patience = int(args.get("early_stopping_patience_vae", 320) or 320)
    max_epochs = int(args.get("epochs_vae", 3840) or 3840)
    rows: List[Dict[str, Any]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold = int(fold_dir.name.replace("fold_", ""))
        path = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        if not path.exists():
            continue
        hist = joblib.load(path)
        metric_name = "val_loss_modelsel" if "val_loss_modelsel" in hist else "val_loss"
        values = np.asarray(hist.get(metric_name, []), dtype=float)
        epochs_completed = int(len(values))
        if epochs_completed:
            best_idx = int(np.nanargmin(values))
            best_epoch = best_idx + 1
            best_val = float(values[best_idx])
        else:
            best_idx = -1
            best_epoch = np.nan
            best_val = np.nan
        row = {
            "run_id": run_id,
            "label": label,
            "fold": fold,
            "best_epoch": best_epoch,
            "early_stop_epoch": epochs_completed,
            "max_epochs_configured": max_epochs,
            "early_stopping_patience": patience,
            "metric_used_for_best": metric_name,
            "best_val_loss_beta_max": best_val,
            "stopped_early": bool(epochs_completed < max_epochs),
            "distance_best_to_early_stop": epochs_completed - best_epoch if epochs_completed and not pd.isna(best_epoch) else np.nan,
        }
        for key in ["train_loss", "train_recon", "train_kld", "val_loss", "val_recon", "val_kld", "val_loss_modelsel", "beta"]:
            arr = np.asarray(hist.get(key, []), dtype=float)
            row[f"{key}_at_best"] = float(arr[best_idx]) if len(arr) and best_idx >= 0 else np.nan
            row[f"{key}_final"] = float(arr[-1]) if len(arr) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["run_id", "fold"])
    ref = df[df["run_id"].eq("locked_current")].set_index("fold")
    for metric in ["best_epoch", "early_stop_epoch", "best_val_loss_beta_max", "val_recon_at_best", "val_kld_at_best"]:
        df[f"delta_vs_locked_{metric}"] = [
            float(row[metric]) - float(ref.loc[int(row["fold"]), metric]) if int(row["fold"]) in ref.index and pd.notna(row[metric]) else np.nan
            for _, row in df.iterrows()
        ]
    return df


def stage_a_metrics(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    path = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))[0]
    df = pd.read_csv(path)
    df = df.rename(columns={"actual_classifier_type": "classifier", "f1_score": "f1"})
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    cols = ["run_id", "label", "fold", "classifier", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "best_clf_params", "did_calibrate"]
    return df[[c for c in cols if c in df.columns]].copy()


def stage_b_foldwise(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    cols = ["run_id", "label", "fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "threshold", "best_inner_auc", "best_params"]
    return df[[c for c in cols if c in df.columns]].copy()


def add_fold_deltas(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    index_cols = ["fold", "classifier"] if "classifier" in out.columns else ["fold"]
    ref = out[out["run_id"].eq("locked_current")].set_index(index_cols)
    for metric in metrics:
        vals = []
        for _, row in out.iterrows():
            key = (int(row["fold"]), row["classifier"]) if "classifier" in out.columns else int(row["fold"])
            if key in ref.index and pd.notna(row.get(metric, np.nan)):
                vals.append(float(row[metric]) - float(ref.loc[key, metric]))
            else:
                vals.append(np.nan)
        out[f"delta_vs_locked_{metric}"] = vals
    return out


def load_predictions(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    df["SiteCode"] = df["SubjectID"].map(sitecode_from_subject)
    df["Manufacturer_mapped"] = df["Manufacturer"].map(normalize_mfr)
    return df


def subgroup_metrics(pred: pd.DataFrame, group_col: str, min_auc_class_n: int = 3) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in pred.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group = keys
        row = {"run_id": run_id, "label": label, group_col: group}
        metrics = binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"])
        if metrics["n_cn"] < min_auc_class_n or metrics["n_ad"] < min_auc_class_n:
            metrics["auc"] = np.nan
            metrics["pr_auc"] = np.nan
            metrics["auc_note"] = f"not reported: requires >= {min_auc_class_n} CN and AD"
        else:
            metrics["auc_note"] = "reported"
        row.update(metrics)
        rows.append(row)
    out = pd.DataFrame(rows)
    key_cols = [group_col]
    ref = out[out["run_id"].eq("locked_current")].set_index(key_cols)
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "cn_fp_rate", "ad_fn_rate"]:
        vals = []
        for _, row in out.iterrows():
            key = row[group_col]
            if key in ref.index and pd.notna(row.get(metric, np.nan)) and pd.notna(ref.loc[key, metric]):
                vals.append(float(row[metric]) - float(ref.loc[key, metric]))
            else:
                vals.append(np.nan)
        out[f"delta_vs_locked_{metric}"] = vals
    return out.sort_values([group_col, "run_id"])


def scanner_leakage_rows(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        for split, path in [
            ("train_dev", run_dir / f"fold_{fold}" / f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row.update({"run_id": run_id, "label": label, "fold": fold, "split": split})
            if "latent_minus_raw_site_acc" not in row:
                if "acc_site_latent" in row and "acc_site_raw" in row:
                    row["latent_minus_raw_site_acc"] = float(row["acc_site_latent"]) - float(row["acc_site_raw"])
                else:
                    row["latent_minus_raw_site_acc"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["fold", "split", "run_id"])


def add_scanner_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ref = out[out["run_id"].eq("locked_current")].set_index(["fold", "split"])
    for metric in ["acc_site_raw", "acc_site_latent", "latent_minus_raw_site_acc", "chance_level"]:
        vals = []
        for _, row in out.iterrows():
            key = (int(row["fold"]), row["split"])
            if key in ref.index and pd.notna(row.get(metric, np.nan)):
                vals.append(float(row[metric]) - float(ref.loc[key, metric]))
            else:
                vals.append(np.nan)
        out[f"delta_vs_locked_{metric}"] = vals
    return out


def manufacturer_value_audit(pred: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH, dtype=str)
    meta["SiteCode_from_SubjectID"] = meta["SubjectID"].map(sitecode_from_subject)
    meta["Site3_zfilled"] = meta["Site3"].apply(lambda x: str(int(float(x))).zfill(3) if pd.notna(x) and str(x).strip() not in {"", "nan"} else "")
    meta["Site3_matches_subject_prefix"] = meta["Site3_zfilled"].eq(meta["SiteCode_from_SubjectID"])
    meta["Manufacturer_mapped"] = meta["Manufacturer"].map(normalize_mfr)
    rows = []
    for (raw, mapped), sub in meta.groupby(["Manufacturer", "Manufacturer_mapped"], dropna=False):
        rows.append(
            {
                "source": "training_ready_metadata",
                "raw_manufacturer": raw,
                "mapped_manufacturer": mapped,
                "n_rows": int(len(sub)),
                "n_unique_subjects": int(sub["SubjectID"].nunique()),
            }
        )
    pred_meta = pred[["SubjectID", "Manufacturer", "Manufacturer_mapped"]].drop_duplicates()
    for (raw, mapped), sub in pred_meta.groupby(["Manufacturer", "Manufacturer_mapped"], dropna=False):
        rows.append(
            {
                "source": "stage_b_predictions_subjects",
                "raw_manufacturer": raw,
                "mapped_manufacturer": mapped,
                "n_rows": int(len(sub)),
                "n_unique_subjects": int(sub["SubjectID"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    site_audit = pd.DataFrame(
        [
            {
                "source": "training_ready_metadata_sitecode",
                "raw_manufacturer": "Site3 vs SubjectID prefix",
                "mapped_manufacturer": "matching_nonmissing",
                "n_rows": int(meta["Site3_matches_subject_prefix"].sum()),
                "n_unique_subjects": int(meta.loc[meta["Site3_matches_subject_prefix"], "SubjectID"].nunique()),
            },
            {
                "source": "training_ready_metadata_sitecode",
                "raw_manufacturer": "Site3 vs SubjectID prefix",
                "mapped_manufacturer": "missing_or_mismatch",
                "n_rows": int((~meta["Site3_matches_subject_prefix"]).sum()),
                "n_unique_subjects": int(meta.loc[~meta["Site3_matches_subject_prefix"], "SubjectID"].nunique()),
            },
        ]
    )
    return pd.concat([out, site_audit], ignore_index=True, sort=False)


def update_failed_optimization_table() -> None:
    df = pd.read_csv(FAILED_TABLE)
    row = {
        "candidate": "block_order_norm_act",
        "category": "VAE block-order hygiene",
        "evaluation_stage": "FULL 5x5",
        "decision": "not promoted: norm_act improved threshold-level specificity/F1 but lowered AUC/PR-AUC",
        "auc": NORM_METRICS["auc"],
        "pr_auc": NORM_METRICS["pr_auc"],
        "balanced_accuracy": NORM_METRICS["balanced_accuracy"],
        "sensitivity": NORM_METRICS["sensitivity"],
        "specificity": NORM_METRICS["specificity"],
        "f1": NORM_METRICS["f1"],
        "delta_auc_vs_locked": NORM_METRICS["auc"] - LOCKED_METRICS["auc"],
        "delta_pr_auc_vs_locked": NORM_METRICS["pr_auc"] - LOCKED_METRICS["pr_auc"],
    }
    df = df[df["candidate"].astype(str).ne(row["candidate"])]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False)
    df.to_csv(FAILED_TABLE, index=False)
    FAILED_TABLE_MD.write_text(md_table(df, max_rows=80), encoding="utf-8")

    text = FINAL_RECOMMENDATION.read_text(encoding="utf-8")
    if "block_order=norm_act" not in text:
        insert = (
            "\nThe block-order norm_act FULL 5x5 confirmation changed only `vae_block_order` from the "
            "default/effective `legacy_act_norm` to `norm_act`, keeping the locked objective, beta, dropout, "
            "channels, scheduler, split, and Stage B readout unchanged. It did not improve ranking metrics: "
            "ROC-AUC decreased from `0.778785` to `0.752187`, and PR-AUC decreased from `0.551832` to "
            "`0.492421`. Although BA/F1 were slightly higher at the selected threshold, the pre-specified "
            "promotion rule required improvement in both AUC and PR-AUC. Therefore block_order=norm_act "
            "should not replace the locked model.\n"
        )
        marker = "The manuscript should present the locked model"
        if marker in text:
            text = text.replace(marker, insert + "\n" + marker)
        else:
            text = text.rstrip() + "\n" + insert
        text = text.replace(
            "Do not promote fc0, dropout010, dropout020, beta65, no_decoder_dropout, final_activation=none, manufacturer-balanced VAE sampling, channel dropout, batch_size=32, ultra-regularized logreg readout, or frozen-latent classifier-selection variants.",
            "Do not promote fc0, dropout010, dropout020, beta65, no_decoder_dropout, block_order=norm_act, final_activation=none, manufacturer-balanced VAE sampling, channel dropout, batch_size=32, ultra-regularized logreg readout, or frozen-latent classifier-selection variants.",
        )
        FINAL_RECOMMENDATION.write_text(text, encoding="utf-8")


def final_recommendation(main_df: pd.DataFrame, mfr_df: pd.DataFrame, site_df: pd.DataFrame, leak_df: pd.DataFrame) -> None:
    locked = main_df[main_df["run_id"].eq("locked_current")].iloc[0]
    norm = main_df[main_df["run_id"].eq("block_order_norm_act")].iloc[0]
    auc_ok = float(norm["auc"]) > float(locked["auc"])
    pr_ok = float(norm["pr_auc"]) >= float(locked["pr_auc"])
    ba_worse = float(norm["balanced_accuracy"]) < float(locked["balanced_accuracy"]) - 0.01
    f1_worse = float(norm["f1"]) < float(locked["f1"]) - 0.01
    lines = [
        "# block_order=norm_act FULL 5x5 Postmortem",
        "",
        "## Primary Result",
        "",
        "| model | AUC | PR-AUC | BA | Sens | Spec | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| locked legacy_act_norm | {locked['auc']:.6f} | {locked['pr_auc']:.6f} | {locked['balanced_accuracy']:.6f} | {locked['sensitivity']:.6f} | {locked['specificity']:.6f} | {locked['f1']:.6f} |",
        f"| norm_act | {norm['auc']:.6f} | {norm['pr_auc']:.6f} | {norm['balanced_accuracy']:.6f} | {norm['sensitivity']:.6f} | {norm['specificity']:.6f} | {norm['f1']:.6f} |",
        "",
        "## Interpretation",
        "",
        (
            "`norm_act` improved selected-threshold specificity and F1 modestly, but it failed the "
            "pre-specified promotion gate because threshold-independent ranking worsened: "
            f"AUC delta={norm['delta_vs_locked_auc']:+.6f}, PR-AUC delta={norm['delta_vs_locked_pr_auc']:+.6f}."
        ),
        "",
        "## Promotion Gate",
        "",
        f"- AUC > locked (`{locked['auc']:.6f}`): {'PASS' if auc_ok else 'FAIL'}",
        f"- PR-AUC >= locked (`{locked['pr_auc']:.6f}`): {'PASS' if pr_ok else 'FAIL'}",
        f"- BA not materially worse: {'FAIL' if ba_worse else 'PASS'}",
        f"- F1 not materially worse: {'FAIL' if f1_worse else 'PASS'}",
        "",
        "**Decision: do not promote `block_order=norm_act`. Keep locked current FULL tanh `[1,0,2]` legacy_act_norm model as the manuscript model.**",
        "",
        "## Safety",
        "",
        "- Training launched by this postmortem: `False`.",
        "- Tensor/metadata/ledger modification: `False`.",
        "- Existing model-output modification: `False`.",
        "- Updated manuscript defense failed-optimization table: `True`, as requested.",
    ]
    (OUTDIR / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    main_df = main_comparison()
    write_table("final_full_confirmation_table", main_df)

    vae = pd.concat(
        [
            load_vae_history("locked_current", "locked current FULL legacy_act_norm [1,0,2]", LOCKED_RUN),
            load_vae_history("block_order_norm_act", "block_order=norm_act FULL [1,0,2]", NORM_RUN),
        ],
        ignore_index=True,
        sort=False,
    )
    vae = add_fold_deltas(
        vae,
        ["best_epoch", "early_stop_epoch", "best_val_loss_beta_max", "val_recon_at_best", "val_kld_at_best"],
    )
    write_table("vae_best_epoch_comparison", vae)

    stage_a = pd.concat(
        [
            stage_a_metrics("locked_current", "locked current FULL legacy_act_norm [1,0,2]", LOCKED_RUN),
            stage_a_metrics("block_order_norm_act", "block_order=norm_act FULL [1,0,2]", NORM_RUN),
        ],
        ignore_index=True,
        sort=False,
    )
    stage_a = add_fold_deltas(stage_a, ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"])
    write_table("stage_a_classifier_auc_comparison", stage_a)

    stage_b = pd.concat(
        [
            stage_b_foldwise("locked_current", "locked current FULL legacy_act_norm [1,0,2]", LOCKED_READOUT),
            stage_b_foldwise("block_order_norm_act", "block_order=norm_act FULL [1,0,2]", NORM_READOUT),
        ],
        ignore_index=True,
        sort=False,
    )
    stage_b = add_fold_deltas(stage_b, ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"])
    write_table("stage_b_foldwise_comparison", stage_b)

    pred = pd.concat(
        [
            load_predictions("locked_current", "locked current FULL legacy_act_norm [1,0,2]", LOCKED_READOUT),
            load_predictions("block_order_norm_act", "block_order=norm_act FULL [1,0,2]", NORM_READOUT),
        ],
        ignore_index=True,
        sort=False,
    )
    mfr = subgroup_metrics(pred, "Manufacturer_mapped")
    write_table("manufacturer_subgroup_comparison", mfr, max_rows=80)
    site = subgroup_metrics(pred, "SiteCode")
    write_table("sitecode_subgroup_comparison", site, max_rows=160)
    mfr_audit = manufacturer_value_audit(pred)
    write_table("manufacturer_value_audit", mfr_audit)

    leak = pd.concat(
        [
            scanner_leakage_rows("locked_current", "locked current FULL legacy_act_norm [1,0,2]", LOCKED_RUN),
            scanner_leakage_rows("block_order_norm_act", "block_order=norm_act FULL [1,0,2]", NORM_RUN),
        ],
        ignore_index=True,
        sort=False,
    )
    leak = add_scanner_deltas(leak)
    write_table("scanner_leakage_foldwise_comparison", leak, max_rows=120)

    final_recommendation(main_df, mfr, site, leak)
    update_failed_optimization_table()

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "locked_run": str(LOCKED_RUN),
        "locked_readout": str(LOCKED_READOUT),
        "candidate_run": str(NORM_RUN),
        "candidate_readout": str(NORM_READOUT),
        "output_dir": str(OUTDIR),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_output_modified": False,
        "failed_optimization_table_updated": True,
    }
    write_json(OUTDIR / "command_log.json", command_log)
    print(main_df[["run_id", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    print(f"Wrote postmortem to {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
