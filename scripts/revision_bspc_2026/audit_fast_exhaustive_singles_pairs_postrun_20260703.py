#!/usr/bin/env python3
"""Read-only post-run audit for FAST exhaustive single/pair channel screen."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


REPO = Path("/home/diego/proyectos/vae_AD")
RUN_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_20260702"
)
LOG_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/"
    "post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_20260702"
)
MANIFEST = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_exhaustive_singles_pairs_preflight_20260702/planned_candidates_singles_pairs.csv"
)
LOCKED_CONTROL = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_channelmean_loss_ablation_beta_matched_20260702/beta_cal_ch102_beta250"
)
OUT = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_exhaustive_singles_pairs_postrun_audit_20260703"
)

BETA = 2.50
HIGH_BETA_THRESHOLD = 0.95 * BETA
LATENT_DIM = 128
BOOTSTRAP_N = 10_000
RNG_SEED = 20260703

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}

LOG_PATTERNS = {
    "traceback": re.compile(r"Traceback", re.I),
    "runtime_error": re.compile(r"RuntimeError", re.I),
    "cuda_oom": re.compile(r"CUDA out of memory|CUDA OOM", re.I),
    "killed": re.compile(r"\bKilled\b|killed process", re.I),
    "no_space": re.compile(r"No space left|no space", re.I),
    "high_beta_failure": re.compile(r"no eligible high-beta|checkpoint.*beta.*fail", re.I),
}


def parse_channels(raw: object) -> list[int]:
    if isinstance(raw, float) and math.isnan(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [int(x) for x in re.findall(r"\d+", text)]


def channel_label(channels: list[int]) -> str:
    return "[" + ",".join(str(c) for c in channels) + "]"


def safe_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def find_one(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    if not matches:
        matches = sorted(path.resolve().glob(pattern)) if path.exists() else []
    return matches[0] if len(matches) == 1 else None


def load_predictions(candidate_dir: Path) -> pd.DataFrame:
    pred_file = find_one(candidate_dir, "all_folds_clf_predictions_MULTI_*.joblib")
    if pred_file is None:
        csv_file = find_one(candidate_dir, "all_folds_clf_predictions_MULTI_*.csv")
        if csv_file is None:
            raise FileNotFoundError(f"No all_folds predictions found in {candidate_dir}")
        pred = pd.read_csv(csv_file)
    else:
        obj = joblib.load(pred_file)
        if isinstance(obj, list):
            pred = pd.concat(obj, ignore_index=True)
        elif isinstance(obj, pd.DataFrame):
            pred = obj.copy()
        else:
            raise TypeError(f"Unsupported predictions object in {pred_file}: {type(obj)}")
    if "classifier_type" in pred.columns:
        pred = pred[pred["classifier_type"].astype(str).str.lower().eq("logreg")].copy()
    required = {"SubjectID", "y_true", "y_score_final", "y_pred"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{candidate_dir} predictions missing columns {sorted(missing)}")
    pred["SubjectID"] = pred["SubjectID"].astype(str)
    return pred


def load_metrics(candidate_dir: Path) -> pd.DataFrame:
    metrics_file = find_one(candidate_dir, "all_folds_metrics_MULTI_*.csv")
    if metrics_file is None:
        raise FileNotFoundError(f"No all_folds metrics found in {candidate_dir}")
    df = pd.read_csv(metrics_file)
    if "actual_classifier_type" in df.columns:
        df = df[df["actual_classifier_type"].astype(str).str.lower().eq("logreg")].copy()
    return df


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_oof_subjects": int(len(y_true)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "pooled_roc_auc": float(roc_auc_score(y_true, y_score)),
        "pooled_pr_auc": float(average_precision_score(y_true, y_score)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pooled_sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "pooled_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "pooled_f1": float(f1_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_history_value(history: dict, key: str, epoch: int) -> float:
    vals = history.get(key)
    if vals is None:
        return np.nan
    idx = max(int(epoch) - 1, 0)
    if idx >= len(vals):
        return np.nan
    try:
        return float(vals[idx])
    except Exception:
        return np.nan


def load_fold_diagnostics(candidate_id: str, candidate_dir: Path, channels: list[int]) -> list[dict]:
    rows: list[dict] = []
    n_channels = len(channels)
    for fold in (1, 2, 3):
        fold_dir = candidate_dir / f"fold_{fold}"
        ckpt_file = fold_dir / f"vae_checkpoint_selection_summary_fold_{fold}.csv"
        hist_file = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        row: dict[str, object] = {
            "candidate": candidate_id,
            "channels": channel_label(channels),
            "n_channels": n_channels,
            "fold": fold,
            "checkpoint_file_exists": ckpt_file.exists(),
            "history_file_exists": hist_file.exists(),
        }
        if ckpt_file.exists():
            ckpt = pd.read_csv(ckpt_file)
            if not ckpt.empty:
                for col in ckpt.columns:
                    row[col] = ckpt.iloc[0][col]
        selected_epoch = int(row.get("selected_epoch", 0) or 0)
        history = {}
        if hist_file.exists():
            history = joblib.load(hist_file)
        for key in ("train_recon", "val_recon", "train_kld", "val_kld", "val_loss_modelsel"):
            row[f"{key}_selected"] = get_history_value(history, key, selected_epoch) if selected_epoch else np.nan
        selected_beta = float(row.get("selected_epoch_beta", np.nan))
        val_recon = float(row.get("val_recon_selected", np.nan))
        val_kld = float(row.get("val_kld_selected", np.nan))
        row["beta_times_val_kld"] = selected_beta * val_kld if np.isfinite(selected_beta * val_kld) else np.nan
        row["rho_beta_kld_over_recon"] = (
            row["beta_times_val_kld"] / val_recon if np.isfinite(val_recon) and val_recon != 0 else np.nan
        )
        row["recon_per_channel"] = val_recon / n_channels if n_channels else np.nan
        row["kld_per_latent_dim"] = val_kld / LATENT_DIM if np.isfinite(val_kld) else np.nan
        row["selected_beta_guard_pass"] = bool(selected_beta >= HIGH_BETA_THRESHOLD) if np.isfinite(selected_beta) else False
        row["active_units"] = np.nan
        row["active_units_recoverability"] = "not present in latent_qc_metrics.csv or history joblib"
        row["nan_flag"] = any(
            not np.isfinite(float(row.get(k, np.nan)))
            for k in ("val_recon_selected", "val_kld_selected", "val_loss_modelsel_selected")
        )
        rows.append(row)
    return rows


def scan_logs() -> pd.DataFrame:
    rows = []
    for log_file in sorted(LOG_ROOT.glob("*.log")):
        text = log_file.read_text(errors="replace")
        for name, pattern in LOG_PATTERNS.items():
            hits = pattern.findall(text)
            if hits:
                rows.append(
                    {
                        "log_file": str(log_file),
                        "pattern": name,
                        "n_hits": len(hits),
                        "first_context": first_context(text, pattern),
                    }
                )
    return pd.DataFrame(rows)


def first_context(text: str, pattern: re.Pattern, span: int = 120) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    start = max(0, m.start() - span)
    end = min(len(text), m.end() + span)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> dict[str, object]:
    a2 = a[["SubjectID", "y_true", "y_score_final"]].rename(columns={"y_score_final": "score_a"})
    b2 = b[["SubjectID", "y_true", "y_score_final"]].rename(columns={"y_score_final": "score_b"})
    m = a2.merge(b2, on="SubjectID", suffixes=("_a", "_b"))
    if not (m["y_true_a"].to_numpy() == m["y_true_b"].to_numpy()).all():
        raise ValueError(f"Label mismatch in bootstrap {label_a} vs {label_b}")
    y = m["y_true_a"].to_numpy().astype(int)
    score_a = m["score_a"].to_numpy(float)
    score_b = m["score_b"].to_numpy(float)
    base_auc = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    base_pr = average_precision_score(y, score_a) - average_precision_score(y, score_b)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(RNG_SEED)
    auc_d = np.empty(BOOTSTRAP_N)
    pr_d = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        s0 = rng.choice(idx0, size=len(idx0), replace=True)
        s1 = rng.choice(idx1, size=len(idx1), replace=True)
        idx = np.concatenate([s0, s1])
        yy = y[idx]
        auc_d[i] = roc_auc_score(yy, score_a[idx]) - roc_auc_score(yy, score_b[idx])
        pr_d[i] = average_precision_score(yy, score_a[idx]) - average_precision_score(yy, score_b[idx])
    return {
        "comparison": f"{label_a} minus {label_b}",
        "model_a": label_a,
        "model_b": label_b,
        "n_common_subjects": int(len(m)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "delta_auc": float(base_auc),
        "delta_auc_ci_low": float(np.percentile(auc_d, 2.5)),
        "delta_auc_ci_high": float(np.percentile(auc_d, 97.5)),
        "p_delta_auc_gt0": float((auc_d > 0).mean()),
        "delta_pr_auc": float(base_pr),
        "delta_pr_auc_ci_low": float(np.percentile(pr_d, 2.5)),
        "delta_pr_auc_ci_high": float(np.percentile(pr_d, 97.5)),
        "p_delta_pr_auc_gt0": float((pr_d > 0).mean()),
        "n_bootstrap": BOOTSTRAP_N,
        "bootstrap_seed": RNG_SEED,
    }


def save_heatmap(matrix: np.ndarray, title: str, output: Path, fmt: str = ".3f") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(7), [f"ch{i}" for i in range(7)])
    ax.set_yticks(range(7), [f"ch{i}" for i in range(7)])
    ax.set_title(title)
    for i in range(7):
        for j in range(7):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_singles_plot(singles: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = singles["channels"].tolist()
    x = np.arange(len(singles))
    y = singles["mean_auc"].to_numpy(float)
    err = singles["sd_auc"].to_numpy(float)
    ax.bar(x, y, yerr=err, capsize=4, color="#4C78A8")
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("Fold mean ROC-AUC")
    ax.set_ylim(max(0.45, np.nanmin(y - err) - 0.03), min(1.0, np.nanmax(y + err) + 0.03))
    ax.set_title("Exhaustive single-channel FAST screen")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_top10_plot(summary: pd.DataFrame, output: Path) -> None:
    top = summary.sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False).head(10).copy()
    top = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(top))
    ax.barh(y, top["pooled_roc_auc"], color="#59A14F")
    ax.set_yticks(y, top["channels"])
    ax.set_xlabel("Pooled OOF ROC-AUC")
    ax.set_title("Top 10 FAST single/pair candidates")
    ax.set_xlim(max(0.55, top["pooled_roc_auc"].min() - 0.03), min(1.0, top["pooled_roc_auc"].max() + 0.03))
    for yi, val in zip(y, top["pooled_roc_auc"]):
        ax.text(val + 0.002, yi, f"{val:.3f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    command_log = {
        "script": str(Path(__file__).resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(RUN_ROOT),
        "log_root": str(LOG_ROOT),
        "manifest": str(MANIFEST),
        "locked_control": str(LOCKED_CONTROL),
        "guardrails": {
            "read_only_inputs": True,
            "did_train": False,
            "did_refit_classifier": False,
            "did_run_triples": False,
            "did_run_oasis": False,
            "did_modify_manuscript": False,
        },
    }

    manifest = pd.read_csv(MANIFEST)
    manifest["channel_indices"] = manifest["channels"].apply(parse_channels)
    manifest["channels_label"] = manifest["channel_indices"].apply(channel_label)
    manifest["resolved_output_dir"] = manifest["output_dir"].apply(lambda p: str(Path(p).resolve()))

    candidate_rows: list[dict] = []
    pooled_rows: list[dict] = []
    diagnostics_rows: list[dict] = []
    prediction_cache: dict[str, pd.DataFrame] = {}

    for _, row in manifest.iterrows():
        candidate = str(row["run_label"])
        channels = row["channel_indices"]
        candidate_dir = Path(row["output_dir"])
        metrics = load_metrics(candidate_dir)
        preds = load_predictions(candidate_dir)
        prediction_cache[candidate] = preds
        fold_auc = metrics["auc"].to_numpy(float)
        fold_pr = metrics["pr_auc"].to_numpy(float)
        pooled = binary_metrics(
            preds["y_true"].to_numpy(), preds["y_score_final"].to_numpy(), preds["y_pred"].to_numpy()
        )
        duplicate_n = int(preds["SubjectID"].duplicated().sum())
        diag = load_fold_diagnostics(candidate, candidate_dir.resolve(), channels)
        diagnostics_rows.extend(diag)
        candidate_rows.append(
            {
                "candidate": candidate,
                "status": row.get("status", ""),
                "output_dir": str(candidate_dir),
                "resolved_output_dir": str(candidate_dir.resolve()),
                "channels": channel_label(channels),
                "channel_indices": " ".join(map(str, channels)),
                "channel_names": "; ".join(CHANNEL_NAMES[c] for c in channels),
                "cardinality": int(row["cardinality"]),
                "fold_rows": int(len(metrics)),
                "fold1_auc": fold_auc[0] if len(fold_auc) > 0 else np.nan,
                "fold2_auc": fold_auc[1] if len(fold_auc) > 1 else np.nan,
                "fold3_auc": fold_auc[2] if len(fold_auc) > 2 else np.nan,
                "mean_auc": float(np.mean(fold_auc)) if len(fold_auc) else np.nan,
                "sd_auc": float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else np.nan,
                "fold1_pr_auc": fold_pr[0] if len(fold_pr) > 0 else np.nan,
                "fold2_pr_auc": fold_pr[1] if len(fold_pr) > 1 else np.nan,
                "fold3_pr_auc": fold_pr[2] if len(fold_pr) > 2 else np.nan,
                "mean_pr_auc": float(np.mean(fold_pr)) if len(fold_pr) else np.nan,
                "sd_pr_auc": float(np.std(fold_pr, ddof=1)) if len(fold_pr) > 1 else np.nan,
                "mean_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
                "mean_sensitivity": float(metrics["sensitivity"].mean()),
                "mean_specificity": float(metrics["specificity"].mean()),
                "mean_f1": float(metrics["f1_score"].mean()),
                "duplicate_oof_subject_rows": duplicate_n,
                **pooled,
            }
        )
        pooled_rows.append({"candidate": candidate, "channels": channel_label(channels), **pooled})

    control_metrics = load_metrics(LOCKED_CONTROL)
    control_preds = load_predictions(LOCKED_CONTROL)
    prediction_cache["locked_ch102"] = control_preds
    control_pooled = binary_metrics(
        control_preds["y_true"].to_numpy(), control_preds["y_score_final"].to_numpy(), control_preds["y_pred"].to_numpy()
    )
    control_summary = {
        "candidate": "locked_ch102",
        "status": "locked_control",
        "output_dir": str(LOCKED_CONTROL),
        "resolved_output_dir": str(LOCKED_CONTROL.resolve()),
        "channels": "[1,0,2]",
        "channel_indices": "1 0 2",
        "channel_names": "; ".join(CHANNEL_NAMES[c] for c in [1, 0, 2]),
        "cardinality": 3,
        "fold_rows": int(len(control_metrics)),
        "mean_auc": float(control_metrics["auc"].mean()),
        "sd_auc": float(control_metrics["auc"].std(ddof=1)),
        "mean_pr_auc": float(control_metrics["pr_auc"].mean()),
        "sd_pr_auc": float(control_metrics["pr_auc"].std(ddof=1)),
        "mean_balanced_accuracy": float(control_metrics["balanced_accuracy"].mean()),
        "mean_sensitivity": float(control_metrics["sensitivity"].mean()),
        "mean_specificity": float(control_metrics["specificity"].mean()),
        "mean_f1": float(control_metrics["f1_score"].mean()),
        "duplicate_oof_subject_rows": int(control_preds["SubjectID"].duplicated().sum()),
        **control_pooled,
    }

    summary = pd.DataFrame(candidate_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    diag_df = pd.DataFrame(diagnostics_rows)

    # Fold diagnostics to candidate-level diagnostics.
    diag_candidate = (
        diag_df.groupby("candidate", dropna=False)
        .agg(
            selected_epoch_mean=("selected_epoch", "mean"),
            selected_epoch_min=("selected_epoch", "min"),
            selected_epoch_max=("selected_epoch", "max"),
            selected_beta_min=("selected_epoch_beta", "min"),
            selected_beta_mean=("selected_epoch_beta", "mean"),
            val_recon_mean=("val_recon_selected", "mean"),
            val_kld_mean=("val_kld_selected", "mean"),
            rho_mean=("rho_beta_kld_over_recon", "mean"),
            rho_min=("rho_beta_kld_over_recon", "min"),
            high_beta_guard_pass_all=("selected_beta_guard_pass", "all"),
            nan_flags=("nan_flag", "sum"),
            active_units_min=("active_units", "min"),
            active_units_mean=("active_units", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(diag_candidate, on="candidate", how="left")

    singles = summary[summary["cardinality"].eq(1)].sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)
    pairs = summary[summary["cardinality"].eq(2)].sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)
    best_single = singles.iloc[0]
    best_pair = pairs.iloc[0]

    # Add pair improvement columns against constituent singles.
    single_auc = {
        parse_channels(r["channel_indices"])[0]: r["pooled_roc_auc"] for _, r in singles.iterrows()
    }
    single_pr = {
        parse_channels(r["channel_indices"])[0]: r["pooled_pr_auc"] for _, r in singles.iterrows()
    }
    pair_rows = []
    for _, r in pairs.iterrows():
        ch = parse_channels(r["channel_indices"])
        d = r.to_dict()
        d["delta_auc_vs_first_single"] = r["pooled_roc_auc"] - single_auc.get(ch[0], np.nan)
        d["delta_auc_vs_second_single"] = r["pooled_roc_auc"] - single_auc.get(ch[1], np.nan)
        d["delta_auc_vs_best_constituent_single"] = r["pooled_roc_auc"] - max(
            single_auc.get(ch[0], np.nan), single_auc.get(ch[1], np.nan)
        )
        d["delta_pr_auc_vs_first_single"] = r["pooled_pr_auc"] - single_pr.get(ch[0], np.nan)
        d["delta_pr_auc_vs_second_single"] = r["pooled_pr_auc"] - single_pr.get(ch[1], np.nan)
        d["delta_pr_auc_vs_best_constituent_single"] = r["pooled_pr_auc"] - max(
            single_pr.get(ch[0], np.nan), single_pr.get(ch[1], np.nan)
        )
        pair_rows.append(d)
    pairs = pd.DataFrame(pair_rows).sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)

    log_issues = scan_logs()
    candidate_has_log_issue = {}
    if not log_issues.empty:
        for cand in summary["candidate"]:
            candidate_has_log_issue[cand] = log_issues["log_file"].str.contains(cand, regex=False).any()
    else:
        candidate_has_log_issue = {cand: False for cand in summary["candidate"]}

    locked_pr = control_summary["pooled_pr_auc"]
    locked_auc = control_summary["pooled_roc_auc"]
    gates = []
    for _, r in summary.iterrows():
        active_min = r.get("active_units_min", np.nan)
        gates.append(
            {
                "candidate": r["candidate"],
                "channels": r["channels"],
                "cardinality": r["cardinality"],
                "pooled_roc_auc": r["pooled_roc_auc"],
                "pooled_pr_auc": r["pooled_pr_auc"],
                "fold_auc_sd": r["sd_auc"],
                "gate_fold_auc_sd_le_0p07": bool(r["sd_auc"] <= 0.07),
                "active_units_min": active_min,
                "gate_active_units_ge_50": (
                    bool(active_min >= 50) if np.isfinite(active_min) else "UNKNOWN_not_recoverable"
                ),
                "gate_pr_auc_within_0p02_of_locked": bool(r["pooled_pr_auc"] >= locked_pr - 0.02),
                "gate_high_beta_checkpoint_100pct": bool(r["high_beta_guard_pass_all"]),
                "gate_no_training_pathology": bool((r["nan_flags"] == 0) and (not candidate_has_log_issue.get(r["candidate"], False))),
                "gate_auc_superiority_ge_0p015_vs_locked": bool(r["pooled_roc_auc"] >= locked_auc + 0.015),
            }
        )
    gates_df = pd.DataFrame(gates)

    bootstrap_rows = [
        paired_bootstrap(
            prediction_cache[str(best_pair["candidate"])],
            prediction_cache[str(best_single["candidate"])],
            str(best_pair["candidate"]),
            str(best_single["candidate"]),
        ),
        paired_bootstrap(
            prediction_cache[str(best_pair["candidate"])],
            prediction_cache["locked_ch102"],
            str(best_pair["candidate"]),
            "locked_ch102",
        ),
        paired_bootstrap(
            prediction_cache[str(best_single["candidate"])],
            prediction_cache["locked_ch102"],
            str(best_single["candidate"]),
            "locked_ch102",
        ),
    ]
    boot_df = pd.DataFrame(bootstrap_rows)

    # Pair matrices; diagonal holds single-channel values for context.
    auc_mat = np.full((7, 7), np.nan)
    pr_mat = np.full((7, 7), np.nan)
    for _, r in singles.iterrows():
        ch = parse_channels(r["channel_indices"])[0]
        auc_mat[ch, ch] = r["pooled_roc_auc"]
        pr_mat[ch, ch] = r["pooled_pr_auc"]
    for _, r in pairs.iterrows():
        ch = parse_channels(r["channel_indices"])
        auc_mat[ch[0], ch[1]] = auc_mat[ch[1], ch[0]] = r["pooled_roc_auc"]
        pr_mat[ch[0], ch[1]] = pr_mat[ch[1], ch[0]] = r["pooled_pr_auc"]

    # Integrity checks.
    candidate_sets = manifest["channels_label"].tolist()
    unexpected = []
    if RUN_ROOT.exists():
        dirs = [p.name for p in RUN_ROOT.iterdir() if p.is_dir() or p.is_symlink()]
        unexpected = sorted(set(dirs) - set(manifest["run_label"].astype(str)))
    missing_candidates = [
        str(r["run_label"]) for _, r in manifest.iterrows() if not Path(r["output_dir"]).exists()
    ]
    duplicate_sets = pd.Series(candidate_sets)[pd.Series(candidate_sets).duplicated()].tolist()
    fold_row_fail = summary[~summary["fold_rows"].eq(3)][["candidate", "fold_rows"]]
    duplicate_oof = summary[summary["duplicate_oof_subject_rows"].gt(0)][["candidate", "duplicate_oof_subject_rows"]]
    beta_fail = diag_df[~diag_df["selected_beta_guard_pass"]][
        ["candidate", "fold", "selected_epoch_beta", "selected_epoch"]
    ]

    summary_out = summary.sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)
    summary_out.to_csv(OUT / "candidate_summary_all.csv", index=False)
    (OUT / "candidate_summary_all.md").write_text(safe_markdown(summary_out), encoding="utf-8")

    singles_out = singles.sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)
    singles_out.to_csv(OUT / "singles_ranking.csv", index=False)
    (OUT / "singles_ranking.md").write_text(safe_markdown(singles_out), encoding="utf-8")

    pairs_out = pairs.sort_values(["pooled_roc_auc", "pooled_pr_auc"], ascending=False)
    pairs_out.to_csv(OUT / "pairs_ranking.csv", index=False)
    (OUT / "pairs_ranking.md").write_text(safe_markdown(pairs_out), encoding="utf-8")

    pooled_df.to_csv(OUT / "pooled_oof_metrics.csv", index=False)
    diag_df.to_csv(OUT / "vae_diagnostics_by_fold.csv", index=False)
    diag_candidate.to_csv(OUT / "vae_diagnostics_by_candidate.csv", index=False)
    boot_df.to_csv(OUT / "paired_bootstrap_comparisons.csv", index=False)
    (OUT / "paired_bootstrap_comparisons.md").write_text(safe_markdown(boot_df), encoding="utf-8")
    gates_df.to_csv(OUT / "eligibility_gates.csv", index=False)
    (OUT / "eligibility_gates.md").write_text(safe_markdown(gates_df), encoding="utf-8")

    save_heatmap(auc_mat, "Pair ROC-AUC heatmap (diagonal = singles)", OUT / "pair_auc_heatmap.pdf")
    save_heatmap(pr_mat, "Pair PR-AUC heatmap (diagonal = singles)", OUT / "pair_pr_auc_heatmap.pdf")
    save_singles_plot(singles_out.sort_values("channel_indices"), OUT / "singles_performance.pdf")
    save_top10_plot(summary_out, OUT / "top10_candidates.pdf")

    # Recommendation.
    within_001 = pairs_out[pairs_out["pooled_roc_auc"] >= best_pair["pooled_roc_auc"] - 0.01].head(3)
    best_pair_vs_locked = boot_df[boot_df["comparison"].eq(f"{best_pair['candidate']} minus locked_ch102")].iloc[0]
    best_pair_vs_single = boot_df[
        boot_df["comparison"].eq(f"{best_pair['candidate']} minus {best_single['candidate']}")
    ].iloc[0]
    eligible_pair_names = gates_df[
        gates_df["candidate"].isin(pairs_out["candidate"])
        & gates_df["gate_fold_auc_sd_le_0p07"].eq(True)
        & gates_df["gate_pr_auc_within_0p02_of_locked"].eq(True)
        & gates_df["gate_high_beta_checkpoint_100pct"].eq(True)
        & gates_df["gate_no_training_pathology"].eq(True)
    ]["candidate"].tolist()
    if (
        best_pair["pooled_roc_auc"] < max(best_single["pooled_roc_auc"], locked_auc) + 0.005
        or best_pair_vs_locked["delta_auc_ci_low"] <= 0
        or best_pair_vs_single["delta_auc_ci_low"] <= 0
    ):
        phase_choice = "D. Stop channel expansion for now"
        phase_reason = (
            "The best pair did not clear uncertainty-aware superiority over both the best single and locked control."
        )
    elif len(eligible_pair_names) > 1:
        phase_choice = "B. Expand eligible pairs within 0.01 pooled ROC-AUC of the best pair, capped at 3"
        phase_reason = "Multiple stable pairs remain close enough to justify a bounded Phase 1C diagnostic."
    else:
        phase_choice = "A. Expand only the best pair"
        phase_reason = "Only the leading pair has a plausible performance margin after gates."

    phase_md = f"""# Phase 1C recommendation

Recommendation: **{phase_choice}**.

Reason: {phase_reason}

Best single: `{best_single['candidate']}` {best_single['channels']} pooled ROC-AUC={best_single['pooled_roc_auc']:.4f}, PR-AUC={best_single['pooled_pr_auc']:.4f}.

Best pair: `{best_pair['candidate']}` {best_pair['channels']} pooled ROC-AUC={best_pair['pooled_roc_auc']:.4f}, PR-AUC={best_pair['pooled_pr_auc']:.4f}.

Locked control `[1,0,2]`: pooled ROC-AUC={locked_auc:.4f}, PR-AUC={locked_pr:.4f}.

Pairs within 0.01 pooled ROC-AUC of best pair, capped at 3:

{safe_markdown(within_001[['candidate','channels','pooled_roc_auc','pooled_pr_auc','sd_auc','rho_mean']])}

Bootstrap comparisons are paired and stratified at subject level with {BOOTSTRAP_N:,} resamples. Positive point estimates alone are not treated as proof of improvement.

No triples were launched by this audit.
"""
    (OUT / "phase1c_recommendation.md").write_text(phase_md, encoding="utf-8")

    integrity_md = f"""# Post-run integrity report

Run root: `{RUN_ROOT}`

Log root: `{LOG_ROOT}`

Manifest: `{MANIFEST}`

Locked control: `{LOCKED_CONTROL}`

## Candidate accounting

- Manifest candidates: {len(manifest)}
- Unique channel sets: {len(set(candidate_sets))}
- Singles: {int((manifest['cardinality'] == 1).sum())}
- Pairs: {int((manifest['cardinality'] == 2).sum())}
- Missing candidate directories: {len(missing_candidates)}
- Duplicate channel sets: {len(duplicate_sets)}
- Unexpected directories under run root: {len(unexpected)}

Missing candidates: {missing_candidates if missing_candidates else 'none'}

Duplicate sets: {duplicate_sets if duplicate_sets else 'none'}

Unexpected directories: {unexpected if unexpected else 'none'}

## Fold and OOF checks

Candidates with fold-row count not equal to 3:

{safe_markdown(fold_row_fail) if not fold_row_fail.empty else 'None.'}

Candidates with duplicated OOF subject rows:

{safe_markdown(duplicate_oof) if not duplicate_oof.empty else 'None.'}

## High-beta checkpoint guard

Required selected beta >= {HIGH_BETA_THRESHOLD:.3f}.

Failed selected-beta guards:

{safe_markdown(beta_fail) if not beta_fail.empty else 'None.'}

## Log scan

Problem patterns scanned: {', '.join(LOG_PATTERNS)}

{safe_markdown(log_issues) if not log_issues.empty else 'No Traceback, RuntimeError, CUDA OOM, killed-process, no-space, or high-beta failure patterns were found in launch logs.'}

## Active units

Active latent-unit counts were requested if recoverable read-only. The saved `latent_qc_metrics.csv` files in this FAST run do not include active-unit counts, and the saved history joblibs contain losses/KLD/beta but not fold latent matrices. Therefore the active-unit gate is reported as `UNKNOWN_not_recoverable` rather than inferred from checkpoints.

## Guardrails

- did_train=false
- did_refit_classifier=false
- did_run_triples=false
- did_run_beta_search=false
- did_run_oasis=false
- did_modify_inputs=false
"""
    (OUT / "postrun_integrity_report.md").write_text(integrity_md, encoding="utf-8")

    command_log["outputs"] = sorted(p.name for p in OUT.iterdir())
    command_log["summary"] = {
        "manifest_candidates": int(len(manifest)),
        "unique_channel_sets": int(len(set(candidate_sets))),
        "singles": int((manifest["cardinality"] == 1).sum()),
        "pairs": int((manifest["cardinality"] == 2).sum()),
        "missing_candidates": missing_candidates,
        "duplicate_sets": duplicate_sets,
        "unexpected_dirs": unexpected,
        "best_single": str(best_single["candidate"]),
        "best_pair": str(best_pair["candidate"]),
        "phase1c_recommendation": phase_choice,
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    print(json.dumps(command_log["summary"], indent=2))


if __name__ == "__main__":
    main()
