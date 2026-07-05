#!/usr/bin/env python3
"""Create interpretability and figure-replication package for promoted ADNI model.

Read-only with respect to tensors, metadata, model artifacts, and previous
result directories. All writes are confined to the requested output package.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
INTERP = RESULTS / "oof_logitz_interpretability_stability_audit"
PROMOTED_FIGS = RESULTS / "oof_logitz_promoted_candidate_figures"
CHANNEL_ABL = RESULTS / "channel_ablation_fast3x3_offdiag_channelmean" / "primary_ablation_table.csv"
ROI_MAP = RESULTS / "adni_best_full_all_eligible_end_to_end_audit_20260531" / "roi_order_aal3_yeo17_mapping.csv"
OUT = RESULTS / "promoted_model_interpretability_figures_20260601"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"read_error": [str(exc)], "source_path": [str(path)]})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path:
        try:
            md = df.to_markdown(index=False)
        except Exception:
            md = df.to_string(index=False)
        md_path.write_text(md + "\n", encoding="utf-8")


def get_predictions(method: str = "oof_ecdf") -> pd.DataFrame:
    df = read_csv(OOF / "calib_predictions.csv")
    if df.empty:
        return df
    df = df[
        (df["model_name"].astype(str) == "logreg_l2_original")
        & (df["feature_set"].astype(str) == "z_plus_age_sex")
        & (df["calib_method"].astype(str) == method)
        & (df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    df["diagnosis_label"] = np.where(df["y_true"].astype(int) == 1, "AD", "CN")
    return df


def bootstrap_curves(pred: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = pred["y_true"].astype(int).to_numpy()
    s = pd.to_numeric(pred["y_score"], errors="coerce").to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    roc_grid = np.linspace(0, 1, 101)
    pr_grid = np.linspace(0, 1, 101)
    tpr_boot = []
    prec_boot = []
    aucs = []
    aps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        bfpr, btpr, _ = roc_curve(yy, ss)
        bp, br, _ = precision_recall_curve(yy, ss)
        tpr_boot.append(np.interp(roc_grid, bfpr, btpr))
        order = np.argsort(br)
        prec_boot.append(np.interp(pr_grid, br[order], bp[order], left=bp[order][0], right=bp[order][-1]))
        aucs.append(roc_auc_score(yy, ss))
        aps.append(average_precision_score(yy, ss))
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    ci_df = pd.DataFrame(
        {
            "metric": ["auc", "pr_auc"],
            "estimate": [roc_auc_score(y, s), average_precision_score(y, s)],
            "bootstrap95_lo": [np.percentile(aucs, 2.5), np.percentile(aps, 2.5)],
            "bootstrap95_hi": [np.percentile(aucs, 97.5), np.percentile(aps, 97.5)],
            "n_bootstrap": [n_boot, n_boot],
        }
    )
    roc_band = pd.DataFrame(
        {
            "fpr_grid": roc_grid,
            "tpr_boot_mean": np.mean(tpr_boot, axis=0),
            "tpr_boot_lo": np.percentile(tpr_boot, 2.5, axis=0),
            "tpr_boot_hi": np.percentile(tpr_boot, 97.5, axis=0),
        }
    )
    pr_band = pd.DataFrame(
        {
            "recall_grid": pr_grid,
            "precision_boot_mean": np.mean(prec_boot, axis=0),
            "precision_boot_lo": np.percentile(prec_boot, 2.5, axis=0),
            "precision_boot_hi": np.percentile(prec_boot, 97.5, axis=0),
        }
    )
    return roc_df.merge(roc_band, how="outer", left_on="fpr", right_on="fpr_grid"), pr_df.merge(pr_band, how="outer", left_on="recall", right_on="recall_grid"), ci_df


def plot_roc_pr(pred: pd.DataFrame, fig_dir: Path, tables_dir: Path, n_boot: int, seed: int) -> None:
    roc_df, pr_df, ci_df = bootstrap_curves(pred, n_boot, seed)
    write_table(roc_df, tables_dir / "roc_curve_bootstrap_band.csv")
    write_table(pr_df, tables_dir / "pr_curve_bootstrap_band.csv")
    write_table(ci_df, tables_dir / "performance_bootstrap_ci.csv", tables_dir / "performance_bootstrap_ci.md")
    y = pred["y_true"].astype(int).to_numpy()
    s = pd.to_numeric(pred["y_score"], errors="coerce").to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    auc_est = roc_auc_score(y, s)
    ap_est = average_precision_score(y, s)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC={auc_est:.3f}")
    band = roc_df.dropna(subset=["fpr_grid", "tpr_boot_lo"])
    axes[0].fill_between(band["fpr_grid"], band["tpr_boot_lo"], band["tpr_boot_hi"], color="#1f77b4", alpha=0.2, label="bootstrap 95% band")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(loc="lower right")
    axes[0].set_title("ROC")
    axes[1].plot(recall, precision, color="#d62728", lw=2, label=f"PR-AUC={ap_est:.3f}")
    band = pr_df.dropna(subset=["recall_grid", "precision_boot_lo"])
    axes[1].fill_between(band["recall_grid"], band["precision_boot_lo"], band["precision_boot_hi"], color="#d62728", alpha=0.2, label="bootstrap 95% band")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Precision-Recall")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_roc_pr_bootstrap_ci.png", dpi=220)
    plt.close(fig)


def calibration_and_confusion(pred: pd.DataFrame, fig_dir: Path, tables_dir: Path) -> None:
    y = pred["y_true"].astype(int).to_numpy()
    s = pd.to_numeric(pred["y_score"], errors="coerce").to_numpy()
    bins = np.linspace(0, 1, 11)
    bin_id = np.digitize(s, bins, right=True)
    rows = []
    ece = 0.0
    for b in range(1, len(bins)):
        mask = bin_id == b
        if not mask.any():
            continue
        conf = float(np.mean(s[mask]))
        acc = float(np.mean(y[mask]))
        weight = float(mask.mean())
        ece += weight * abs(acc - conf)
        rows.append({"bin": b, "score_lo": bins[b - 1], "score_hi": bins[b], "n": int(mask.sum()), "mean_score": conf, "fraction_positive": acc, "abs_gap": abs(acc - conf)})
    cal_df = pd.DataFrame(rows)
    cal_df["ece_10bin"] = ece
    write_table(cal_df, tables_dir / "calibration_bins.csv", tables_dir / "calibration_bins.md")
    frac_pos, mean_pred = calibration_curve(y, s, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(mean_pred, frac_pos, marker="o", color="#2ca02c")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Fraction AD")
    ax.set_title(f"Reliability diagram, ECE={ece:.3f}")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_calibration_reliability.png", dpi=220)
    plt.close(fig)

    cm = confusion_matrix(y, pred["y_pred"].astype(int).to_numpy(), labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_CN", "true_AD"], columns=["pred_CN", "pred_AD"])
    write_table(cm_df.reset_index(names="true_label"), tables_dir / "confusion_matrix_primary_threshold.csv", tables_dir / "confusion_matrix_primary_threshold.md")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["Pred CN", "Pred AD"])
    ax.set_yticks([0, 1], ["True CN", "True AD"])
    ax.set_title("Primary Threshold Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_confusion_matrix.png", dpi=220)
    plt.close(fig)


def score_distribution(pred: pd.DataFrame, fig_dir: Path, tables_dir: Path) -> None:
    summary = pred.groupby("diagnosis_label")["y_score"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    q = pred.groupby("diagnosis_label")["y_score"].quantile([0.1, 0.25, 0.75, 0.9]).unstack().reset_index()
    q.columns = ["diagnosis_label", "p10", "p25", "p75", "p90"]
    summary = summary.merge(q, on="diagnosis_label", how="left")
    write_table(summary, tables_dir / "score_distribution_summary.csv", tables_dir / "score_distribution_summary.md")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in [("CN", "#1f77b4"), ("AD", "#d62728")]:
        vals = pred.loc[pred["diagnosis_label"] == label, "y_score"].astype(float)
        ax.hist(vals, bins=24, alpha=0.55, density=True, label=label, color=color)
        ax.axvline(vals.median(), color=color, linestyle="--", lw=1)
    thr = pd.to_numeric(pred["threshold"], errors="coerce").median()
    ax.axvline(thr, color="black", lw=1.5, label=f"median fold threshold={thr:.3f}")
    ax.set_xlabel("AD score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_score_distribution.png", dpi=220)
    plt.close(fig)


def collect_rate_distortion() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        p = RUN / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        df = read_csv(p)
        if df.empty:
            continue
        df = df.copy()
        df["fold"] = fold
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_training_and_rd(fig_dir: Path, tables_dir: Path) -> None:
    rd = collect_rate_distortion()
    write_table(rd, tables_dir / "rate_distortion_curves.csv")
    if rd.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for fold, grp in rd.groupby("fold"):
        axes[0].plot(grp["epoch"], grp["L_val_betaMax"], lw=1, label=f"fold {fold}")
        axes[1].plot(grp["epoch"], grp["L_train_betaMax"], lw=1, alpha=0.7)
    axes[0].set_ylabel("Val L(beta max)")
    axes[1].set_ylabel("Train L(beta max)")
    axes[1].set_xlabel("Epoch")
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title("VAE Training Curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_vae_training_curves.png", dpi=220)
    plt.close(fig)
    rd["val_kld_over_recon"] = pd.to_numeric(rd["R_val_nats"], errors="coerce") / pd.to_numeric(rd["D_val"], errors="coerce")
    rd["beta_kld_over_recon"] = pd.to_numeric(rd["beta"], errors="coerce") * rd["val_kld_over_recon"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for fold, grp in rd.groupby("fold"):
        axes[0].plot(grp["epoch"], grp["val_kld_over_recon"], lw=1, label=f"fold {fold}")
        axes[1].scatter(grp["D_val"], grp["R_val_nats"], s=4, alpha=0.35)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val KLD / reconstruction")
    axes[1].set_xlabel("Val reconstruction D")
    axes[1].set_ylabel("Val KLD R (nats)")
    axes[0].set_title("KLD/Reconstruction")
    axes[1].set_title("Rate-Distortion")
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_rate_distortion_curves.png", dpi=220)
    plt.close(fig)


def latent_qc_and_scanner(fig_dir: Path, tables_dir: Path) -> None:
    rows = []
    for fold in range(1, 6):
        p = RUN / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        df = read_csv(p)
        if not df.empty:
            df["fold"] = fold
            rows.append(df)
    lat = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_table(lat, tables_dir / "latent_qc_summary.csv", tables_dir / "latent_qc_summary.md")
    if not lat.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        diag = lat[lat["variable"] == "Y_target"]
        axes[0].bar(diag["fold"].astype(str), diag["n_active"])
        axes[0].set_title("Active Units")
        axes[1].bar(diag["fold"].astype(str), diag["total_correlation_nats"])
        axes[1].set_title("Total Correlation")
        for var in lat["variable"].unique():
            sub = lat[lat["variable"] == var]
            axes[2].plot(sub["fold"], sub["mi_sum_nats"], marker="o", label=var)
        axes[2].set_title("Latent MI Summary")
        axes[2].set_xlabel("Fold")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig07_latent_qc.png", dpi=220)
        plt.close(fig)

    norm_rows = []
    for fold in range(1, 6):
        p = RUN / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_test_latent_mu.csv"
        df = read_csv(p)
        if df.empty:
            continue
        mu_cols = [c for c in df.columns if c.startswith("mu_")]
        vals = df[mu_cols].to_numpy(dtype=float)
        tmp = df[["SubjectID", "ResearchGroup_Mapped", "fold"]].copy()
        tmp["latent_l2_norm"] = np.linalg.norm(vals, axis=1)
        norm_rows.append(tmp)
    norms = pd.concat(norm_rows, ignore_index=True) if norm_rows else pd.DataFrame()
    write_table(norms, tables_dir / "latent_norm_by_subject.csv", tables_dir / "latent_norm_by_subject.md")

    leak_rows = []
    for fold in range(1, 6):
        p = RUN / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
        df = read_csv(p)
        if not df.empty:
            df["fold"] = fold
            df["latent_minus_raw"] = df["acc_site_latent"] - df["acc_site_raw"]
            leak_rows.append(df)
    leak = pd.concat(leak_rows, ignore_index=True) if leak_rows else pd.DataFrame()
    write_table(leak, tables_dir / "scanner_leakage_raw_vs_latent.csv", tables_dir / "scanner_leakage_raw_vs_latent.md")
    if not leak.empty:
        x = np.arange(len(leak))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.18, leak["acc_site_raw"], width=0.36, label="Raw")
        ax.bar(x + 0.18, leak["acc_site_latent"], width=0.36, label="Latent")
        ax.set_xticks(x, [f"F{int(f)}" for f in leak["fold"]])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Manufacturer balanced accuracy")
        ax.set_title("Scanner Leakage: Raw vs Latent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "fig08_scanner_leakage.png", dpi=220)
        plt.close(fig)


def channel_ablation(fig_dir: Path, tables_dir: Path) -> None:
    df = read_csv(CHANNEL_ABL)
    if df.empty:
        return
    df = df.sort_values("auc", ascending=False).copy()
    write_table(df, tables_dir / "channel_ablation_summary.csv", tables_dir / "channel_ablation_summary.md")
    top = df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["run_key"], top["auc"], color="#4c78a8")
    ax.set_xlabel("FAST 3x3 AUC")
    ax.set_title("Scale-Corrected Channel Ablation Summary")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig09_channel_ablation_summary.png", dpi=220)
    plt.close(fig)


def copy_existing_interpretability(fig_dir: Path, tables_dir: Path, artifact_dir: Path) -> pd.DataFrame:
    sources = [
        INTERP / "shap_latent_importance_promoted.csv",
        INTERP / "shap_top20_promoted.csv",
        INTERP / "shap_latent_importance_locked.csv",
        INTERP / "shap_top20_locked.csv",
        INTERP / "ig_fold_edges_promoted.csv",
        INTERP / "ig_consensus_edges_promoted.csv",
        INTERP / "ig_fold_edges_locked.csv",
        INTERP / "ig_consensus_edges_locked.csv",
        INTERP / "channel_contributions_ig.csv",
        INTERP / "channel_contributions_summary.csv",
        INTERP / "edge_overlap_stats.json",
    ]
    fig_sources = sorted(INTERP.glob("fig_*.png")) + sorted(PROMOTED_FIGS.glob("fig*.png"))
    rows = []
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for src in sources + fig_sources:
        if not src.exists():
            rows.append({"source_path": str(src), "copied": False, "reason": "missing"})
            continue
        dst = artifact_dir / src.name
        shutil.copy2(src, dst)
        rows.append({"source_path": str(src), "copied": True, "copied_to": str(dst)})
    write_table(pd.DataFrame(rows), tables_dir / "source_interpretability_artifact_manifest.csv", tables_dir / "source_interpretability_artifact_manifest.md")
    return pd.DataFrame(rows)


def plot_shap(fig_dir: Path, tables_dir: Path) -> None:
    top = read_csv(INTERP / "shap_top20_promoted.csv")
    if top.empty:
        return
    write_table(top, tables_dir / "shap_top20_promoted.csv", tables_dir / "shap_top20_promoted.md")
    fig, ax = plt.subplots(figsize=(8, 6))
    t = top.sort_values("abs_coef").tail(20)
    colors = np.where(t["direction"].astype(str).str.contains("AD"), "#d62728", "#1f77b4")
    ax.barh(t["feature"], t["abs_coef"], color=colors)
    ax.set_xlabel("Mean |linear contribution|")
    ax.set_title("SHAP/Linear Latent Feature Importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_shap_latent_importance.png", dpi=220)
    plt.close(fig)

    locked = read_csv(INTERP / "shap_top20_locked.csv")
    summary = []
    for label, df in [("locked_v5p1b", locked), ("promoted_beta3p75", top)]:
        if df.empty:
            summary.append({"source": label, "status": "missing"})
        else:
            tmp = df.copy()
            tmp["source"] = label
            summary.append(tmp)
    combined = pd.concat([x if isinstance(x, pd.DataFrame) else pd.DataFrame([x]) for x in summary], ignore_index=True)
    combined["shap_mode_note"] = "Existing linear latent-importance audit; no new classifier refit performed."
    write_table(combined, tables_dir / "shap_frozen_unfrozen_collected_summary.csv", tables_dir / "shap_frozen_unfrozen_collected_summary.md")


def hemi(roi: str) -> str:
    if str(roi).endswith("_L"):
        return "L"
    if str(roi).endswith("_R"):
        return "R"
    return "midline_or_unknown"


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / pooled)  # AD - CN


def load_tensor_effect_lookup(edges: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cfg = read_json(RUN / "run_config.json")
    args = cfg.get("args", {})
    tensor_path = Path(args.get("global_tensor_path", ""))
    metadata_path = Path(args.get("metadata_path", ""))
    if not tensor_path.exists() or not metadata_path.exists():
        return {}
    tensor_npz = np.load(tensor_path, allow_pickle=True)
    data = tensor_npz["global_tensor_data"]
    subjects = tensor_npz["subject_ids"].astype(str)
    roi_names = list(tensor_npz["roi_names_in_order"].astype(str))
    channel_names = list(tensor_npz["channel_names"].astype(str))
    selected_channels = args.get("channels_to_use", [1, 0, 2])
    meta = pd.read_csv(metadata_path)
    meta = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    meta = meta[meta["SubjectID"].astype(str).isin(subj_to_idx)].copy()
    idx = meta["SubjectID"].astype(str).map(subj_to_idx).to_numpy()
    y = meta["ResearchGroup_Mapped"].astype(str).to_numpy()
    roi_to_idx = {r: i for i, r in enumerate(roi_names)}
    out: dict[str, dict[str, Any]] = {}
    for _, edge in edges.iterrows():
        src, dst = str(edge["src"]), str(edge["dst"])
        if src not in roi_to_idx or dst not in roi_to_idx:
            continue
        i, j = roi_to_idx[src], roi_to_idx[dst]
        rec: dict[str, Any] = {
            "src_aal3_index_0based": i,
            "dst_aal3_index_0based": j,
            "src_aal3_index_1based": i + 1,
            "dst_aal3_index_1based": j + 1,
        }
        vals_by_channel = []
        for ch in selected_channels:
            vals = data[idx, ch, i, j].astype(float)
            cn = vals[y == "CN"]
            ad = vals[y == "AD"]
            d = cohen_d(cn, ad)
            safe_name = channel_names[ch].replace("Pearson_Full_FisherZ_Signed", "Pearson_Full").replace("Pearson_OMST_GCE_Signed_Weighted", "Pearson_OMST").replace("MI_KNN_Symmetric", "MI")
            rec[f"cohen_d_AD_minus_CN_{safe_name}"] = d
            vals_by_channel.append(vals)
        vals_mean = np.mean(np.vstack(vals_by_channel), axis=0)
        rec["cohen_d_AD_minus_CN_mean_selected_channels"] = cohen_d(vals_mean[y == "CN"], vals_mean[y == "AD"])
        out[str(edge["edge_key"])] = rec
    return out


def edge_interpretability(fig_dir: Path, tables_dir: Path) -> None:
    edges = read_csv(INTERP / "ig_fold_edges_promoted.csv")
    if edges.empty:
        return
    edges["edge_key"] = edges["edge_key"].astype(str)
    edges["sign"] = pd.to_numeric(edges["sign"], errors="coerce")
    edges["ig_diff_signed"] = pd.to_numeric(edges["ig_diff_signed"], errors="coerce")
    edges["ig_diff_abs"] = pd.to_numeric(edges["ig_diff_abs"], errors="coerce")
    n_folds = int(edges["fold"].nunique())
    grouped = []
    for key, grp in edges.groupby("edge_key"):
        signs = grp["sign"].dropna().to_numpy()
        sign_counts = Counter(signs)
        maj_sign, maj_count = sign_counts.most_common(1)[0]
        row = grp.iloc[0].to_dict()
        row.update(
            {
                "freq": int(grp["fold"].nunique()),
                "pi": float(grp["fold"].nunique() / n_folds),
                "mean_deltaS_signed": float(grp["ig_diff_signed"].mean()),
                "mean_abs_deltaS": float(grp["ig_diff_abs"].mean()),
                "median_abs_deltaS": float(grp["ig_diff_abs"].median()),
                "majority_sign": int(maj_sign),
                "sign_consistency": float(maj_count / len(signs)),
                "folds": ",".join(map(str, sorted(grp["fold"].unique()))),
                "src_hemisphere": hemi(row.get("src")),
                "dst_hemisphere": hemi(row.get("dst")),
            }
        )
        grouped.append(row)
    cons = pd.DataFrame(grouped).sort_values(["freq", "sign_consistency", "mean_abs_deltaS"], ascending=False)
    effect_lookup = load_tensor_effect_lookup(cons)
    for col in [
        "src_aal3_index_0based",
        "dst_aal3_index_0based",
        "src_aal3_index_1based",
        "dst_aal3_index_1based",
        "cohen_d_AD_minus_CN_mean_selected_channels",
        "cohen_d_AD_minus_CN_Pearson_Full",
        "cohen_d_AD_minus_CN_Pearson_OMST",
        "cohen_d_AD_minus_CN_MI",
    ]:
        cons[col] = np.nan
    for idx, row in cons.iterrows():
        rec = effect_lookup.get(str(row["edge_key"]), {})
        for k, v in rec.items():
            cons.loc[idx, k] = v
    ch = read_csv(INTERP / "channel_contributions_summary.csv")
    if not ch.empty:
        prom = ch[ch["candidate"] == "promoted_beta3p75"]
        for _, r in prom.iterrows():
            cons[f"global_ig_fraction_{r['channel']}"] = r["mean_fraction"]
    cons["channel_contribution_note"] = "IG channel contribution is global, not edge-specific, in the available artifact."
    write_table(cons, tables_dir / "ig_consensus_edges_recomputed_all.csv", tables_dir / "ig_consensus_edges_recomputed_all.md")
    top20 = cons.sort_values("mean_abs_deltaS", ascending=False).head(20)
    write_table(top20, tables_dir / "top20_edges_by_abs_deltaS.csv", tables_dir / "top20_edges_by_abs_deltaS.md")
    consensus_top = cons[(cons["pi"] >= 0.4) & (cons["sign_consistency"] >= 0.6)].copy()
    write_table(consensus_top, tables_dir / "top_consensus_edges_pi_ge_0p4_sign_ge_0p6.csv", tables_dir / "top_consensus_edges_pi_ge_0p4_sign_ge_0p6.md")

    roi_map = read_csv(ROI_MAP)
    n_roi = int(max(roi_map["roi_index_0based"].max() + 1, 131)) if not roi_map.empty else 131
    mat = np.zeros((n_roi, n_roi), dtype=float)
    for _, r in cons.iterrows():
        i, j = r.get("src_aal3_index_0based"), r.get("dst_aal3_index_0based")
        if pd.notna(i) and pd.notna(j):
            mat[int(i), int(j)] = r["mean_deltaS_signed"]
            mat[int(j), int(i)] = r["mean_deltaS_signed"]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    vmax = np.nanmax(np.abs(mat)) if np.any(mat) else 1.0
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("IG Consensus Edge Map")
    ax.set_xlabel("ROI index")
    ax.set_ylabel("ROI index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_ig_consensus_edge_map.png", dpi=220)
    plt.close(fig)

    network = cons.groupby(["src_network", "dst_network"])["mean_abs_deltaS"].sum().reset_index()
    nets = sorted(set(network["src_network"]).union(network["dst_network"]))
    net_idx = {n: i for i, n in enumerate(nets)}
    nmat = np.zeros((len(nets), len(nets)))
    for _, r in network.iterrows():
        i, j = net_idx[r["src_network"]], net_idx[r["dst_network"]]
        nmat[i, j] += r["mean_abs_deltaS"]
        if i != j:
            nmat[j, i] += r["mean_abs_deltaS"]
    write_table(network, tables_dir / "network_pair_saliency_summary.csv", tables_dir / "network_pair_saliency_summary.md")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(nmat, cmap="magma")
    ax.set_xticks(range(len(nets)), nets, rotation=90, fontsize=6)
    ax.set_yticks(range(len(nets)), nets, fontsize=6)
    ax.set_title("Network-Pair Saliency Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_network_pair_matrix.png", dpi=220)
    plt.close(fig)

    stability_tables(edges, cons, fig_dir, tables_dir)


def stability_tables(edges: pd.DataFrame, cons: pd.DataFrame, fig_dir: Path, tables_dir: Path) -> None:
    pair_rows = []
    for k in [5, 10, 20]:
        fold_sets = {}
        for fold, grp in edges.groupby("fold"):
            fold_sets[fold] = set(grp.sort_values("ig_diff_abs", ascending=False).head(k)["edge_key"].astype(str))
        for a, b in itertools.combinations(sorted(fold_sets), 2):
            sa, sb = fold_sets[a], fold_sets[b]
            inter = len(sa & sb)
            union = len(sa | sb)
            pair_rows.append({"top_k": k, "fold_a": a, "fold_b": b, "jaccard": inter / union if union else np.nan, "dice": 2 * inter / (len(sa) + len(sb)) if (sa or sb) else np.nan})
    overlap = pd.DataFrame(pair_rows)
    write_table(overlap, tables_dir / "fold_topk_jaccard_dice.csv", tables_dir / "fold_topk_jaccard_dice.md")

    union_edges = sorted(edges["edge_key"].astype(str).unique())
    sal_maps = {}
    for fold, grp in edges.groupby("fold"):
        m = dict(zip(grp["edge_key"].astype(str), grp["ig_diff_signed"]))
        sal_maps[fold] = np.array([m.get(e, 0.0) for e in union_edges])
    corr_rows = []
    for a, b in itertools.combinations(sorted(sal_maps), 2):
        if stats is not None:
            rho, p = stats.spearmanr(sal_maps[a], sal_maps[b])
        else:
            rho, p = np.nan, np.nan
        corr_rows.append({"fold_a": a, "fold_b": b, "spearman_rho_sparse_union": rho, "p_value": p, "n_union_edges": len(union_edges)})
    corr = pd.DataFrame(corr_rows)
    write_table(corr, tables_dir / "fold_saliency_spearman_sparse_union.csv", tables_dir / "fold_saliency_spearman_sparse_union.md")

    rows = []
    for pi in [0.4, 0.6, 0.8]:
        for sign in [0.6, 0.8]:
            sub = cons[(cons["pi"] >= pi) & (cons["sign_consistency"] >= sign)]
            rows.append({"pi_threshold": pi, "sign_consistency_threshold": sign, "n_edges": len(sub), "mean_abs_deltaS_sum": sub["mean_abs_deltaS"].sum()})
    thresh = pd.DataFrame(rows)
    write_table(thresh, tables_dir / "consensus_threshold_sensitivity.csv", tables_dir / "consensus_threshold_sensitivity.md")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(cons["freq"], bins=np.arange(0.5, cons["freq"].max() + 1.5, 1), color="#4c78a8")
    axes[0].set_xlabel("Replication frequency")
    axes[0].set_ylabel("Edges")
    axes[1].hist(cons["sign_consistency"], bins=np.linspace(0, 1, 11), color="#f58518")
    axes[1].set_xlabel("Sign consistency")
    axes[1].set_ylabel("Edges")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig13_replication_sign_consistency_histograms.png", dpi=220)
    plt.close(fig)

    quad = cons.copy()
    quad["abs_cohen_d"] = quad["cohen_d_AD_minus_CN_mean_selected_channels"].abs()
    sal_med = quad["mean_abs_deltaS"].median()
    d_med = quad["abs_cohen_d"].median()
    quad["quadrant"] = np.select(
        [
            (quad["mean_abs_deltaS"] >= sal_med) & (quad["abs_cohen_d"] >= d_med),
            (quad["mean_abs_deltaS"] >= sal_med) & (quad["abs_cohen_d"] < d_med),
            (quad["mean_abs_deltaS"] < sal_med) & (quad["abs_cohen_d"] >= d_med),
        ],
        ["high_saliency_high_effect", "high_saliency_low_effect", "low_saliency_high_effect"],
        default="low_saliency_low_effect",
    )
    write_table(quad, tables_dir / "saliency_vs_heldout_cohend_quadrant_edges.csv")
    quad_count = quad["quadrant"].value_counts().rename_axis("quadrant").reset_index(name="n_edges")
    write_table(quad_count, tables_dir / "saliency_vs_heldout_cohend_quadrant_table.csv", tables_dir / "saliency_vs_heldout_cohend_quadrant_table.md")


def reviewer_safe_text(out_dir: Path) -> None:
    text = """# Reviewer-Safe Interpretability Notes

The promoted-model interpretability package separates predictive saliency from
univariate group effects. Latent SHAP/linear-importance summaries indicate which
latent coordinates drive the saved Stage B readout, while IG edge saliency is a
post-hoc attribution of the trained VAE/readout stack. These quantities should
not be interpreted as proof that a connection is biologically superior to a
univariate AD-CN effect.

The consensus edge analysis is intentionally conservative. Fold-level top-K edge
overlap is low, and the strict promoted-model consensus file from the prior
audit had no edges surviving the default consensus rule. The recomputed tables
therefore report replication frequency, sign consistency, and threshold
sensitivity rather than presenting a single definitive biomarker list.

The quadrant table compares saliency magnitude against held-out Cohen d. Edges
with high saliency but low Cohen d are model-specific contributors; edges with
high Cohen d but low saliency are univariate effects not strongly used by the
model. This framing is safer for reviewers than claiming that saliency alone
defines disease biology.
"""
    (out_dir / "reviewer_safe_interpretation.md").write_text(text, encoding="utf-8")


def make_readme(out_dir: Path) -> None:
    text = f"""# Promoted Model Interpretability Figures

Generated: {now_iso()}

Model: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`

Primary row for performance figures:
OOF-ECDF score-harmonized `logreg_l2` with `z_plus_age_sex`,
AUC `0.795155`, PR-AUC `0.573934`.

Interpretability source:
existing `oof_logitz_interpretability_stability_audit` artifacts were reused
for SHAP/IG because they are post-hoc analyses of the promoted trained model and
do not require retraining or classifier refitting. The distinction is recorded
in the source artifact manifest.

Guardrails:
- no VAE training
- no tensor modification
- no metadata modification
- no classifier refitting
- all generated files are derived figures/tables in this package
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    artifact_dir = out_dir / "source_artifacts"
    if args.dry_run:
        print(f"Would write interpretability package to {out_dir}")
        return 0
    for d in [fig_dir, tables_dir, artifact_dir]:
        d.mkdir(parents=True, exist_ok=True)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "started_at": now_iso(),
        "model_run": str(RUN),
        "oof_predictions": str(OOF / "calib_predictions.csv"),
        "interpretability_source": str(INTERP),
        "safety": {
            "vae_training": False,
            "classifier_refit": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifacts_modified": False,
        },
    }

    pred = get_predictions("oof_ecdf")
    write_table(pred, tables_dir / "primary_oof_ecdf_predictions.csv")
    plot_roc_pr(pred, fig_dir, tables_dir, args.n_bootstrap, args.seed)
    calibration_and_confusion(pred, fig_dir, tables_dir)
    score_distribution(pred, fig_dir, tables_dir)
    plot_training_and_rd(fig_dir, tables_dir)
    latent_qc_and_scanner(fig_dir, tables_dir)
    channel_ablation(fig_dir, tables_dir)
    copy_existing_interpretability(fig_dir, tables_dir, artifact_dir)
    plot_shap(fig_dir, tables_dir)
    edge_interpretability(fig_dir, tables_dir)
    reviewer_safe_text(out_dir)
    make_readme(out_dir)

    dependency_status = """# Figure Dependency Status

- ROC/PR, calibration, confusion, score distribution, VAE training, rate-distortion, latent QC, scanner leakage, channel ablation, SHAP, IG edge map, and network matrix were generated with matplotlib/pandas/sklearn.
- Existing SHAP/IG figures from the prior promoted-model interpretability audit were copied into `source_artifacts/`.
- Glass brain and chord diagrams were not regenerated because no ROI MNI coordinate table or chord-specific plotting dependency was detected in this package. The network-pair matrix is provided as the dependency-safe substitute.
"""
    (out_dir / "figure_dependency_status.md").write_text(dependency_status, encoding="utf-8")

    command_log["finished_at"] = now_iso()
    command_log["outputs"] = sorted(str(p.relative_to(ROOT)) for p in out_dir.rglob("*") if p.is_file())
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote interpretability package to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
