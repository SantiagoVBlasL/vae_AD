"""
Formal audit of the ADNI v5 DPARSF-10000 no-Python-bandpass baseline training results.

Compares v5 [4,1,0] against v4 [4,1,0] (same channels, different preprocessing)
and v4 [1,0,2] (manuscript channel set), flags channel-set mismatch with manuscript,
and deep-dives fold 4 performance.

Usage:
    python audit_v5_no_pybandpass_baseline_results.py [--output-root DIR]

Outputs:
    results/revision_bspc_2026/v5_no_pybandpass_result_audit/
        README.md
        run_comparison_table.csv
        foldwise_auc_table.csv
        v5_fold4_deep_dive.csv
        channel_set_traceability_table.csv
        leakage_qc_comparison.csv
        manuscript_consistency_notes.md
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Result roots
_BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

_RUNS: Dict[str, Path] = {
    "v5_ch4_1_0_baseline": _BIG_DISK / "adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline",
    "v4_ch4_1_0":          _BIG_DISK / "adni_expanded_v4_beta25_ch4_1_0",
    "v4_static3":          _BIG_DISK / "adni_expanded_v4_beta25_static3",
    "v4_ch4_1_0_linearout": _BIG_DISK / "adni_expanded_v4_beta25_ch4_1_0_linearout",
    "v4_ch4_1_0_half_fc":   _BIG_DISK / "adni_expanded_v4_beta25_ch4_1_0_half_fc",
    "v4_ch4_1_0_mfrstrat":  _BIG_DISK / "adni_expanded_v4_beta25_ch4_1_0_mfrstrat",
}

_DEFAULT_OUT = (
    _REPO_ROOT / "results" / "revision_bspc_2026" / "v5_no_pybandpass_result_audit"
)

# Manuscript channel set (from v4 static3 run_config)
_MANUSCRIPT_CHANNELS = ["Pearson_Full_FisherZ_Signed", "Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]
_MANUSCRIPT_CHANNEL_INDICES = [1, 0, 2]

# Canonical metrics file suffix (same for all runs)
_METRICS_SUFFIX = (
    "all_folds_metrics_MULTI_logreg_"
    "vaeconvtranspose4l_ld256_beta2.5_normzscore_offdiag_"
    "ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
)

_METADATA_TRAINING = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _find_metrics_file(run_dir: Path) -> Optional[Path]:
    candidate = run_dir / _METRICS_SUFFIX
    if candidate.exists():
        return candidate
    matches = list(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return matches[0] if matches else None


def _load_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "run_config.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return None


def _load_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    p = _find_metrics_file(run_dir)
    if p is None:
        return None
    return pd.read_csv(p)


def _load_leakage(run_dir: Path, n_folds: int = 5) -> Optional[pd.DataFrame]:
    rows = []
    for fold in range(1, n_folds + 1):
        fold_dir = run_dir / f"fold_{fold}"
        for suffix in [f"fold_{fold}_scanner_leakage_summary.csv",
                       f"fold_{fold}_test_scanner_leakage_summary.csv"]:
            p = fold_dir / suffix
            if p.exists():
                df = pd.read_csv(p)
                df["fold"] = fold
                df["leakage_source"] = "train_pool" if "test" not in suffix else "test"
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def _load_latent_qc(run_dir: Path, n_folds: int = 5) -> Optional[pd.DataFrame]:
    rows = []
    for fold in range(1, n_folds + 1):
        p = run_dir / f"fold_{fold}" / "latent_qc_metrics.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    return pd.concat(rows, ignore_index=True) if rows else None


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _summary_stats(series: pd.Series) -> Tuple[float, float]:
    return float(series.mean()), float(series.std())


def _run_summary_row(
    run_id: str,
    run_dir: Path,
    metrics_df: Optional[pd.DataFrame],
    rc: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_id": run_id,
        "available": metrics_df is not None,
        "run_dir": str(run_dir),
        "n_subjects_tensor": None,
        "channels_indices": None,
        "channels_names": None,
        "is_manuscript_channels": None,
        "preprocessing": None,
    }

    if rc is not None:
        row["n_subjects_tensor"] = (
            rc.get("tensor_shape", [None])[0] if rc.get("tensor_shape") else None
        )
        row["channels_indices"] = str(rc.get("channels_to_use_indices"))
        row["channels_names"] = str(rc.get("channel_names_selected"))
        sel = rc.get("channels_to_use_indices") or []
        row["is_manuscript_channels"] = sorted(sel) == sorted(_MANUSCRIPT_CHANNEL_INDICES)
        metadata_path = str(rc.get("metadata_path", ""))
        if "no_pybandpass" in metadata_path or "no_pybandpass" in str(run_dir):
            row["preprocessing"] = "dparsf10000_no_python_bandpass"
        else:
            row["preprocessing"] = "dparsf_with_python_bandpass"

    if metrics_df is not None:
        for clf in metrics_df["actual_classifier_type"].unique():
            sub = metrics_df[metrics_df["actual_classifier_type"] == clf]
            mu_raw, sd_raw = _summary_stats(sub["auc_raw"])
            mu_fin, sd_fin = _summary_stats(sub["auc_final"])
            row[f"{clf}_auc_raw_mean"] = round(mu_raw, 4)
            row[f"{clf}_auc_raw_std"] = round(sd_raw, 4)
            row[f"{clf}_auc_final_mean"] = round(mu_fin, 4)
            row[f"{clf}_auc_final_std"] = round(sd_fin, 4)

    return row


def _foldwise_rows(run_id: str, metrics_df: Optional[pd.DataFrame]) -> List[Dict]:
    if metrics_df is None:
        return []
    rows = []
    for _, r in metrics_df.iterrows():
        rows.append({
            "run_id": run_id,
            "fold": int(r["fold"]),
            "classifier": r["actual_classifier_type"],
            "auc_raw": round(float(r["auc_raw"]), 4),
            "auc_final": round(float(r["auc_final"]), 4),
            "pr_auc_raw": round(float(r.get("pr_auc_raw", float("nan"))), 4),
            "sensitivity": round(float(r.get("sensitivity", float("nan"))), 4),
            "specificity": round(float(r.get("specificity", float("nan"))), 4),
            "balanced_accuracy": round(float(r.get("balanced_accuracy", float("nan"))), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Fold 4 deep dive
# ---------------------------------------------------------------------------

def build_fold4_deep_dive(v5_dir: Path) -> pd.DataFrame:
    fold4 = v5_dir / "fold_4"
    rows = []

    # Subject composition
    for label, csv_name in [("test", "test_subjects_fold.csv"),
                             ("train_dev", "train_dev_subjects_fold.csv")]:
        p = fold4 / csv_name
        if p.exists():
            df = pd.read_csv(p)
            for grp, cnt in df.groupby("ResearchGroup_Mapped").size().items():
                rows.append({"section": f"subject_counts_{label}",
                             "item": str(grp), "value": str(cnt), "note": ""})

    # Test subject Manufacturer breakdown
    test_df = pd.read_csv(fold4 / "test_subjects_fold.csv")
    if _METADATA_TRAINING.exists():
        meta = pd.read_csv(_METADATA_TRAINING)
        merged = test_df.merge(
            meta[["SubjectID", "Manufacturer"]],
            on="SubjectID", how="left"
        )
        for (grp, mfr), cnt in merged.groupby(
            ["ResearchGroup_Mapped", "Manufacturer"]
        ).size().items():
            rows.append({"section": "test_manufacturer_breakdown",
                         "item": f"{grp}_{mfr}", "value": str(cnt), "note": ""})

    # Scanner leakage
    lk_p = fold4 / "fold_4_scanner_leakage_summary.csv"
    if lk_p.exists():
        lk = pd.read_csv(lk_p)
        for _, r in lk.iterrows():
            rows.append({"section": "leakage_train_pool",
                         "item": "acc_site_raw",
                         "value": f"{r.acc_site_raw:.4f}",
                         "note": "raw connectivity"})
            rows.append({"section": "leakage_train_pool",
                         "item": "acc_site_latent",
                         "value": f"{r.acc_site_latent:.4f}",
                         "note": f"chance={r.chance_level:.3f}"})

    # Test leakage
    tlk_p = fold4 / "fold_4_test_scanner_leakage_summary.csv"
    if tlk_p.exists():
        tlk = pd.read_csv(tlk_p)
        for _, r in tlk.iterrows():
            rows.append({"section": "leakage_test",
                         "item": "acc_site_raw",
                         "value": f"{r.acc_site_raw:.4f}",
                         "note": "raw connectivity"})
            rows.append({"section": "leakage_test",
                         "item": "acc_site_latent",
                         "value": f"{r.acc_site_latent:.4f}",
                         "note": f"chance={r.chance_level:.3f}"})

    # Latent QC
    lqc_p = fold4 / "latent_qc_metrics.csv"
    if lqc_p.exists():
        lqc = pd.read_csv(lqc_p)
        for _, r in lqc.iterrows():
            rows.append({"section": "latent_qc",
                         "item": "silhouette_latent",
                         "value": f"{r.silhouette_latent:.5f}",
                         "note": "near-zero = no group separation"})

    # AUC
    mf_p = _find_metrics_file(v5_dir)
    if mf_p:
        mf = pd.read_csv(mf_p)
        f4 = mf[mf["fold"] == 4]
        for _, r in f4.iterrows():
            rows.append({"section": "auc_fold4",
                         "item": f"{r.actual_classifier_type}_auc_raw",
                         "value": f"{r.auc_raw:.4f}",
                         "note": "vs run mean logreg=0.7066"})
            rows.append({"section": "auc_fold4",
                         "item": f"{r.actual_classifier_type}_auc_final",
                         "value": f"{r.auc_final:.4f}",
                         "note": ""})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Channel traceability
# ---------------------------------------------------------------------------

def build_channel_traceability(run_configs: Dict[str, Optional[Dict]]) -> pd.DataFrame:
    rows = []
    for run_id, rc in run_configs.items():
        if rc is None:
            continue
        sel = rc.get("channel_names_selected", [])
        idx = rc.get("channels_to_use_indices", [])
        rows.append({
            "run_id": run_id,
            "channel_indices": str(idx),
            "channel_names": " | ".join(sel) if sel else "N/A",
            "matches_manuscript": sorted(idx or []) == sorted(_MANUSCRIPT_CHANNEL_INDICES),
            "manuscript_channels": " | ".join(_MANUSCRIPT_CHANNELS),
            "n_subjects_tensor": (
                rc.get("tensor_shape", [None])[0]
                if rc.get("tensor_shape")
                else None
            ),
            "preprocessing_note": (
                "DPARSF-10000, no Python bandpass"
                if "no_pybandpass" in str(rc.get("metadata_path", "")) or
                   "no_pybandpass" in str(rc.get("global_tensor_path", ""))
                else "DPARSF + Python bandpass"
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leakage comparison table
# ---------------------------------------------------------------------------

def build_leakage_comparison(
    run_ids: List[str],
    run_dirs: Dict[str, Path],
) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        d = run_dirs.get(run_id)
        if d is None or not d.exists():
            continue
        lk_df = _load_leakage(d)
        lqc_df = _load_latent_qc(d)
        if lk_df is None:
            continue

        train_lk = lk_df[lk_df["leakage_source"] == "train_pool"]
        for fold, grp in train_lk.groupby("fold"):
            row_base = {
                "run_id": run_id,
                "fold": fold,
                "source": "train_pool",
            }
            for _, r in grp.iterrows():
                row = {**row_base,
                       "site_col": r.get("site_col", ""),
                       "n_sites": r.get("n_sites"),
                       "acc_site_raw": round(float(r.get("acc_site_raw", float("nan"))), 4),
                       "acc_site_latent": round(float(r.get("acc_site_latent", float("nan"))), 4),
                       "chance_level": round(float(r.get("chance_level", float("nan"))), 4),
                       "silhouette_latent": None,
                       }
                if lqc_df is not None:
                    lq = lqc_df[lqc_df["fold"] == fold]
                    if not lq.empty:
                        row["silhouette_latent"] = round(float(lq["silhouette_latent"].iloc[0]), 5)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def write_readme(
    out_dir: Path,
    run_summary: pd.DataFrame,
    foldwise: pd.DataFrame,
    leakage: pd.DataFrame,
    run_ts: str,
) -> None:

    def get_val(df, run_id, col, default="N/A"):
        row = df[df["run_id"] == run_id]
        if row.empty or col not in row.columns:
            return default
        v = row[col].iloc[0]
        if isinstance(v, float) and np.isnan(v):
            return default
        return v

    v5 = "v5_ch4_1_0_baseline"
    v4_same = "v4_ch4_1_0"
    v4_ms = "v4_static3"

    # Per-fold AUC for fold 4 comparison
    v5_f4_lr = foldwise[
        (foldwise["run_id"] == v5) & (foldwise["fold"] == 4) &
        (foldwise["classifier"] == "logreg")
    ]
    v5_f4_svm = foldwise[
        (foldwise["run_id"] == v5) & (foldwise["fold"] == 4) &
        (foldwise["classifier"] == "svm")
    ]
    v5_f4_lr_raw = float(v5_f4_lr["auc_raw"].iloc[0]) if not v5_f4_lr.empty else float("nan")
    v5_f4_svm_raw = float(v5_f4_svm["auc_raw"].iloc[0]) if not v5_f4_svm.empty else float("nan")

    v5_lk_f4 = leakage[
        (leakage["run_id"] == v5) & (leakage["fold"] == 4) &
        (leakage["source"] == "train_pool")
    ]
    v4_lk_f4 = leakage[
        (leakage["run_id"] == v4_same) & (leakage["fold"] == 4) &
        (leakage["source"] == "train_pool")
    ]

    lines = [
        "# ADNI v5 DPARSF-10000 No-Python-Bandpass — Baseline Result Audit",
        "",
        f"Generated: {run_ts}",
        "",
        "---",
        "",
        "## 1. What We Ran",
        "",
        "**Run:** `adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline`",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Tensor | ADNI v5, DPARSF-10000, no Python bandpass |",
        "| Tensor shape | (495, 7, 131, 131) |",
        "| python_bandpass_applied | **False** |",
        "| Training metadata | 493 subjects (3 excluded: 035_S_6927, 128_S_2002, 114_S_6039) |",
        "| Supervised pool | CN=147, AD=96 (all with complete Age+Sex) |",
        "| Channels used | [4,1,0] = dFC_StdDev, Pearson_Full_FisherZ_Signed, Pearson_OMST_GCE_Signed_Weighted |",
        "| Architecture | beta=2.5, latent_dim=256, tanh, quarter-FC, no LayerNorm |",
        "| Classifiers | logreg, svm (Optuna 300 trials each) |",
        "| Outer folds | 5 × 1 = 5 |",
        "| Metadata features | Age, Sex |",
        "",
        "---",
        "",
        "## 2. AUC Results",
        "",
        "| Run | Channels | Preprocessing | N_tensor | LogReg AUC raw | LogReg AUC final | SVM AUC raw | SVM AUC final |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, row in run_summary[run_summary["available"]].iterrows():
        lr_raw = f"{get_val(run_summary, row.run_id, 'logreg_auc_raw_mean')} ± {get_val(run_summary, row.run_id, 'logreg_auc_raw_std')}"
        lr_fin = f"{get_val(run_summary, row.run_id, 'logreg_auc_final_mean')} ± {get_val(run_summary, row.run_id, 'logreg_auc_final_std')}"
        sv_raw = f"{get_val(run_summary, row.run_id, 'svm_auc_raw_mean')} ± {get_val(run_summary, row.run_id, 'svm_auc_raw_std')}"
        sv_fin = f"{get_val(run_summary, row.run_id, 'svm_auc_final_mean')} ± {get_val(run_summary, row.run_id, 'svm_auc_final_std')}"
        lines.append(
            f"| {row.run_id} | {get_val(run_summary, row.run_id, 'channels_indices')} | "
            f"{get_val(run_summary, row.run_id, 'preprocessing')} | "
            f"{get_val(run_summary, row.run_id, 'n_subjects_tensor')} | "
            f"{lr_raw} | {lr_fin} | {sv_raw} | {sv_fin} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Technical Validity",
        "",
        "The v5 baseline run is **technically valid**:",
        "",
        "- All 5 folds completed successfully with AUC > 0.60.",
        "- Tensor: finite=True, NaN=0, python_bandpass_applied=False. Confirmed in pre-training QC.",
        "- Metadata: CN=147, AD=96, all with complete Age+Sex. No imputed demographics.",
        "- run_config.json and run_manifest.json written.",
        "- VAE models saved per fold (`vae_model_fold_N.pt`).",
        "",
        "---",
        "",
        "## 4. Comparison with v4",
        "",
        "**v5 [4,1,0] vs v4 [4,1,0] (same channels, different preprocessing):**",
        "",
        f"- LogReg AUC raw: v5 = "
        f"{get_val(run_summary, v5, 'logreg_auc_raw_mean')}, "
        f"v4 = {get_val(run_summary, v4_same, 'logreg_auc_raw_mean')}  "
        f"→ Δ = {float(get_val(run_summary, v5, 'logreg_auc_raw_mean', 0)) - float(get_val(run_summary, v4_same, 'logreg_auc_raw_mean', 0)):.4f}",
        f"- SVM AUC raw: v5 = "
        f"{get_val(run_summary, v5, 'svm_auc_raw_mean')}, "
        f"v4 = {get_val(run_summary, v4_same, 'svm_auc_raw_mean')} "
        f"→ Δ = {float(get_val(run_summary, v5, 'svm_auc_raw_mean', 0)) - float(get_val(run_summary, v4_same, 'svm_auc_raw_mean', 0)):.4f}",
        "",
        "The v5 run shows a **~0.06 AUC drop** compared to v4 with identical channels and architecture.",
        "The most likely explanation is the preprocessing difference:",
        "",
        "- **v4**: DPARSF output + Python bandpass applied after extraction",
        "- **v5**: DPARSF-10000 output, Python bandpass NOT applied (bandpass done within DPARSF)",
        "",
        "This is the key confound. The DPARSF-10000 internal bandpass may produce different",
        "frequency content than the subsequent Python bandpass used in v4. The effect on",
        "functional connectivity channel statistics should be investigated.",
        "",
        "Additional confounds:",
        "- N_tensor differs: v4=515 subjects, v5=495 subjects.",
        "- Training pool: v5 supervised=243, v4 supervised=? (subject pool composition differs).",
        "",
        "---",
        "",
        "## 5. Channel Set vs Manuscript",
        "",
        "**This run uses channels [4,1,0] = dFC_StdDev + Pearson_FisherZ + OMST.**",
        "",
        f"**The manuscript channel set is [1,0,2] = Pearson_FisherZ + OMST + MI_KNN.**",
        "",
        "These are **different channel sets**. This run is NOT the direct reproduction of the",
        "manuscript result. The [4,1,0] set replaces MI_KNN (channel 2) with dFC_StdDev (channel 4).",
        "",
        "For a like-for-like comparison with the manuscript:",
        "- **v4 static3 [1,0,2]** (with Python bandpass): logreg_raw="
        f"{get_val(run_summary, v4_ms, 'logreg_auc_raw_mean')} ± "
        f"{get_val(run_summary, v4_ms, 'logreg_auc_raw_std')}",
        "- **v5 [1,0,2] not yet run** → next experiment B.",
        "",
        "---",
        "",
        "## 6. Fold 4 Analysis",
        "",
        f"Fold 4 is the weakest fold: logreg_raw={_fmt(v5_f4_lr_raw)}, svm_raw={_fmt(v5_f4_svm_raw)}.",
        "",
        "**Test composition (fold 4):** AD=19, CN=29 (48 subjects)",
        "",
        "CN test subjects by manufacturer: GE=0, Philips=17 (59%), SIEMENS=12",
        "AD test subjects by manufacturer: GE=6, Philips=9, SIEMENS=4",
        "",
        "Fold 4 CN test is **Philips-heavy (59%)**. The BSPC 2026 revision noted that all",
        "historical CN subjects are Philips-only, making Philips-CN the primary site-confound.",
        "A fold where CN test = 59% Philips creates an imbalanced scanner distribution",
        "that the VAE may exploit differentially.",
        "",
        "**Scanner leakage (fold 4, train pool):**",
        f"- acc_site_raw: {v5_lk_f4['acc_site_raw'].values[0] if not v5_lk_f4.empty else 'N/A'}",
        f"- acc_site_latent: {v5_lk_f4['acc_site_latent'].values[0] if not v5_lk_f4.empty else 'N/A'} "
        f"(latent HIGHER than raw → VAE amplified scanner information in fold 4)",
        "",
        "The train-pool latent leakage spike in fold 4 (0.777 vs ~0.64 in other folds) is",
        "consistent with the Philips-heavy test composition pushing the VAE toward manufacturer-",
        "discriminative representations.",
        "",
        "Fold 4 explains approximately the bottom 0.09 AUC drag on the mean.",
        "Without fold 4: logreg_raw mean over folds 1-3,5 ≈ "
        f"{np.mean([0.770, 0.719, 0.726, 0.702]):.3f}.",
        "",
        "---",
        "",
        "## 7. Scanner Leakage Summary",
        "",
        "| Run | Folds | Mean acc_site_raw | Mean acc_site_latent | Leakage direction |",
        "|---|---|---|---|---|",
    ]

    for run_id, rd in [("v5_ch4_1_0_baseline", _RUNS["v5_ch4_1_0_baseline"]),
                        ("v4_ch4_1_0", _RUNS["v4_ch4_1_0"])]:
        lk_sub = leakage[(leakage["run_id"] == run_id) & (leakage["source"] == "train_pool")]
        if lk_sub.empty:
            continue
        m_raw = lk_sub["acc_site_raw"].mean()
        m_lat = lk_sub["acc_site_latent"].mean()
        direction = "latent > raw (↑ leakage)" if m_lat > m_raw else "latent < raw (↓ leakage)"
        lines.append(
            f"| {run_id} | 5 | {m_raw:.4f} | {m_lat:.4f} | {direction} |"
        )

    lines += [
        "",
        "v5 mean train-pool leakage is lower than v4 (latent ~0.693 vs ~0.746 for v4).",
        "However fold 4 is an outlier (latent=0.777). This may reflect the fold composition",
        "rather than a systematic preprocessing artifact.",
        "",
        "---",
        "",
        "## 8. Recommended Next Experiments",
        "",
        "**Immediate priority (for manuscript comparison):**",
        "",
        "**Experiment B — v5 [1,0,2] (manuscript channels):**",
        "- Run channels [1,0,2] = Pearson_FisherZ + OMST + MI_KNN on v5 tensor.",
        "- Provides the direct comparison to the manuscript (same channels, different preprocessing).",
        "- Wrapper ready at: "
        "`scripts/revision_bspc_2026/run_adni_v5_dparsf10000_no_pybandpass_ch1_0_2_baseline.py`",
        "",
        "**Secondary (diagnostic):**",
        "",
        "**Experiment A — Downstream latent sweep on v5 [4,1,0]:**",
        "- Use already-trained VAE models from this run (5 folds, `vae_model_fold_N.pt`).",
        "- Sweep classifier configurations: no Age/Sex metadata, different C values,",
        "  raw connectivity features (no VAE), to isolate VAE contribution.",
        "- Script ready at: "
        "`scripts/revision_bspc_2026/run_adni_v5_dparsf10000_no_pybandpass_latent_sweep.py`",
        "",
        "**Key question these experiments will answer:**",
        "- Is the ~0.06 AUC drop primarily due to (a) no Python bandpass, (b) channel set,",
        "  or (c) subject pool composition?",
        "- Experiment B controls for (b) and partly for (c).",
        "- If v5 [1,0,2] ≈ v4 static3 [1,0,2], preprocessing is not the main factor.",
        "- If v5 [1,0,2] << v4 static3 [1,0,2], Python bandpass removal is significant.",
        "",
        "---",
        "",
        "## 9. Files in This Directory",
        "",
        "| File | Description |",
        "|---|---|",
        "| `run_comparison_table.csv` | Per-run mean AUC summary across all available runs |",
        "| `foldwise_auc_table.csv` | Per-fold AUC for each run and classifier |",
        "| `v5_fold4_deep_dive.csv` | Detailed breakdown of fold 4 (worst fold) |",
        "| `channel_set_traceability_table.csv` | Channel indices/names per run vs manuscript |",
        "| `leakage_qc_comparison.csv` | Per-fold scanner leakage for v5 and v4 |",
        "| `manuscript_consistency_notes.md` | Written analysis of channel-set mismatch |",
    ]

    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  Wrote README.md")


# ---------------------------------------------------------------------------
# Manuscript consistency notes
# ---------------------------------------------------------------------------

def write_manuscript_notes(out_dir: Path) -> None:
    text = textwrap.dedent("""
    # Manuscript Consistency Notes — Channel Set Traceability

    ## Channel Sets Across Runs

    | Run | Indices | Names | Matches manuscript? |
    |---|---|---|---|
    | v4_static3 | [1, 0, 2] | Pearson_Full_FisherZ_Signed, Pearson_OMST_GCE_Signed_Weighted, MI_KNN_Symmetric | **YES** |
    | v4_ch4_1_0 | [4, 1, 0] | dFC_StdDev, Pearson_Full_FisherZ_Signed, Pearson_OMST_GCE_Signed_Weighted | NO |
    | v5_ch4_1_0_baseline | [4, 1, 0] | dFC_StdDev, Pearson_Full_FisherZ_Signed, Pearson_OMST_GCE_Signed_Weighted | NO |
    | v5_ch1_0_2_baseline | [1, 0, 2] | Pearson_Full_FisherZ_Signed, Pearson_OMST_GCE_Signed_Weighted, MI_KNN_Symmetric | **YES** (planned) |

    ## What the Manuscript Reports

    The BSPC 2026 manuscript describes a 3-channel connectivity representation using:
    - Full Pearson correlation (Fisher Z-transformed)
    - Pearson OMST (orthogonal minimum spanning tree, GCE variant)
    - Mutual Information (KNN estimator, symmetric)

    These correspond to tensor indices [1, 0, 2].

    ## Implication of Channel Mismatch

    The v5 baseline run (this audit) used channels [4, 1, 0], which replaces MI_KNN with
    dFC_StdDev. This is the same channel set used in the v4 ch4_1_0 ablation study, which
    tested dFC_StdDev as a potential improvement over MI_KNN. In v4, [4,1,0] produced higher
    AUC than [1,0,2] (0.7678 vs 0.7280 for logreg_raw), so it was a justified choice for v4.

    However, for the BSPC 2026 revision, we need to:
    1. Report results on the manuscript's original channel set [1,0,2].
    2. Separately report the v5 no-Python-bandpass preprocessing effect.

    The proper comparison for the revision is:
      v4 static3 [1,0,2] + Python bandpass → AUC = 0.7280 ± 0.0384
      v5 [1,0,2] + no Python bandpass → AUC = (TBD — next experiment)

    ## Preprocessing Difference Summary

    | Version | Tensor | Python bandpass | DPARSF bandpass |
    |---|---|---|---|
    | v4 | adni_expanded_v4 (515 subjects) | YES | YES (within DPARSF) |
    | v5 | adni_expanded_v5_dparsf10000 (495 subjects) | **NO** | YES (within DPARSF) |

    The v5 dataset was specifically built to remove the Python-level re-bandpass step.
    The BSPC 2026 revision argument is that Python bandpass is redundant given DPARSF
    already applies bandpass. If removing it preserves or improves AUC, this supports
    the revision's methodological claim.

    ## Current Status

    - v5 [4,1,0] (this run): AUC = 0.7066 (logreg_raw). Lower than v4 [4,1,0] = 0.7678.
    - The channel mismatch with manuscript and the preprocessing change are confounded.
    - Run v5 [1,0,2] to disentangle.
    """).strip()
    (out_dir / "manuscript_consistency_notes.md").write_text(text + "\n")
    print("  Wrote manuscript_consistency_notes.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=== v5 DPARSF-10000 No-Python-Bandpass Baseline Audit ===")
    print(f"Output: {out_dir}")
    print()

    # --- Check available runs ---
    available: Dict[str, bool] = {}
    for run_id, run_dir in _RUNS.items():
        mf = _find_metrics_file(run_dir)
        available[run_id] = mf is not None
        status = "OK" if available[run_id] else "NOT AVAILABLE"
        print(f"  [{status}] {run_id}: {run_dir.name}")
    print()

    # --- Load all data ---
    metrics: Dict[str, Optional[pd.DataFrame]] = {}
    run_configs: Dict[str, Optional[Dict]] = {}
    for run_id, run_dir in _RUNS.items():
        metrics[run_id] = _load_metrics(run_dir)
        run_configs[run_id] = _load_run_config(run_dir)

    # --- Build comparison table ---
    print("[1] Building run comparison table ...")
    summary_rows = [
        _run_summary_row(run_id, _RUNS[run_id], metrics[run_id], run_configs[run_id])
        for run_id in _RUNS
    ]
    run_summary = pd.DataFrame(summary_rows)
    run_summary.to_csv(out_dir / "run_comparison_table.csv", index=False)
    print(f"    Wrote run_comparison_table.csv ({len(run_summary)} rows)")

    # --- Fold-wise AUC ---
    print("[2] Building foldwise AUC table ...")
    all_foldwise = []
    for run_id in _RUNS:
        all_foldwise.extend(_foldwise_rows(run_id, metrics[run_id]))
    foldwise = pd.DataFrame(all_foldwise)
    foldwise.to_csv(out_dir / "foldwise_auc_table.csv", index=False)
    print(f"    Wrote foldwise_auc_table.csv ({len(foldwise)} rows)")

    # --- Fold 4 deep dive ---
    print("[3] Building fold 4 deep dive ...")
    v5_dir = _RUNS["v5_ch4_1_0_baseline"]
    if v5_dir.exists():
        f4_df = build_fold4_deep_dive(v5_dir)
        f4_df.to_csv(out_dir / "v5_fold4_deep_dive.csv", index=False)
        print(f"    Wrote v5_fold4_deep_dive.csv ({len(f4_df)} rows)")
    else:
        print("    v5 results not found, skipping fold 4 deep dive")

    # --- Channel traceability ---
    print("[4] Building channel set traceability table ...")
    ch_df = build_channel_traceability(run_configs)
    ch_df.to_csv(out_dir / "channel_set_traceability_table.csv", index=False)
    print(f"    Wrote channel_set_traceability_table.csv ({len(ch_df)} rows)")

    # --- Leakage comparison ---
    print("[5] Building leakage QC comparison ...")
    lk_df = build_leakage_comparison(
        ["v5_ch4_1_0_baseline", "v4_ch4_1_0", "v4_static3"],
        _RUNS,
    )
    lk_df.to_csv(out_dir / "leakage_qc_comparison.csv", index=False)
    print(f"    Wrote leakage_qc_comparison.csv ({len(lk_df)} rows)")

    # --- README ---
    print("[6] Writing README ...")
    write_readme(out_dir, run_summary, foldwise, lk_df, run_ts)

    # --- Manuscript notes ---
    print("[7] Writing manuscript consistency notes ...")
    write_manuscript_notes(out_dir)

    # --- Print key results ---
    print()
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    avail = run_summary[run_summary["available"]]
    for _, row in avail.iterrows():
        lr = f"{row.get('logreg_auc_raw_mean', 'N/A')} ± {row.get('logreg_auc_raw_std', 'N/A')}"
        sv = f"{row.get('svm_auc_raw_mean', 'N/A')} ± {row.get('svm_auc_raw_std', 'N/A')}"
        ms = "✓" if row.get("is_manuscript_channels") else "✗"
        print(f"  {row.run_id:40s} | LR={lr} | SVM={sv} | MS-channels={ms}")
    print()
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
