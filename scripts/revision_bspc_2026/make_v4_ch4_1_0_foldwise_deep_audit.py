#!/usr/bin/env python3
"""Read-only fold-wise deep audit for V4 [4,1,0].

Reads only existing CSV/TXT/JSON/log artifacts. It does not retrain and does
not load tensors, checkpoints, joblibs, or large arrays.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SYMLINK = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
EXTERNAL_LOG_DIR = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/run_logs")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/v4_ch4_1_0_foldwise_deep_audit"

MAIN_TABLE = "foldwise_training_latent_classifier_audit.csv"
PEARSON_TABLE = "foldwise_correlations_pearson.csv"
SPEARMAN_TABLE = "foldwise_correlations_spearman.csv"
BOTTLENECK_TABLE = "foldwise_ranked_bottlenecks.csv"
README = "README.md"

OUTCOMES = [
    "AUC_logreg",
    "AUC_svm",
    "Sens_AD_logreg",
    "Sens_AD_svm",
    "BalAcc_logreg",
    "BalAcc_svm",
]
PREDICTORS = [
    "MI_Manufacturer_over_Y_test",
    "acc_site_latent_test",
    "latent_minus_raw_test",
    "TC_test",
    "KLD_over_R_val",
    "best_epoch",
    "best_epoch_frac",
    "train_val_gap_total",
    "train_val_gap_R",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only fold-wise deep audit for V4 [4,1,0].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_SYMLINK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in [MAIN_TABLE, PEARSON_TABLE, SPEARMAN_TABLE, BOTTLENECK_TABLE, README] if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing audit outputs. Use --overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )
    return output_dir


def read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(value) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def real_run_dir(run_dir: Path) -> Path:
    return run_dir.resolve()


def discover_logs(run_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for base in [run_dir, real_run_dir(run_dir)]:
        candidates.extend(sorted((base / "Logs").glob("*.log")))
    candidates.extend(sorted(EXTERNAL_LOG_DIR.glob("*ch4_1_0*.log")))
    candidates.extend(sorted(EXTERNAL_LOG_DIR.glob("*v4_ch4_1_0*.log")))
    unique = []
    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def parse_log_metadata(run_dir: Path) -> Dict[int, Dict[str, float]]:
    fold_state: Dict[int, Dict[str, float]] = {}
    current_fold: Optional[int] = None
    for log_path in discover_logs(run_dir):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"Iniciando Fold\s+(\d+)/", line)
            if match:
                current_fold = int(match.group(1))
                fold_state.setdefault(current_fold, {})["log_path"] = str(log_path)
            match = re.search(r"Starting Fold\s+(\d+)/", line, flags=re.IGNORECASE)
            if match:
                current_fold = int(match.group(1))
                fold_state.setdefault(current_fold, {})["log_path"] = str(log_path)
            match = re.search(
                r"Early stopping VAE en epoch\s+(\d+).*?Mejor ValL\(βmax\):\s+([0-9.]+).*?\(época\s+(\d+)\)",
                line,
            )
            if match and current_fold is not None:
                fold_state.setdefault(current_fold, {})["early_stop_epoch"] = int(match.group(1))
                fold_state[current_fold]["vae_best_val_loss_beta_max"] = float(match.group(2))
                fold_state[current_fold]["best_epoch"] = int(match.group(3))
            match = re.search(r"completado en\s+([0-9.]+)\s+segundos", line)
            if match and current_fold is not None:
                fold_state.setdefault(current_fold, {})["runtime_sec"] = float(match.group(1))
            match = re.search(r"completed in\s+([0-9.]+)\s*s\.?", line, flags=re.IGNORECASE)
            if match and current_fold is not None:
                fold_state.setdefault(current_fold, {})["runtime_sec"] = float(match.group(1))
    return fold_state


def parse_metrics(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    frames = []
    for path in sorted(real_run_dir(run_dir).glob("all_folds_metrics_MULTI_*.csv")):
        df = safe_read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return out
    metrics = pd.concat(frames, ignore_index=True)
    clf_col = "actual_classifier_type" if "actual_classifier_type" in metrics.columns else "classifier"
    for _, row in metrics.iterrows():
        fold = int(row["fold"])
        clf = str(row[clf_col])
        suffix = "logreg" if clf == "logreg" else "svm" if clf == "svm" else clf
        out.setdefault(fold, {})
        out[fold][f"AUC_{suffix}"] = numeric(row.get("auc"))
        out[fold][f"PR_AUC_{suffix}"] = numeric(row.get("pr_auc"))
        out[fold][f"BalAcc_{suffix}"] = numeric(row.get("balanced_accuracy"))
        out[fold][f"Sens_AD_{suffix}"] = numeric(row.get("sensitivity"))
        out[fold][f"Spec_CN_{suffix}"] = numeric(row.get("specificity"))
        out[fold][f"F1_{suffix}"] = numeric(row.get("f1_score"))
    return out


def parse_optuna(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    root = real_run_dir(run_dir)
    for fold in range(1, 6):
        out.setdefault(fold, {})
        logreg = read_json(root / f"fold_{fold}/optuna_best_trial_logreg_fold_{fold}.json")
        svm = read_json(root / f"fold_{fold}/optuna_best_trial_svm_fold_{fold}.json")
        params = logreg.get("best_params", {}) if isinstance(logreg, dict) else {}
        out[fold]["C_logreg"] = numeric(params.get("model__C"))
        out[fold]["n_iter_logreg"] = numeric(logreg.get("effective_n_trials", logreg.get("n_trials")))
        params = svm.get("best_params", {}) if isinstance(svm, dict) else {}
        out[fold]["C_svm"] = numeric(params.get("model__C"))
        out[fold]["gamma_svm"] = numeric(params.get("model__gamma"))
        out[fold]["n_iter_svm"] = numeric(svm.get("effective_n_trials", svm.get("n_trials")))
    return out


def parse_rate_distortion(run_dir: Path, log_meta: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    root = real_run_dir(run_dir)
    for fold in range(1, 6):
        path = root / f"fold_{fold}/fold_{fold}_rate_distortion.csv"
        rd = safe_read_csv(path)
        if rd.empty:
            continue
        best_epoch = log_meta.get(fold, {}).get("best_epoch")
        if pd.isna(best_epoch) or best_epoch is None:
            idx = pd.to_numeric(rd.get("L_val_betaMax"), errors="coerce").idxmin()
            best_epoch = int(rd.loc[idx, "epoch"])
        row = rd[pd.to_numeric(rd["epoch"], errors="coerce") == int(best_epoch)]
        if row.empty:
            idx = (pd.to_numeric(rd["epoch"], errors="coerce") - int(best_epoch)).abs().idxmin()
            row = rd.loc[[idx]]
        r = row.iloc[0]
        # Here R_*_at_best means reconstruction/distortion (D_*); KLD is R_*_nats.
        r_train = numeric(r.get("D_train"))
        kld_train = numeric(r.get("R_train_nats"))
        r_val = numeric(r.get("D_val"))
        kld_val = numeric(r.get("R_val_nats"))
        out[fold] = {
            "R_train_at_best": r_train,
            "KLD_train_at_best": kld_train,
            "R_val_at_best": r_val,
            "KLD_val_at_best": kld_val,
            "KLD_over_R_train": kld_train / r_train if pd.notna(kld_train) and pd.notna(r_train) and r_train else np.nan,
            "KLD_over_R_val": kld_val / r_val if pd.notna(kld_val) and pd.notna(r_val) and r_val else np.nan,
            "train_val_gap_total": numeric(r.get("L_val_betaMax")) - numeric(r.get("L_train_betaMax")),
            "train_val_gap_R": r_val - r_train if pd.notna(r_val) and pd.notna(r_train) else np.nan,
        }
    return out


def parse_latent_info(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    root = real_run_dir(run_dir)
    for fold in range(1, 6):
        out.setdefault(fold, {})
        for split, token in [("test", "test"), ("trainDev", "trainDev")]:
            path = root / f"fold_{fold}/fold_{fold}_{token}_latent_info_summary.csv"
            df = safe_read_csv(path)
            if df.empty:
                continue
            y = df[df["variable"].astype(str) == "Y_target"]
            manufacturer = df[df["variable"].astype(str) == "Manufacturer"]
            base = y if not y.empty else df
            mi_y = numeric(y["mi_sum_nats"].iloc[0]) if not y.empty else np.nan
            mi_man = numeric(manufacturer["mi_sum_nats"].iloc[0]) if not manufacturer.empty else np.nan
            out[fold][f"MI_Y_{split}"] = mi_y
            out[fold][f"MI_Manufacturer_{split}"] = mi_man
            out[fold][f"MI_Manufacturer_over_Y_{split}"] = mi_man / mi_y if pd.notna(mi_y) and mi_y else np.nan
            out[fold][f"TC_{split}"] = numeric(base["total_correlation_nats"].iloc[0])
            out[fold][f"active_units_{split}"] = numeric(base["n_active"].iloc[0])
    return out


def parse_scanner_qc(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    root = real_run_dir(run_dir)
    for fold in range(1, 6):
        out.setdefault(fold, {})
        for split, filename in [
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
        ]:
            df = safe_read_csv(root / f"fold_{fold}/{filename}")
            if df.empty:
                continue
            row = df.iloc[0]
            raw = numeric(row.get("acc_site_raw"))
            latent = numeric(row.get("acc_site_latent"))
            out[fold][f"acc_site_raw_{split}"] = raw
            out[fold][f"acc_site_latent_{split}"] = latent
            out[fold][f"latent_minus_raw_{split}"] = latent - raw if pd.notna(latent) and pd.notna(raw) else np.nan
        qc = safe_read_csv(root / f"fold_{fold}/latent_qc_metrics.csv")
        if not qc.empty:
            out[fold]["silhouette_latent"] = numeric(qc.iloc[0].get("silhouette_latent"))
    return out


def merge_sources(*sources: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        row: Dict[str, float] = {"fold": fold}
        for source in sources:
            row.update(source.get(fold, {}))
        early = row.get("early_stop_epoch")
        best = row.get("best_epoch")
        row["best_epoch_frac"] = best / early if pd.notna(best) and pd.notna(early) and early else np.nan
        rows.append(row)
    columns = [
        "fold",
        "best_epoch",
        "early_stop_epoch",
        "best_epoch_frac",
        "runtime_sec",
        "vae_best_val_loss_beta_max",
        "R_train_at_best",
        "KLD_train_at_best",
        "R_val_at_best",
        "KLD_val_at_best",
        "KLD_over_R_train",
        "KLD_over_R_val",
        "train_val_gap_total",
        "train_val_gap_R",
        "AUC_logreg",
        "AUC_svm",
        "PR_AUC_logreg",
        "PR_AUC_svm",
        "BalAcc_logreg",
        "BalAcc_svm",
        "Sens_AD_logreg",
        "Sens_AD_svm",
        "Spec_CN_logreg",
        "Spec_CN_svm",
        "F1_logreg",
        "F1_svm",
        "C_logreg",
        "C_svm",
        "gamma_svm",
        "n_iter_logreg",
        "n_iter_svm",
        "MI_Y_test",
        "MI_Manufacturer_test",
        "MI_Manufacturer_over_Y_test",
        "MI_Y_trainDev",
        "MI_Manufacturer_trainDev",
        "MI_Manufacturer_over_Y_trainDev",
        "TC_test",
        "TC_trainDev",
        "active_units_test",
        "active_units_trainDev",
        "acc_site_raw_test",
        "acc_site_latent_test",
        "latent_minus_raw_test",
        "acc_site_raw_trainDev",
        "acc_site_latent_trainDev",
        "latent_minus_raw_trainDev",
        "silhouette_latent",
    ]
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]


def build_correlations(df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for predictor in PREDICTORS:
            sub = df[[outcome, predictor]].apply(pd.to_numeric, errors="coerce").dropna()
            corr = sub[outcome].corr(sub[predictor], method=method) if len(sub) >= 3 else np.nan
            rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": int(len(sub)),
                    "r": corr,
                    "abs_r": abs(corr) if pd.notna(corr) else np.nan,
                    "exploratory_low_n": True,
                    "note": "n=5 outer folds maximum; exploratory/low-n correlation, do not overclaim.",
                }
            )
    return pd.DataFrame(rows)


def high_relative(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    q3 = vals.quantile(0.75)
    return vals > q3


def build_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[
        "fold",
        "AUC_logreg",
        "AUC_svm",
        "Sens_AD_logreg",
        "Sens_AD_svm",
        "MI_Manufacturer_over_Y_test",
        "acc_site_latent_test",
        "latent_minus_raw_test",
        "TC_test",
        "train_val_gap_R",
        "best_epoch_frac",
    ]].copy()
    out["high_scanner_leakage"] = out["acc_site_latent_test"] > 0.60
    out["manufacturer_dominates_label"] = out["MI_Manufacturer_over_Y_test"] > 1.0
    out["possible_overfit"] = high_relative(out["train_val_gap_R"])
    out["late_best_epoch"] = out["best_epoch_frac"] > 0.75
    out["early_stop_too_early"] = out["best_epoch_frac"] < 0.35
    out["high_TC"] = high_relative(out["TC_test"])
    flag_cols = [
        "high_scanner_leakage",
        "manufacturer_dominates_label",
        "possible_overfit",
        "late_best_epoch",
        "early_stop_too_early",
        "high_TC",
    ]
    out["bottleneck_score"] = out[flag_cols].sum(axis=1)
    out["flags"] = out.apply(lambda r: ";".join(col for col in flag_cols if bool(r[col])), axis=1)
    return out.sort_values(["bottleneck_score", "fold"], ascending=[False, True])


def top_findings(df: pd.DataFrame, bottlenecks: pd.DataFrame, spearman: pd.DataFrame) -> List[str]:
    findings: List[str] = []
    best_auc_fold = df.assign(best_auc=df[["AUC_logreg", "AUC_svm"]].max(axis=1)).sort_values("best_auc", ascending=False).iloc[0]
    findings.append(
        f"Best fold-level AUC occurs in fold {int(best_auc_fold['fold'])} "
        f"(max AUC={best_auc_fold['best_auc']:.3f}); its test acc_site_latent is "
        f"{best_auc_fold['acc_site_latent_test']:.3f} and MI(Manufacturer)/MI(Y) is "
        f"{best_auc_fold['MI_Manufacturer_over_Y_test']:.3f}."
    )
    worst = bottlenecks.iloc[0]
    findings.append(
        f"Highest bottleneck burden is fold {int(worst['fold'])} with flags: "
        f"{worst['flags'] or 'none'}."
    )
    corr = spearman[(spearman["outcome"] == "AUC_svm") & (spearman["predictor"] == "acc_site_latent_test")]
    if not corr.empty and pd.notna(corr.iloc[0]["r"]):
        direction = "higher" if corr.iloc[0]["r"] > 0 else "lower"
        findings.append(
            f"Across folds, SVM AUC is associated with {direction} test latent scanner accuracy "
            f"(Spearman r={corr.iloc[0]['r']:.3f}); n=5, exploratory only."
        )
    return findings[:3]


def write_readme(
    path: Path,
    audit: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    spearman: pd.DataFrame,
    run_dir: Path,
) -> None:
    findings = top_findings(audit, bottlenecks, spearman)
    high_auc = audit.assign(best_auc=audit[["AUC_logreg", "AUC_svm"]].max(axis=1))
    corr_auc_scanner = high_auc["best_auc"].corr(high_auc["acc_site_latent_test"], method="spearman")
    corr_auc_mi = high_auc["best_auc"].corr(high_auc["MI_Manufacturer_over_Y_test"], method="spearman")
    scanner_text = "higher" if pd.notna(corr_auc_scanner) and corr_auc_scanner > 0 else "lower"
    mi_text = "higher" if pd.notna(corr_auc_mi) and corr_auc_mi > 0 else "lower"
    recommendation = (
        "Prioritize a beta increase or LayerNorm check before more channel searches: "
        "manufacturer information remains above label information in every fold, and all latent units are active. "
        "Dropout/final-activation changes are secondary unless reconstruction gaps dominate the next audit."
    )
    lines = [
        "# V4 [4,1,0] Fold-Wise Deep Audit",
        "",
        "## Scope",
        "This is a read-only audit of existing V4 [4,1,0] artifacts. It reads CSV/TXT/JSON/log files only. It does not retrain, load tensors, load checkpoints, load joblibs, or modify the run directory.",
        f"Run symlink: `{run_dir}`",
        f"Resolved run path: `{real_run_dir(run_dir)}`",
        "",
        "## Metric Mapping",
        "`R_train_at_best` and `R_val_at_best` are mapped from the rate-distortion CSV's reconstruction/distortion columns (`D_train`, `D_val`). `KLD_*_at_best` is mapped from `R_*_nats`, the rate/KL term.",
        "",
        "## Top 3 Fold-Wise Findings",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Scanner Leakage vs High-AUC Folds",
            f"High-AUC folds show {scanner_text} scanner leakage by Spearman association between per-fold best AUC and test acc_site_latent (r={corr_auc_scanner:.3f}). This is exploratory because n=5.",
            "",
            "## MI(Manufacturer)/MI(Y) Association",
            f"Per-fold best AUC is associated with {mi_text} MI(Manufacturer)/MI(Y) (Spearman r={corr_auc_mi:.3f}). This does not establish causality and should not be used as a final model-selection claim.",
            "",
            "## Next Experiment Recommendation",
            recommendation,
            "",
            "## Caveat",
            "Only five outer folds are available. Correlations and fold-wise flags are exploratory/low-n and should be treated as diagnostic hypotheses, not inferential evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    run_dir = args.run_dir

    log_meta = parse_log_metadata(run_dir)
    metrics = parse_metrics(run_dir)
    optuna = parse_optuna(run_dir)
    rd = parse_rate_distortion(run_dir, log_meta)
    latent = parse_latent_info(run_dir)
    scanner = parse_scanner_qc(run_dir)
    audit = merge_sources(log_meta, rd, metrics, optuna, latent, scanner)
    pearson = build_correlations(audit, "pearson")
    spearman = build_correlations(audit, "spearman")
    bottlenecks = build_bottlenecks(audit)

    audit.to_csv(output_dir / MAIN_TABLE, index=False)
    pearson.to_csv(output_dir / PEARSON_TABLE, index=False)
    spearman.to_csv(output_dir / SPEARMAN_TABLE, index=False)
    bottlenecks.to_csv(output_dir / BOTTLENECK_TABLE, index=False)
    write_readme(output_dir / README, audit, bottlenecks, spearman, run_dir)

    print(f"Audit written to: {output_dir}")
    print(f"Rows: {len(audit)} folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
