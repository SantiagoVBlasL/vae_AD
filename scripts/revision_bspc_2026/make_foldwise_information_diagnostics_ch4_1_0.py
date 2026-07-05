#!/usr/bin/env python3
"""Fold-wise information diagnostics for the ADNI V4 [4,1,0] mu/logvar sweep.

This is read-only with respect to the completed CVAE run. It joins the new
architecture-free classifier sweep with existing fold QC artifacts: VAE
rate-distortion logs, latent information summaries, scanner leakage summaries,
prediction counts, and reconstruction distribution summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_SWEEP_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/latent_representation_sweep_mu_logvar_ch4_1_0"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/foldwise_information_diagnostics_mu_logvar_ch4_1_0"
)
DEFAULT_TANH_AUDIT_README = (
    PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/tanh_saturation_ch4_1_0/README.md"
)
DEFAULT_LAYERNORM_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_layernorm"
)

MAIN_TABLE = "foldwise_information_diagnostics_ch4_1_0.csv"
PEARSON_TABLE = "foldwise_auc_diagnostic_correlations_pearson.csv"
SPEARMAN_TABLE = "foldwise_auc_diagnostic_correlations_spearman.csv"
RECON_TABLE = "reconstruction_distribution_summaries_wide.csv"
README = "README.md"
MANIFEST = "foldwise_information_diagnostics_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tanh-audit-readme", type=Path, default=DEFAULT_TANH_AUDIT_README)
    parser.add_argument("--layernorm-dir", type=Path, default=DEFAULT_LAYERNORM_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [MAIN_TABLE, PEARSON_TABLE, SPEARMAN_TABLE, RECON_TABLE, README, MANIFEST]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated diagnostic outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def numeric(value: Any) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return out or "unknown"


def history_epochs(fold_dir: Path, fold: int) -> Dict[str, Any]:
    path = fold_dir / f"vae_train_history_fold_{fold}.joblib"
    if not path.exists():
        return {}
    try:
        hist = joblib.load(path)
    except Exception:
        return {}
    if not isinstance(hist, dict):
        return {}
    val_key = "val_loss_modelsel" if "val_loss_modelsel" in hist else "val_loss"
    vals = pd.to_numeric(pd.Series(hist.get(val_key, [])), errors="coerce")
    if vals.empty:
        return {}
    best_idx = int(vals.idxmin())
    out = {
        "history_path": str(path),
        "best_epoch_from_history": best_idx + 1,
        "early_stop_epoch_from_history": int(len(vals)),
    }
    for key in ["train_loss", "train_recon", "train_kld", "val_loss", "val_recon", "val_kld", "val_loss_modelsel", "beta"]:
        values = hist.get(key)
        if isinstance(values, list) and best_idx < len(values):
            out[f"{key}_at_best_history"] = numeric(values[best_idx])
    return out


def parse_rate_distortion(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = run_dir.resolve()
    for fold in range(1, 6):
        fold_dir = root / f"fold_{fold}"
        row: Dict[str, Any] = {"fold": fold}
        hist = history_epochs(fold_dir, fold)
        row.update(hist)
        rd = safe_read_csv(fold_dir / f"fold_{fold}_rate_distortion.csv")
        if not rd.empty:
            val_loss = pd.to_numeric(rd.get("L_val_betaMax"), errors="coerce")
            if val_loss.notna().any():
                idx = int(val_loss.idxmin())
            else:
                idx = int(pd.to_numeric(rd.get("D_val"), errors="coerce").idxmin())
            best = rd.loc[idx]
            row.update(
                {
                    "best_epoch": int(numeric(best.get("epoch"))),
                    "early_stop_epoch": int(pd.to_numeric(rd.get("epoch"), errors="coerce").max()),
                    "beta_at_best": numeric(best.get("beta")),
                    "recon_train_at_best": numeric(best.get("D_train")),
                    "recon_val_at_best": numeric(best.get("D_val")),
                    "kl_train_at_best": numeric(best.get("R_train_nats")),
                    "kl_val_at_best": numeric(best.get("R_val_nats")),
                    "loss_train_beta_max_at_best": numeric(best.get("L_train_betaMax")),
                    "loss_val_beta_max_at_best": numeric(best.get("L_val_betaMax")),
                }
            )
        if pd.isna(row.get("best_epoch", np.nan)) and "best_epoch_from_history" in row:
            row["best_epoch"] = row["best_epoch_from_history"]
        if pd.isna(row.get("early_stop_epoch", np.nan)) and "early_stop_epoch_from_history" in row:
            row["early_stop_epoch"] = row["early_stop_epoch_from_history"]
        row["reconstruction_gap_val_minus_train"] = (
            row.get("recon_val_at_best", np.nan) - row.get("recon_train_at_best", np.nan)
            if pd.notna(row.get("recon_val_at_best", np.nan)) and pd.notna(row.get("recon_train_at_best", np.nan))
            else np.nan
        )
        row["kl_gap_val_minus_train"] = (
            row.get("kl_val_at_best", np.nan) - row.get("kl_train_at_best", np.nan)
            if pd.notna(row.get("kl_val_at_best", np.nan)) and pd.notna(row.get("kl_train_at_best", np.nan))
            else np.nan
        )
        row["loss_gap_val_minus_train"] = (
            row.get("loss_val_beta_max_at_best", np.nan) - row.get("loss_train_beta_max_at_best", np.nan)
            if pd.notna(row.get("loss_val_beta_max_at_best", np.nan))
            and pd.notna(row.get("loss_train_beta_max_at_best", np.nan))
            else np.nan
        )
        row["best_epoch_frac_of_training"] = (
            row.get("best_epoch", np.nan) / row.get("early_stop_epoch", np.nan)
            if pd.notna(row.get("best_epoch", np.nan)) and pd.notna(row.get("early_stop_epoch", np.nan)) and row.get("early_stop_epoch")
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_latent_info(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = run_dir.resolve()
    for fold in range(1, 6):
        row: Dict[str, Any] = {"fold": fold}
        for split, token in [("trainDev", "trainDev"), ("test", "test")]:
            df = safe_read_csv(root / f"fold_{fold}/fold_{fold}_{token}_latent_info_summary.csv")
            if df.empty:
                continue
            y = df[df["variable"].astype(str) == "Y_target"]
            manufacturer = df[df["variable"].astype(str) == "Manufacturer"]
            base = y if not y.empty else df
            mi_y = numeric(y["mi_sum_nats"].iloc[0]) if not y.empty else np.nan
            mi_man = numeric(manufacturer["mi_sum_nats"].iloc[0]) if not manufacturer.empty else np.nan
            row[f"MI_Y_{split}"] = mi_y
            row[f"MI_Manufacturer_{split}"] = mi_man
            row[f"MI_Manufacturer_over_Y_{split}"] = mi_man / mi_y if pd.notna(mi_y) and mi_y else np.nan
            row[f"active_units_{split}"] = numeric(base["n_active"].iloc[0]) if "n_active" in base.columns else np.nan
            row[f"frac_active_{split}"] = numeric(base["frac_active"].iloc[0]) if "frac_active" in base.columns else np.nan
            row[f"TC_{split}"] = numeric(base["total_correlation_nats"].iloc[0]) if "total_correlation_nats" in base.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_scanner_leakage(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = run_dir.resolve()
    for fold in range(1, 6):
        row: Dict[str, Any] = {"fold": fold}
        for split, filename in [
            ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            df = safe_read_csv(root / f"fold_{fold}/{filename}")
            if df.empty:
                continue
            first = df.iloc[0]
            row[f"acc_site_raw_{split}"] = numeric(first.get("acc_site_raw"))
            row[f"acc_site_raw_std_{split}"] = numeric(first.get("acc_site_raw_std"))
            row[f"acc_site_latent_{split}"] = numeric(first.get("acc_site_latent"))
            row[f"acc_site_latent_std_{split}"] = numeric(first.get("acc_site_latent_std"))
            row[f"site_chance_level_{split}"] = numeric(first.get("chance_level"))
            row[f"latent_minus_raw_site_acc_{split}"] = (
                row[f"acc_site_latent_{split}"] - row[f"acc_site_raw_{split}"]
                if pd.notna(row.get(f"acc_site_latent_{split}")) and pd.notna(row.get(f"acc_site_raw_{split}"))
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_distribution_summaries(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = run_dir.resolve()
    stat_cols = ["mean", "std", "min", "max", "p05", "p95"]
    for fold in range(1, 6):
        row: Dict[str, Any] = {"fold": fold}
        for kind in ["raw", "norm", "recon"]:
            df = safe_read_csv(root / f"fold_{fold}/fold_{fold}_dist_{kind}.csv")
            if df.empty or "channel" not in df.columns:
                continue
            for _, channel_row in df.iterrows():
                channel = slug(channel_row["channel"])
                for stat in stat_cols:
                    if stat in df.columns:
                        row[f"dist_{kind}_{channel}_{stat}"] = numeric(channel_row.get(stat))
            mins = pd.to_numeric(df.get("min"), errors="coerce")
            maxs = pd.to_numeric(df.get("max"), errors="coerce")
            row[f"dist_{kind}_global_min"] = float(mins.min()) if mins.notna().any() else np.nan
            row[f"dist_{kind}_global_max"] = float(maxs.max()) if maxs.notna().any() else np.nan
        row["recon_any_tanh_boundary_hit"] = (
            (row.get("dist_recon_global_min", np.nan) <= -0.999) or (row.get("dist_recon_global_max", np.nan) >= 0.999)
            if pd.notna(row.get("dist_recon_global_min", np.nan)) and pd.notna(row.get("dist_recon_global_max", np.nan))
            else np.nan
        )
        row["norm_any_outside_tanh_range"] = (
            (row.get("dist_norm_global_min", np.nan) < -1.0) or (row.get("dist_norm_global_max", np.nan) > 1.0)
            if pd.notna(row.get("dist_norm_global_min", np.nan)) and pd.notna(row.get("dist_norm_global_max", np.nan))
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_sweep_tables(sweep_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = sweep_dir / "latent_representation_sweep_metrics.csv"
    fold_path = sweep_dir / "latent_representation_sweep_fold_metrics.csv"
    pred_path = sweep_dir / "latent_representation_sweep_predictions.csv"
    summary = safe_read_csv(summary_path)
    folds = safe_read_csv(fold_path)
    preds = safe_read_csv(pred_path)
    if summary.empty or folds.empty or preds.empty:
        raise FileNotFoundError(f"Missing completed sweep outputs under {sweep_dir}")
    return summary, folds, preds


def ensure_count_aliases(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    out = fold_metrics.copy()
    aliases = {
        "fp": "false_positives",
        "fn": "false_negatives",
        "tp": "true_positives",
        "tn": "true_negatives",
    }
    for src, dst in aliases.items():
        if src in out.columns:
            out[dst] = out[src]
    return out


def merge_fold_diagnostics(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rd = parse_rate_distortion(run_dir)
    latent = parse_latent_info(run_dir)
    scanner = parse_scanner_leakage(run_dir)
    dist = parse_distribution_summaries(run_dir)
    diag = rd.merge(latent, on="fold", how="outer").merge(scanner, on="fold", how="outer").merge(dist, on="fold", how="outer")
    return diag, dist


def build_correlations(main: pd.DataFrame, method: str) -> pd.DataFrame:
    predictors = {
        "MI(Y)": "MI_Y_test",
        "MI(Manufacturer)": "MI_Manufacturer_test",
        "MI(Manufacturer)/MI(Y)": "MI_Manufacturer_over_Y_test",
        "acc_site_latent": "acc_site_latent_test",
        "TC": "TC_test",
        "reconstruction_gap": "reconstruction_gap_val_minus_train",
        "best_epoch": "best_epoch",
    }
    rows: List[Dict[str, Any]] = []
    for (representation, classifier), grp in main.groupby(["representation", "classifier"], sort=False):
        for label, col in predictors.items():
            sub = grp[["roc_auc", col]].apply(pd.to_numeric, errors="coerce").dropna()
            corr = sub["roc_auc"].corr(sub[col], method=method) if len(sub) >= 3 else np.nan
            rows.append(
                {
                    "method": method,
                    "representation": representation,
                    "classifier": classifier,
                    "outcome": "AUC",
                    "predictor": label,
                    "predictor_column": col,
                    "n_folds": int(len(sub)),
                    "r": corr,
                    "abs_r": abs(corr) if pd.notna(corr) else np.nan,
                    "exploratory_low_n": True,
                    "note": "Computed across five outer folds per representation/classifier; diagnostic only.",
                }
            )
    return pd.DataFrame(rows)


def best_row(summary: pd.DataFrame, mask: pd.Series) -> pd.Series:
    sub = summary[mask & (summary["status"] == "ok")].copy()
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.sort_values(["roc_auc_mean_fold", "balanced_accuracy_mean_fold"], ascending=False).iloc[0]


def format_auc(row: pd.Series) -> str:
    if row.empty:
        return "unavailable"
    return f"{row['roc_auc_mean_fold']:.4f} +/- {row['roc_auc_sd_fold']:.4f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *rows])


def count_complete_folds(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for fold in range(1, 6) if (path / f"fold_{fold}").exists())


def read_tanh_note(path: Path, diag: pd.DataFrame) -> Tuple[bool, str]:
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "final_activation=linear" in text and "Recommendation: test" in text:
            return True, "prior tanh saturation audit explicitly recommended a controlled final_activation=linear run"
    boundary = pd.to_numeric(diag.get("recon_any_tanh_boundary_hit"), errors="coerce")
    outside = pd.to_numeric(diag.get("norm_any_outside_tanh_range"), errors="coerce")
    if boundary.fillna(0).astype(bool).any() and outside.fillna(0).astype(bool).any():
        return True, "reconstruction summaries hit tanh boundaries while normalized inputs exceed [-1, 1]"
    return False, "current summary artifacts do not by themselves justify changing final activation"


def write_readme(
    path: Path,
    summary: pd.DataFrame,
    main: pd.DataFrame,
    diag: pd.DataFrame,
    pearson: pd.DataFrame,
    sweep_dir: Path,
    run_dir: Path,
    tanh_audit_readme: Path,
    layernorm_dir: Path,
) -> None:
    ok = summary[summary["status"] == "ok"].copy()
    best = ok.sort_values(["roc_auc_mean_fold", "balanced_accuracy_mean_fold"], ascending=False).iloc[0]
    mu_best = best_row(summary, summary["representation"] == "mu")
    logvar_best = best_row(
        summary,
        summary["representation"].isin(["logvar", "mu_plus_logvar", "mu_plus_entropy_summary", "mu_plus_logvar_PCA"]),
    )
    pure_logvar = best_row(summary, summary["representation"] == "logvar")
    delta = (
        float(logvar_best["roc_auc_mean_fold"]) - float(mu_best["roc_auc_mean_fold"])
        if not mu_best.empty and not logvar_best.empty
        else np.nan
    )
    if pd.notna(delta) and delta >= 0.02:
        logvar_help = "yes, materially"
    elif pd.notna(delta) and delta >= 0.005:
        logvar_help = "yes, modestly"
    else:
        logvar_help = "no material evidence"

    active_mean = pd.to_numeric(diag.get("active_units_test"), errors="coerce").mean()
    frac_active_mean = pd.to_numeric(diag.get("frac_active_test"), errors="coerce").mean()
    tc_mean = pd.to_numeric(diag.get("TC_test"), errors="coerce").mean()
    mi_ratio_mean = pd.to_numeric(diag.get("MI_Manufacturer_over_Y_test"), errors="coerce").mean()
    acc_site_latent_mean = pd.to_numeric(diag.get("acc_site_latent_test"), errors="coerce").mean()
    bottleneck_text = (
        "entangled/over-expressive rather than underpowered"
        if pd.notna(frac_active_mean) and frac_active_mean > 0.8 and pd.notna(mi_ratio_mean) and mi_ratio_mean > 1.0
        else "not clearly classifiable from the available summaries"
    )
    enough_text = (
        "potentially enough for an architecture-free increment"
        if pd.notna(delta) and delta >= 0.02 and float(best["roc_auc_mean_fold"]) >= 0.80
        else "not enough as a final AUC-improvement step"
    )
    linear_ok, linear_reason = read_tanh_note(tanh_audit_readme, diag)
    linear_text = "justified as the next controlled retrain" if linear_ok else "not justified yet"
    layernorm_complete = count_complete_folds(layernorm_dir)
    layernorm_text = (
        "defer LayerNorm; the available LayerNorm output is incomplete and the current signal points first to activation/reconstruction and classifier-use diagnostics"
        if layernorm_complete < 5
        else "defer LayerNorm unless a direct completed comparison beats the current frozen-latent results"
    )

    top_cols = ["representation", "classifier", "roc_auc_mean_fold", "roc_auc_sd_fold", "pr_auc_mean_fold", "balanced_accuracy_mean_fold"]
    top_table = markdown_table(ok[top_cols].head(8))
    corr_top = (
        pearson.sort_values("abs_r", ascending=False)
        .head(8)[["representation", "classifier", "predictor", "n_folds", "r"]]
    )
    corr_top_table = markdown_table(corr_top)

    lines = [
        "# ADNI-Only Mu/Logvar Diagnostic README",
        "",
        "## Scope",
        "This report uses the original saved Sex-stratified outer folds from the completed ADNI V4 [4,1,0] beta=2.5 run. It does not retrain the CVAE, does not modify architecture, does not use OASIS, and does not use Manufacturer as a predictive feature.",
        "",
        f"- Source run: `{run_dir}`",
        f"- Sweep directory: `{sweep_dir}`",
        "- Manufacturer-stratified CVAE results should remain scanner-balanced sensitivity analysis, not the main model.",
        "",
        "## Compact Sweep Summary",
        "",
        top_table,
        "",
        "## Logvar Result",
        f"`logvar`/posterior-uncertainty features show {logvar_help} of improving classification over the best `mu` representation in this sweep.",
        f"- Best `mu`: {format_auc(mu_best)} via `{mu_best.get('classifier', 'NA')}`.",
        f"- Best logvar-related representation: {format_auc(logvar_best)} via `{logvar_best.get('representation', 'NA')}` + `{logvar_best.get('classifier', 'NA')}`.",
        f"- Delta vs best `mu`: {delta:.4f}" if pd.notna(delta) else "- Delta vs best `mu`: unavailable",
        f"- Pure `logvar` best: {format_auc(pure_logvar)}.",
        "",
        "## Bottleneck Interpretation",
        f"The bottleneck appears {bottleneck_text}. Mean test active units={active_mean:.1f}, frac active={frac_active_mean:.3f}, TC={tc_mean:.2f}, MI(Manufacturer)/MI(Y)={mi_ratio_mean:.2f}, and acc_site_latent={acc_site_latent_mean:.3f}.",
        "",
        "## Architecture-Free Step",
        f"The current architecture-free downstream sweep is {enough_text}. Use it as a diagnostic and possibly a small classifier-side improvement, but do not treat it as a replacement for addressing the representation/reconstruction issue unless its held-out AUC gain is robust in the table above.",
        "",
        "## Final Activation",
        f"`final_activation=linear` is {linear_text}: {linear_reason}. This is a next-run recommendation only; no retraining was done here.",
        "",
        "## LayerNorm",
        layernorm_text + ".",
        "",
        "## Strongest AUC Correlations",
        "",
        corr_top_table,
        "",
        "## Caveat",
        "Fold-wise correlations use only five outer folds per representation/classifier. They are diagnostic hypotheses, not inferential evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    sweep_dir = resolve(args.sweep_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)

    summary, fold_metrics, _preds = load_sweep_tables(sweep_dir)
    fold_metrics = ensure_count_aliases(fold_metrics)
    diag, recon_wide = merge_fold_diagnostics(run_dir)
    main = fold_metrics.merge(diag, on="fold", how="left")

    pearson = build_correlations(main, "pearson")
    spearman = build_correlations(main, "spearman")

    main.to_csv(outdir / MAIN_TABLE, index=False)
    pearson.to_csv(outdir / PEARSON_TABLE, index=False)
    spearman.to_csv(outdir / SPEARMAN_TABLE, index=False)
    recon_wide.to_csv(outdir / RECON_TABLE, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()),
        "sweep_dir": str(sweep_dir),
        "sweep_dir_realpath": str(sweep_dir.resolve()),
        "output_dir": str(outdir),
        "output_dir_realpath": str(outdir.resolve()),
        "vae_retrained": False,
        "architecture_modified": False,
        "manufacturer_used_as_predictive_feature": False,
        "n_rows": int(main.shape[0]),
        "n_correlation_rows_pearson": int(pearson.shape[0]),
        "n_correlation_rows_spearman": int(spearman.shape[0]),
    }
    (outdir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(
        outdir / README,
        summary,
        main,
        diag,
        pearson,
        sweep_dir,
        run_dir,
        resolve(args.tanh_audit_readme),
        resolve(args.layernorm_dir),
    )

    top = summary[summary["status"] == "ok"].head(10)
    print("Top sweep rows:")
    print(
        top[
            [
                "representation",
                "classifier",
                "roc_auc_mean_fold",
                "roc_auc_sd_fold",
                "roc_auc_pooled",
                "pr_auc_mean_fold",
                "balanced_accuracy_mean_fold",
            ]
        ].to_string(index=False)
    )
    print(f"Diagnostics written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
