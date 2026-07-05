#!/usr/bin/env python3
"""Create compact paper-style V4 channel comparison tables and figures.

Reads only existing comparison CSV outputs. Does not load tensors,
checkpoints, joblibs, or run artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "results/revision_bspc_2026/v4_channel_fullrun_comparison"
FIGURE_DIR = COMPARISON_DIR / "figures"

MODEL_CSV = COMPARISON_DIR / "model_comparison_v4_channel_fullruns.csv"
SCANNER_CSV = COMPARISON_DIR / "scanner_latent_comparison_v4_channel_fullruns.csv"
LATENT_CSV = COMPARISON_DIR / "latent_info_comparison_v4_channel_fullruns.csv"
RUNTIME_CSV = COMPARISON_DIR / "runtime_comparison_v4_channel_fullruns.csv"

PAPER_TABLE_CSV = COMPARISON_DIR / "paper_table_v4_channel_fullrun_comparison.csv"
PAPER_TABLE_MD = COMPARISON_DIR / "paper_table_v4_channel_fullrun_comparison.md"
INTERPRETATION_MD = COMPARISON_DIR / "interpretation_v4_channel_fullrun_comparison.md"
PERFORMANCE_FIG = FIGURE_DIR / "figure_v4_channel_performance.png"
LEAKAGE_FIG = FIGURE_DIR / "figure_v4_scanner_leakage.png"

RUN_LABELS = {
    "v4_static3_ch1_0_2": "[1,0,2]",
    "v4_ch4_1": "[4,1]",
    "v4_ch4_1_0": "[4,1,0]",
}


def fmt_pm(mean: float, sd: float) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(sd):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {sd:.3f}"


def fmt3(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.3f}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(MODEL_CSV),
        pd.read_csv(SCANNER_CSV),
        pd.read_csv(LATENT_CSV),
        pd.read_csv(RUNTIME_CSV),
    )


def build_paper_table(
    model: pd.DataFrame,
    scanner: pd.DataFrame,
    latent: pd.DataFrame,
) -> pd.DataFrame:
    test_scanner = scanner[scanner["split"] == "test"][
        ["run_name", "acc_site_latent_mean"]
    ].rename(columns={"acc_site_latent_mean": "acc_site_latent test"})
    test_latent = latent[latent["split"] == "test"][
        ["run_name", "mi_manufacturer_over_y_ratio"]
    ].rename(columns={"mi_manufacturer_over_y_ratio": "MI(Manufacturer)/MI(Y)"})

    merged = model.merge(test_scanner, on="run_name", how="left").merge(
        test_latent, on="run_name", how="left"
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "run_name": row["run_name"],
                "channels": RUN_LABELS.get(row["run_name"], row.get("channels_indices", "")),
                "classifier": row["classifier"],
                "AUC mean+/-SD": fmt_pm(row["auc_mean_fold"], row["auc_std_fold"]),
                "PR-AUC mean+/-SD": fmt_pm(row["pr_auc_mean_fold"], row["pr_auc_std_fold"]),
                "balanced accuracy": fmt3(row["balanced_accuracy_mean_fold"]),
                "sensitivity AD": fmt3(row["sensitivity_AD_mean_fold"]),
                "specificity CN": fmt3(row["specificity_CN_mean_fold"]),
                "F1": fmt3(row["f1_mean_fold"]),
                "Brier": fmt3(row["brier_pooled"]),
                "MI(Manufacturer)/MI(Y)": fmt3(row["MI(Manufacturer)/MI(Y)"]),
                "acc_site_latent test": fmt3(row["acc_site_latent test"]),
                "runtime minutes": fmt3(row["runtime_total_min"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame) -> None:
    lines = [
        "# V4 Channel Full-Run Paper Table",
        "",
        table.to_markdown(index=False),
        "",
    ]
    PAPER_TABLE_MD.write_text("\n".join(lines), encoding="utf-8")


def plot_performance(model: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)
    fig.patch.set_facecolor("white")

    plot_df = model.copy()
    plot_df["label"] = plot_df["channels_indices"] + "\n" + plot_df["classifier"].str.upper()
    x = np.arange(len(plot_df))

    axes[0].bar(
        x,
        plot_df["auc_mean_fold"],
        yerr=plot_df["auc_std_fold"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )
    axes[0].set_title("Mean-fold AUC")
    axes[0].set_ylim(0.65, 0.82)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.38
    axes[1].bar(
        x - width / 2,
        plot_df["sensitivity_AD_mean_fold"],
        width=width,
        color="#F58518",
        edgecolor="black",
        linewidth=0.5,
        label="Sensitivity AD",
    )
    axes[1].bar(
        x + width / 2,
        plot_df["specificity_CN_mean_fold"],
        width=width,
        color="#54A24B",
        edgecolor="black",
        linewidth=0.5,
        label="Specificity CN",
    )
    axes[1].set_title("Operating balance at threshold 0.5")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Metric")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(PERFORMANCE_FIG, bbox_inches="tight")
    plt.close(fig)


def plot_scanner_leakage(scanner: pd.DataFrame, latent: pd.DataFrame) -> None:
    scanner_test = scanner[scanner["split"] == "test"].copy()
    latent_test = latent[latent["split"] == "test"].copy()
    merged = scanner_test.merge(
        latent_test[["run_name", "mi_manufacturer_over_y_ratio"]],
        on="run_name",
        how="left",
    )
    merged["channels"] = merged["run_name"].map(RUN_LABELS)
    x = np.arange(len(merged))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), dpi=200)
    fig.patch.set_facecolor("white")

    axes[0].bar(
        x,
        merged["acc_site_latent_mean"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.5,
    )
    if "chance_level_mean" in merged:
        chance = merged["chance_level_mean"].dropna()
        if len(chance):
            axes[0].axhline(float(chance.iloc[0]), color="black", linestyle="--", linewidth=1.0, label="Chance")
            axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Test scanner/manufacturer leakage")
    axes[0].set_ylabel("Latent site accuracy")
    axes[0].set_ylim(0.0, 0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(merged["channels"], rotation=0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        merged["mi_manufacturer_over_y_ratio"],
        color="#B279A2",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Test latent information ratio")
    axes[1].set_ylabel("MI(Manufacturer) / MI(Y)")
    axes[1].set_ylim(0.0, max(2.0, merged["mi_manufacturer_over_y_ratio"].max() * 1.15))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(merged["channels"], rotation=0)
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(LEAKAGE_FIG, bbox_inches="tight")
    plt.close(fig)


def write_interpretation() -> None:
    text = """# V4 Channel Full-Run Interpretation

## Summary
- `[4,1,0]` + LogReg has the highest mean-fold AUC.
- `[4,1]` + SVM has the strongest sensitivity/specificity balance.
- All `dFC_StdDev`-containing variants outperform static3 by AUC.
- `[4,1,0]` reduces MI(Manufacturer)/MI(Y) compared with static3 and `[4,1]`, but `acc_site_latent` remains high, so manufacturer confounding is not resolved.
- Do not claim scanner invariance.

## Current Recommendation
- Recommended current performance candidate: `[4,1,0]`.
- Recommended parsimonious robustness candidate: `[4,1]`.

## Method Note
This summary uses only existing comparison CSV outputs. It does not retrain, load tensors, load checkpoints, or load joblibs.
"""
    INTERPRETATION_MD.write_text(text, encoding="utf-8")


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model, scanner, latent, _runtime = load_inputs()
    paper_table = build_paper_table(model, scanner, latent)
    paper_table.to_csv(PAPER_TABLE_CSV, index=False)
    write_markdown_table(paper_table)
    plot_performance(model)
    plot_scanner_leakage(scanner, latent)
    write_interpretation()
    print(f"Wrote: {PAPER_TABLE_CSV}")
    print(f"Wrote: {PAPER_TABLE_MD}")
    print(f"Wrote: {PERFORMANCE_FIG}")
    print(f"Wrote: {LEAKAGE_FIG}")
    print(f"Wrote: {INTERPRETATION_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
