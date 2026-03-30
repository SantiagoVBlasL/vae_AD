"""
aggregate_scanner_leakage.py
----------------------------
Aggregate manufacturer-leakage (scanner decodability) results across 5 CV folds
for the BSPC 2026 revision.

Two leakage contexts are aggregated separately:
  - train_cv : CV-internal estimate on the training partition (all 5 folds)
  - test_held : held-out test-set estimate (available for folds 3, 4, 5 only)

Representations: connectome_norm, latent_mu
Target label: Manufacturer (3 classes: Philips, SIEMENS, GE MEDICAL SYSTEMS)
Metric: balanced accuracy (already computed per fold, loaded here)
Chance level: 0.333...

Outputs:
  results/revision_bspc_2026/scanner_leakage/
    scanner_leakage_aggregate.csv
    scanner_leakage_summary.md
    barplot_scanner_leakage.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLD_ROOT = PROJECT_ROOT / "results" / "vae_3channels_beta65_pro"
OUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "scanner_leakage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CHANCE = 1 / 3
REPRESENTATIONS = ["connectome_norm", "latent_mu"]


# ---------------------------------------------------------------------------
# Load raw fold records
# ---------------------------------------------------------------------------
def _load_leakage_csv(path: Path, context: str, fold_num: int) -> pd.DataFrame:
    """Read one leakage CSV and tag it with context and fold_num."""
    df = pd.read_csv(path)
    df["context"] = context
    df["fold_num"] = fold_num
    return df


def load_all_leakage() -> pd.DataFrame:
    """Collect all available leakage records from fold directories."""
    records = []
    for k in range(1, N_FOLDS + 1):
        fold_dir = FOLD_ROOT / f"fold_{k}"

        # --- train CV leakage (present in all 5 folds) ---
        train_path = fold_dir / f"fold_{k}_scanner_leakage.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Expected train leakage file: {train_path}")
        records.append(_load_leakage_csv(train_path, "train_cv", k))

        # --- held-out test leakage (only folds 3, 4, 5) ---
        test_path = fold_dir / f"fold_{k}_test_scanner_leakage.csv"
        if test_path.exists():
            records.append(_load_leakage_csv(test_path, "test_held", k))

    df = pd.concat(records, ignore_index=True)
    # Normalise representation names from fold tags
    df["representation"] = df["representation"].str.strip()
    return df


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one aggregate row per (representation, context):
      mean_bacc, std_bacc, n_folds, chance, fold-level values as list
    """
    rows = []
    for rep in REPRESENTATIONS:
        for ctx in ["train_cv", "test_held"]:
            sub = df[(df["representation"] == rep) & (df["context"] == ctx)]
            if sub.empty:
                continue
            fold_vals = sub["balanced_accuracy_mean"].values
            n_samples_total = sub["n_samples"].sum()
            row = {
                "representation": rep,
                "context": ctx,
                "n_folds": len(sub),
                "n_samples_total": int(n_samples_total),
                "chance_level": CHANCE,
                "mean_bacc": fold_vals.mean(),
                "std_bacc": fold_vals.std(ddof=1) if len(fold_vals) > 1 else np.nan,
                "min_bacc": fold_vals.min(),
                "max_bacc": fold_vals.max(),
                "fold_values": ";".join(f"{v:.4f}" for v in fold_vals),
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_barplot(agg: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: representation × context, with chance reference."""
    contexts = ["train_cv", "test_held"]
    ctx_labels = {"train_cv": "Train CV (n=5 folds)", "test_held": "Held-out test (n=3 folds)"}
    rep_labels = {"connectome_norm": "Connectome (raw)", "latent_mu": "Latent μ"}
    rep_colors = {"connectome_norm": "#4C72B0", "latent_mu": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    bar_width = 0.35
    x = np.arange(len(REPRESENTATIONS))

    for ax_idx, ctx in enumerate(contexts):
        ax = axes[ax_idx]
        sub = agg[agg["context"] == ctx]
        for i, rep in enumerate(REPRESENTATIONS):
            row = sub[sub["representation"] == rep]
            if row.empty:
                ax.bar(i, 0, bar_width, color=rep_colors[rep], alpha=0.5,
                       label=rep_labels[rep])
                continue
            val = row["mean_bacc"].values[0]
            err = row["std_bacc"].values[0]
            ax.bar(i, val, bar_width,
                   color=rep_colors[rep], alpha=0.85,
                   label=rep_labels[rep],
                   yerr=np.nan_to_num(err, nan=0.0),
                   capsize=5, error_kw={"elinewidth": 1.5})
            ax.text(i, val + (err if not np.isnan(err) else 0) + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Chance ({CHANCE:.2f})")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
                   label="Balanced chance (0.50)")
        ax.set_title(ctx_labels[ctx], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([rep_labels[r] for r in REPRESENTATIONS], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Balanced accuracy" if ax_idx == 0 else "")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Scanner (Manufacturer) Decodability — 3-class balanced accuracy\n"
        "Philips / SIEMENS / GE MEDICAL SYSTEMS",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def make_summary_md(agg: pd.DataFrame, raw: pd.DataFrame) -> str:
    lines = [
        "# Scanner Leakage Aggregate Summary — BSPC 2026 Revision",
        "",
        f"*Chance level: {CHANCE:.4f} (3 classes: Philips, SIEMENS, GE)*",
        "",
        "---",
        "",
        "## Aggregate balanced accuracy by representation and context",
        "",
        "| Representation | Context | n_folds | Mean ± SD | Min | Max | vs Chance |",
        "|----------------|---------|---------|-----------|-----|-----|-----------|",
    ]
    for _, row in agg.iterrows():
        sd_str = f"{row['std_bacc']:.3f}" if not np.isnan(row["std_bacc"]) else "—"
        delta = row["mean_bacc"] - CHANCE
        lines.append(
            f"| {row['representation']} | {row['context']} | {row['n_folds']} "
            f"| {row['mean_bacc']:.3f} ± {sd_str} "
            f"| {row['min_bacc']:.3f} | {row['max_bacc']:.3f} "
            f"| {delta:+.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    # Pull train_cv and test_held values for the narrative
    def _get(rep, ctx, field):
        r = agg[(agg["representation"] == rep) & (agg["context"] == ctx)]
        return r[field].values[0] if not r.empty else np.nan

    cn_train = _get("connectome_norm", "train_cv", "mean_bacc")
    lm_train = _get("latent_mu", "train_cv", "mean_bacc")
    cn_test = _get("connectome_norm", "test_held", "mean_bacc")
    lm_test = _get("latent_mu", "test_held", "mean_bacc")

    def _fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "N/A"

    lines += [
        f"**Train-CV leakage** (within training partition, 5 folds):",
        f"- connectome_norm: {_fmt(cn_train)} vs chance {CHANCE:.3f}",
        f"- latent_mu: {_fmt(lm_train)} vs chance {CHANCE:.3f}",
        "",
        f"**Test-held leakage** (held-out test sets, folds 3–5 only):",
        f"- connectome_norm: {_fmt(cn_test)} vs chance {CHANCE:.3f}",
        f"- latent_mu: {_fmt(lm_test)} vs chance {CHANCE:.3f}",
        "",
        "### Key conclusions for manuscript",
        "",
    ]

    # Automated verdict
    test_verdict_conn = (
        "LEAKAGE PRESENT" if (not np.isnan(cn_test) and cn_test > 0.45)
        else "NEAR CHANCE" if not np.isnan(cn_test)
        else "NO TEST DATA"
    )
    test_verdict_lat = (
        "LEAKAGE PRESENT" if (not np.isnan(lm_test) and lm_test > 0.45)
        else "NEAR CHANCE" if not np.isnan(lm_test)
        else "NO TEST DATA"
    )

    lines += [
        f"- Connectome-space (test held-out): **{test_verdict_conn}** (bal.acc = {_fmt(cn_test)})",
        f"- Latent μ-space (test held-out): **{test_verdict_lat}** (bal.acc = {_fmt(lm_test)})",
        "",
        "**Note on train vs test discrepancy**: train-CV leakage is inflated because",
        "the training partition is class-imbalanced toward Philips (CN=Philips only).",
        "The held-out test estimate is more reliable but only 3 folds are available.",
        "",
        "**Manuscript disclosure recommendation**:",
        "> A post-hoc scanner decodability test was performed to assess whether",
        "> manufacturer identity is decodable from the connectome or from the VAE",
        "> latent representations. On held-out test subjects (folds 3–5), balanced",
        f"> accuracy was {_fmt(cn_test)} (connectome) and {_fmt(lm_test)} (latent μ),",
        f"> compared to chance ({CHANCE:.3f}). The {test_verdict_lat.lower()} in the",
        "> latent space confirms that scanner information is [partially/not fully]",
        "> removed by the β-VAE regularisation.",
    ]

    lines += [
        "",
        "---",
        "",
        "## Fold-level values",
        "",
        "| Representation | Context | Fold | Balanced Accuracy |",
        "|----------------|---------|------|-------------------|",
    ]
    for _, row in raw.iterrows():
        lines.append(
            f"| {row['representation']} | {row['context']} "
            f"| fold_{row['fold_num']} | {row['balanced_accuracy_mean']:.4f} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading scanner leakage files...")
    raw = load_all_leakage()
    print(f"  Loaded {len(raw)} records from {raw['fold_num'].nunique()} fold directories")

    print("Aggregating...")
    agg = aggregate(raw)

    # Save aggregate CSV
    agg_csv = OUT_DIR / "scanner_leakage_aggregate.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"  Saved: {agg_csv}")

    # Save barplot
    plot_path = OUT_DIR / "barplot_scanner_leakage.png"
    make_barplot(agg, plot_path)
    print(f"  Saved: {plot_path}")

    # Save summary markdown
    md_path = OUT_DIR / "scanner_leakage_summary.md"
    md_path.write_text(make_summary_md(agg, raw))
    print(f"  Saved: {md_path}")

    # Terminal summary
    print()
    print("=" * 62)
    print("  SCANNER LEAKAGE AGGREGATE — TERMINAL SUMMARY")
    print("=" * 62)
    print(f"  Chance level (3-class): {CHANCE:.4f}")
    print()
    for ctx in ["train_cv", "test_held"]:
        sub = agg[agg["context"] == ctx]
        if sub.empty:
            continue
        n_folds = sub["n_folds"].max()
        ctx_label = "Train CV     " if ctx == "train_cv" else "Test held-out"
        print(f"  {ctx_label} ({n_folds} folds):")
        for rep in REPRESENTATIONS:
            row = sub[sub["representation"] == rep]
            if row.empty:
                continue
            m = row["mean_bacc"].values[0]
            s = row["std_bacc"].values[0]
            delta = m - CHANCE
            verdict = "[WARNING — leakage]" if m > 0.50 else "[moderate]" if m > 0.40 else "[near chance]"
            sd_str = f"±{s:.3f}" if not np.isnan(s) else "±n/a"
            print(f"    {rep:20s}: {m:.3f} {sd_str:8s} (Δchance={delta:+.3f}) {verdict}")
    print()
    print(f"  Outputs → {OUT_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
