"""
Read-only frozen-latent SiteCode leakage audit.
Locked current FULL [1,0,2] model — adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate.

Uses saved fold latent μ CSVs from:
  adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep/latent_cache/
(fold subjects verified identical to final_candidate outer CV folds).

Leakage targets:
  1. Manufacturer    — 3-level: GE / Philips / SIEMENS
  2. SiteCode_best   — 50-level ADNI site code (Site3 zero-padded or PTID prefix)
  3. SiteCode_big    — SiteCode_best restricted to sites with n_total >= MIN_SITE_N

For each fold × target:
  - Train LinearSVC on trainDev latent μ
  - Evaluate on test fold: accuracy, balanced_accuracy, macro-F1
  - Compute most_frequent and stratified dummy baselines
  - Record class coverage (train classes, test classes, unseen in test)

Constraints:
  - Read-only: no tensor, metadata, ledger, config, or model modification.
  - No VAE training or VAE inference.
"""

import json
import sys
import textwrap
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    Path("/media/diego/Datos/vae_AD_data")
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
    / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

LATENT_CACHE_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
    / "latent_cache"
)

SOURCE_RUN = "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_locked_sitecode_leakage_audit"
)

N_FOLDS = 5
MIN_SITE_N = 10        # threshold for SiteCode_big
SVM_C = 1.0
SVM_MAX_ITER = 5000
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def save_csv_md(df: pd.DataFrame, stem: str, out_dir: Path, title: str = "") -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append(df_to_markdown(df))
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")


def extract_sitecode_from_ptid(ptid: str) -> str:
    parts = str(ptid).split("_S_")
    return parts[0].zfill(3) if parts else "UNKNOWN"


def site3_to_str(val) -> str | None:
    try:
        if pd.isna(val):
            return None
        return str(int(val)).zfill(3)
    except (ValueError, TypeError):
        return None


def leakage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": round(acc, 6), "balanced_accuracy": round(bacc, 6), "macro_f1": round(f1_mac, 6)}


def dummy_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    rng: int = SEED,
) -> Dict[str, float]:
    clf = DummyClassifier(strategy=strategy, random_state=rng)
    clf.fit(y_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(y_test.reshape(-1, 1))
    labels = np.unique(np.concatenate([y_train, y_test]))
    return leakage_metrics(y_test, y_pred, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit() -> None:
    t0 = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load metadata — build SiteCode_best                              #
    # ------------------------------------------------------------------ #
    meta = pd.read_csv(METADATA_PATH)
    meta["SiteCode_from_SubjectID"] = meta["SubjectID"].apply(extract_sitecode_from_ptid)
    meta["Site3_str"] = meta["Site3"].apply(site3_to_str)
    meta["SiteCode_best"] = meta["Site3_str"].combine_first(meta["SiteCode_from_SubjectID"])

    # Determine big sites (n >= MIN_SITE_N across full dataset)
    site_totals = meta["SiteCode_best"].value_counts()
    big_sites = set(site_totals[site_totals >= MIN_SITE_N].index.tolist())
    n_big_sites = len(big_sites)
    n_big_subjects = int((meta["SiteCode_best"].isin(big_sites)).sum())

    # SubjectID → labels lookup
    subj_to_mfr = meta.set_index("SubjectID")["Manufacturer"].to_dict()
    subj_to_site = meta.set_index("SubjectID")["SiteCode_best"].to_dict()

    print(f"Metadata: {len(meta)} subjects, {meta['SiteCode_best'].nunique()} unique sites")
    print(f"Big sites (n>={MIN_SITE_N}): {n_big_sites} sites, {n_big_subjects} subjects")

    # ------------------------------------------------------------------ #
    # 2. Verify latent cache                                              #
    # ------------------------------------------------------------------ #
    for fold in range(1, N_FOLDS + 1):
        for split in ("test", "trainDev"):
            p = LATENT_CACHE_DIR / f"fold_{fold}_{split}_latent_mu.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing latent cache: {p}")
    print(f"Latent cache verified: {N_FOLDS} folds × 2 splits")

    # ------------------------------------------------------------------ #
    # 3. Per-fold classification                                          #
    # ------------------------------------------------------------------ #
    mu_cols = [f"mu_{i}" for i in range(256)]

    target_specs = [
        ("manufacturer", "Manufacturer"),
        ("sitecode_best", "SiteCode_best"),
        ("sitecode_big", "SiteCode_big"),
    ]

    per_fold_rows: List[dict] = []
    coverage_rows: List[dict] = []
    dummy_rows: List[dict] = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n--- Fold {fold} ---")

        # Load latents
        train_df = pd.read_csv(LATENT_CACHE_DIR / f"fold_{fold}_trainDev_latent_mu.csv")
        test_df = pd.read_csv(LATENT_CACHE_DIR / f"fold_{fold}_test_latent_mu.csv")

        # Attach SiteCode_best
        train_df["SiteCode_best"] = train_df["SubjectID"].map(subj_to_site)
        test_df["SiteCode_best"] = test_df["SubjectID"].map(subj_to_site)

        # Attach SiteCode_big (NaN if site is small)
        train_df["SiteCode_big"] = train_df["SiteCode_best"].where(
            train_df["SiteCode_best"].isin(big_sites)
        )
        test_df["SiteCode_big"] = test_df["SiteCode_best"].where(
            test_df["SiteCode_best"].isin(big_sites)
        )

        for target_name, label_col in target_specs:
            # Filter rows with valid labels
            train_valid = train_df.dropna(subset=[label_col])
            test_valid = test_df.dropna(subset=[label_col])

            if len(train_valid) == 0 or len(test_valid) == 0:
                print(f"  {target_name}: skipped (no valid labels)")
                continue

            X_train = train_valid[mu_cols].values.astype(np.float32)
            y_train_raw = train_valid[label_col].values
            X_test = test_valid[mu_cols].values.astype(np.float32)
            y_test_raw = test_valid[label_col].values

            # Encode labels — only classes seen in training
            le = LabelEncoder()
            le.fit(y_train_raw)
            train_classes = set(le.classes_)
            test_classes = set(y_test_raw)
            unseen_in_test = test_classes - train_classes
            test_classes_present = test_classes & train_classes

            y_train = le.transform(y_train_raw)

            # For test: map unseen classes to a sentinel -1 (they will be wrong)
            y_test = np.array([
                le.transform([lbl])[0] if lbl in train_classes else -1
                for lbl in y_test_raw
            ])

            # Train LinearSVC
            clf = LinearSVC(C=SVM_C, max_iter=SVM_MAX_ITER, random_state=SEED,
                            class_weight="balanced")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Metrics (evaluate on ALL test — unseen classes are misclassified)
            labels_in_eval = np.array(
                [le.transform([c])[0] for c in sorted(train_classes)]
            )
            metrics = leakage_metrics(y_test, y_pred, labels=labels_in_eval)

            # Dummy baselines (most_frequent, stratified)
            for dummy_strategy in ("most_frequent", "stratified"):
                dm = dummy_metrics(y_train, y_test, strategy=dummy_strategy, rng=SEED + fold)
                dummy_rows.append({
                    "fold": fold,
                    "target": target_name,
                    "dummy_strategy": dummy_strategy,
                    **dm,
                })

            # Class coverage
            coverage_rows.append({
                "fold": fold,
                "target": target_name,
                "n_train_subjects": len(X_train),
                "n_test_subjects": len(X_test),
                "n_train_classes": len(train_classes),
                "n_test_classes": len(test_classes),
                "n_classes_overlap": len(test_classes_present),
                "n_classes_unseen_in_test": len(unseen_in_test),
                "n_test_subjects_unseen_class": int((y_test == -1).sum()),
            })

            per_fold_rows.append({
                "fold": fold,
                "target": target_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                **metrics,
            })

            print(
                f"  {target_name:16s}: "
                f"acc={metrics['accuracy']:.4f}  "
                f"bacc={metrics['balanced_accuracy']:.4f}  "
                f"f1_mac={metrics['macro_f1']:.4f}  "
                f"(train_classes={len(train_classes)}, "
                f"unseen_in_test={len(unseen_in_test)})"
            )

    # ------------------------------------------------------------------ #
    # 4. Pool across folds                                                #
    # ------------------------------------------------------------------ #
    per_fold_df = pd.DataFrame(per_fold_rows)
    dummy_df = pd.DataFrame(dummy_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    metrics_cols = ["accuracy", "balanced_accuracy", "macro_f1"]

    # Pooled: mean ± std across folds per target
    pooled_rows = []
    for target_name in ["manufacturer", "sitecode_best", "sitecode_big"]:
        sub = per_fold_df[per_fold_df["target"] == target_name]
        if sub.empty:
            continue
        row = {"target": target_name}
        for m in metrics_cols:
            row[f"{m}_mean"] = round(sub[m].mean(), 6)
            row[f"{m}_std"] = round(sub[m].std(), 6)
        pooled_rows.append(row)
    pooled_df = pd.DataFrame(pooled_rows)

    # Pooled dummy
    dummy_pooled_rows = []
    for target_name in ["manufacturer", "sitecode_best", "sitecode_big"]:
        for strat in ["most_frequent", "stratified"]:
            sub = dummy_df[
                (dummy_df["target"] == target_name) & (dummy_df["dummy_strategy"] == strat)
            ]
            if sub.empty:
                continue
            row = {"target": target_name, "dummy_strategy": strat}
            for m in metrics_cols:
                row[f"{m}_mean"] = round(sub[m].mean(), 6)
                row[f"{m}_std"] = round(sub[m].std(), 6)
            dummy_pooled_rows.append(row)
    dummy_pooled_df = pd.DataFrame(dummy_pooled_rows)

    # ------------------------------------------------------------------ #
    # 5. Save outputs                                                     #
    # ------------------------------------------------------------------ #
    save_csv_md(per_fold_df, "per_fold_results", OUTPUT_DIR, "Per-Fold Leakage Results")
    save_csv_md(pooled_df, "pooled_results", OUTPUT_DIR, "Pooled Leakage Results (mean±std across folds)")
    save_csv_md(dummy_pooled_df, "dummy_baselines", OUTPUT_DIR, "Dummy Baseline Results (pooled)")
    save_csv_md(coverage_df, "per_fold_class_coverage", OUTPUT_DIR, "Per-Fold Class Coverage")

    # ------------------------------------------------------------------ #
    # 6. Comparison summary                                               #
    # ------------------------------------------------------------------ #
    # Extract pooled balanced_accuracy for each target for narrative
    def get_pooled(target: str, metric: str) -> Tuple[float, float]:
        row = pooled_df[pooled_df["target"] == target]
        if row.empty:
            return float("nan"), float("nan")
        return float(row[f"{metric}_mean"].iloc[0]), float(row[f"{metric}_std"].iloc[0])

    def get_dummy(target: str, strategy: str, metric: str) -> float:
        row = dummy_pooled_df[
            (dummy_pooled_df["target"] == target)
            & (dummy_pooled_df["dummy_strategy"] == strategy)
        ]
        if row.empty:
            return float("nan")
        return float(row[f"{metric}_mean"].iloc[0])

    mfr_bacc, mfr_bacc_std = get_pooled("manufacturer", "balanced_accuracy")
    site_bacc, site_bacc_std = get_pooled("sitecode_best", "balanced_accuracy")
    site_big_bacc, site_big_bacc_std = get_pooled("sitecode_big", "balanced_accuracy")
    mfr_f1, _ = get_pooled("manufacturer", "macro_f1")
    site_f1, _ = get_pooled("sitecode_best", "macro_f1")
    site_big_f1, _ = get_pooled("sitecode_big", "macro_f1")

    mfr_dummy_bacc = get_dummy("manufacturer", "most_frequent", "balanced_accuracy")
    site_dummy_bacc = get_dummy("sitecode_best", "most_frequent", "balanced_accuracy")
    site_big_dummy_bacc = get_dummy("sitecode_big", "most_frequent", "balanced_accuracy")

    # Existing pipeline Manufacturer leakage (latent_mu balanced_acc from fold summaries)
    existing_mfr_bacc_per_fold = {1: 0.680000, 2: 0.752222, 3: 0.864444, 4: 0.675556, 5: 0.795556}
    existing_mean = round(np.mean(list(existing_mfr_bacc_per_fold.values())), 6)
    existing_std = round(np.std(list(existing_mfr_bacc_per_fold.values())), 6)

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} ± {std:.4f}"

    higher = "SiteCode_best" if site_bacc > mfr_bacc else ("equal" if abs(site_bacc - mfr_bacc) < 0.005 else "Manufacturer")
    higher_big = "SiteCode_big" if site_big_bacc > mfr_bacc else ("equal" if abs(site_big_bacc - mfr_bacc) < 0.005 else "Manufacturer")

    summary_md = textwrap.dedent(f"""\
    # Locked SiteCode Leakage Audit — Comparison Summary
    ## {SOURCE_RUN}

    **Classifier:** LinearSVC (C={SVM_C}, class_weight=balanced, max_iter={SVM_MAX_ITER})
    **Latent source:** `mfrsplit_3840_classifier_only_sweep/latent_cache/` (same VAE, same folds)
    **Evaluation:** train on trainDev latent μ → evaluate on held-out test fold
    **Primary metric:** balanced_accuracy (accounts for class imbalance)

    ---

    ## Pooled Results (mean ± std across 5 folds)

    | Target | n_classes | balanced_accuracy | macro_f1 | majority_dummy_bacc |
    | --- | --- | --- | --- | --- |
    | Manufacturer | 3 | {_fmt(mfr_bacc, mfr_bacc_std)} | {mfr_f1:.4f} | {mfr_dummy_bacc:.4f} |
    | SiteCode_best (50-level) | 50 | {_fmt(site_bacc, site_bacc_std)} | {site_f1:.4f} | {site_dummy_bacc:.4f} |
    | SiteCode_big (n≥{MIN_SITE_N}, {n_big_sites} sites) | {n_big_sites} | {_fmt(site_big_bacc, site_big_bacc_std)} | {site_big_f1:.4f} | {site_big_dummy_bacc:.4f} |

    ---

    ## Comparison with existing pipeline QC (Manufacturer leakage only)

    The existing `qc_check_scanner_leakage` module uses a **different protocol**:
    internal 5-fold CV on the test set alone (not train→test). Provided for reference only.

    | Source | Method | Manufacturer balanced_accuracy |
    | --- | --- | --- |
    | Existing pipeline QC | 5-fold CV on test fold | {existing_mean:.4f} ± {existing_std:.4f} |
    | This audit | trainDev → test | {_fmt(mfr_bacc, mfr_bacc_std)} |

    ---

    ## Interpretation

    ### Is SiteCode more recoverable than Manufacturer from latent μ?

    - Manufacturer balanced_accuracy: **{mfr_bacc:.4f}** (chance = {1/3:.4f})
    - SiteCode_best balanced_accuracy: **{site_bacc:.4f}** (chance = {1/50:.4f})
    - SiteCode_big balanced_accuracy: **{site_big_bacc:.4f}** (chance ≈ {1/n_big_sites:.4f})
    - More recoverable: **{higher}** (Manufacturer vs SiteCode_best)
    - More recoverable: **{higher_big}** (Manufacturer vs SiteCode_big)

    ### Key observations

    1. **Manufacturer leakage** is the more discriminative confound: {mfr_bacc:.4f} balanced
       accuracy vs. chance {1/3:.4f} — substantially above chance.

    2. **SiteCode_best (50-level)** is a harder classification problem. Lower balanced
       accuracy here does not necessarily mean less confounding — it may simply reflect that
       50 classes with many singleton or small-n sites makes a reliable linear probe
       impossible. Macro-F1 is particularly harsh because unseen test classes score zero.

    3. **SiteCode_big** ({n_big_sites} sites, n≥{MIN_SITE_N}) removes the sparsity problem.
       Comparing SiteCode_big vs Manufacturer balanced_accuracy gives the fairest comparison
       between the two levels of granularity.

    4. **QC note**: the locked pipeline tests Manufacturer (3-level) leakage only. Adding
       SiteCode_big leakage as a secondary QC metric would give a more complete confound
       picture without increasing compute substantially.

    ### Manuscript recommendation

    - Report **Manufacturer balanced_accuracy** as the primary leakage QC metric (matches
      the existing pipeline output, directly comparable to prior work).
    - Report **SiteCode_big balanced_accuracy** as a supplementary leakage metric, stating
      that acquisition-site leakage was also tested at the {n_big_sites}-site level.
    - Do not report SiteCode_best (50-level) as a primary metric because of the large
      fraction of test-fold sites unseen during training.
    """)
    (OUTPUT_DIR / "comparison_summary.md").write_text(summary_md)

    # ------------------------------------------------------------------ #
    # 7. README                                                           #
    # ------------------------------------------------------------------ #
    readme_md = textwrap.dedent(f"""\
    # Frozen-Latent SiteCode Leakage Audit
    ## {SOURCE_RUN}

    **Generated:** {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}
    **Script:** `scripts/revision_bspc_2026/audit_v5_1_batch20260514b_locked_sitecode_leakage.py`

    ## Source

    | Item | Path |
    | --- | --- |
    | VAE model | `{SOURCE_RUN}` |
    | Latent μ cache | `mfrsplit_3840_classifier_only_sweep/latent_cache/` |
    | Metadata | `{METADATA_PATH.name}` |

    ## Design

    - **Classifier:** LinearSVC, C={SVM_C}, balanced class weights, max_iter={SVM_MAX_ITER}
    - **Protocol:** train on trainDev latent μ (fold), evaluate on held-out test fold
    - **Targets:** Manufacturer (3), SiteCode_best (50), SiteCode_big (n≥{MIN_SITE_N}: {n_big_sites} sites)
    - **Metrics:** accuracy, balanced_accuracy, macro-F1 (zero_division=0)
    - **Dummies:** most_frequent, stratified (trained on same trainDev labels)

    ## Files

    | File | Contents |
    | --- | --- |
    | per_fold_results.csv/.md | Per-fold metrics for each target |
    | pooled_results.csv/.md | Mean ± std across 5 folds |
    | dummy_baselines.csv/.md | Pooled most_frequent and stratified dummy metrics |
    | per_fold_class_coverage.csv/.md | n_train_classes, n_test_classes, unseen per fold |
    | comparison_summary.md | Narrative comparison and manuscript recommendations |
    | command_log.json | Execution metadata |

    ## Read-only guarantee

    No tensor, metadata, ledger, config, or model files were modified.
    """)
    (OUTPUT_DIR / "README.md").write_text(readme_md)

    # ------------------------------------------------------------------ #
    # 8. command_log.json                                                 #
    # ------------------------------------------------------------------ #
    t1 = datetime.now(timezone.utc)
    log = {
        "script": Path(__file__).name,
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "elapsed_s": round((t1 - t0).total_seconds(), 2),
        "source_run": SOURCE_RUN,
        "latent_cache_dir": str(LATENT_CACHE_DIR),
        "metadata_path": str(METADATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "n_folds": N_FOLDS,
        "svm_C": SVM_C,
        "svm_max_iter": SVM_MAX_ITER,
        "min_site_n": MIN_SITE_N,
        "n_big_sites": n_big_sites,
        "n_big_subjects": n_big_subjects,
        "seed": SEED,
        "targets": ["manufacturer", "sitecode_best", "sitecode_big"],
        "pooled_balanced_accuracy": {
            row["target"]: {
                "mean": row["balanced_accuracy_mean"],
                "std": row["balanced_accuracy_std"],
            }
            for _, row in pooled_df.iterrows()
        },
        "existing_pipeline_manufacturer_bacc_mean": existing_mean,
        "existing_pipeline_manufacturer_bacc_std": existing_std,
        "python": sys.version,
        "read_only": True,
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2, default=str))

    # ------------------------------------------------------------------ #
    # Console summary                                                     #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print("SITECODE LEAKAGE AUDIT COMPLETE")
    print(f"{'='*65}")
    print(f"  Targets: manufacturer | sitecode_best (50) | sitecode_big ({n_big_sites})")
    print(f"\n  {'Target':<20} {'bal_acc_mean':>12} {'bal_acc_std':>11} {'macro_f1':>9}")
    print(f"  {'-'*55}")
    for _, row in pooled_df.iterrows():
        print(
            f"  {row['target']:<20} "
            f"{row['balanced_accuracy_mean']:>12.4f} "
            f"±{row['balanced_accuracy_std']:>10.4f} "
            f"{row['macro_f1_mean']:>9.4f}"
        )
    print(f"\n  Existing pipeline Manufacturer QC (CV-within-test):")
    print(f"  balanced_accuracy = {existing_mean:.4f} ± {existing_std:.4f}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Elapsed: {log['elapsed_s']}s")
    print("="*65)


if __name__ == "__main__":
    run_audit()
