#!/usr/bin/env python3
"""Read-only split identity audit for v5.1c Fold-1 horizon5120 diagnostic.

Compares Fold 1 split artifacts between:
  - full v5.1c horizon4480/cycles56 run
  - fold-only v5.1c horizon5120/cycles64 diagnostic

The script only reads completed split artifacts and writes audit outputs under a
new results directory. It does not train or modify run outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_RUN = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5"
)
FOLDONLY_RUN = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "adni_v5_1c_recover035_ch1_0_2_fold1_horizon5120_cycles64"
)
OUT_DIR = PROJECT_ROOT / (
    "results/revision_bspc_2026/v5_1c_fold1_5120_split_identity_audit"
)
TARGET_FOLD = 1
RECOVERED_SUBJECT = "035_S_6927"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--full-run", type=Path, default=FULL_RUN)
    parser.add_argument("--foldonly-run", type=Path, default=FOLDONLY_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metadata_path_from_run(run_dir: Path) -> Path:
    cfg = read_json(run_dir / "run_config.json")
    raw = cfg.get("metadata_path") or cfg.get("args", {}).get("metadata_path")
    if raw is None:
        raw = cfg.get("paths", {}).get("metadata_path")
    if raw is None:
        raise RuntimeError(f"Could not locate metadata_path in {run_dir / 'run_config.json'}")
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metadata(run_dir: Path) -> pd.DataFrame:
    path = metadata_path_from_run(run_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "tensor_index" not in df.columns:
        raise RuntimeError(f"Metadata lacks tensor_index column: {path}")
    return df


def subject_table(run_dir: Path, split_name: str) -> pd.DataFrame:
    fold_dir = run_dir / f"fold_{TARGET_FOLD}"
    if split_name == "classifier_test":
        path = fold_dir / "test_subjects_fold.csv"
    elif split_name == "classifier_train_dev":
        path = fold_dir / "train_dev_subjects_fold.csv"
    else:
        raise ValueError(split_name)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.insert(0, "position", np.arange(1, len(df) + 1))
    return df


def vae_split_table(run_dir: Path, meta: pd.DataFrame, split_name: str) -> pd.DataFrame:
    fold_dir = run_dir / f"fold_{TARGET_FOLD}"
    pool_path = fold_dir / "vae_training_pool_tensor_idx.npy"
    train_local_path = fold_dir / "vae_actual_train_idx_local_to_pool.npy"
    val_local_path = fold_dir / "vae_internal_val_idx_local_to_pool.npy"
    for path in [pool_path, train_local_path, val_local_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    pool_tensor_idx = np.load(pool_path)
    if split_name == "vae_training_pool":
        tensor_idx = pool_tensor_idx
    elif split_name == "vae_actual_train":
        tensor_idx = pool_tensor_idx[np.load(train_local_path)]
    elif split_name == "vae_internal_val":
        tensor_idx = pool_tensor_idx[np.load(val_local_path)]
    else:
        raise ValueError(split_name)

    map_cols = [
        c
        for c in [
            "tensor_index",
            "SubjectID",
            "ResearchGroup_Mapped",
            "Diagnosis",
            "Manufacturer",
            "Site3",
            "Age",
            "Sex",
        ]
        if c in meta.columns
    ]
    mapper = meta[map_cols].rename(columns={"tensor_index": "tensor_idx"})
    df = pd.DataFrame({"tensor_idx": tensor_idx.astype(int)})
    df.insert(0, "position", np.arange(1, len(df) + 1))
    df = df.merge(mapper, on="tensor_idx", how="left")
    return df


def enrich_subject_table(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "Manufacturer" in df.columns and "Site3" in df.columns:
        return df
    map_cols = [
        c
        for c in [
            "tensor_index",
            "Manufacturer",
            "Site3",
            "Age",
            "Sex",
            "Diagnosis",
        ]
        if c in meta.columns
    ]
    mapper = meta[map_cols].rename(columns={"tensor_index": "tensor_idx"})
    return df.merge(mapper, on="tensor_idx", how="left")


def compare_split(
    split_name: str,
    full_df: pd.DataFrame,
    foldonly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_ids = full_df["SubjectID"].astype(str).tolist()
    fold_ids = foldonly_df["SubjectID"].astype(str).tolist()
    full_pos = {sid: i + 1 for i, sid in enumerate(full_ids)}
    fold_pos = {sid: i + 1 for i, sid in enumerate(fold_ids)}
    all_ids = sorted(set(full_ids) | set(fold_ids))
    rows: list[dict[str, Any]] = []
    full_lookup = full_df.set_index("SubjectID", drop=False)
    fold_lookup = foldonly_df.set_index("SubjectID", drop=False)
    for sid in all_ids:
        frow = full_lookup.loc[sid] if sid in full_lookup.index else pd.Series(dtype=object)
        orow = fold_lookup.loc[sid] if sid in fold_lookup.index else pd.Series(dtype=object)
        rows.append(
            {
                "split_name": split_name,
                "SubjectID": sid,
                "present_in_full": sid in full_pos,
                "present_in_foldonly": sid in fold_pos,
                "position_full": full_pos.get(sid, np.nan),
                "position_foldonly": fold_pos.get(sid, np.nan),
                "same_position": full_pos.get(sid) == fold_pos.get(sid),
                "ResearchGroup_Mapped_full": frow.get("ResearchGroup_Mapped", np.nan),
                "ResearchGroup_Mapped_foldonly": orow.get("ResearchGroup_Mapped", np.nan),
                "Manufacturer_full": frow.get("Manufacturer", np.nan),
                "Manufacturer_foldonly": orow.get("Manufacturer", np.nan),
                "Site3_full": frow.get("Site3", np.nan),
                "Site3_foldonly": orow.get("Site3", np.nan),
                "tensor_idx_full": frow.get("tensor_idx", np.nan),
                "tensor_idx_foldonly": orow.get("tensor_idx", np.nan),
            }
        )
    comparison = pd.DataFrame(rows)
    summary = {
        "split_name": split_name,
        "n_full": len(full_ids),
        "n_foldonly": len(fold_ids),
        "sets_identical": set(full_ids) == set(fold_ids),
        "order_identical": full_ids == fold_ids,
        "symmetric_difference_n": len(set(full_ids) ^ set(fold_ids)),
        "full_only_subjects": sorted(set(full_ids) - set(fold_ids)),
        "foldonly_only_subjects": sorted(set(fold_ids) - set(full_ids)),
        "recovered_035_location_full": "present" if RECOVERED_SUBJECT in full_pos else "absent",
        "recovered_035_location_foldonly": "present" if RECOVERED_SUBJECT in fold_pos else "absent",
        "recovered_035_position_full": full_pos.get(RECOVERED_SUBJECT),
        "recovered_035_position_foldonly": fold_pos.get(RECOVERED_SUBJECT),
    }
    return comparison, summary


def counts_table(split_name: str, run_label: str, df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["ResearchGroup_Mapped", "Manufacturer"] if c in df.columns]
    if len(group_cols) < 2:
        return pd.DataFrame()
    out = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(group_cols)
    )
    out.insert(0, "run_label", run_label)
    out.insert(0, "split_name", split_name)
    return out


def write_split_md(
    path: Path,
    title: str,
    detail: pd.DataFrame,
    summaries: list[dict[str, Any]],
    counts: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for summary in summaries:
            f.write(f"## {summary['split_name']}\n\n")
            f.write(f"- n_full: `{summary['n_full']}`\n")
            f.write(f"- n_foldonly: `{summary['n_foldonly']}`\n")
            f.write(f"- sets_identical: `{summary['sets_identical']}`\n")
            f.write(f"- order_identical: `{summary['order_identical']}`\n")
            f.write(f"- symmetric_difference_n: `{summary['symmetric_difference_n']}`\n")
            f.write(f"- 035_S_6927 full: `{summary['recovered_035_location_full']}`")
            if summary["recovered_035_position_full"]:
                f.write(f" at position `{summary['recovered_035_position_full']}`")
            f.write("\n")
            f.write(f"- 035_S_6927 foldonly: `{summary['recovered_035_location_foldonly']}`")
            if summary["recovered_035_position_foldonly"]:
                f.write(f" at position `{summary['recovered_035_position_foldonly']}`")
            f.write("\n\n")
            if summary["full_only_subjects"] or summary["foldonly_only_subjects"]:
                f.write(f"- full_only_subjects: `{summary['full_only_subjects']}`\n")
                f.write(f"- foldonly_only_subjects: `{summary['foldonly_only_subjects']}`\n\n")
        f.write("## Diagnosis x Manufacturer Counts\n\n")
        if counts.empty:
            f.write("_No counts available._\n\n")
        else:
            f.write(counts.to_markdown(index=False))
            f.write("\n\n")
        f.write("## Subject-Level Comparison\n\n")
        if detail.empty:
            f.write("_No rows._\n")
        else:
            f.write(detail.to_markdown(index=False))
            f.write("\n")


def main() -> int:
    args = parse_args()
    full_run = resolve(args.full_run)
    foldonly_run = resolve(args.foldonly_run)
    out_dir = resolve(args.output_dir)

    for run_dir in [full_run, foldonly_run]:
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        if not (run_dir / "run_config.json").exists():
            raise FileNotFoundError(run_dir / "run_config.json")
        if not (run_dir / f"fold_{TARGET_FOLD}").exists():
            raise FileNotFoundError(run_dir / f"fold_{TARGET_FOLD}")

    full_meta = load_metadata(full_run)
    fold_meta = load_metadata(foldonly_run)

    full_test = enrich_subject_table(subject_table(full_run, "classifier_test"), full_meta)
    fold_test = enrich_subject_table(subject_table(foldonly_run, "classifier_test"), fold_meta)
    full_train = enrich_subject_table(subject_table(full_run, "classifier_train_dev"), full_meta)
    fold_train = enrich_subject_table(subject_table(foldonly_run, "classifier_train_dev"), fold_meta)

    split_pairs = {
        "classifier_test": (full_test, fold_test),
        "classifier_train_dev": (full_train, fold_train),
        "vae_training_pool": (
            vae_split_table(full_run, full_meta, "vae_training_pool"),
            vae_split_table(foldonly_run, fold_meta, "vae_training_pool"),
        ),
        "vae_actual_train": (
            vae_split_table(full_run, full_meta, "vae_actual_train"),
            vae_split_table(foldonly_run, fold_meta, "vae_actual_train"),
        ),
        "vae_internal_val": (
            vae_split_table(full_run, full_meta, "vae_internal_val"),
            vae_split_table(foldonly_run, fold_meta, "vae_internal_val"),
        ),
    }

    detail_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    count_parts: list[pd.DataFrame] = []
    for split, (full_df, fold_df) in split_pairs.items():
        comp, summary = compare_split(split, full_df, fold_df)
        detail_parts.append(comp)
        summaries.append(summary)
        count_parts.append(counts_table(split, "full_horizon4480", full_df))
        count_parts.append(counts_table(split, "foldonly_horizon5120", fold_df))

    detail = pd.concat(detail_parts, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    counts = pd.concat([c for c in count_parts if not c.empty], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    test_detail = detail[detail["split_name"] == "classifier_test"].copy()
    train_detail = detail[detail["split_name"] != "classifier_test"].copy()
    test_summaries = [s for s in summaries if s["split_name"] == "classifier_test"]
    train_summaries = [s for s in summaries if s["split_name"] != "classifier_test"]
    test_counts = counts[counts["split_name"] == "classifier_test"].copy()
    train_counts = counts[counts["split_name"] != "classifier_test"].copy()

    test_detail.to_csv(out_dir / "fold1_test_subject_comparison.csv", index=False)
    train_detail.to_csv(out_dir / "fold1_train_subject_comparison.csv", index=False)
    summary_df.to_csv(out_dir / "fold1_split_identity_summary.csv", index=False)
    counts.to_csv(out_dir / "fold1_diagnosis_manufacturer_counts.csv", index=False)

    write_split_md(
        out_dir / "fold1_test_subject_comparison.md",
        "Fold 1 Test Subject Comparison",
        test_detail,
        test_summaries,
        test_counts,
    )
    write_split_md(
        out_dir / "fold1_train_subject_comparison.md",
        "Fold 1 Train / VAE Split Comparison",
        train_detail,
        train_summaries,
        train_counts,
    )

    all_sets = bool(summary_df["sets_identical"].all())
    all_orders = bool(summary_df["order_identical"].all())
    test_summary = summary_df[summary_df["split_name"] == "classifier_test"].iloc[0]
    train_summary = summary_df[summary_df["split_name"] == "classifier_train_dev"].iloc[0]
    recovered_locations = summary_df[
        [
            "split_name",
            "recovered_035_location_full",
            "recovered_035_position_full",
            "recovered_035_location_foldonly",
            "recovered_035_position_foldonly",
        ]
    ]
    decision = (
        "PASS: Fold 1 split identity is exact for all audited classifier and VAE split artifacts."
        if all_sets and all_orders
        else "FAIL: Fold 1 split identity differs in at least one audited split."
    )
    decision_text = f"""# Fold 1 Split Identity Decision

Decision: **{decision}**

Compared runs:

- Full v5.1c horizon4480/cycles56: `{full_run}`
- Fold-only v5.1c horizon5120/cycles64: `{foldonly_run}`

Classifier split summary:

- n_test_full: `{int(test_summary['n_full'])}`
- n_test_foldonly: `{int(test_summary['n_foldonly'])}`
- test sets identical: `{bool(test_summary['sets_identical'])}`
- test order identical: `{bool(test_summary['order_identical'])}`
- test symmetric_difference_n: `{int(test_summary['symmetric_difference_n'])}`
- n_train_dev_full: `{int(train_summary['n_full'])}`
- n_train_dev_foldonly: `{int(train_summary['n_foldonly'])}`
- train/dev sets identical: `{bool(train_summary['sets_identical'])}`
- train/dev order identical: `{bool(train_summary['order_identical'])}`
- train/dev symmetric_difference_n: `{int(train_summary['symmetric_difference_n'])}`

035_S_6927 location:

{recovered_locations.to_markdown(index=False)}

Interpretation:

The fold-only diagnostic is split-comparable to the full v5.1c Fold 1 only if all audited sets and orders are identical. The VAE actual train/internal validation splits are included because representation learning can change if internal validation assignment changes.

Safety:

No training was launched. No tensor, metadata, ledger, config, or existing model-output files were modified.
"""
    (out_dir / "fold1_split_identity_decision.md").write_text(decision_text, encoding="utf-8")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "full_run": str(full_run),
        "foldonly_run": str(foldonly_run),
        "target_fold": TARGET_FOLD,
        "read_only": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "existing_model_output_modified": False,
        "decision": decision,
        "outputs": [
            "fold1_test_subject_comparison.csv",
            "fold1_test_subject_comparison.md",
            "fold1_train_subject_comparison.csv",
            "fold1_train_subject_comparison.md",
            "fold1_split_identity_decision.md",
            "command_log.json",
        ],
    }
    (out_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(decision_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
