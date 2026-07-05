#!/usr/bin/env python3
"""Foldwise split preview for the recover035 FULL 5x5 metadata-rescue retrain.

Loads patched_metadata_candidate.csv, runs the same outer 5-fold stratified
split as the launcher, and reports per-fold counts with 035_S_6927 flagged.

Outputs to: results/revision_bspc_2026/recover035_full5x5_split_preview_detail/
Also writes the standard split_preview_csv and split_preview_summary_csv paths
from the config so the launcher's --skip-preview-write can reuse them.

Read-only. Does not train, modify tensor, metadata, ledger, configs, or model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_recover035_full5x5.json"
)
DEFAULT_OUTPUT = RESULTS / "recover035_full5x5_split_preview_detail"

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    return view.to_markdown(index=False) + "\n"


def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tensor_idx" not in df.columns and "tensor_index" in df.columns:
        df = df.rename(columns={"tensor_index": "tensor_idx"})
    df["SubjectID"] = df["SubjectID"].astype(str)
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].astype(str)
    df["Manufacturer"] = df["Manufacturer"].map(normalize_manufacturer)
    df["Sex"] = df["Sex"].fillna("UNKNOWN").astype(str)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["tensor_idx"] = df["tensor_idx"].astype(int)
    return df


def strat_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        out[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    out["contains_035"] = bool((df["SubjectID"] == RECOVER_SUBJECT).any())
    return out


def run_split(meta: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    params = config["parameters"]
    seed = int(params["seed"])
    n_splits = int(params["outer_folds"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    strat_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, strat_cols)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()

    summary_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    recover_info: Dict[str, Any] = {"test_fold": None, "trainDev_folds": []}

    for fold, (train_dev_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(cn_ad)), y_outer), start=1
    ):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_cols = ["ResearchGroup_Mapped", *params["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_cols)
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + fold + 9,
            shuffle=True,
        )
        components = [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_pool.iloc[train_local]),
            ("vae_internal_val", vae_pool.iloc[val_local]),
        ]
        for component, df in components:
            row: Dict[str, Any] = {"fold": fold, "split_component": component}
            row.update(count_fields(df))
            summary_rows.append(row)

        # Track 035 placement
        if (test["SubjectID"] == RECOVER_SUBJECT).any():
            recover_info["test_fold"] = fold
        if (train_dev["SubjectID"] == RECOVER_SUBJECT).any():
            recover_info["trainDev_folds"].append(fold)

        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subject_rows.append({
                    "fold": fold,
                    "split_component": split_name,
                    "SubjectID": r["SubjectID"],
                    "tensor_idx": int(r["tensor_idx"]),
                    "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                    "Manufacturer": r["Manufacturer"],
                    "Sex": r["Sex"],
                    "is_recover_subject": bool(r["SubjectID"] == RECOVER_SUBJECT),
                    "is_excluded_subject": bool(r["SubjectID"] == EXCLUDED_SUBJECT),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(subject_rows), recover_info


def write_recover_subject_summary(
    outdir: Path, recover_info: Dict[str, Any], subject_df: pd.DataFrame
) -> None:
    lines = [
        f"# {RECOVER_SUBJECT} Split Placement",
        "",
        f"- **Outer-test fold**: {recover_info['test_fold']}",
        f"- **Classifier train_dev folds**: {sorted(recover_info['trainDev_folds'])}",
        "",
        "In the outer-test fold, 035_S_6927 contributes to classifier evaluation as an AD test subject.",
        "In all other folds, 035_S_6927 is in the train_dev pool and the VAE pool.",
        "",
        "## Classifier test fold rows for 035_S_6927",
        "",
    ]
    rows_035 = subject_df[
        subject_df["is_recover_subject"]
        & subject_df["split_component"].eq("classifier_test")
    ]
    if not rows_035.empty:
        lines.append(rows_035.to_markdown(index=False))
    else:
        lines.append("_(not found in test splits)_")
    lines.append("")
    (outdir / "recover_subject_fold_placement.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    outdir = resolve(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    meta_path = resolve(Path(config["paths"]["metadata_path"]))
    if not meta_path.exists():
        print(f"ERROR: metadata not found: {meta_path}", flush=True)
        return 1

    df = load_metadata(meta_path)

    # Verify 035 present, 128 absent from classifier pool
    clf_pool = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    n_recover = int((clf_pool["SubjectID"] == RECOVER_SUBJECT).sum())
    n_excluded = int((clf_pool["SubjectID"] == EXCLUDED_SUBJECT).sum())
    print(f"Metadata loaded: n={len(df)} (CN={int((df['ResearchGroup_Mapped']=='CN').sum())}, MCI={int((df['ResearchGroup_Mapped']=='MCI').sum())}, AD={int((df['ResearchGroup_Mapped']=='AD').sum())})")
    print(f"{RECOVER_SUBJECT} in classifier pool: {n_recover} (expected 1)")
    print(f"{EXCLUDED_SUBJECT} in classifier pool: {n_excluded} (expected 0)")

    summary, subjects, recover_info = run_split(df, config)
    print(f"{RECOVER_SUBJECT} outer-test fold: {recover_info['test_fold']}")
    print(f"{RECOVER_SUBJECT} train_dev folds: {sorted(recover_info['trainDev_folds'])}")

    # Write detailed split preview
    summary.to_csv(outdir / "recover035_split_preview_summary.csv", index=False)
    (outdir / "recover035_split_preview_summary.md").write_text(md_table(summary, max_rows=30), encoding="utf-8")
    subjects.to_csv(outdir / "recover035_split_preview_subjects.csv", index=False)

    write_recover_subject_summary(outdir, recover_info, subjects)

    # Also write to standard config paths for launcher compatibility
    std_summary_path = resolve(Path(config["paths"]["split_preview_summary_csv"]))
    std_subjects_path = resolve(Path(config["paths"]["split_preview_csv"]))
    std_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(std_summary_path, index=False)
    subjects.to_csv(std_subjects_path, index=False)
    print(f"Written: {std_subjects_path}")
    print(f"Written: {std_summary_path}")

    # Print summary table (classifier components only)
    clf_summary = summary[summary["split_component"].isin(["classifier_test", "classifier_train_dev"])]
    print(clf_summary[["fold", "split_component", "n", "AD", "CN", "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips", "Sex_F", "Sex_M", "contains_035"]].to_string(index=False))

    command_log = {
        "created_utc": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "config": str(resolve(args.config)),
        "metadata": str(meta_path),
        "output_dir": str(outdir),
        "recover_subject_test_fold": recover_info["test_fold"],
        "recover_subject_traindev_folds": sorted(recover_info["trainDev_folds"]),
        "recover_subject_in_clf_pool": n_recover,
        "excluded_subject_in_clf_pool": n_excluded,
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote split preview to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
