#!/usr/bin/env python3
"""Audit VAE checkpoint availability for ADNI V4 [4,1,0] tanh baseline.

The audit is read-only. It scans the completed baseline run for per-fold VAE
checkpoints and reports whether downstream-guided checkpoint selection can be
performed without retraining.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/checkpoint_availability_audit"
)

GENERATED_FILES = [
    "checkpoint_availability_by_fold.csv",
    "checkpoint_availability_summary.json",
    "README.md",
]

EPOCH_PATTERNS = [
    re.compile(r"(?:epoch|ep|e)[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"_(\d{3,5})(?:\.pt|\.pth)$", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains audit outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def infer_epoch(path: Path) -> Optional[int]:
    text = path.name
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def checkpoint_role(path: Path, fold: int) -> str:
    if path.name == f"vae_model_fold_{fold}.pt":
        return "final_best_checkpoint"
    if "checkpoint" in path.name.lower():
        return "periodic_checkpoint"
    return "unknown_checkpoint"


def discover_checkpoints(fold_dir: Path, fold: int) -> List[Path]:
    patterns = [
        f"vae_model_fold_{fold}.pt",
        "vae_checkpoint*.pt",
        "vae_checkpoint*.pth",
        "checkpoint*.pt",
        "checkpoint*.pth",
        "vae_checkpoints/*.pt",
        "vae_checkpoints/*.pth",
    ]
    found: Dict[Path, None] = {}
    for pattern in patterns:
        for path in fold_dir.glob(pattern):
            if path.is_file():
                found[path] = None
    return sorted(found.keys(), key=lambda p: (infer_epoch(p) is None, infer_epoch(p) or 10**9, str(p)))


def read_manifest_paths(fold_dir: Path) -> List[Path]:
    patterns = ["vae_checkpoint_manifest*.csv", "vae_checkpoint_manifest*.json", "vae_checkpoints/*manifest*.json"]
    found: Dict[Path, None] = {}
    for pattern in patterns:
        for path in fold_dir.glob(pattern):
            if path.is_file():
                found[path] = None
    return sorted(found.keys())


def make_readme(run_dir: Path, outdir: Path, rows: pd.DataFrame, summary: Dict[str, Any]) -> None:
    per_fold_lines = []
    for fold, group in rows.groupby("fold"):
        paths = group["checkpoint_path"].tolist()
        can_select = bool(group["can_select_without_retraining"].iloc[0])
        per_fold_lines.append(
            f"- Fold {fold}: {len(paths)} checkpoint(s); selection without retraining = `{can_select}`"
        )
    if not per_fold_lines:
        per_fold_lines.append("- No fold checkpoints were found.")

    conclusion = (
        "Multiple checkpoints are available in every fold, so downstream checkpoint selection can proceed without retraining."
        if summary["can_select_without_retraining_all_folds"]
        else "The completed baseline does not contain multiple VAE checkpoints per fold. A new checkpoint-cadence run is required before leakage-safe downstream checkpoint selection can be evaluated."
    )

    lines = [
        "# V4 [4,1,0] Checkpoint Availability Audit",
        "",
        "Read-only audit of the completed tanh beta=2.5 ADNI V4 [4,1,0] run.",
        "",
        f"- Source run: `{run_dir}`",
        f"- Source realpath: `{run_dir.resolve() if run_dir.exists() else 'missing'}`",
        f"- Folds audited: {summary['folds_audited']}",
        f"- Total checkpoints found: {summary['total_checkpoints_found']}",
        f"- Can select without retraining in all folds: `{summary['can_select_without_retraining_all_folds']}`",
        "",
        "## Per Fold",
        "",
        *per_fold_lines,
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Outputs",
        "",
        "- `checkpoint_availability_by_fold.csv`",
        "- `checkpoint_availability_summary.json`",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    rows: List[Dict[str, Any]] = []

    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        checkpoints = discover_checkpoints(fold_dir, fold) if fold_dir.exists() else []
        manifests = read_manifest_paths(fold_dir) if fold_dir.exists() else []
        can_select = len(checkpoints) >= 2
        if checkpoints:
            for path in checkpoints:
                rows.append(
                    {
                        "fold": fold,
                        "fold_dir": str(fold_dir),
                        "fold_dir_exists": fold_dir.exists(),
                        "checkpoint_path": str(path),
                        "checkpoint_realpath": str(path.resolve()),
                        "checkpoint_name": path.name,
                        "checkpoint_role": checkpoint_role(path, fold),
                        "epoch_inferable": infer_epoch(path),
                        "manifest_paths": ";".join(str(p) for p in manifests),
                        "n_checkpoints_in_fold": len(checkpoints),
                        "can_select_without_retraining": can_select,
                    }
                )
        else:
            rows.append(
                {
                    "fold": fold,
                    "fold_dir": str(fold_dir),
                    "fold_dir_exists": fold_dir.exists(),
                    "checkpoint_path": "",
                    "checkpoint_realpath": "",
                    "checkpoint_name": "",
                    "checkpoint_role": "missing",
                    "epoch_inferable": None,
                    "manifest_paths": ";".join(str(p) for p in manifests),
                    "n_checkpoints_in_fold": 0,
                    "can_select_without_retraining": False,
                }
            )

    df = pd.DataFrame(rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()) if run_dir.exists() else None,
        "folds_audited": 5,
        "total_checkpoints_found": int((df["checkpoint_path"].astype(str) != "").sum()),
        "n_folds_with_multiple_checkpoints": int(
            df.groupby("fold")["n_checkpoints_in_fold"].max().ge(2).sum()
        ),
        "can_select_without_retraining_all_folds": bool(
            df.groupby("fold")["n_checkpoints_in_fold"].max().ge(2).all()
        ),
        "no_retraining_performed": True,
    }

    df.to_csv(outdir / "checkpoint_availability_by_fold.csv", index=False)
    (outdir / "checkpoint_availability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_readme(run_dir, outdir, df, summary)

    display_cols = [
        "fold",
        "n_checkpoints_in_fold",
        "checkpoint_name",
        "checkpoint_role",
        "epoch_inferable",
        "can_select_without_retraining",
    ]
    print(df[display_cols].to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
