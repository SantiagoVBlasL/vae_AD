#!/usr/bin/env python3
"""Find ADNI ROI signal sources for the v5 DPARSF-only rebuild.

This inventories ROI time-series files only. It does not read tensors,
checkpoints, model outputs, or compute connectivity.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_dparsf_only_rebuild"
    / "roi_signal_inventory"
)
DEFAULT_ROOTS = [
    PROJECT_ROOT / "data",
    Path("/media/diego/My_Book_Diego"),
]

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)
ROI_FILE_RE = re.compile(r"^ROISignals_.*?(\d{3}_S_\d{4}).*\.(txt|mat)$", re.IGNORECASE)

EXCLUDE_DIR_NAMES = {
    ".git",
    ".cache",
    "__pycache__",
    "node_modules",
    "wandb",
    "checkpoints",
    "checkpoint",
    "model_checkpoints",
    "trained_models",
    "plots",
    "figures",
    "Logs",
    "logs",
    "lost+found",
}
EXCLUDE_SUFFIXES = {".npz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".png", ".jpg", ".jpeg", ".gif", ".html", ".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find all ADNI ROISignals txt/mat files and summarize DPARSF-bandpass provenance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--roots", type=Path, nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-txt-shape-mb", type=float, default=10.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_subject(value: str) -> str:
    match = SUBJECT_RE.search(str(value).upper())
    return match.group(1).upper() if match else ""


def should_prune_dir(path: Path) -> bool:
    name = path.name
    if name in EXCLUDE_DIR_NAMES:
        return True
    low = str(path).lower()
    if "/site-packages/" in low or "/.conda/" in low:
        return True
    return False


def is_candidate_roi_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".mat"}:
        return False
    if suffix in EXCLUDE_SUFFIXES:
        return False
    name = path.name
    full = str(path)
    if ROI_FILE_RE.match(name):
        return True
    return "ROISignals" in full and SUBJECT_RE.search(name) is not None


def provenance_tokens(path: Path) -> List[str]:
    terms = [
        "adni_passband_20260510",
        "MARTIN59",
        "MARTIN_20260429_PHILIPS10",
        "PHILIPS",
        "ResultsAAL3",
        "ROISignals",
        "ROISignals_AAL3_FunImgARWSDCFN",
        "FunImg",
        "FunImgARWSDCFN",
        "FunImgARWSDCF",
        "ARWSDCFN",
        "ARWSDCF",
        "ARWSDC",
        "bandpass",
        "passband",
        "pasabandas",
        "AAL3",
    ]
    low = str(path).lower()
    out = [term for term in terms if term.lower() in low]
    return list(dict.fromkeys(out))


def suspected_dparsf_bandpass(tokens: Sequence[str]) -> bool:
    low = {token.lower() for token in tokens}
    return bool(
        {"funimgarwsdcfn", "arwsdcfn", "bandpass", "passband", "pasabandas"}.intersection(low)
        or "roisignals_aal3_funimgarwsdcfn" in low
    )


def source_batch(path: Path) -> str:
    text = str(path)
    low = text.lower()
    if "adni_passband_20260510" in low:
        return "martin_passband_20260510"
    if "martin59" in low:
        return "martin59_arwsdcf"
    if "martin_20260429_philips10" in low:
        return "martin_20260429_philips10"
    if "philips_cn_stress" in low:
        return "philips_cn_stress"
    if "subjectsdata" in low:
        return "original_subjectsdata"
    if "roisignals_aal3_funimgarwsdcfn" in low:
        return "unknown_funimgarwsdcfn"
    if "roisignals" in low:
        return "unknown_roisignals"
    return "unknown"


def source_root_for(path: Path) -> Path:
    parent = path.parent
    parts = list(parent.parts)
    for idx, part in enumerate(parts):
        if part.startswith("ROISignals"):
            return Path(*parts[: idx + 1])
    return parent


def read_txt_shape(path: Path, max_mb: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "txt_shape": "",
        "txt_rows": np.nan,
        "txt_cols": np.nan,
        "txt_shape_status": "not_read",
        "txt_nan_count": np.nan,
    }
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            out["txt_shape_status"] = f"not_read_gt_{max_mb:g}mb"
            return out
        try:
            arr = np.loadtxt(path, delimiter=",")
            status = "ok_comma_delimited"
        except Exception:
            arr = np.loadtxt(path)
            status = "ok_whitespace_delimited"
        shape = tuple(int(x) for x in arr.shape)
        out.update(
            {
                "txt_shape": str(shape),
                "txt_rows": int(arr.shape[0]) if arr.ndim >= 1 else 1,
                "txt_cols": int(arr.shape[1]) if arr.ndim >= 2 else 1,
                "txt_shape_status": status,
                "txt_nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
            }
        )
    except Exception as exc:
        out["txt_shape_status"] = f"failed: {exc}"
    return out


def find_files(roots: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    for root in roots:
        root = resolve(root)
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirs[:] = [name for name in dirs if not should_prune_dir(current_path / name)]
            for name in files:
                path = current_path / name
                if path.suffix.lower() in EXCLUDE_SUFFIXES:
                    continue
                if is_candidate_roi_file(path):
                    found.append(path)
    return sorted(found)


def pair_files(paths: Sequence[Path], max_txt_shape_mb: float) -> pd.DataFrame:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in paths:
        sid = normalize_subject(path.name)
        if not sid:
            continue
        src_root = source_root_for(path)
        key = (sid, str(src_root))
        row = grouped.setdefault(
            key,
            {
                "SubjectID": sid,
                "source_root": str(src_root),
                "source_batch": source_batch(path),
                "provenance_tokens": "",
                "suspected_dparsf_bandpass": False,
                "txt_path": "",
                "mat_path": "",
                "txt_size_bytes": np.nan,
                "mat_size_bytes": np.nan,
                "root_search_path": "",
            },
        )
        tokens = provenance_tokens(path)
        existing = [token for token in str(row.get("provenance_tokens", "")).split("|") if token]
        row["provenance_tokens"] = "|".join(list(dict.fromkeys(existing + tokens)))
        row["suspected_dparsf_bandpass"] = bool(row["suspected_dparsf_bandpass"] or suspected_dparsf_bandpass(tokens))
        if path.suffix.lower() == ".txt":
            row["txt_path"] = str(path)
            row["txt_size_bytes"] = int(path.stat().st_size)
            row.update(read_txt_shape(path, max_txt_shape_mb))
        elif path.suffix.lower() == ".mat":
            row["mat_path"] = str(path)
            row["mat_size_bytes"] = int(path.stat().st_size)
    df = pd.DataFrame(grouped.values())
    if df.empty:
        return pd.DataFrame(
            columns=[
                "SubjectID",
                "txt_path",
                "mat_path",
                "source_root",
                "source_batch",
                "txt_size_bytes",
                "mat_size_bytes",
                "txt_shape",
                "txt_rows",
                "txt_cols",
                "provenance_tokens",
                "suspected_dparsf_bandpass",
            ]
        )
    for col in ["txt_shape", "txt_rows", "txt_cols", "txt_shape_status", "txt_nan_count"]:
        if col not in df.columns:
            df[col] = np.nan if col not in {"txt_shape", "txt_shape_status"} else ""
    return df.sort_values(["SubjectID", "suspected_dparsf_bandpass", "source_batch", "source_root"], ascending=[True, False, True, True])


def choose_preferred_source(sub: pd.DataFrame) -> pd.Series:
    ranked = sub.copy()
    ranked["_has_txt"] = ranked["txt_path"].astype(str).ne("").astype(int)
    ranked["_has_mat"] = ranked["mat_path"].astype(str).ne("").astype(int)
    ranked["_dparsf"] = ranked["suspected_dparsf_bandpass"].astype(bool).astype(int)
    ranked["_batch_rank"] = ranked["source_batch"].map(
        {
            "martin_passband_20260510": 0,
            "martin59_arwsdcf": 1,
            "martin_20260429_philips10": 2,
            "unknown_funimgarwsdcfn": 3,
            "unknown_roisignals": 8,
        }
    ).fillna(9)
    ranked["_cols_ok"] = pd.to_numeric(ranked["txt_cols"], errors="coerce").isin([131, 166, 170]).astype(int)
    ranked = ranked.sort_values(["_has_txt", "_dparsf", "_cols_ok", "_has_mat", "_batch_rank"], ascending=[False, False, False, False, True])
    return ranked.iloc[0]


def build_subject_summary(files: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if files.empty:
        return pd.DataFrame()
    for sid, sub in files.groupby("SubjectID", sort=True):
        preferred = choose_preferred_source(sub)
        rows.append(
            {
                "SubjectID": sid,
                "n_signal_sources": int(len(sub)),
                "n_dparsf_bandpass_sources": int(sub["suspected_dparsf_bandpass"].astype(bool).sum()),
                "has_any_txt": bool(sub["txt_path"].astype(str).ne("").any()),
                "has_any_mat": bool(sub["mat_path"].astype(str).ne("").any()),
                "has_dparsf_bandpass_txt": bool(
                    (sub["suspected_dparsf_bandpass"].astype(bool) & sub["txt_path"].astype(str).ne("")).any()
                ),
                "preferred_txt_path": preferred.get("txt_path", ""),
                "preferred_mat_path": preferred.get("mat_path", ""),
                "preferred_source_root": preferred.get("source_root", ""),
                "preferred_source_batch": preferred.get("source_batch", ""),
                "preferred_txt_shape": preferred.get("txt_shape", ""),
                "preferred_txt_rows": preferred.get("txt_rows", np.nan),
                "preferred_txt_cols": preferred.get("txt_cols", np.nan),
                "preferred_suspected_dparsf_bandpass": bool(preferred.get("suspected_dparsf_bandpass", False)),
                "all_source_batches": "|".join(sorted(set(sub["source_batch"].astype(str)))),
                "all_source_roots": "|".join(sorted(set(sub["source_root"].astype(str)))),
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID")


def build_duplicate_report(files: pd.DataFrame) -> pd.DataFrame:
    if files.empty:
        return pd.DataFrame()
    dup = files.groupby("SubjectID").filter(lambda sub: len(sub) > 1).copy()
    if dup.empty:
        return pd.DataFrame(
            columns=["SubjectID", "n_signal_sources", "source_batch", "source_root", "txt_path", "mat_path", "suspected_dparsf_bandpass"]
        )
    dup["n_signal_sources"] = dup.groupby("SubjectID")["SubjectID"].transform("size")
    return dup[
        [
            "SubjectID",
            "n_signal_sources",
            "source_batch",
            "source_root",
            "txt_path",
            "mat_path",
            "txt_shape",
            "suspected_dparsf_bandpass",
            "provenance_tokens",
        ]
    ].sort_values(["SubjectID", "source_batch", "source_root"])


def build_missing_report(files: pd.DataFrame) -> pd.DataFrame:
    if files.empty:
        return pd.DataFrame()
    missing = files[(files["txt_path"].astype(str).eq("")) | (files["mat_path"].astype(str).eq(""))].copy()
    return missing[
        [
            "SubjectID",
            "source_batch",
            "source_root",
            "txt_path",
            "mat_path",
            "suspected_dparsf_bandpass",
            "provenance_tokens",
        ]
    ].sort_values(["SubjectID", "source_batch", "source_root"])


def write_readme(path: Path, roots: Sequence[Path], files: pd.DataFrame, summary: pd.DataFrame, duplicates: pd.DataFrame, missing: pd.DataFrame) -> None:
    n_subjects = int(summary["SubjectID"].nunique()) if not summary.empty else 0
    n_pref_dparsf = int(summary["preferred_suspected_dparsf_bandpass"].astype(bool).sum()) if not summary.empty else 0
    n_pref_txt = int(summary["has_any_txt"].astype(bool).sum()) if not summary.empty else 0
    lines = [
        "# ADNI ROI Signal Inventory",
        "",
        "Recursive inventory only. No tensors/checkpoints were read, no connectivity matrices were computed, and no training was run.",
        "",
        "## Search Roots",
        "",
    ]
    lines.extend(f"- `{resolve(root)}`" for root in roots)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Signal source rows: `{len(files)}`",
            f"- Unique subjects with any ROI signal source: `{n_subjects}`",
            f"- Subjects with preferred txt source: `{n_pref_txt}`",
            f"- Subjects with preferred suspected DPARSF-bandpass source: `{n_pref_dparsf}`",
            f"- Subjects with duplicate/multiple signal sources: `{duplicates['SubjectID'].nunique() if not duplicates.empty else 0}`",
            f"- Source rows missing txt or mat pair: `{len(missing)}`",
            "",
            "## v5 Use",
            "",
            "The manifest builder should select one DPARSF-bandpass ROI signal source per subject and must not use existing connectivity tensors/matrices. Subjects without a confirmed DPARSF-bandpass txt signal remain blockers for a full v5 rebuild.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    roots = [resolve(root) for root in args.roots]
    paths = find_files(roots)
    files = pair_files(paths, args.max_txt_shape_mb)
    summary = build_subject_summary(files)
    duplicates = build_duplicate_report(files)
    missing = build_missing_report(files)

    files.to_csv(output_dir / "roi_signal_files_all.csv", index=False)
    summary.to_csv(output_dir / "roi_signal_subject_summary.csv", index=False)
    duplicates.to_csv(output_dir / "duplicate_signal_sources.csv", index=False)
    missing.to_csv(output_dir / "missing_txt_or_mat.csv", index=False)
    write_readme(output_dir / "README.md", roots, files, summary, duplicates, missing)

    print(f"Wrote ROI signal inventory to {output_dir}")
    print(f"Signal source rows: {len(files)}")
    print(f"Unique subjects: {summary['SubjectID'].nunique() if not summary.empty else 0}")
    print(f"Preferred suspected DPARSF-bandpass subjects: {int(summary['preferred_suspected_dparsf_bandpass'].astype(bool).sum()) if not summary.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
