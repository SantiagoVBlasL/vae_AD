#!/usr/bin/env python3
"""Audit historical feature-extraction sources for v5 DPARSF-only rebuild.

Read-only with respect to source files. It inspects notebooks/scripts for
bandpass, ROI-reduction, reorder, target-length, TR, output, and channel clues.
It does not compute connectivity and does not train.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_dparsf_only_rebuild"
    / "source_code_audit"
)

PRIMARY_SOURCE_FILES = [
    PROJECT_ROOT / "notebooks" / "01_feature_extraction_manual.ipynb",
    PROJECT_ROOT / "scripts" / "feature_extraction_manual.py",
    PROJECT_ROOT / "notebooks" / "01_feature_extraction_manual_COVID.ipynb",
    PROJECT_ROOT / "scripts" / "feature_extraction_manual_COVID.py",
]

SEARCH_TERMS = [
    "ROI_SIGNALS_DIR_PATH_AAL3",
    "ROISignals",
    "ROISignals_AAL3",
    "FunImgARWSDCFN",
    "FunImgARWSDCF",
    "ARWSDCFN",
    "ARWSDCF",
    "LOW_CUT_HZ",
    "HIGH_CUT_HZ",
    "butter",
    "filtfilt",
    "bandpass",
    "_bandpass_filter_signals",
    "TARGET_LEN_TS",
    "TR_SECONDS",
    "ROI_MNI_V7_vol",
    "aal3_131_manual_network_order",
    "_orient_and_reduce_rois",
    "_reorder_rois_by_network",
    "Yeo",
    "CONNECTIVITY_CHANNEL_NAMES",
    "Pearson_OMST",
    "Pearson_Full_FisherZ",
    "MI_KNN",
    "dFC",
    "DistanceCorr",
    "Granger",
    "GLOBAL_TENSOR",
    "np.savez",
]

SELF_AUDIT_EXCLUDE_NAMES = {
    "audit_feature_extraction_sources_and_bandpass.py",
    "find_all_adni_roi_signal_sources.py",
    "build_adni_v5_dparsf_only_subject_manifest.py",
}

PARAM_NAMES = [
    "BASE_PATH_AAL3",
    "ROI_SIGNALS_DIR_PATH_AAL3",
    "ROI_FILENAME_TEMPLATE",
    "AAL3_META_PATH",
    "AAL3_MANUAL_NETWORK_MAPPING_CSV",
    "LOW_CUT_HZ",
    "HIGH_CUT_HZ",
    "FILTER_ORDER",
    "TR_SECONDS",
    "TARGET_LEN_TS",
    "RAW_DATA_EXPECTED_COLUMNS",
    "FINAL_N_ROIS_EXPECTED",
    "CONNECTIVITY_CHANNEL_NAMES",
    "N_CHANNELS",
    "OUTPUT_CONNECTIVITY_DIR_NAME",
]


@dataclass(frozen=True)
class SourceText:
    path: Path
    source_type: str
    label: str
    text: str
    line_offset: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit historical feature extraction code for bandpass/provenance parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def discover_related_sources() -> List[Path]:
    paths: List[Path] = []
    for path in PRIMARY_SOURCE_FILES:
        if path.exists():
            paths.append(path)
    for base in [PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.name in SELF_AUDIT_EXCLUDE_NAMES:
                continue
            low = str(path).lower()
            if (
                "feature_extraction" in low
                or "dparsf" in low
                or "bandpass" in low
                or "build_adni_expanded" in path.name
            ):
                paths.append(path)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def notebook_code_sources(path: Path) -> List[SourceText]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    out: List[SourceText] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        out.append(SourceText(path=path, source_type="notebook_code", label=f"cell_{idx}", text=text))
    return out


def file_sources(path: Path) -> List[SourceText]:
    if path.suffix == ".ipynb":
        return notebook_code_sources(path)
    try:
        return [SourceText(path=path, source_type="python_script", label="file", text=path.read_text(encoding="utf-8", errors="replace"))]
    except Exception:
        return []


def classify_hit(line: str) -> str:
    low = line.lower()
    if "bandpass" in low or "butter" in low or "filtfilt" in low or "low_cut" in low or "high_cut" in low:
        return "bandpass"
    if "roi_mni_v7" in low or "170" in low or "131" in low or "reduce_rois" in low or "final_n_rois" in low:
        return "roi_reduction"
    if "yeo" in low or "manual_network_order" in low or "reorder" in low:
        return "roi_reorder"
    if "target_len" in low or "tr_seconds" in low or "tr =" in low:
        return "timebase"
    if "connectivity_channel" in low or "pearson" in low or "granger" in low or "distancecorr" in low:
        return "channels"
    if "roisignals" in low or "base_path" in low or "funimg" in low or "arwsdc" in low:
        return "input_source"
    if "global_tensor" in low or "savez" in low or "output" in low:
        return "output"
    return "other"


def matched_terms(line: str) -> str:
    low = line.lower()
    return "|".join(term for term in SEARCH_TERMS if term.lower() in low)


def audit_hits(sources: Sequence[SourceText]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source in sources:
        for lineno, line in enumerate(source.text.splitlines(), start=1):
            if not matched_terms(line):
                continue
            rows.append(
                {
                    "source_file": rel(source.path),
                    "source_type": source.source_type,
                    "location": source.label if source.source_type.startswith("notebook") else str(lineno),
                    "category": classify_hit(line),
                    "matched_terms": matched_terms(line),
                    "snippet": " ".join(line.strip().split()),
                }
            )
    return pd.DataFrame(rows)


def safe_literal(value: ast.AST) -> str:
    try:
        return repr(ast.literal_eval(value))
    except Exception:
        return ast.unparse(value) if hasattr(ast, "unparse") else ""


def assignment_parameters_from_python(path: Path) -> List[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    line_map = text.splitlines()
    for node in ast.walk(tree):
        targets: List[str] = []
        value_node: Optional[ast.AST] = None
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value_node = node.value
        if not value_node:
            continue
        for name in targets:
            if name not in PARAM_NAMES:
                continue
            rows.append(
                {
                    "source_file": rel(path),
                    "source_type": "python_script",
                    "location": str(getattr(node, "lineno", "")),
                    "parameter": name,
                    "value": safe_literal(value_node),
                    "raw_line": line_map[getattr(node, "lineno", 1) - 1].strip() if line_map else "",
                }
            )
    return rows


def assignment_parameters_from_text(source: SourceText) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(r"^\s*([A-Z][A-Z0-9_]+)\s*=\s*(.+)")
    for lineno, line in enumerate(source.text.splitlines(), start=1):
        match = pattern.match(line)
        if not match or match.group(1) not in PARAM_NAMES:
            continue
        rows.append(
            {
                "source_file": rel(source.path),
                "source_type": source.source_type,
                "location": source.label if source.source_type.startswith("notebook") else str(lineno),
                "parameter": match.group(1),
                "value": match.group(2).strip(),
                "raw_line": line.strip(),
            }
        )
    return rows


def audit_parameters(sources: Sequence[SourceText]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    parsed_python = set()
    for source in sources:
        if source.path.suffix == ".py" and str(source.path) not in parsed_python:
            parsed_python.add(str(source.path))
            rows.extend(assignment_parameters_from_python(source.path))
        rows.extend(assignment_parameters_from_text(source))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["source_file", "source_type", "location", "parameter", "value", "raw_line", "interpretation"])
    df["interpretation"] = df["parameter"].map(parameter_interpretation)
    return df.sort_values(["parameter", "source_file", "location"])


def parameter_interpretation(name: str) -> str:
    if name in {"LOW_CUT_HZ", "HIGH_CUT_HZ", "FILTER_ORDER"}:
        return "historical Python temporal filter parameter"
    if name in {"TR_SECONDS", "TARGET_LEN_TS"}:
        return "timebase / homogenization parameter"
    if name in {"RAW_DATA_EXPECTED_COLUMNS", "FINAL_N_ROIS_EXPECTED", "AAL3_META_PATH"}:
        return "AAL3 170-to-131 ROI reduction parameter"
    if name == "AAL3_MANUAL_NETWORK_MAPPING_CSV":
        return "Yeo/network ROI reorder parameter"
    if name == "CONNECTIVITY_CHANNEL_NAMES":
        return "connectivity channel order"
    if name in {"BASE_PATH_AAL3", "ROI_SIGNALS_DIR_PATH_AAL3", "ROI_FILENAME_TEMPLATE"}:
        return "ROI signal input source parameter"
    if name == "OUTPUT_CONNECTIVITY_DIR_NAME":
        return "tensor/output location parameter"
    return ""


def write_hits_markdown(path: Path, hits: pd.DataFrame) -> None:
    lines = ["# Extraction Source Hits", ""]
    if hits.empty:
        lines.append("No matching source hits found.")
    else:
        for (source_file, category), sub in hits.groupby(["source_file", "category"], sort=True):
            lines.extend([f"## {source_file} - {category}", ""])
            for _, row in sub.head(80).iterrows():
                lines.append(f"- `{row['location']}` `{row['matched_terms']}`: {row['snippet']}")
            if len(sub) > 80:
                lines.append(f"- ... truncated {len(sub) - 80} additional hits for this section.")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, sources: Sequence[Path], hits: pd.DataFrame, params: pd.DataFrame) -> None:
    has_python_bandpass = bool(
        not hits.empty
        and hits["category"].eq("bandpass").any()
        and hits["snippet"].str.contains("butter|filtfilt|_bandpass_filter_signals", case=False, regex=True).any()
    )
    has_roi_reduction = bool(not hits.empty and hits["category"].eq("roi_reduction").any())
    has_reorder = bool(not hits.empty and hits["category"].eq("roi_reorder").any())
    channel_hits = hits[hits["category"] == "channels"] if not hits.empty else pd.DataFrame()
    lines = [
        "# Feature Extraction Source Audit",
        "",
        "Read-only source-code audit. No connectivity computation, tensor generation, inference, or training was run.",
        "",
        "## Summary",
        "",
        f"- Source files inspected: `{len(sources)}`",
        f"- Matching source hits: `{len(hits)}`",
        f"- Parameter assignments detected: `{len(params)}`",
        f"- Historical Python bandpass code detected: `{'YES' if has_python_bandpass else 'NO'}`",
        f"- ROI 170->131 reduction evidence detected: `{'YES' if has_roi_reduction else 'NO'}`",
        f"- Yeo/network reorder evidence detected: `{'YES' if has_reorder else 'NO'}`",
        f"- Channel-name/order evidence rows: `{len(channel_hits)}`",
        "",
        "## v5 Implication",
        "",
        "For `adni_expanded_v5_passband_dparsf_only`, the historical Python bandpass branch must remain disabled. DPARSF-bandpass ROI signals should be treated as already temporally filtered, and connectivity must be recomputed uniformly for every final training subject.",
        "",
        "## Inspected Sources",
        "",
    ]
    lines.extend(f"- `{rel(path)}`" for path in sources)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    source_paths = discover_related_sources()
    sources: List[SourceText] = []
    for path in source_paths:
        sources.extend(file_sources(path))

    hits = audit_hits(sources)
    params = audit_parameters(sources)

    hits.to_csv(output_dir / "extraction_source_hits.csv", index=False)
    params.to_csv(output_dir / "extraction_parameters_detected.csv", index=False)
    write_hits_markdown(output_dir / "extraction_source_hits.md", hits)
    write_readme(output_dir / "README.md", source_paths, hits, params)

    print(f"Wrote source audit to {output_dir}")
    print(f"Source files inspected: {len(source_paths)}")
    print(f"Matching source hits: {len(hits)}")
    print(f"Parameter assignments detected: {len(params)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
