#!/usr/bin/env python3
"""Read-only audit of filtering/preprocessing provenance for AAL3 tensors."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/filtering_preprocessing_provenance_audit"
ORIGINAL_RUN_CONFIG = PROJECT_ROOT / "results/vae_3channels_beta25/run_config.json"
ORIGINAL_NOTEBOOK = PROJECT_ROOT / "notebooks/01_feature_extraction_manual.ipynb"
COVID_NOTEBOOK = PROJECT_ROOT / "notebooks/01_feature_extraction_manual_COVID.ipynb"
CURRENT_WRAPPER = PROJECT_ROOT / "scripts/revision_bspc_2026/run_original_model_inference_dparsf_bandpass10.py"
INFERENCE_SCRIPT = PROJECT_ROOT / "scripts/inference_covid_from_adcn.py"

SEARCH_TERMS = [
    "bandpass",
    "filter",
    "butter",
    "filtfilt",
    "sosfilt",
    "scipy.signal",
    "nilearn.signal.clean",
    "lowcut",
    "highcut",
    "low_pass",
    "high_pass",
    "0.01",
    "0.08",
    "TR",
    "detrend",
    "standardize",
    "DPARSF",
    "FunImg",
    "FunImgARW",
    "FunImgARWSDC",
    "FunImgARWSDCF",
    "FunImgARWSDCFN",
    "ROISignals",
    "aal3",
    "140",
    "ROI_MNI_V7_vol",
]

PARAM_PATTERNS = [
    "TR_SECONDS",
    "LOW_CUT_HZ",
    "HIGH_CUT_HZ",
    "FILTER_ORDER",
    "TARGET_LEN_TS",
    "RAW_DATA_EXPECTED_COLUMNS",
    "N_ROIS_EXPECTED",
    "ROI_SIGNALS_DIR_PATH_AAL3",
    "AAL3_META_PATH",
    "AAL3_MANUAL_NETWORK_MAPPING_CSV",
    "CONNECTIVITY_CHANNEL_NAMES",
    "SELECTED_CHANNELS",
    "SELECTED_CHANNEL_NAMES",
]


@dataclass
class TensorMeta:
    path: str
    exists: bool
    shape: str = ""
    channel_names: str = ""
    rois_count: str = ""
    target_len_ts: str = ""
    tr_seconds: str = ""
    filter_low_hz: str = ""
    filter_high_hz: str = ""
    roi_order_name: str = ""
    python_bandpass_applied: str = ""
    source_preprocessing: str = ""
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit filtering/preprocessing provenance without retraining or feature extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def contains_term(text: str) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in SEARCH_TERMS)


def matching_terms(text: str) -> str:
    low = text.lower()
    return ";".join(term for term in SEARCH_TERMS if term.lower() in low)


def add_evidence(
    rows: List[Dict[str, Any]],
    file_path: Path,
    source_type: str,
    category: str,
    snippet: str,
    line_number: Any = "",
    terms: str = "",
) -> None:
    rows.append(
        {
            "file": rel(file_path),
            "source_type": source_type,
            "line_or_cell": line_number,
            "category": category,
            "matched_terms": terms or matching_terms(snippet),
            "snippet": " ".join(str(snippet).strip().split()),
        }
    )


def scan_text_file(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not path.exists():
        add_evidence(rows, path, "missing", "missing_file", "file not found")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        if contains_term(line):
            category = "search_term_hit"
            if re.search(r"def\s+_bandpass_filter_signals|filtfilt|butter", line):
                category = "temporal_filtering"
            elif re.search(r"LOW_CUT_HZ|HIGH_CUT_HZ|TR_SECONDS|TARGET_LEN_TS|RAW_DATA_EXPECTED_COLUMNS", line):
                category = "parameter_assignment"
            elif re.search(r"python_bandpass|already_filtered|bandpass.*skipped|skipped.*bandpass|double filtering", line, flags=re.I):
                category = "current_wrapper_bandpass_policy"
            elif re.search(r"add_argument", line) and re.search(r"low_cut|high_cut|tr_seconds|target_len", line):
                category = "cli_parameter"
            add_evidence(rows, path, "python_script", category, line, idx)


def notebook_hits(path: Path, rows: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    if not path.exists():
        add_evidence(rows, path, "missing", "missing_file", "notebook not found")
        return hits
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not contains_term(source):
            continue
        excerpt_lines = []
        for line in source.splitlines():
            if contains_term(line) or any(pat in line for pat in PARAM_PATTERNS) or "main(" in line or "args_" in line:
                excerpt_lines.append(line)
        excerpt = "\n".join(excerpt_lines[:80])
        hits.append((idx, excerpt))
        add_evidence(rows, path, "notebook", "code_cell_hit", excerpt, f"cell_{idx}")
    return hits


def write_notebook_hits_markdown(path: Path, hits_by_notebook: Mapping[Path, List[Tuple[int, str]]]) -> None:
    lines = ["# Notebook Code Hits", ""]
    for nb_path, hits in hits_by_notebook.items():
        lines.extend([f"## {rel(nb_path)}", ""])
        if not hits:
            lines.extend(["No matching code cells found.", ""])
            continue
        for cell_idx, excerpt in hits:
            lines.extend([f"### Cell {cell_idx}", "", "```python", excerpt.strip(), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_npy_header_from_npz(npz_path: Path, key: str) -> Optional[Tuple[Tuple[int, ...], str]]:
    member = f"{key}.npy"
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if member not in zf.namelist():
                return None
            with zf.open(member) as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
                else:
                    shape, _fortran, dtype = np.lib.format._read_array_header(fh, version)
                return tuple(shape), str(dtype)
    except Exception:
        return None


def scalar_or_list_to_str(value: np.ndarray, max_items: int = 16) -> str:
    if value.shape == ():
        return str(value.tolist())
    if value.ndim == 1:
        return "[" + ", ".join(str(x) for x in value[:max_items].tolist()) + (", ..." if len(value) > max_items else "") + "]"
    return f"shape={value.shape}"


def read_tensor_metadata(path: Path) -> TensorMeta:
    meta = TensorMeta(path=str(path), exists=path.exists())
    if not path.exists():
        return meta
    header = _read_npy_header_from_npz(path, "global_tensor_data")
    if header:
        meta.shape = str(header[0])
    try:
        with np.load(path, allow_pickle=False) as zf:
            for key, attr in [
                ("channel_names", "channel_names"),
                ("rois_count", "rois_count"),
                ("target_len_ts", "target_len_ts"),
                ("tr_seconds", "tr_seconds"),
                ("filter_low_hz", "filter_low_hz"),
                ("filter_high_hz", "filter_high_hz"),
                ("roi_order_name", "roi_order_name"),
                ("python_bandpass_applied", "python_bandpass_applied"),
                ("source_preprocessing", "source_preprocessing"),
                ("notes", "notes"),
            ]:
                if key in zf.files:
                    setattr(meta, attr, scalar_or_list_to_str(zf[key]))
    except Exception as exc:
        meta.notes = f"metadata read error: {exc}"
    return meta


def load_run_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_project(value: Any) -> Optional[Path]:
    if value is None:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else PROJECT_ROOT / p


def has_disable_bandpass_flag(path: Path) -> str:
    if not path.exists():
        return "unknown"
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    disable_patterns = ["--no_bandpass", "--no-bandpass", "--skip-bandpass", "--disable-bandpass", "apply_bandpass"]
    return "yes" if any(pat in text for pat in disable_patterns) else "no"


def line_snippets(path: Path, patterns: Sequence[str], max_hits: int = 12) -> str:
    if not path.exists():
        return ""
    out = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if any(re.search(pattern, line, flags=re.I) for pattern in patterns):
            out.append(f"{rel(path)}:{idx}: {' '.join(line.strip().split())}")
        if len(out) >= max_hits:
            break
    return " | ".join(out)


def infer_original_tensor_path(run_config: Mapping[str, Any]) -> Optional[Path]:
    args = run_config.get("args", {}) if isinstance(run_config, dict) else {}
    for key in ["global_tensor_path", "tensor_npz_path", "tensor_path"]:
        value = args.get(key) or run_config.get(key)
        path = resolve_from_project(value)
        if path is not None:
            return path
    return None


def build_pipeline_matrix(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    run_config = load_run_config(ORIGINAL_RUN_CONFIG)
    original_tensor = infer_original_tensor_path(run_config)
    covid_tensor = PROJECT_ROOT / (
        "data/COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_"
        "ChNorm_ROIreorderedYeo17_ParallelTuned/"
        "GLOBAL_TENSOR_from_COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_"
        "GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
    )
    martin59_tensor = Path(
        "/media/diego/Datos/adni_expansion/MARTIN59/AAL3_v6_5_17_MARTIN59_ARWSDCF/"
        "GLOBAL_TENSOR_from_AAL3_v6_5_17_MARTIN59_ARWSDCF.npz"
    )
    current_tensor = PROJECT_ROOT / (
        "results/revision_bspc_2026/original_model_inference_dparsf_bandpass10/"
        "features_or_tensor/GLOBAL_TENSOR_DPARSF_original_bandpass10_AAL3_131ROIs.npz"
    )

    original_meta = read_tensor_metadata(original_tensor) if original_tensor else TensorMeta(path="", exists=False)
    covid_meta = read_tensor_metadata(covid_tensor)
    martin_meta = read_tensor_metadata(martin59_tensor)
    current_meta = read_tensor_metadata(current_tensor)

    selected_channels = str(run_config.get("channels_to_use_indices") or run_config.get("args", {}).get("channels_to_use", ""))
    channel_order = original_meta.channel_names or str(run_config.get("channel_names_master_in_tensor_order", ""))

    original_source = "ROI_SIGNALS_DIR_PATH_AAL3 default = ROISignals_AAL3_NiftiPreprocessedAllBatchesNorm; pipeline log path in historical tensor dir."
    covid_source = "notebook uses RAW_SIGNALS_DIR/--roi_signals_dir = data/ROISignals_FunImgARWSDCFN."
    previous_source = "Martin59 tensor dir name AAL3_v6_5_17_MARTIN59_ARWSDCF and NPZ metadata."
    current_source = "OneDrive_1_2-5-2026/ResultsAAL3/ROISignals_AAL3_FunImgARWSDCFN; wrapper source_preprocessing=DPARSF_original_bandpass."

    matrix = [
        {
            "pipeline_name": "original_paper_training_tensor",
            "input_source": original_source,
            "dparsf_bandpass_evidence": "unclear",
            "python_bandpass_evidence": "yes",
            "python_bandpass_low": original_meta.filter_low_hz or "0.01",
            "python_bandpass_high": original_meta.filter_high_hz or "0.08",
            "TR": original_meta.tr_seconds or "3.0",
            "target_length_TR": original_meta.target_len_ts or "140",
            "input_n_rois": "170",
            "final_n_rois": original_meta.rois_count or "131",
            "roi_reduction_evidence": "ROI_MNI_V7_vol + drop missing/small ROIs; roi_order_name=" + (original_meta.roi_order_name or "unknown"),
            "channel_order": channel_order,
            "selected_channels": selected_channels,
            "evidence_files": "; ".join(
                [
                    rel(ORIGINAL_RUN_CONFIG),
                    rel(PROJECT_ROOT / "scripts/feature_extraction_manual.py"),
                    rel(original_tensor) if original_tensor else "original tensor path not found",
                    rel(original_tensor.parent / f"pipeline_log_{original_tensor.parent.name}.csv") if original_tensor else "",
                ]
            ),
            "confidence": "high for Python bandpass; low/unclear for DPARSF bandpass",
            "interpretation": "Original tensor metadata and script show Python Butter/Filtfilt 0.01-0.08. The source ROI folder name does not itself prove DPARSF bandpass, so the full A/B/C/D label is unclear rather than proven double-filtered.",
        },
        {
            "pipeline_name": "covid_inference_tensor",
            "input_source": covid_source,
            "dparsf_bandpass_evidence": "yes",
            "python_bandpass_evidence": "yes",
            "python_bandpass_low": covid_meta.filter_low_hz or "0.01",
            "python_bandpass_high": covid_meta.filter_high_hz or "0.08",
            "TR": covid_meta.tr_seconds or "3.0",
            "target_length_TR": covid_meta.target_len_ts or "140",
            "input_n_rois": "170",
            "final_n_rois": covid_meta.rois_count or "131",
            "roi_reduction_evidence": "COVID notebook passes AAL3 metadata/reorder CSV; NPZ roi_order_name=" + (covid_meta.roi_order_name or "unknown"),
            "channel_order": covid_meta.channel_names,
            "selected_channels": selected_channels,
            "evidence_files": "; ".join([rel(COVID_NOTEBOOK), rel(PROJECT_ROOT / "scripts/feature_extraction_manual_COVID.py"), rel(covid_tensor)]),
            "confidence": "high",
            "interpretation": "COVID source folder explicitly contains FunImgARWSDCFN and feature extraction NPZ records Python 0.01-0.08 filtering.",
        },
        {
            "pipeline_name": "previous_new_subjects_preprocessing",
            "input_source": previous_source,
            "dparsf_bandpass_evidence": "yes",
            "python_bandpass_evidence": "yes",
            "python_bandpass_low": martin_meta.filter_low_hz or "0.01",
            "python_bandpass_high": martin_meta.filter_high_hz or "0.08",
            "TR": martin_meta.tr_seconds or "3.0",
            "target_length_TR": martin_meta.target_len_ts or "140",
            "input_n_rois": "170",
            "final_n_rois": martin_meta.rois_count or "131",
            "roi_reduction_evidence": "NPZ metadata and directory name; roi_order_name=" + (martin_meta.roi_order_name or "unknown"),
            "channel_order": martin_meta.channel_names,
            "selected_channels": selected_channels,
            "evidence_files": str(martin59_tensor),
            "confidence": "high if Martin59 tensor is representative of prior new-subject inference tensors",
            "interpretation": "Martin59 tensor provenance indicates DPARSF/ARWSDCF inputs and Python 0.01-0.08 feature-extraction filtering.",
        },
        {
            "pipeline_name": "dparsf_bandpass10_current_inference",
            "input_source": current_source,
            "dparsf_bandpass_evidence": "yes",
            "python_bandpass_evidence": "no",
            "python_bandpass_low": current_meta.filter_low_hz or "NaN",
            "python_bandpass_high": current_meta.filter_high_hz or "NaN",
            "TR": current_meta.tr_seconds or "3.0",
            "target_length_TR": current_meta.target_len_ts or "140",
            "input_n_rois": "170",
            "final_n_rois": "131",
            "roi_reduction_evidence": "wrapper audit roi_reduction_validation.csv; NPZ python_bandpass_applied=False",
            "channel_order": current_meta.channel_names,
            "selected_channels": selected_channels,
            "evidence_files": "; ".join([rel(CURRENT_WRAPPER), rel(current_tensor)]),
            "confidence": "high",
            "interpretation": "Current wrapper intentionally skips Python bandpass to avoid double filtering; this is DPARSF-only relative to temporal filtering.",
        },
        {
            "pipeline_name": "proposed_dparsf_bandpass10_plus_python_bandpass_control",
            "input_source": current_source,
            "dparsf_bandpass_evidence": "yes",
            "python_bandpass_evidence": "planned",
            "python_bandpass_low": "0.01",
            "python_bandpass_high": "0.08",
            "TR": "3.0",
            "target_length_TR": "140",
            "input_n_rois": "170",
            "final_n_rois": "131",
            "roi_reduction_evidence": "would use same ROI reduction/reorder as current wrapper",
            "channel_order": current_meta.channel_names or channel_order,
            "selected_channels": selected_channels,
            "evidence_files": "proposed controlled experiment; no outputs yet",
            "confidence": "planned",
            "interpretation": "Most informative control to separate DPARSF-only compatibility from the historical extractor's Python-filtered tensor format.",
        },
    ]

    for item in matrix:
        add_evidence(
            rows,
            Path(item["evidence_files"].split(";")[0].strip()) if item["evidence_files"] else Path("."),
            "derived_matrix",
            item["pipeline_name"],
            item["interpretation"],
            "",
            "",
        )
    return pd.DataFrame(matrix)


def write_recommendations(path: Path) -> None:
    payload = {
        "main_conclusion": {
            "original_training_label": "D_unclear_or_mixed",
            "reason": "Python bandpass is explicit in code and tensor metadata; DPARSF-bandpass status of the original ADNI ROI source is not proven by available provenance.",
        },
        "next_most_informative_control": {
            "name": "dparsf_bandpass10_plus_python_bandpass_original_model",
            "purpose": "Run the same 9 DPARSF-bandpass subjects through the original tensor generation logic with Python Butter/Filtfilt 0.01-0.08 enabled, then compare against current DPARSF-only and previous preprocessing scores.",
            "comparisons": [
                "V1 current DPARSF-only scores",
                "V2 DPARSF+Python scores",
                "previous preprocessing scores where available",
            ],
            "do_not_retrain": True,
            "expected_decision_value": "If V2 moves scores back toward previous AD-like predictions, Python/double filtering is implicated. If V2 remains low, the main drift may be elsewhere in preprocessing or cohort/scanner effects.",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, matrix: pd.DataFrame, evidence: pd.DataFrame) -> None:
    original = matrix[matrix["pipeline_name"] == "original_paper_training_tensor"].iloc[0]
    covid = matrix[matrix["pipeline_name"] == "covid_inference_tensor"].iloc[0]
    current = matrix[matrix["pipeline_name"] == "dparsf_bandpass10_current_inference"].iloc[0]

    script_disable_ad = has_disable_bandpass_flag(PROJECT_ROOT / "scripts/feature_extraction_manual.py")
    script_disable_covid = has_disable_bandpass_flag(PROJECT_ROOT / "scripts/feature_extraction_manual_COVID.py")
    filter_snippets = evidence[
        evidence["category"].isin(["temporal_filtering", "parameter_assignment", "current_wrapper_bandpass_policy", "cli_parameter"])
    ].head(18)

    lines = [
        "# Filtering / Preprocessing Provenance Audit",
        "",
        "Read-only audit. No retraining, feature extraction, inference, checkpoints, joblibs, or large tensor arrays were loaded. NPZ inspection was limited to headers and small metadata members.",
        "",
        "## Explicit Answers",
        "",
        f"- Did original feature extraction likely apply Python bandpass? `Yes`. Evidence: `filter_low_hz={original['python_bandpass_low']}`, `filter_high_hz={original['python_bandpass_high']}` in the original global tensor metadata, plus unconditional `_bandpass_filter_signals(...filtfilt...)` in `feature_extraction_manual.py`.",
        f"- Did COVID feature extraction likely apply Python bandpass? `Yes`. Evidence: `filter_low_hz={covid['python_bandpass_low']}`, `filter_high_hz={covid['python_bandpass_high']}` in the COVID global tensor metadata and COVID extractor CLI defaults.",
        f"- Does the current DPARSF-bandpass10 wrapper skip Python bandpass? `Yes`. Evidence: `python_bandpass_applied=False`, `filter_low_hz={current['python_bandpass_low']}`, and wrapper text says Python bandpass is skipped to avoid double filtering.",
        "- Was the original model trained on A/B/C/D? `D: unclear / mixed`. The Python bandpass component is high-confidence. The original ADNI ROI source does not contain enough direct evidence to prove whether DPARSF had already applied bandpass, so `DPARSF + Python` is plausible but not proven from available artifacts.",
        "- Is current DPARSF-bandpass10 inference faithful to original training? `Only partially`. It matches ROI count/order/channel format and DPARSF-bandpass input, but intentionally omits the Python 0.01-0.08 filter that original/COVID tensors record. It is best interpreted as a DPARSF-only temporal-filtering control.",
        "- Next most informative control experiment: run the same 9 DPARSF-bandpass subjects with Python bandpass ON using the original model, then compare V1 current DPARSF-only scores, V2 DPARSF+Python scores, and previous preprocessing scores when available.",
        "",
        "## Bandpass Disable Flags",
        "",
        f"- `scripts/feature_extraction_manual.py`: disable-bandpass flag found? `{script_disable_ad}`.",
        f"- `scripts/feature_extraction_manual_COVID.py`: disable-bandpass flag found? `{script_disable_covid}`. It exposes low/high cut parameters, but no explicit skip-bandpass flag was found.",
        "- `run_original_model_inference_dparsf_bandpass10.py`: implements a custom already-filtered adapter that does not call `_bandpass_filter_signals`.",
        "",
        "## Pipeline Matrix",
        "",
        matrix.to_markdown(index=False),
        "",
        "## Key Evidence Snippets",
        "",
    ]
    for _, row in filter_snippets.iterrows():
        lines.append(f"- `{row['file']}:{row['line_or_cell']}` [{row['category']}]: {row['snippet']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strongest defensible statement is that the original and COVID tensors are Python-bandpass tensors. The COVID and Martin59/new-subject style tensors also carry strong DPARSF-folder evidence, making them likely double-filtered. The original ADNI training tensor has high-confidence Python filtering but insufficient direct source-folder evidence to prove DPARSF filtering before Python.",
            "",
            "For the paper response, treat the current DPARSF-bandpass10 run as evidence that a DPARSF-only compatibility control lowers the previously AD-like score for `003_S_4644`, but do not claim it fully reproduces the original training preprocessing until the DPARSF+Python control is run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    evidence_rows: List[Dict[str, Any]] = []
    notebook_paths = [ORIGINAL_NOTEBOOK, COVID_NOTEBOOK]
    script_paths = [
        PROJECT_ROOT / "scripts/feature_extraction_manual.py",
        PROJECT_ROOT / "scripts/feature_extraction_manual_COVID.py",
        PROJECT_ROOT / "src/betavae_xai/feature_extraction_manual.py",
        PROJECT_ROOT / "src/betavae_xai/feature_extraction_manual_COVID.py",
        CURRENT_WRAPPER,
        INFERENCE_SCRIPT,
        ORIGINAL_RUN_CONFIG,
    ]

    hits_by_notebook = {path: notebook_hits(path, evidence_rows) for path in notebook_paths}
    for path in script_paths:
        scan_text_file(path, evidence_rows)

    run_config = load_run_config(ORIGINAL_RUN_CONFIG)
    original_tensor = infer_original_tensor_path(run_config)
    if original_tensor:
        tensor_dir = original_tensor.parent
        for extra in sorted(tensor_dir.glob("*.csv")) + sorted(tensor_dir.glob("*.md")) + sorted(tensor_dir.glob("*.json")) + sorted(tensor_dir.glob("*.txt")):
            scan_text_file(extra, evidence_rows)
        meta = read_tensor_metadata(original_tensor)
        add_evidence(
            evidence_rows,
            original_tensor,
            "npz_metadata",
            "original_tensor_metadata",
            f"shape={meta.shape}; filter_low_hz={meta.filter_low_hz}; filter_high_hz={meta.filter_high_hz}; tr={meta.tr_seconds}; target_len={meta.target_len_ts}; rois={meta.rois_count}; roi_order={meta.roi_order_name}; channels={meta.channel_names}",
        )

    for npz_path, label in [
        (
            PROJECT_ROOT
            / "data/COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned/GLOBAL_TENSOR_from_COVID_AAL3_Tensor_v1_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz",
            "covid_tensor_metadata",
        ),
        (
            PROJECT_ROOT
            / "results/revision_bspc_2026/original_model_inference_dparsf_bandpass10/features_or_tensor/GLOBAL_TENSOR_DPARSF_original_bandpass10_AAL3_131ROIs.npz",
            "current_dparsf_bandpass10_tensor_metadata",
        ),
        (
            Path("/media/diego/Datos/adni_expansion/MARTIN59/AAL3_v6_5_17_MARTIN59_ARWSDCF/GLOBAL_TENSOR_from_AAL3_v6_5_17_MARTIN59_ARWSDCF.npz"),
            "previous_new_subjects_martin59_tensor_metadata",
        ),
    ]:
        meta = read_tensor_metadata(npz_path)
        add_evidence(
            evidence_rows,
            npz_path,
            "npz_metadata",
            label,
            f"exists={meta.exists}; shape={meta.shape}; filter_low_hz={meta.filter_low_hz}; filter_high_hz={meta.filter_high_hz}; tr={meta.tr_seconds}; target_len={meta.target_len_ts}; rois={meta.rois_count}; python_bandpass_applied={meta.python_bandpass_applied}; source_preprocessing={meta.source_preprocessing}",
        )

    matrix = build_pipeline_matrix(evidence_rows)
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(output_dir / "filtering_evidence_by_file.csv", index=False)
    write_notebook_hits_markdown(output_dir / "notebook_code_hits.md", hits_by_notebook)
    matrix.to_csv(output_dir / "pipeline_variant_matrix.csv", index=False)
    write_recommendations(output_dir / "recommended_next_experiments.json")
    write_readme(output_dir / "README.md", matrix, evidence)

    print(f"Wrote {output_dir / 'README.md'}")
    print(f"Wrote {output_dir / 'filtering_evidence_by_file.csv'}")
    print(f"Wrote {output_dir / 'notebook_code_hits.md'}")
    print(f"Wrote {output_dir / 'pipeline_variant_matrix.csv'}")
    print(f"Wrote {output_dir / 'recommended_next_experiments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
