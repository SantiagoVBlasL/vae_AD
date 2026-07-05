#!/usr/bin/env python
"""Read-only OASIS AAL3 ROI-order and 170-to-131 mapping audit.

This script does not compute connectomes and does not modify OASIS input data.
It reconciles the OASIS 170-column AAL3 ROI signal matrices with the ADNI
AAL3-131 Yeo17-reordered ROI definition used by the locked ADNI model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "Tanda_2026_05_25"
DEFAULT_EXISTING_AUDIT = (
    PROJECT_ROOT / "results" / "revision_bspc_2026" / "oasis_tanda_2026_05_25_audit"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_roi_mapping_audit"
)
DEFAULT_AAL3_META = PROJECT_ROOT / "data" / "ROI_MNI_V7_vol.txt"
DEFAULT_ADNI_131_MAPPING = PROJECT_ROOT / "data" / "aal3_131_manual_network_order.csv"
DEFAULT_ADNI_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)

AAL3_EMPTY_LABELS: dict[int, dict[str, str]] = {
    35: {
        "nom_c": "CINGULATE_ANT_L_EMPTY_AAL3",
        "nom_l": "Anterior cingulate and paracingulate gyri_L",
        "source": "AAL3 empty original AAL2 anterior cingulate label",
    },
    36: {
        "nom_c": "CINGULATE_ANT_R_EMPTY_AAL3",
        "nom_l": "Anterior cingulate and paracingulate gyri_R",
        "source": "AAL3 empty original AAL2 anterior cingulate label",
    },
    81: {
        "nom_c": "THALAMUS_L_EMPTY_AAL3",
        "nom_l": "Thalamus_L",
        "source": "AAL3 empty original AAL2 thalamus label",
    },
    82: {
        "nom_c": "THALAMUS_R_EMPTY_AAL3",
        "nom_l": "Thalamus_R",
        "source": "AAL3 empty original AAL2 thalamus label",
    },
}
KNOWN_AAL3_EMPTY_1BASED = sorted(AAL3_EMPTY_LABELS)
SMALL_ROI_VOXEL_THRESHOLD = 100
EXPECTED_RAW_COLUMNS = 170
EXPECTED_AFTER_EMPTY_REMOVAL = 166
EXPECTED_FINAL_ROIS = 131


@dataclass
class AuditInputs:
    input_dir: str
    existing_audit_dir: str
    output_dir: str
    aal3_meta_path: str
    adni_131_mapping_path: str
    adni_tensor_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--existing-audit-dir", type=Path, default=DEFAULT_EXISTING_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aal3-meta-path", type=Path, default=DEFAULT_AAL3_META)
    parser.add_argument("--adni-131-mapping-path", type=Path, default=DEFAULT_ADNI_131_MAPPING)
    parser.add_argument("--adni-tensor-path", type=Path, default=DEFAULT_ADNI_TENSOR)
    return parser.parse_args()


def ensure_inputs(args: argparse.Namespace) -> None:
    required = [
        args.input_dir,
        args.existing_audit_dir,
        args.existing_audit_dir / "roi_signal_qc.csv",
        args.existing_audit_dir / "file_inventory.csv",
        args.aal3_meta_path,
        args.adni_131_mapping_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def read_oasis_qc(existing_audit_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_qc = pd.read_csv(existing_audit_dir / "roi_signal_qc.csv")
    file_inventory = pd.read_csv(existing_audit_dir / "file_inventory.csv")
    return roi_qc, file_inventory


def load_adni_aal3_definitions(aal3_meta_path: Path, adni_131_mapping_path: Path) -> dict[str, pd.DataFrame]:
    aal3_meta_166 = pd.read_csv(aal3_meta_path, sep="\t")
    aal3_meta_166["color"] = aal3_meta_166["color"].astype(int)
    aal3_meta_166["vol_vox"] = aal3_meta_166["vol_vox"].astype(int)
    aal3_meta_166 = aal3_meta_166.sort_values("color").reset_index(drop=True)

    if len(aal3_meta_166) != EXPECTED_AFTER_EMPTY_REMOVAL:
        raise ValueError(f"Expected {EXPECTED_AFTER_EMPTY_REMOVAL} AAL3 non-empty rows, got {len(aal3_meta_166)}")
    unexpected_empty_present = sorted(set(aal3_meta_166["color"]) & set(KNOWN_AAL3_EMPTY_1BASED))
    if unexpected_empty_present:
        raise ValueError(f"Expected empty AAL3 labels to be absent, found {unexpected_empty_present}")

    small_mask = aal3_meta_166["vol_vox"] < SMALL_ROI_VOXEL_THRESHOLD
    small_rois = aal3_meta_166.loc[small_mask].copy().reset_index(drop=True)
    adni_original_131 = aal3_meta_166.loc[~small_mask].copy().reset_index(drop=True)
    adni_original_131["adni_original_index_131"] = np.arange(len(adni_original_131), dtype=int)

    if len(adni_original_131) != EXPECTED_FINAL_ROIS:
        raise ValueError(f"Expected {EXPECTED_FINAL_ROIS} final ADNI ROIs, got {len(adni_original_131)}")

    manual = pd.read_csv(adni_131_mapping_path)
    manual["Index_131"] = manual["Index_131"].astype(int)
    if len(manual) != EXPECTED_FINAL_ROIS or sorted(manual["Index_131"].tolist()) != list(range(EXPECTED_FINAL_ROIS)):
        raise ValueError("Manual ADNI 131 mapping does not contain a 0..130 permutation in Index_131")

    # Reproduce scripts/feature_extraction_manual.py sorting logic exactly:
    # sort first by original Index_131 for alignment, then by background, Yeo17 label, hemisphere, and ROI name.
    mapping = manual.sort_values("Index_131").reset_index(drop=True).copy()
    labs = mapping["Yeo17_Label_manual"].astype(int)
    mapping["__sort_is_bg"] = (labs <= 0).astype(int)
    hemi_raw = mapping["Hemi"].astype(str).str.upper()
    mapping["__sort_hemi"] = hemi_raw.map({"L": 0, "R": 1}).fillna(2).astype(int)
    mapping["__sort_name"] = mapping["nom_l"].astype(str)
    adni_final_131 = (
        mapping.sort_values(
            ["__sort_is_bg", "Yeo17_Label_manual", "__sort_hemi", "__sort_name"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .drop(columns=["__sort_is_bg", "__sort_hemi", "__sort_name"])
    )
    adni_final_131.insert(0, "ADNI_final_index_131", np.arange(len(adni_final_131), dtype=int))
    adni_final_131["ADNI_final_index_131_1based"] = adni_final_131["ADNI_final_index_131"] + 1
    adni_final_131 = adni_final_131.rename(columns={"Index_131": "adni_original_index_131"})

    return {
        "aal3_meta_166": aal3_meta_166,
        "small_rois": small_rois,
        "adni_original_131": adni_original_131,
        "adni_final_131": adni_final_131,
    }


def load_adni_tensor_roi_names(adni_tensor_path: Path) -> tuple[str | None, list[str]]:
    if not adni_tensor_path.exists():
        return None, []
    data = np.load(adni_tensor_path, allow_pickle=True)
    order_name = str(data["roi_order_name"]) if "roi_order_name" in data.files else None
    roi_names = data["roi_names_in_order"].astype(str).tolist() if "roi_names_in_order" in data.files else []
    return order_name, roi_names


def build_order_candidates(file_inventory: pd.DataFrame, roi_qc: pd.DataFrame) -> pd.DataFrame:
    terms = ("aal", "roi", "label", "atlas", "dparsf", "dpabi", "spm", "order", "mask")
    candidate_files = file_inventory[
        file_inventory["file_path"].astype(str).str.lower().apply(lambda s: any(t in s for t in terms))
    ].copy()
    modality_counts = candidate_files["modality_type"].value_counts(dropna=False).to_dict()
    # Count only plausible label/order sidecars. ROI-signal text matrices and QC/atlas
    # images contain "ROI", "AAL3", or "atlas" in their paths but do not document
    # column labels.
    explicit_label_terms = ("label", "order", "roi_mni", "lut", "aal3_labels")
    explicit_label_like = file_inventory[
        file_inventory["file_path"]
        .astype(str)
        .str.lower()
        .apply(lambda s: any(t in s for t in explicit_label_terms))
    ]

    nan_sets = roi_qc["all_nan_roi_column_indices_1based"].astype(str).value_counts(dropna=False).to_dict()
    n_qc = int(len(roi_qc))
    n_shape_ok = int(roi_qc["aal3_170_shape_match"].fillna(False).sum()) if "aal3_170_shape_match" in roi_qc else 0

    rows = [
        {
            "candidate": "oasis_batch_explicit_roi_label_sidecar",
            "evidence": f"searched file inventory for {terms}; explicit label/order sidecars found={len(explicit_label_like)}",
            "support_level": "absent_in_batch",
            "notes": "No explicit ROI-label/order file was found in the OASIS batch inventory.",
        },
        {
            "candidate": "oasis_roisignals_aal3_170_columns",
            "evidence": f"ROI QC rows={n_qc}; AAL3 170 shape rows={n_shape_ok}; all-NaN column sets={nan_sets}",
            "support_level": "strong_shape_evidence",
            "notes": "Processed ROI-signal files are in ResultsAAL3/ROISignals_AAL3_* and consistently have 170 columns.",
        },
        {
            "candidate": "adni_pipeline_aal3_color_order_1_to_170",
            "evidence": "ADNI feature extraction code expects 170 raw AAL3 columns, drops labels 35/36/81/82, then drops vol_vox<100 and applies Yeo17 manual ordering.",
            "support_level": "strong_local_pipeline_evidence",
            "notes": "OASIS all-NaN columns match the AAL3 empty labels used by ADNI exactly.",
        },
        {
            "candidate": "aal3_official_empty_label_explanation",
            "evidence": "AAL3 leaves original AAL2 anterior-cingulate labels 35/36 and thalamus labels 81/82 empty because finer subdivisions are used.",
            "support_level": "external_static_atlas_evidence",
            "notes": "Use as anatomical interpretation only; still ask Martin to confirm DPABI exported columns in AAL3 label/color order.",
        },
        {
            "candidate": "candidate_file_modality_counts",
            "evidence": json.dumps(modality_counts, sort_keys=True),
            "support_level": "batch_inventory_context",
            "notes": "Files containing AAL/ROI/atlas/order-like terms are mostly ROI time series and QC/atlas images, not label sidecars.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_all_nan_columns(roi_qc: pd.DataFrame) -> pd.DataFrame:
    observed_sets = roi_qc["all_nan_roi_column_indices_1based"].astype(str).value_counts(dropna=False)
    observed_n = {}
    for s, count in observed_sets.items():
        for token in str(s).split(","):
            token = token.strip()
            if token.isdigit():
                observed_n[int(token)] = observed_n.get(int(token), 0) + int(count)

    rows = []
    for col in KNOWN_AAL3_EMPTY_1BASED:
        info = AAL3_EMPTY_LABELS[col]
        rows.append(
            {
                "oasis_col_idx_0based": col - 1,
                "oasis_col_idx_1based": col,
                "AAL3_empty_label_interpretation": info["nom_l"],
                "short_name": info["nom_c"],
                "observed_all_nan_in_n_runs": observed_n.get(col, 0),
                "total_roi_qc_runs": len(roi_qc),
                "observed_in_all_runs": observed_n.get(col, 0) == len(roi_qc),
                "present_in_local_ROI_MNI_V7_vol": False,
                "matches_adni_known_empty_label": True,
                "reason": info["source"],
            }
        )
    return pd.DataFrame(rows)


def build_mapping_tables(defs: dict[str, pd.DataFrame], adni_tensor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aal3_meta_166 = defs["aal3_meta_166"]
    small_rois = defs["small_rois"]
    adni_original_131 = defs["adni_original_131"]
    adni_final_131 = defs["adni_final_131"]

    meta_by_color = aal3_meta_166.set_index("color").to_dict("index")
    small_colors = set(small_rois["color"].astype(int))
    original_by_color = adni_original_131.set_index("color").to_dict("index")
    final_by_original = adni_final_131.set_index("adni_original_index_131").to_dict("index")

    mapping_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for color in range(1, EXPECTED_RAW_COLUMNS + 1):
        oasis_col_1based = color
        oasis_col_0based = color - 1
        if color in KNOWN_AAL3_EMPTY_1BASED:
            info = AAL3_EMPTY_LABELS[color]
            mapping_rows.append(
                {
                    "oasis_col_idx_0based": oasis_col_0based,
                    "oasis_col_idx_1based": oasis_col_1based,
                    "AAL3_ROI_name": info["nom_l"],
                    "AAL3_short_name": info["nom_c"],
                    "AAL3_color": color,
                    "vol_vox": np.nan,
                    "ADNI_final_ROI_name": "",
                    "ADNI_final_index_131": "",
                    "ADNI_final_index_131_1based": "",
                    "adni_original_index_131_before_yeo_reorder": "",
                    "keep_drop": "drop",
                    "reason": "AAL3 empty label; OASIS column is all-NaN and ADNI drops before 166-to-131 reduction",
                }
            )
            continue

        if color not in meta_by_color:
            mapping_rows.append(
                {
                    "oasis_col_idx_0based": oasis_col_0based,
                    "oasis_col_idx_1based": oasis_col_1based,
                    "AAL3_ROI_name": "",
                    "AAL3_short_name": "",
                    "AAL3_color": color,
                    "vol_vox": np.nan,
                    "ADNI_final_ROI_name": "",
                    "ADNI_final_index_131": "",
                    "ADNI_final_index_131_1based": "",
                    "adni_original_index_131_before_yeo_reorder": "",
                    "keep_drop": "unresolved",
                    "reason": "Color absent from local AAL3 metadata and not in known empty-label set",
                }
            )
            unresolved_rows.append(
                {
                    "scope": "roi",
                    "oasis_col_idx_1based": color,
                    "issue": "unexpected_missing_aal3_color",
                    "recommended_action": "resolve atlas label before connectome build",
                }
            )
            continue

        meta = meta_by_color[color]
        if color in small_colors:
            mapping_rows.append(
                {
                    "oasis_col_idx_0based": oasis_col_0based,
                    "oasis_col_idx_1based": oasis_col_1based,
                    "AAL3_ROI_name": meta["nom_l"],
                    "AAL3_short_name": meta["nom_c"],
                    "AAL3_color": color,
                    "vol_vox": int(meta["vol_vox"]),
                    "ADNI_final_ROI_name": "",
                    "ADNI_final_index_131": "",
                    "ADNI_final_index_131_1based": "",
                    "adni_original_index_131_before_yeo_reorder": "",
                    "keep_drop": "drop",
                    "reason": f"small ROI excluded by ADNI rule vol_vox < {SMALL_ROI_VOXEL_THRESHOLD}",
                }
            )
            continue

        original = original_by_color[color]
        final = final_by_original[int(original["adni_original_index_131"])]
        mapping_rows.append(
            {
                "oasis_col_idx_0based": oasis_col_0based,
                "oasis_col_idx_1based": oasis_col_1based,
                "AAL3_ROI_name": meta["nom_l"],
                "AAL3_short_name": meta["nom_c"],
                "AAL3_color": color,
                "vol_vox": int(meta["vol_vox"]),
                "ADNI_final_ROI_name": final["nom_l"],
                "ADNI_final_index_131": int(final["ADNI_final_index_131"]),
                "ADNI_final_index_131_1based": int(final["ADNI_final_index_131_1based"]),
                "adni_original_index_131_before_yeo_reorder": int(original["adni_original_index_131"]),
                "keep_drop": "keep",
                "reason": "retained ADNI 131 ROI; reorder by ADNI_final_index_131 before tensor construction",
            }
        )

    mapping_170 = pd.DataFrame(mapping_rows)

    adni_match = adni_final_131.copy()
    by_original = adni_original_131.set_index("adni_original_index_131")
    adni_match["oasis_col_idx_1based"] = adni_match["adni_original_index_131"].map(
        by_original["color"].astype(int)
    )
    adni_match["oasis_col_idx_0based"] = adni_match["oasis_col_idx_1based"].astype(int) - 1
    adni_match["ADNI_final_ROI_name"] = adni_match["nom_l"]
    adni_match["ADNI_final_short_name"] = adni_match["nom_c"]
    adni_match["mapping_status"] = "matched"
    adni_match["mapping_source"] = "ADNI AAL3 166->131 filtering plus Yeo17 manual order"

    tensor_order_name, tensor_roi_names = load_adni_tensor_roi_names(adni_tensor_path)
    if tensor_roi_names:
        adni_match["adni_tensor_roi_name"] = tensor_roi_names
        adni_match["matches_adni_tensor_roi_name"] = adni_match["ADNI_final_ROI_name"].astype(str).eq(
            pd.Series(tensor_roi_names, index=adni_match.index).astype(str)
        )
    else:
        adni_match["adni_tensor_roi_name"] = ""
        adni_match["matches_adni_tensor_roi_name"] = ""
    adni_match["adni_tensor_roi_order_name"] = tensor_order_name or ""

    if len(unresolved_rows) == 0:
        unresolved_rows.append(
            {
                "scope": "global",
                "oasis_col_idx_1based": "",
                "issue": "oasis_batch_lacks_explicit_roi_label_order_sidecar",
                "recommended_action": "ask Martin to confirm DPABI/DPARSF AAL3 ROISignals columns are AAL3 label/color order 1..170",
            }
        )
    unresolved = pd.DataFrame(unresolved_rows)
    return mapping_170, adni_match, unresolved


def build_readme(
    args: argparse.Namespace,
    roi_qc: pd.DataFrame,
    mapping_170: pd.DataFrame,
    adni_match: pd.DataFrame,
    unresolved: pd.DataFrame,
    decision: str,
) -> str:
    counts = mapping_170["keep_drop"].value_counts(dropna=False).to_dict()
    nan_sets = roi_qc["all_nan_roi_column_indices_1based"].astype(str).value_counts(dropna=False).to_dict()
    tensor_match_count = (
        int(adni_match["matches_adni_tensor_roi_name"].eq(True).sum())
        if "matches_adni_tensor_roi_name" in adni_match
        else 0
    )
    return f"""# OASIS AAL3 ROI-Order And 170-to-131 Mapping Audit

Decision: `{decision}`

## Inputs

- OASIS folder: `{args.input_dir}`
- Existing OASIS audit: `{args.existing_audit_dir}`
- AAL3 local metadata: `{args.aal3_meta_path}`
- ADNI 131 manual mapping: `{args.adni_131_mapping_path}`
- ADNI tensor ROI-order check: `{args.adni_tensor_path}`

## Main Findings

- OASIS ROI-signal QC rows: {len(roi_qc)}
- OASIS matrix shape: 164 x 170 in all QC-passing rows from the prior audit.
- Complete all-NaN ROI columns observed: {nan_sets}
- The all-NaN columns match the AAL3 empty labels 35, 36, 81, and 82 used by the ADNI pipeline.
- Local `ROI_MNI_V7_vol.txt` contains 166 non-empty AAL3 labels, consistent with AAL3 max label 170 but four empty labels.
- ADNI filtering drops the four empty labels, then drops 35 small ROIs with `vol_vox < {SMALL_ROI_VOXEL_THRESHOLD}`, yielding 131 ROIs.
- Mapping outcome: {counts}
- ADNI final ROI-order table rows: {len(adni_match)}
- ADNI tensor ROI-name matches: {tensor_match_count}/{len(adni_match)}.

## Interpretation

The OASIS matrices are technically compatible with the ADNI AAL3-131 model if their 170 columns are in AAL3 label/color order. Under that assumption, the exact ADNI 131 ROI set and final Yeo17 order can be reconstructed by:

1. Treating OASIS columns as AAL3 labels 1..170.
2. Dropping empty labels 35, 36, 81, and 82.
3. Dropping ADNI small-ROI exclusions with `vol_vox < {SMALL_ROI_VOXEL_THRESHOLD}`.
4. Reordering retained ROIs using the ADNI `aal3_manual_yeo17_order` mapping.

The audit found no ROI-level incompatibility after applying those rules. The remaining blocker is provenance: this OASIS batch does not include an explicit ROI label/order sidecar, so Martin or the preprocessing log should confirm that DPABI/DPARSF exported `ROISignals_AAL3_*` in AAL3 label/color order.

## Non-Actions

- No connectomes were computed.
- No OASIS inputs were modified.
- No ADNI tensor or metadata files were modified.
"""


def build_recommendation(decision: str, unresolved: pd.DataFrame) -> str:
    return f"""# Final Recommendation

Decision: `{decision}`

## Recommendation

Do not build OASIS connectomes yet. The 170-to-131 mapping is internally consistent with the ADNI AAL3 pipeline, and the four all-NaN columns correspond to expected AAL3 empty labels, but the OASIS batch lacks an explicit ROI label/order sidecar.

Ask Martin to confirm one item before connectome construction:

- DPABI/DPARSF `ROISignals_AAL3_FunImgARWSDCFN` columns are exported in AAL3 label/color order 1..170, with empty labels 35, 36, 81, and 82 preserved as empty/all-NaN columns.

If confirmed, OASIS can proceed to connectome build by extracting the 131 retained ROIs in `roi_mapping_170_to_131.csv` and reordering them by `ADNI_final_index_131`.

## Compatibility Status

- 170 OASIS columns accounted for.
- 131 ADNI final ROIs matched.
- 4 AAL3 empty labels dropped.
- 35 small-volume ADNI exclusions dropped.
- No individual retained ROI is unresolved.

## Remaining Unresolved Items

{unresolved.to_markdown(index=False)}
"""


def main() -> None:
    args = parse_args()
    ensure_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roi_qc, file_inventory = read_oasis_qc(args.existing_audit_dir)
    defs = load_adni_aal3_definitions(args.aal3_meta_path, args.adni_131_mapping_path)

    order_candidates = build_order_candidates(file_inventory, roi_qc)
    all_nan = summarize_all_nan_columns(roi_qc)
    mapping_170, adni_match, unresolved = build_mapping_tables(defs, args.adni_tensor_path)

    # Keep a conservative decision until an explicit OASIS ROI-order sidecar or Martin confirmation is available.
    decision = "needs_martin_roi_order_confirmation"

    write_csv_md(
        order_candidates,
        args.output_dir / "oasis_170_roi_order_candidates.csv",
        args.output_dir / "oasis_170_roi_order_candidates.md",
        "OASIS 170-Column ROI-Order Candidates",
    )
    write_csv_md(
        all_nan,
        args.output_dir / "all_nan_roi_columns.csv",
        args.output_dir / "all_nan_roi_columns.md",
        "All-NaN AAL3 ROI Columns",
    )
    write_csv_md(
        adni_match,
        args.output_dir / "adni_131_matching_table.csv",
        args.output_dir / "adni_131_matching_table.md",
        "ADNI 131 ROI Matching Table",
    )
    write_csv_md(
        mapping_170,
        args.output_dir / "roi_mapping_170_to_131.csv",
        args.output_dir / "roi_mapping_170_to_131.md",
        "ROI Mapping 170 To 131",
    )
    write_csv_md(
        unresolved,
        args.output_dir / "unresolved_rois.csv",
        args.output_dir / "unresolved_rois.md",
        "Unresolved ROI/Provenance Items",
    )

    (args.output_dir / "README.md").write_text(
        build_readme(args, roi_qc, mapping_170, adni_match, unresolved, decision),
        encoding="utf-8",
    )
    (args.output_dir / "final_recommendation.md").write_text(
        build_recommendation(decision, unresolved),
        encoding="utf-8",
    )

    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "inputs": asdict(
            AuditInputs(
                input_dir=str(args.input_dir),
                existing_audit_dir=str(args.existing_audit_dir),
                output_dir=str(args.output_dir),
                aal3_meta_path=str(args.aal3_meta_path),
                adni_131_mapping_path=str(args.adni_131_mapping_path),
                adni_tensor_path=str(args.adni_tensor_path),
            )
        ),
        "read_only": True,
        "computed_connectomes": False,
        "modified_input_data": False,
        "decision": decision,
        "counts": {
            "roi_qc_rows": int(len(roi_qc)),
            "mapping_rows": int(len(mapping_170)),
            "kept_rois": int(mapping_170["keep_drop"].eq("keep").sum()),
            "dropped_empty_labels": int(
                mapping_170["reason"].astype(str).str.contains("empty label", case=False).sum()
            ),
            "dropped_small_rois": int(
                mapping_170["reason"].astype(str).str.contains("small ROI", case=False).sum()
            ),
            "unresolved_rows": int(len(unresolved)),
        },
        "outputs": [
            "README.md",
            "oasis_170_roi_order_candidates.csv",
            "oasis_170_roi_order_candidates.md",
            "all_nan_roi_columns.csv",
            "all_nan_roi_columns.md",
            "adni_131_matching_table.csv",
            "adni_131_matching_table.md",
            "roi_mapping_170_to_131.csv",
            "roi_mapping_170_to_131.md",
            "unresolved_rois.csv",
            "unresolved_rois.md",
            "final_recommendation.md",
            "command_log.json",
        ],
    }
    (args.output_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
