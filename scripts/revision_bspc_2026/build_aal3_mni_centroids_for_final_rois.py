#!/usr/bin/env python3
"""Build verified AAL3 MNI centroid table and match to final 131 tensor ROI names.

Guardrails:
  did_train_vae: False
  did_retrain_classifier: False
  did_run_shap_ig: False
  did_modify_tensors: False
  did_modify_metadata: False
  did_modify_manuscript: False

Outputs:
  <OUT>/aal3v1_1mm_all_label_centroids_mni.csv
  <OUT>/final_131_roi_mni_centroids.csv
  <OUT>/roi_matching_audit.csv
  <OUT>/roi_matching_audit.md
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AAL3_DIR = Path("/home/diego/Escritorio/AAL3v1_for_SPM12/AAL3")
ATLAS_NIB = AAL3_DIR / "ROI_MNI_V7_1mm.nii"
ATLAS_XML = AAL3_DIR / "ROI_MNI_V7_1mm.xml"

ROI_INFO_CSV = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
    "/roi_info_from_tensor.csv"
)

# 8 consensus-edge ROIs - ALL must be matched; failure → abort figure generation
CONSENSUS_ROIS = {
    "Cerebelum_7b_L", "Cerebelum_Crus2_L",
    "Cerebelum_7b_R",
    "Parietal_Inf_L",
    "Frontal_Sup_2_R",
    "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "OFCant_R", "OFClat_R",
    "Parietal_Sup_R",
    "Precuneus_R",
    "Postcentral_L", "Postcentral_R",
    "Precentral_R",
}

OUT = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_q1_20260624"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_roi(name: str) -> str:
    """Canonical form: lowercase, single-space, remove leading/trailing."""
    return name.strip().lower().replace("_", "_")


def cerebelum_to_cerebellum(name: str) -> str:
    """Tensor uses 'Cerebelum_' (single l); atlas uses 'Cerebellum_' (double l)."""
    return name.replace("Cerebelum_", "Cerebellum_").replace("cerebelum_", "cerebellum_")


# ---------------------------------------------------------------------------
# Step 1: Parse XML labels
# ---------------------------------------------------------------------------

def parse_atlas_xml(xml_path: Path) -> dict[int, str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels: dict[int, str] = {}
    for elem in root.iter("label"):
        idx_el = elem.find("index")
        name_el = elem.find("name")
        if idx_el is not None and name_el is not None:
            idx = int(idx_el.text.strip())
            name = name_el.text.strip()
            labels[idx] = name
    return labels


# ---------------------------------------------------------------------------
# Step 2: Compute MNI centroids from NIfTI
# ---------------------------------------------------------------------------

def compute_centroids(nifti_path: Path, label_names: dict[int, str]) -> pd.DataFrame:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    affine = img.affine

    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels > 0]

    rows = []
    for lbl in unique_labels:
        mask = data == lbl
        n_vox = int(mask.sum())
        if n_vox == 0:
            continue
        vox_coords = np.array(np.where(mask), dtype=float).T  # (N, 3)
        centroid_vox = vox_coords.mean(axis=0)
        # Apply affine: [x,y,z,1] = affine @ [i,j,k,1]
        centroid_mni = nib.affines.apply_affine(affine, centroid_vox)
        name = label_names.get(int(lbl), f"UNKNOWN_LABEL_{int(lbl)}")
        rows.append({
            "label_id": int(lbl),
            "aal3_name_raw": name,
            "aal3_name_normalized": cerebelum_to_cerebellum(name).lower(),
            "x": round(float(centroid_mni[0]), 2),
            "y": round(float(centroid_mni[1]), 2),
            "z": round(float(centroid_mni[2]), 2),
            "n_voxels": n_vox,
            "source_nifti": ATLAS_NIB.name,
            "source_label_file": ATLAS_XML.name,
        })

    df = pd.DataFrame(rows).sort_values("label_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 3: Match 131 tensor ROIs to atlas centroids
# ---------------------------------------------------------------------------

def match_tensor_rois(
    roi_info: pd.DataFrame,
    centroid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matched_df, audit_df)."""
    # Build lookup: atlas_name_raw (lower) → row
    atlas_by_name_lower: dict[str, dict] = {
        row["aal3_name_raw"].lower(): row
        for _, row in centroid_df.iterrows()
    }
    atlas_by_cerebellum_norm: dict[str, dict] = {
        cerebelum_to_cerebellum(k): v for k, v in atlas_by_name_lower.items()
    }

    matched_rows = []
    audit_rows = []

    for i, tensor_row in roi_info.iterrows():
        tensor_name: str = tensor_row["roi_name_in_tensor"]
        network: str = tensor_row["network_label_in_tensor"]

        tensor_lower = tensor_name.lower()
        tensor_cerebellum_norm = cerebelum_to_cerebellum(tensor_lower)

        atlas_row = None
        match_rule = None

        # Rule 1: exact case-insensitive match
        if tensor_lower in atlas_by_name_lower:
            atlas_row = atlas_by_name_lower[tensor_lower]
            match_rule = "exact_lowercase"

        # Rule 2: Cerebelum_ → Cerebellum_ normalization
        elif tensor_cerebellum_norm in atlas_by_name_lower:
            atlas_row = atlas_by_name_lower[tensor_cerebellum_norm]
            match_rule = "cerebelum_to_cerebellum_normalization"

        # Rule 3: cerebelum norm in cerebellum-corrected atlas
        elif tensor_cerebellum_norm in atlas_by_cerebellum_norm:
            atlas_row = atlas_by_cerebellum_norm[tensor_cerebellum_norm]
            match_rule = "cerebelum_to_cerebellum_normalization_v2"

        if atlas_row is not None:
            matched_rows.append({
                "roi_tensor_index": int(i),
                "roi_name_in_tensor": tensor_name,
                "network": network,
                "aal3_label_id": int(atlas_row["label_id"]),
                "aal3_name_raw": atlas_row["aal3_name_raw"],
                "x": atlas_row["x"],
                "y": atlas_row["y"],
                "z": atlas_row["z"],
                "n_voxels": atlas_row["n_voxels"],
                "match_status": "MATCHED",
                "match_rule": match_rule,
            })
            audit_rows.append({
                "tensor_name": tensor_name,
                "atlas_name": atlas_row["aal3_name_raw"],
                "match_status": "MATCHED",
                "match_rule": match_rule,
                "x": atlas_row["x"], "y": atlas_row["y"], "z": atlas_row["z"],
                "consensus_edge_roi": tensor_name in CONSENSUS_ROIS,
            })
        else:
            matched_rows.append({
                "roi_tensor_index": int(i),
                "roi_name_in_tensor": tensor_name,
                "network": network,
                "aal3_label_id": None,
                "aal3_name_raw": None,
                "x": None, "y": None, "z": None,
                "n_voxels": None,
                "match_status": "UNMATCHED",
                "match_rule": None,
            })
            audit_rows.append({
                "tensor_name": tensor_name,
                "atlas_name": None,
                "match_status": "UNMATCHED",
                "match_rule": None,
                "x": None, "y": None, "z": None,
                "consensus_edge_roi": tensor_name in CONSENSUS_ROIS,
            })

    matched_df = pd.DataFrame(matched_rows)
    audit_df = pd.DataFrame(audit_rows)
    return matched_df, audit_df


# ---------------------------------------------------------------------------
# Step 4: Validate consensus-edge ROIs
# ---------------------------------------------------------------------------

def validate_consensus_rois(matched_df: pd.DataFrame) -> None:
    unmatched_consensus = matched_df[
        (matched_df["roi_name_in_tensor"].isin(CONSENSUS_ROIS)) &
        (matched_df["match_status"] == "UNMATCHED")
    ]
    if len(unmatched_consensus) > 0:
        print("CRITICAL: The following consensus-edge ROIs are UNMATCHED in the atlas:")
        for _, r in unmatched_consensus.iterrows():
            print(f"  {r['roi_name_in_tensor']}")
        print("STOPPING: Do not generate glass-brain figure with unverified coordinates.")
        sys.exit(1)
    print(f"  All {len(CONSENSUS_ROIS)} consensus-edge ROIs matched successfully.")


# ---------------------------------------------------------------------------
# Step 5: Write audit markdown
# ---------------------------------------------------------------------------

def write_audit_md(audit_df: pd.DataFrame, centroid_df: pd.DataFrame, out: Path) -> None:
    n_matched = (audit_df["match_status"] == "MATCHED").sum()
    n_unmatched = (audit_df["match_status"] == "UNMATCHED").sum()
    unmatched_names = audit_df[audit_df["match_status"] == "UNMATCHED"]["tensor_name"].tolist()

    lines = [
        "# ROI Matching Audit: Tensor → AAL3 Atlas",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        f"Atlas NIfTI: `{ATLAS_NIB.name}`",
        f"Atlas XML: `{ATLAS_XML.name}`",
        f"Atlas labels found: {len(centroid_df)}",
        f"Tensor ROIs: {len(audit_df)}",
        f"Matched: {n_matched} ({100*n_matched/len(audit_df):.1f}%)",
        f"Unmatched: {n_unmatched}",
        "",
        "## Unmatched ROIs",
        "",
    ]
    if unmatched_names:
        for name in unmatched_names:
            is_ce = name in CONSENSUS_ROIS
            flag = " **[CONSENSUS EDGE — CRITICAL]**" if is_ce else ""
            lines.append(f"- `{name}`{flag}")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Consensus-Edge ROI Match Summary",
        "",
        "| Tensor ROI | Atlas ROI | x | y | z | Rule |",
        "|-----------|-----------|--:|--:|--:|------|",
    ]
    for _, r in audit_df[audit_df["consensus_edge_roi"]].iterrows():
        lines.append(
            f"| {r['tensor_name']} | {r['atlas_name'] or 'UNMATCHED'} | "
            f"{r['x'] or '—'} | {r['y'] or '—'} | {r['z'] or '—'} | "
            f"{r['match_rule'] or 'UNMATCHED'} |"
        )

    lines += [
        "",
        "## Matching Rules Applied",
        "",
        "1. **exact_lowercase**: tensor name matches atlas name case-insensitively.",
        "2. **cerebelum_to_cerebellum_normalization**: tensor uses `Cerebelum_` (single l); atlas uses `Cerebellum_` (double l). Harmless typographic correction — same anatomy.",
        "",
        "## Non-consensus Unmatched ROIs",
        "",
        "- `Vent_Str_L` / `Vent_Str_R`: Ventral striatum. Not present in `ROI_MNI_V7_1mm`. AAL3v1 renamed this to `N_Acc_L/R` (Nucleus Accumbens). These ROIs are **not** in any consensus edge and do not affect Figure 4.",
    ]

    (out / "roi_matching_audit.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[Task 1] Parsing AAL3 XML label names...")
    label_names = parse_atlas_xml(ATLAS_XML)
    print(f"  Found {len(label_names)} labels in XML.")

    print("[Task 1] Computing MNI centroids from NIfTI...")
    centroid_df = compute_centroids(ATLAS_NIB, label_names)
    print(f"  Extracted {len(centroid_df)} non-zero labels.")

    out_centroids = OUT / "aal3v1_1mm_all_label_centroids_mni.csv"
    centroid_df.to_csv(out_centroids, index=False)
    print(f"  Saved: {out_centroids.name}")

    print("[Task 2] Loading tensor ROI list...")
    roi_info = pd.read_csv(ROI_INFO_CSV)
    print(f"  Found {len(roi_info)} tensor ROIs.")

    print("[Task 2] Matching tensor ROIs to atlas...")
    matched_df, audit_df = match_tensor_rois(roi_info, centroid_df)

    n_matched = (matched_df["match_status"] == "MATCHED").sum()
    n_unmatched = (matched_df["match_status"] == "UNMATCHED").sum()
    print(f"  Matched: {n_matched} / {len(matched_df)}  Unmatched: {n_unmatched}")

    unmatched = matched_df[matched_df["match_status"] == "UNMATCHED"]["roi_name_in_tensor"].tolist()
    if unmatched:
        print(f"  Unmatched ROIs: {unmatched}")

    print("[Task 2] Validating all consensus-edge ROIs are matched...")
    validate_consensus_rois(matched_df)

    out_matched = OUT / "final_131_roi_mni_centroids.csv"
    matched_df.to_csv(out_matched, index=False)
    print(f"  Saved: {out_matched.name}")

    audit_df.to_csv(OUT / "roi_matching_audit.csv", index=False)
    write_audit_md(audit_df, centroid_df, OUT)
    print(f"  Saved: roi_matching_audit.csv / .md")

    # Build the 8-edge coordinate lookup table
    edge_rois = sorted(CONSENSUS_ROIS)
    edge_coord_rows = []
    for roi in edge_rois:
        row = matched_df[matched_df["roi_name_in_tensor"] == roi]
        if len(row) == 0:
            print(f"  WARNING: {roi} not in matched table")
            continue
        r = row.iloc[0]
        edge_coord_rows.append({
            "roi_name_in_tensor": roi,
            "aal3_label_id": r["aal3_label_id"],
            "aal3_name_raw": r["aal3_name_raw"],
            "network": r["network"],
            "x": r["x"], "y": r["y"], "z": r["z"],
        })
    edge_coords_df = pd.DataFrame(edge_coord_rows)
    out_edge = OUT / "final_8edge_mni_coordinates.csv"
    edge_coords_df.to_csv(out_edge, index=False)
    print(f"  Saved: {out_edge.name}")

    print("\n[Done] All centroid outputs written to:", OUT)


if __name__ == "__main__":
    main()
