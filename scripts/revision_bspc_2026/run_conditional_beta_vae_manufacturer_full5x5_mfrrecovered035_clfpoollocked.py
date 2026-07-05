#!/usr/bin/env python3
"""Launcher for conditional Manufacturer beta-VAE FULL 5x5 — recovered-035 VAE pool, classifier pool locked.

Identical to ``run_conditional_beta_vae_manufacturer_fast3x3_cleanmfr.py`` except:

1. A branch-local patched metadata CSV is created before any training:
   - Base: training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv (646 rows from Datos)
   - Patch: 035_S_6927 appended with Manufacturer=SIEMENS (recovered from idaSearch direct evidence)
   - Result: 647-row patched CSV saved under metadata/training_ready_metadata_v5_1b_mfrrecovered035.csv
   - 128_S_2002 remains unpatched (unresolvable, no direct Manufacturer evidence)

2. The Stage A --metadata_path is overridden to point to this patched metadata CSV.

3. 035_S_6927 is explicitly excluded from the supervised classifier pool with
   --classifier_exclude_subject_ids, so the AD/CN readout remains unchanged
   (n=396, CN=300, AD=96).  It remains eligible for the VAE pool.

4. With --vae_required_metadata_cols Manufacturer:
   - 035_S_6927 now has Manufacturer=SIEMENS → passes the filter → stays in VAE pool
   - 128_S_2002 is still absent from the patched metadata → NaN Manufacturer → still removed

Expected outcome vs cleanmfr:
  - cleanmfr: 2 subjects removed per fold (035_S_6927, 128_S_2002)
  - mfrrecovered035_clfpoollocked: 1 subject removed per fold (128_S_2002 only)
  - classifier AD/CN pool remains locked to cleanmfr: n=396, CN=300, AD=96

Default behavior is dry-run/no training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import py_compile
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"
PREFLIGHT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_manufacturer_fast3x3_preflight"
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"
BIG_DISK_RUN_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked/runs")
DATOS_MOUNT = Path("/media/diego/Datos")

TRAIN_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
AGGREGATOR_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/aggregate_conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked.py"
INTEGRITY_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/audit_conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked_integrity.py"

# Metadata paths
SOURCE_TRAINING_READY_METADATA = DATOS_MOUNT / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
FULL_SUBJECT_METADATA = DATOS_MOUNT / "vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_metadata_v5_1_batch20260514b_no_pybandpass.csv"
PATCHED_METADATA_SUBPATH = Path("metadata/training_ready_metadata_v5_1b_mfrrecovered035_clfpoollocked.csv")

# Patch constants
PATCHED_SUBJECT = "035_S_6927"
CLASSIFIER_EXCLUDED_SUBJECT = PATCHED_SUBJECT
PATCHED_MANUFACTURER = "SIEMENS"
PATCHED_TENSOR_INDEX = 256
UNRESOLVED_SUBJECT = "128_S_2002"
PATCH_EVIDENCE_SOURCE = "data/idaSearch_4_03_2026.csv Imaging Protocol field (Image ID 1436478)"
PATCH_EVIDENCE_CONFIDENCE = "direct"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
N_FOLDS = 5
EPOCHS_VAE = 4480
CYCLICAL_BETA_N_CYCLES = 56
EARLY_STOPPING_PATIENCE_VAE = 320
EXPECTED_CANDIDATES = [
    "ch1_0_2_baseline_unconditioned",
    "ch1_0_2_decoder_only_manufacturer",
]

FULL5X5_TARGET_VALIDATION_ROWS = [
    {
        "selected_target": "[1,0,2]",
        "candidate_id": "ch1_0_2_decoder_only_manufacturer",
        "matched_baseline": "ch1_0_2_baseline_unconditioned",
        "evidence_source": "conditional Manufacturer VAE training-maturity audit",
        "auc_delta_vs_matched_baseline": "positive",
        "pr_auc_delta_vs_matched_baseline": "positive",
        "manufacturer_leakage_reduction": "positive",
        "right_censored_or_near_horizon_folds": "2/3",
        "decision": "selected_for_controlled_FULL_5x5_confirmation",
    },
    {
        "selected_target": "[1]",
        "candidate_id": "ch1_decoder_only_manufacturer",
        "matched_baseline": "ch1_baseline_unconditioned",
        "evidence_source": "conditional Manufacturer VAE training-maturity audit",
        "auc_delta_vs_matched_baseline": "not_selected",
        "pr_auc_delta_vs_matched_baseline": "not_selected",
        "manufacturer_leakage_reduction": "not_selected",
        "right_censored_or_near_horizon_folds": "not_selected",
        "decision": "not_FULL_5x5_target",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_datos_drive() -> None:
    if not DATOS_MOUNT.exists() or not os.path.ismount(str(DATOS_MOUNT)):
        raise SystemExit(
            "Datos drive is not mounted.\n"
            "  Run:  udisksctl mount -b /dev/sda1\n"
            "  or:   sudo mount /dev/sda1 /media/diego/Datos\n"
            "Then retry."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--preflight-root", type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--big-disk-run-root", type=Path, default=BIG_DISK_RUN_ROOT)
    parser.add_argument("--python-executable", default=PYTHON)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--no-external-symlink", action="store_true")
    parser.add_argument("--skip-aggregation", action="store_true")
    parser.add_argument("--status", action="store_true", help="Print per-candidate status and exit (read-only).")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_matrix(preflight_root: Path) -> pd.DataFrame:
    path = preflight_root / "experiment_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validated preflight matrix: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(["candidate_id", "channels", "vae_conditioning_mode", "vae_conditioning_vars", "readout_feature_set"]) - set(df.columns))
    if missing:
        raise ValueError(f"Preflight matrix missing columns: {missing}")
    missing_candidates = [c for c in EXPECTED_CANDIDATES if c not in set(df["candidate_id"].astype(str))]
    if missing_candidates:
        raise ValueError(f"Preflight matrix missing expected FULL 5x5 candidates: {missing_candidates}")
    df = df[df["candidate_id"].astype(str).isin(EXPECTED_CANDIDATES)].copy()
    order = {candidate_id: idx for idx, candidate_id in enumerate(EXPECTED_CANDIDATES)}
    df["_order"] = df["candidate_id"].map(order)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    if "condition_id" not in df.columns:
        df["condition_id"] = df["candidate_id"].astype(str).str.replace("^ch1_0_2_", "", regex=True).str.replace("^ch1_", "", regex=True)
    if "corr_lambda" not in df.columns:
        df["corr_lambda"] = 0.0
    return df


def load_preflight_config(preflight_root: Path, candidate_id: str) -> Dict[str, Any]:
    path = preflight_root / "configs" / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing preflight config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_channels(value: str) -> List[int]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError(f"Invalid channels value: {value!r}")
    return parsed


def selected_rows(matrix: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "all":
        return matrix.copy()
    rows = matrix[matrix["candidate_id"].eq(candidate)].copy()
    if rows.empty:
        valid = ", ".join(matrix["candidate_id"].tolist())
        raise ValueError(f"Unknown candidate={candidate!r}. Valid: all, {valid}")
    return rows


def command_values_after(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    values: List[str] = []
    for token in tokens[list(tokens).index(flag) + 1:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def command_value(tokens: Sequence[str], flag: str) -> Optional[str]:
    values = command_values_after(tokens, flag)
    return values[0] if values else None


def replace_flag_value(tokens: List[str], flag: str, new_values: Sequence[str]) -> List[str]:
    if flag not in tokens:
        return tokens + [flag, *list(new_values)]
    idx = tokens.index(flag)
    end = idx + 1
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    return tokens[: idx + 1] + list(new_values) + tokens[end:]


def remove_token(tokens: Sequence[str], token: str) -> List[str]:
    out = list(tokens)
    while token in out:
        out.remove(token)
    return out


def ensure_qc_nuisance_cols(tokens: List[str]) -> List[str]:
    return replace_flag_value(tokens, "--qc_nuisance_cols", ["Manufacturer", "SiteCode", "Sex", "Age_Group"])


def candidate_dirs(output_root: Path, big_disk_run_root: Path, candidate_id: str, no_external_symlink: bool) -> Dict[str, Path]:
    local_run = output_root / "runs" / candidate_id
    big_run = local_run if no_external_symlink else big_disk_run_root / candidate_id
    return {"local_run": local_run, "big_run": big_run}


def readout_dir_for(local_run: Path, readout_feature_set: str) -> Path:
    return local_run / f"classifier_only_readout_{readout_feature_set}"


def create_patched_metadata(
    output_root: Path,
    source_metadata_path: Path,
    full_metadata_path: Path,
    dry_run: bool,
) -> Tuple[Path, Dict[str, Any]]:
    """Create branch-local patched metadata CSV with PATCHED_SUBJECT's Manufacturer filled."""
    patched_path = output_root / PATCHED_METADATA_SUBPATH

    # If already exists and verified, reuse it
    if patched_path.exists():
        try:
            p = pd.read_csv(patched_path)
            row_p = p[p["SubjectID"].astype(str) == PATCHED_SUBJECT]
            if len(row_p) == 1 and str(row_p.iloc[0]["Manufacturer"]) == PATCHED_MANUFACTURER:
                return patched_path, {
                    "status": "already_exists_verified",
                    "patched_path": str(patched_path),
                    "n_rows_patched_csv": len(p),
                    "subject_present": True,
                    "manufacturer_correct": True,
                }
        except Exception as exc:
            pass
        return patched_path, {"status": "already_exists_unverified", "patched_path": str(patched_path)}

    # Load source training-ready metadata (646 rows)
    if not source_metadata_path.exists():
        raise FileNotFoundError(f"Source metadata not found: {source_metadata_path}")
    source = pd.read_csv(source_metadata_path)
    n_source = len(source)
    n_nan_mfr_source = int(source["Manufacturer"].isna().sum()) if "Manufacturer" in source.columns else -1

    # Verify PATCHED_SUBJECT is not already in source
    already_in_source = PATCHED_SUBJECT in source["SubjectID"].astype(str).values
    if already_in_source:
        raise RuntimeError(
            f"{PATCHED_SUBJECT} is already in source metadata ({source_metadata_path}). "
            "No patch needed — check whether cleanmfr was updated."
        )

    # Get full metadata row for PATCHED_SUBJECT
    if not full_metadata_path.exists():
        raise FileNotFoundError(f"Full subject metadata not found: {full_metadata_path}")
    full = pd.read_csv(full_metadata_path)
    subject_rows = full[full["SubjectID"].astype(str) == PATCHED_SUBJECT].copy()
    if subject_rows.empty:
        raise RuntimeError(f"Subject {PATCHED_SUBJECT} not found in full metadata: {full_metadata_path}")

    new_row = subject_rows.iloc[[0]].copy()
    original_mfr = str(new_row.iloc[0]["Manufacturer"]) if "Manufacturer" in new_row.columns else "NOT_FOUND"

    # Apply the patch: only Manufacturer and training_ready
    new_row["Manufacturer"] = PATCHED_MANUFACTURER
    # Match training_ready=True to be consistent with the source CSV (all True)
    new_row["training_ready"] = True

    patched = pd.concat([source, new_row], ignore_index=True, sort=False)
    n_patched = len(patched)
    n_nan_mfr_patched = int(patched["Manufacturer"].isna().sum()) if "Manufacturer" in patched.columns else -1

    audit_info = {
        "status": "created",
        "source_metadata_path": str(source_metadata_path),
        "full_metadata_path": str(full_metadata_path),
        "patched_path": str(patched_path),
        "n_rows_source": n_source,
        "n_rows_patched": n_patched,
        "n_nan_manufacturer_before": n_nan_mfr_source + 2,  # 035+128 are outside source, contribute NaN via tensor join
        "n_nan_manufacturer_after": 1,  # only 128_S_2002 remains unresolvable
        "patched_subject": PATCHED_SUBJECT,
        "original_manufacturer": original_mfr,
        "patched_manufacturer": PATCHED_MANUFACTURER,
        "training_ready_patched_to": True,
        "unresolved_subject": UNRESOLVED_SUBJECT,
        "unresolved_reason": "no_direct_manufacturer_evidence_historical_desde_cero_anomalous_signal",
        "fields_patched": ["Manufacturer", "training_ready"],
        "fields_not_patched": ["Age", "Sex"],
    }

    if not dry_run:
        patched_path.parent.mkdir(parents=True, exist_ok=True)
        patched.to_csv(patched_path, index=False)
        audit_info["sha256_source"] = sha256_file(source_metadata_path)
        audit_info["sha256_patched"] = sha256_file(patched_path)

    return patched_path, audit_info


def write_patch_audit(output_root: Path, audit_info: Dict[str, Any]) -> None:
    """Write patch audit files to output_root."""
    output_root.mkdir(parents=True, exist_ok=True)

    # patched_subjects.csv/.md
    patched_df = pd.DataFrame([{
        "SubjectID": PATCHED_SUBJECT,
        "tensor_index": PATCHED_TENSOR_INDEX,
        "ResearchGroup_Mapped": "AD",
        "original_Manufacturer": audit_info.get("original_manufacturer", "nan"),
        "patched_Manufacturer": PATCHED_MANUFACTURER,
        "evidence_source": PATCH_EVIDENCE_SOURCE,
        "evidence_confidence": PATCH_EVIDENCE_CONFIDENCE,
        "patch_type": "append_row_to_training_ready_csv",
        "other_fields_patched": "training_ready=True",
        "fields_not_patched": "Age,Sex (remain NaN; unrecoverable in this branch)",
        "patch_justification": (
            "Manufacturer=SIEMENS recovered from idaSearch_4_03_2026.csv Imaging Protocol "
            "field for Image ID 1436478 (direct evidence). Root cause of original NaN: "
            "metadata propagation failure in v5 build — subject present in tensor but not "
            "carried into training-ready metadata CSV."
        ),
    }])
    patched_df.to_csv(output_root / "patched_subjects.csv", index=False)
    (output_root / "patched_subjects.md").write_text(
        patched_df.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    # unresolved_subjects.csv/.md
    unresolved_df = pd.DataFrame([{
        "SubjectID": UNRESOLVED_SUBJECT,
        "tensor_index": 363,
        "ResearchGroup_Mapped": "NaN",
        "original_Manufacturer": "NaN",
        "patch_status": "remain_excluded",
        "reason": (
            "No direct Manufacturer evidence exists for 128_S_2002. Subject has no ADNI "
            "record in idaSearch or any ADNI metadata source. Signal originates from "
            "desde_cero historical preprocessed data with anomalous characteristics "
            "(96.9% near-zero values, unknown scale, max=168170). Excluded from all runs."
        ),
        "cleanmfr_outcome": "removed_from_vae_pool_per_fold (no_change_in_mfrrecovered035)",
    }])
    unresolved_df.to_csv(output_root / "unresolved_subjects.csv", index=False)
    (output_root / "unresolved_subjects.md").write_text(
        unresolved_df.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    # before_after_missing_manufacturer_counts.csv/.md
    ba_df = pd.DataFrame([
        {
            "scenario": "cleanmfr (both subjects absent from metadata, NaN via tensor left-join)",
            "n_subjects_in_tensor": 648,
            "n_subjects_in_metadata_csv": 646,
            "n_manufacturer_nan_or_missing": 2,
            "subjects_with_nan_manufacturer": f"{PATCHED_SUBJECT},{UNRESOLVED_SUBJECT}",
            "vae_pool_subjects_removed_per_fold": 2,
        },
        {
            "scenario": "mfrrecovered035 (035_S_6927 patched to SIEMENS, 128_S_2002 still absent)",
            "n_subjects_in_tensor": 648,
            "n_subjects_in_metadata_csv": 647,
            "n_manufacturer_nan_or_missing": 1,
            "subjects_with_nan_manufacturer": UNRESOLVED_SUBJECT,
            "vae_pool_subjects_removed_per_fold": 1,
        },
    ])
    ba_df.to_csv(output_root / "before_after_missing_manufacturer_counts.csv", index=False)
    (output_root / "before_after_missing_manufacturer_counts.md").write_text(
        ba_df.to_markdown(index=False) + "\n", encoding="utf-8"
    )

    patch_audit = pd.DataFrame([{
        "patch_branch": "mfrrecovered035_clfpoollocked",
        "patched_subject": PATCHED_SUBJECT,
        "patched_manufacturer": PATCHED_MANUFACTURER,
        "supervised_classifier_action": f"explicitly_excluded_via_classifier_exclude_subject_ids={CLASSIFIER_EXCLUDED_SUBJECT}",
        "vae_pool_action": "eligible_if_not_outer_test_and_has_required_Manufacturer",
        "unresolved_subject": UNRESOLVED_SUBJECT,
        "unresolved_action": "remain_excluded_by_missing_required_Manufacturer",
        "expected_classifier_n": 396,
        "expected_classifier_cn": 300,
        "expected_classifier_ad": 96,
        "source_metadata_modified": False,
        "tensor_modified": False,
        "ledger_modified": False,
    }])
    patch_audit.to_csv(output_root / "patch_audit.csv", index=False)
    (output_root / "patch_audit.md").write_text(patch_audit.to_markdown(index=False) + "\n", encoding="utf-8")

    # checksum provenance
    checksum_rows = []
    if "sha256_source" in audit_info:
        checksum_rows.append({
            "file": "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv (source, Datos, unmodified)",
            "n_rows": audit_info.get("n_rows_source", ""),
            "sha256": audit_info.get("sha256_source", ""),
        })
    if "sha256_patched" in audit_info:
        checksum_rows.append({
            "file": "training_ready_metadata_v5_1b_mfrrecovered035.csv (patched, branch-local)",
            "n_rows": audit_info.get("n_rows_patched", ""),
            "sha256": audit_info.get("sha256_patched", ""),
        })
    if checksum_rows:
        ck_df = pd.DataFrame(checksum_rows)
        ck_df.to_csv(output_root / "metadata_checksums.csv", index=False)
        (output_root / "metadata_checksums.md").write_text(
            ck_df.to_markdown(index=False) + "\n", encoding="utf-8"
        )

    # final_patch_decision.md
    decision_lines = [
        "# mfrrecovered035 Metadata Patch Decision",
        "",
        f"**Date:** {now_utc()[:10]}",
        "",
        "## What was patched",
        "",
        f"- **Subject patched:** `{PATCHED_SUBJECT}` (tensor_index=256, ResearchGroup_Mapped=AD)",
        f"- **Field patched:** `Manufacturer` — set from NaN to `{PATCHED_MANUFACTURER}`",
        "- **Field patched:** `training_ready` — set from False to True (to match 646-row training-ready CSV format)",
        f"- **Supervised classifier guard:** `{PATCHED_SUBJECT}` is explicitly excluded via `--classifier_exclude_subject_ids`",
        "- **Fields NOT patched:** `Age` and `Sex` remain NaN (unrecoverable in current sources)",
        "- **Patch type:** Row appended to 646-row training-ready metadata CSV",
        "- **Resulting CSV:** 647 rows (`metadata/training_ready_metadata_v5_1b_mfrrecovered035.csv`)",
        "",
        "## Evidence for Manufacturer=SIEMENS",
        "",
        "Direct evidence from ADNI idaSearch export (`data/idaSearch_4_03_2026.csv`):",
        "  Imaging Protocol field for Image ID 1436478:",
        '  `"Field Strength=3.0;TE=30.0;Manufacturer=SIEMENS;Slice Thickness=3.4;TR=3000.0"`',
        "",
        "Corroborated by:",
        "  - `data/revision_bspc_2026/adni_expanded_v5_passband_dparsf_only/subject_metadata_adni_expanded_v5_passband_dparsf_only.csv`",
        "  - `results/revision_bspc_2026/adni_v5_1c_metadata_recovery_preflight/subject_recovery_evidence.csv`",
        "  - `results/revision_bspc_2026/adni_v5_1c_recover035_dataset_qc/subject_alignment.csv`",
        "",
        "## Why 128_S_2002 is not patched",
        "",
        "128_S_2002 has no ADNI record in any source (not in idaSearch, not in any ADNI metadata CSV).",
        "Signal originates from desde_cero historical preprocessed data with anomalous characteristics.",
        "Previous audits concluded this subject is unresolvable. It remains excluded.",
        "",
        "## What was NOT modified",
        "",
        "- Datos drive metadata is unmodified (source read-only)",
        "- Tensor (GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz) is unmodified",
        "- Ledger is unmodified",
        "- Prior run outputs (cleanmfr or other experiments) are unmodified",
        "- Configs on disk are unmodified",
        "",
        "## Expected VAE pool impact",
        "",
        "- cleanmfr: 2 subjects removed per fold (035_S_6927, 128_S_2002) → VAE pool loses 2 per fold",
        "- mfrrecovered035: 1 subject removed per fold (128_S_2002 only) → VAE pool loses 1 per fold",
        "- Classifier pool: locked to cleanmfr / original AD-CN readout (n=396, CN=300, AD=96)",
        "- 035_S_6927 is restored only to the VAE pool; it must not appear in classifier train/dev/test.",
        "",
        "## Scientific rationale",
        "",
        "The cleanmfr experiment excluded 035_S_6927 from the VAE pool because Manufacturer was missing",
        "in the metadata (metadata propagation failure, not a genuine quality issue).",
        "This patch restores what should have been present from the beginning.",
        "The mfrrecovered035 branch represents the scientifically cleaner version of the cleanmfr experiment:",
        "only genuinely unresolvable Manufacturer-missing subjects are excluded from the VAE pool.",
    ]
    (output_root / "final_patch_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def build_commands(
    row: pd.Series,
    preflight_root: Path,
    output_root: Path,
    big_disk_run_root: Path,
    python_executable: str,
    no_external_symlink: bool,
    force_clean: bool,
    patched_metadata_path: Path,
) -> Dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    cfg = load_preflight_config(preflight_root, candidate_id)
    channels = parse_channels(str(row["channels"]))
    dirs = candidate_dirs(output_root, big_disk_run_root, candidate_id, no_external_symlink=no_external_symlink)
    local_run = dirs["local_run"]
    readout_dir = readout_dir_for(local_run, str(row["readout_feature_set"]))

    stage_a = list(cfg["stage_a_command"])
    stage_b = list(cfg["stage_b_command"])
    stage_a[0] = python_executable
    stage_b[0] = python_executable
    stage_a = replace_flag_value(stage_a, "--output_dir", [str(local_run)])
    stage_a = replace_flag_value(stage_a, "--channels_to_use", [str(x) for x in channels])
    stage_a = replace_flag_value(stage_a, "--outer_folds", [str(N_FOLDS)])
    stage_a = replace_flag_value(stage_a, "--inner_folds", [str(N_FOLDS)])
    stage_a = replace_flag_value(stage_a, "--epochs_vae", [str(EPOCHS_VAE)])
    stage_a = replace_flag_value(stage_a, "--cyclical_beta_n_cycles", [str(CYCLICAL_BETA_N_CYCLES)])
    stage_a = replace_flag_value(stage_a, "--early_stopping_patience_vae", [str(EARLY_STOPPING_PATIENCE_VAE)])
    stage_a = ensure_qc_nuisance_cols(stage_a)
    # Override metadata path to branch-local patched CSV
    stage_a = replace_flag_value(stage_a, "--metadata_path", [str(patched_metadata_path)])
    # mfrrecovered035: require non-missing Manufacturer in VAE pool for all candidates
    stage_a = replace_flag_value(stage_a, "--vae_required_metadata_cols", ["Manufacturer"])
    # clfpoollocked: restore 035_S_6927 to VAE pool only, never to classifier train/dev/test.
    stage_a = replace_flag_value(stage_a, "--classifier_exclude_subject_ids", [CLASSIFIER_EXCLUDED_SUBJECT])
    stage_b = replace_flag_value(stage_b, "--run-dir", [str(local_run)])
    stage_b = replace_flag_value(stage_b, "--output-dir", [str(readout_dir)])
    stage_b = replace_flag_value(stage_b, "--outer-folds", [str(N_FOLDS)])
    stage_b = replace_flag_value(stage_b, "--inner-folds", [str(N_FOLDS)])
    if force_clean and "--overwrite" not in stage_b:
        stage_b.append("--overwrite")
    return {
        "candidate_id": candidate_id,
        "condition_id": str(row["condition_id"]),
        "channels": channels,
        "channel_names": str(row["channel_names"]),
        "vae_conditioning_mode": str(row["vae_conditioning_mode"]),
        "vae_conditioning_vars": str(row["vae_conditioning_vars"]),
        "corr_lambda": float(row["corr_lambda"]),
        "readout_feature_set": str(row["readout_feature_set"]),
        "local_run_dir": local_run,
        "big_run_dir": dirs["big_run"],
        "readout_dir": readout_dir,
        "stage_a_dry_cmd": stage_a,
        "stage_a_real_cmd": remove_token(stage_a, "--dry-run"),
        "stage_b_cmd": stage_b,
    }


def validate_stage_a_command(spec: Dict[str, Any], real: bool, patched_metadata_path: Path) -> List[str]:
    errors: List[str] = []
    cmd = list(spec["stage_a_real_cmd"] if real else spec["stage_a_dry_cmd"])
    if real and "--dry-run" in cmd:
        errors.append(f"{spec['candidate_id']}: real Stage A command still contains --dry-run.")
    if not real and "--dry-run" not in cmd:
        errors.append(f"{spec['candidate_id']}: dry-run Stage A command lacks --dry-run.")
    expected_pairs = {
        "--classifier_types": ["logreg"],
        "--n_iter_logreg": ["1"],
        "--outer_folds": [str(N_FOLDS)],
        "--inner_folds": [str(N_FOLDS)],
        "--epochs_vae": [str(EPOCHS_VAE)],
        "--cyclical_beta_n_cycles": [str(CYCLICAL_BETA_N_CYCLES)],
        "--early_stopping_patience_vae": [str(EARLY_STOPPING_PATIENCE_VAE)],
        "--lr_scheduler_T0": ["80"],
        "--latent_dim": ["256"],
        "--beta_vae": ["2.5"],
        "--dropout_rate_vae": ["0.15"],
        "--vae_dropout_scope": ["legacy_all"],
        "--vae_block_order": ["legacy_act_norm"],
        "--vae_final_activation": ["tanh"],
        "--recon_loss_mode": ["offdiag_channelmean_sum"],
        "--metadata_features": ["Age", "Sex"],
        "--vae_conditioning_mode": [spec["vae_conditioning_mode"]],
        "--vae_conditioning_vars": [spec["vae_conditioning_vars"]],
        "--vae_latent_covariate_corr_lambda": [str(spec["corr_lambda"])],
        "--qc_nuisance_cols": ["Manufacturer", "SiteCode", "Sex", "Age_Group"],
    }
    for flag, expected in expected_pairs.items():
        observed = command_values_after(cmd, flag)
        if observed != expected:
            errors.append(f"{spec['candidate_id']}: expected {flag} {expected}, got {observed}.")
    if "Manufacturer" not in command_values_after(cmd, "--classifier_stratify_cols"):
        errors.append(f"{spec['candidate_id']}: classifier_stratify_cols must include Manufacturer.")
    if "Manufacturer" not in command_values_after(cmd, "--vae_stratify_cols"):
        errors.append(f"{spec['candidate_id']}: vae_stratify_cols must include Manufacturer.")
    if "Manufacturer" in command_values_after(cmd, "--metadata_features"):
        errors.append(f"{spec['candidate_id']}: Manufacturer must not be passed to Stage B classifier metadata features.")
    if "Sex" in command_values_after(cmd, "--classifier_stratify_cols") or "Sex" in command_values_after(cmd, "--vae_stratify_cols"):
        errors.append(f"{spec['candidate_id']}: Sex must remain metadata/covariate only, not stratifier.")
    forbidden = ["--n_iter_svm", "--n_iter_rf", "--n_iter_gb", "--n_iter_xgb", "--n_iter_mlp"]
    present = [flag for flag in forbidden if flag in cmd]
    if present:
        errors.append(f"{spec['candidate_id']}: Stage A contains forbidden classifier trial flags: {present}.")
    # mfrrecovered035: require VAE pool filter for Manufacturer
    if command_values_after(cmd, "--vae_required_metadata_cols") != ["Manufacturer"]:
        errors.append(
            f"{spec['candidate_id']}: mfrrecovered035 requires --vae_required_metadata_cols Manufacturer "
            f"(got {command_values_after(cmd, '--vae_required_metadata_cols')})."
        )
    if command_values_after(cmd, "--classifier_exclude_subject_ids") != [CLASSIFIER_EXCLUDED_SUBJECT]:
        errors.append(
            f"{spec['candidate_id']}: classifier pool must explicitly exclude "
            f"{CLASSIFIER_EXCLUDED_SUBJECT}; got {command_values_after(cmd, '--classifier_exclude_subject_ids')}."
        )
    # mfrrecovered035: metadata path must be the branch-local patched CSV
    metadata = command_value(cmd, "--metadata_path")
    if not metadata or Path(metadata).resolve() != patched_metadata_path.resolve():
        errors.append(
            f"{spec['candidate_id']}: --metadata_path must be the branch-local patched CSV "
            f"({patched_metadata_path}), got {metadata}."
        )
    if not metadata or not Path(metadata).exists():
        errors.append(f"{spec['candidate_id']}: patched metadata path missing or does not exist: {metadata}")
    tensor = command_value(cmd, "--global_tensor_path")
    if not tensor or not Path(tensor).exists():
        errors.append(f"{spec['candidate_id']}: tensor path missing: {tensor}")
    return errors


def validate_stage_b_command(spec: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    cmd = list(spec["stage_b_cmd"])
    expected = {
        "--models": [PRIMARY_MODEL],
        "--readout-feature-sets": [spec["readout_feature_set"]],
        "--outer-folds": [str(N_FOLDS)],
        "--inner-folds": [str(N_FOLDS)],
    }
    for flag, values in expected.items():
        observed = command_values_after(cmd, flag)
        if observed != values:
            errors.append(f"{spec['candidate_id']}: expected Stage B {flag} {values}, got {observed}.")
    if "--reuse-latent-cache" not in cmd:
        errors.append(f"{spec['candidate_id']}: Stage B must include --reuse-latent-cache.")
    if any("manufacturer" in value.lower() for value in command_values_after(cmd, "--readout-feature-sets")):
        errors.append(f"{spec['candidate_id']}: Stage B readout must not pass Manufacturer to classifier.")
    return errors


def validate_specs(specs: Sequence[Dict[str, Any]], real: bool, patched_metadata_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    except Exception as exc:
        rows.append({"candidate_id": "__launcher__", "status": "error", "message": f"py_compile failed: {exc}"})
    for path in [TRAIN_SCRIPT, STAGE_B_SCRIPT, AGGREGATOR_SCRIPT, INTEGRITY_SCRIPT]:
        if not path.exists():
            rows.append({"candidate_id": "__common__", "status": "error", "message": f"Missing script: {path}"})
    for spec in specs:
        errors = validate_stage_a_command(spec, real=real, patched_metadata_path=patched_metadata_path)
        errors.extend(validate_stage_b_command(spec))
        status = "ok" if not errors else "error"
        rows.append({
            "candidate_id": spec["candidate_id"],
            "condition_id": spec["condition_id"],
            "channels": json.dumps(spec["channels"]),
            "vae_conditioning_mode": spec["vae_conditioning_mode"],
            "vae_conditioning_vars": spec["vae_conditioning_vars"],
            "corr_lambda": spec["corr_lambda"],
            "readout_feature_set": spec["readout_feature_set"],
            "local_run_dir": str(spec["local_run_dir"]),
            "big_run_dir": str(spec["big_run_dir"]),
            "readout_dir": str(spec["readout_dir"]),
            "status": status,
            "message": "; ".join(errors),
        })
    return rows


def stage_a_complete(run_dir: Path) -> bool:
    if not (run_dir / "run_config.json").exists():
        return False
    if not list(run_dir.glob("all_folds_metrics_MULTI*.csv")):
        return False
    for fold in range(1, N_FOLDS + 1):
        fold_dir = run_dir / f"fold_{fold}"
        required = [
            fold_dir / f"vae_model_fold_{fold}.pt",
            fold_dir / "vae_norm_params.joblib",
            fold_dir / "train_dev_subjects_fold.csv",
            fold_dir / "test_subjects_fold.csv",
        ]
        if not all(p.exists() for p in required):
            return False
    return True


def stage_b_complete(readout_dir: Path) -> bool:
    required = [
        readout_dir / "classifier_sweep_pooled_metrics.csv",
        readout_dir / "classifier_sweep_foldwise_metrics.csv",
        readout_dir / "classifier_sweep_predictions.csv",
        readout_dir / "classifier_sweep_thresholds_by_fold.csv",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    except Exception:
        return False
    if "readout_feature_set" not in pooled.columns:
        pooled["readout_feature_set"] = "z_plus_age_sex"
    mask = (
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    return bool(mask.any())


def folds_stage_a_complete_count(run_dir: Path) -> int:
    if not run_dir.exists() and not run_dir.is_symlink():
        return 0
    count = 0
    for fold in range(1, N_FOLDS + 1):
        fold_dir = run_dir / f"fold_{fold}"
        required = [
            fold_dir / f"vae_model_fold_{fold}.pt",
            fold_dir / "vae_norm_params.joblib",
            fold_dir / "train_dev_subjects_fold.csv",
            fold_dir / "test_subjects_fold.csv",
        ]
        if all(p.exists() for p in required):
            count += 1
    return count


def aggregate_ready(output_root: Path) -> bool:
    patterns = ["*aggregate*.csv", "*pooled_comparison*.csv", "*comparison_summary*.csv"]
    return any(list(output_root.glob(p)) for p in patterns)


def run_status_audit(args: argparse.Namespace) -> int:
    datos_mounted = DATOS_MOUNT.exists() and os.path.ismount(str(DATOS_MOUNT))
    if not datos_mounted:
        print(f"[WARN] Datos drive not mounted. Stage A status may be incomplete.")

    preflight_root = resolve(args.preflight_root)
    output_root = resolve(args.output_root)
    big_root = args.big_disk_run_root if args.big_disk_run_root.is_absolute() else resolve(args.big_disk_run_root)

    try:
        matrix = read_matrix(preflight_root)
        rows = selected_rows(matrix, args.candidate)
    except FileNotFoundError as exc:
        print(f"[WARN] Cannot read preflight matrix: {exc}")
        candidate_ids = EXPECTED_CANDIDATES if args.candidate == "all" else [args.candidate]
        rows = pd.DataFrame({
            "candidate_id": candidate_ids,
            "readout_feature_set": ["z_plus_age_sex"] * len(candidate_ids),
        })

    patched_path = output_root / PATCHED_METADATA_SUBPATH
    audit_rows: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        candidate_id = str(row["candidate_id"])
        readout_fs = str(row.get("readout_feature_set", "z_plus_age_sex"))
        local_run = output_root / "runs" / candidate_id
        big_run = local_run if args.no_external_symlink else big_root / candidate_id
        readout_dir = readout_dir_for(local_run, readout_fs)

        n_folds = folds_stage_a_complete_count(local_run)
        a_done = stage_a_complete(local_run)
        b_done = stage_b_complete(readout_dir)
        agg = aggregate_ready(output_root)

        if b_done and agg:
            state = "aggregate_ready"
        elif b_done:
            state = "stageB_complete"
        elif a_done:
            state = "stageA_complete"
        elif n_folds > 0:
            state = "partial"
        else:
            state = "missing"

        audit_rows.append({
            "candidate_id": candidate_id,
            "local_run_dir": str(local_run),
            "big_run_dir": str(big_run),
            "big_run_exists": big_run.exists(),
            "folds_stageA_complete": f"{n_folds}/{N_FOLDS}",
            "stageB_readout": "present" if readout_dir.exists() else "absent",
            "metrics_aggregable": "present" if b_done else "absent",
            "state": state,
        })

    df = pd.DataFrame(audit_rows)
    print(f"\n=== Candidate Status Audit (mfrrecovered035_clfpoollocked FULL 5x5) ===")
    print(f"Patched metadata: {patched_path} ({'exists' if patched_path.exists() else 'NOT_YET_CREATED'})")
    print(df.to_string(index=False))
    print()
    if output_root.exists():
        audit_path = output_root / "status_audit.csv"
        df.to_csv(audit_path, index=False)
        print(f"Saved: {audit_path}")
    return 0


def timestamped_quarantine(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.parent / f"{path.name}_quarantine_{stamp}"
    idx = 1
    while candidate.exists() or candidate.is_symlink():
        idx += 1
        candidate = path.parent / f"{path.name}_quarantine_{stamp}_{idx}"
    return candidate


def symlink_target_matches(link_path: Path, target_path: Path) -> bool:
    if not link_path.is_symlink():
        return False
    raw = Path(os.readlink(link_path))
    if not raw.is_absolute():
        raw = link_path.parent / raw
    try:
        return raw.resolve() == target_path.resolve()
    except FileNotFoundError:
        return raw.absolute() == target_path.absolute()


def prepare_run_path(spec: Dict[str, Any], force_clean: bool, no_external_symlink: bool, resume: bool = False) -> List[str]:
    local_run = spec["local_run_dir"]
    big_run = spec["big_run_dir"]
    quarantines: List[str] = []
    local_run.parent.mkdir(parents=True, exist_ok=True)
    if no_external_symlink:
        if local_run.exists() and force_clean:
            q = timestamped_quarantine(local_run)
            shutil.move(str(local_run), str(q))
            quarantines.append(str(q))
        local_run.mkdir(parents=True, exist_ok=True)
        return quarantines
    big_run.mkdir(parents=True, exist_ok=True)
    if any(big_run.iterdir()) and not (force_clean or resume):
        raise RuntimeError(f"Refusing non-empty external run target without --resume or --force-clean: {big_run}")
    if local_run.is_symlink():
        if not symlink_target_matches(local_run, big_run):
            if not force_clean:
                raise RuntimeError(f"{local_run} points to {os.readlink(local_run)}, expected {big_run}; use --force-clean.")
            q = timestamped_quarantine(local_run)
            shutil.move(str(local_run), str(q))
            quarantines.append(str(q))
    elif local_run.exists():
        if not force_clean:
            raise RuntimeError(f"Refusing existing non-symlink run dir: {local_run}; use --force-clean.")
        q = timestamped_quarantine(local_run)
        shutil.move(str(local_run), str(q))
        quarantines.append(str(q))
    if not local_run.exists() and not local_run.is_symlink():
        local_run.symlink_to(big_run, target_is_directory=True)
    if not symlink_target_matches(local_run, big_run):
        raise RuntimeError(f"Failed to prepare symlink {local_run} -> {big_run}")
    if force_clean and any(big_run.iterdir()):
        q = timestamped_quarantine(big_run)
        q.mkdir(parents=True)
        for child in list(big_run.iterdir()):
            shutil.move(str(child), str(q / child.name))
        quarantines.append(str(q))
    return quarantines


def verify_fresh_stage_a(run_dir: Path, started_at: float) -> None:
    stale: List[str] = []
    missing: List[str] = []
    for fold in range(1, N_FOLDS + 1):
        ckpt = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
        elif ckpt.stat().st_mtime < started_at:
            stale.append(str(ckpt))
    if missing or stale:
        raise RuntimeError(f"Stage A checkpoint freshness failed. Missing={missing}; stale={stale}")


def audit_supervised_exclusion_for_specs(output_root: Path, specs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    violations: List[str] = []
    for spec in specs:
        run_dir = Path(spec["local_run_dir"])
        for fold in range(1, N_FOLDS + 1):
            for split_name, filename in [
                ("train_dev", "train_dev_subjects_fold.csv"),
                ("test", "test_subjects_fold.csv"),
            ]:
                path = run_dir / f"fold_{fold}" / filename
                if not path.exists():
                    rows.append({
                        "candidate_id": spec["candidate_id"],
                        "fold": fold,
                        "split": split_name,
                        "path": str(path),
                        "status": "missing",
                        "n": np.nan,
                        "CN": np.nan,
                        "AD": np.nan,
                        "contains_035_S_6927": np.nan,
                        "contains_128_S_2002": np.nan,
                    })
                    continue
                df = pd.read_csv(path)
                subjects = set(df.get("SubjectID", pd.Series(dtype=str)).astype(str).tolist())
                has_035 = CLASSIFIER_EXCLUDED_SUBJECT in subjects
                has_128 = UNRESOLVED_SUBJECT in subjects
                counts = df.get("ResearchGroup_Mapped", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
                status = "ok"
                if has_035 or has_128:
                    status = "violation"
                    violations.append(f"{spec['candidate_id']} fold {fold} {split_name} contains excluded subject")
                rows.append({
                    "candidate_id": spec["candidate_id"],
                    "fold": fold,
                    "split": split_name,
                    "path": str(path),
                    "status": status,
                    "n": int(len(df)),
                    "CN": int(counts.get("CN", 0)),
                    "AD": int(counts.get("AD", 0)),
                    "contains_035_S_6927": bool(has_035),
                    "contains_128_S_2002": bool(has_128),
                })
    audit = pd.DataFrame(rows)
    if not audit.empty:
        output_root.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output_root / "supervised_exclusion_audit.csv", index=False)
        write_markdown_table(output_root / "supervised_exclusion_audit.md", audit)
    if violations:
        raise RuntimeError("Supervised exclusion audit failed: " + "; ".join(violations))
    return audit


def verify_stage_b_classifier_pool_counts(readout_dir: Path, candidate_id: str) -> None:
    pooled_path = readout_dir / "classifier_sweep_pooled_metrics.csv"
    if not pooled_path.exists():
        raise RuntimeError(f"{candidate_id}: Stage B pooled metrics missing: {pooled_path}")
    pooled = pd.read_csv(pooled_path)
    if "readout_feature_set" not in pooled.columns:
        pooled["readout_feature_set"] = "z_plus_age_sex"
    mask = (
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        & pooled["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    )
    rows = pooled.loc[mask].copy()
    if rows.empty:
        raise RuntimeError(f"{candidate_id}: Stage B primary row not found in {pooled_path}")
    row = rows.iloc[0]
    n = int(row.get("n", -1))
    n_cn = int(row.get("n_cn", -1))
    n_ad = int(row.get("n_ad", -1))
    if (n, n_cn, n_ad) != (396, 300, 96):
        raise RuntimeError(
            f"{candidate_id}: classifier pool count violation. "
            f"Expected n=396 CN=300 AD=96, got n={n} CN={n_cn} AD={n_ad}."
        )


def run_logged(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"UTC start: {now_utc()}\n")
        log.write("COMMAND: " + shlex.join(list(cmd)) + "\n")
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
        log.write(f"\nUTC end: {now_utc()}\nEXIT_CODE: {code}\n")
    return int(code)


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
    else:
        path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def write_full5x5_target_validation(output_root: Path) -> None:
    df = pd.DataFrame(FULL5X5_TARGET_VALIDATION_ROWS)
    df.to_csv(output_root / "full5x5_target_validation.csv", index=False)
    write_markdown_table(output_root / "full5x5_target_validation.md", df)


def write_manifest(output_root: Path, rows: List[Dict[str, Any]], dry_run: bool, patched_metadata_path: Path) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "run_manifest.csv", index=False)
    write_markdown_table(output_root / "run_manifest.md", df)
    lines = [
        "# Conditional beta-VAE Manufacturer FULL 5x5 — recovered-035 VAE pool, classifier pool locked",
        "",
        f"Generated UTC: {now_utc()}",
        f"Dry-run only: {dry_run}",
        "",
        "## Metadata patch",
        "",
        f"035_S_6927 appended with Manufacturer=SIEMENS to create a branch-local patched metadata CSV.",
        f"Patched metadata: {patched_metadata_path}",
        "128_S_2002 remains excluded (unresolvable Manufacturer, no ADNI record).",
        "035_S_6927 is explicitly excluded from every supervised classifier train/dev/test split.",
        "Expected classifier pool: n=396, CN=300, AD=96.",
        "",
        "Both FULL 5x5 candidates use --vae_required_metadata_cols Manufacturer.",
        "After the patch: only 128_S_2002 is removed from the VAE pool per fold (1 subject instead of 2).",
        "The supervised classifier pool must remain locked at n=396, CN=300, AD=96; only the VAE pool changes by recovering 035_S_6927.",
        "",
        "## FULL 5x5 target selection",
        "",
        "This controlled FULL 5x5 confirmation targets `[1,0,2]`, not `[1]`, because the training-maturity audit",
        "identified `ch1_0_2_decoder_only_manufacturer` as the strongest valid candidate: positive AUC delta,",
        "positive PR-AUC delta, positive Manufacturer-leakage reduction, and 2/3 right-censored or near-horizon folds.",
        "The matched pair is therefore `ch1_0_2_baseline_unconditioned` versus",
        "`ch1_0_2_decoder_only_manufacturer`.",
        "",
        "Stage A trains FULL 5x5 VAE (4480 epochs / 56 cycles) with dummy canonical logreg (n_iter_logreg=1), ignored for ranking.",
        "Stage B is classifier-only logreg_l2 on z plus Age/Sex with true inner-CV OOF thresholds.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_full5x5_target_validation(output_root)
    return df


def main() -> int:
    args = parse_args()

    if args.status:
        return run_status_audit(args)

    if args.confirm_training and args.dry_run:
        raise SystemExit("Use either --dry-run or --confirm-training, not both.")
    if not args.confirm_training:
        args.dry_run = True

    check_datos_drive()

    preflight_root = resolve(args.preflight_root)
    output_root = resolve(args.output_root)
    big_root = args.big_disk_run_root if args.big_disk_run_root.is_absolute() else resolve(args.big_disk_run_root)

    # Create patched metadata (branch-local; does NOT modify Datos)
    output_root.mkdir(parents=True, exist_ok=True)
    patched_metadata_path, patch_audit_info = create_patched_metadata(
        output_root=output_root,
        source_metadata_path=SOURCE_TRAINING_READY_METADATA,
        full_metadata_path=FULL_SUBJECT_METADATA,
        dry_run=False,  # always create it; it's a local file, not a Datos modification
    )
    write_patch_audit(output_root, patch_audit_info)

    matrix = read_matrix(preflight_root)
    rows = selected_rows(matrix, args.candidate)
    specs = [
        build_commands(
            row,
            preflight_root=preflight_root,
            output_root=output_root,
            big_disk_run_root=big_root,
            python_executable=args.python_executable,
            no_external_symlink=args.no_external_symlink,
            force_clean=args.force_clean,
            patched_metadata_path=patched_metadata_path,
        )
        for _, row in rows.iterrows()
    ]
    validation_rows = validate_specs(specs, real=args.confirm_training, patched_metadata_path=patched_metadata_path)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output_root / "launcher_validation.csv", index=False)
    write_markdown_table(output_root / "launcher_validation.md", validation)
    if validation["status"].eq("error").any():
        print(validation.to_string(index=False))
        raise SystemExit("Launcher validation failed; no training launched.")

    manifest_rows: List[Dict[str, Any]] = []
    for spec in specs:
        a_done = stage_a_complete(spec["local_run_dir"])
        b_done = stage_b_complete(spec["readout_dir"])
        if (a_done or b_done) and not args.dry_run and not (args.resume or args.force_clean):
            raise SystemExit(f"{spec['candidate_id']} already has outputs; use --resume or --force-clean.")
        manifest_rows.append({
            "candidate_id": spec["candidate_id"],
            "condition_id": spec["condition_id"],
            "channels": json.dumps(spec["channels"]),
            "channel_names": spec["channel_names"],
            "vae_conditioning_mode": spec["vae_conditioning_mode"],
            "vae_conditioning_vars": spec["vae_conditioning_vars"],
            "corr_lambda": spec["corr_lambda"],
            "readout_feature_set": spec["readout_feature_set"],
            "run_dir": str(spec["local_run_dir"]),
            "big_run_dir": str(spec["big_run_dir"]),
            "readout_dir": str(spec["readout_dir"]),
            "metadata_path_used": str(patched_metadata_path),
            "stage_a_completed_before": bool(a_done),
            "stage_b_completed_before": bool(b_done),
            "stage_a_command": shlex.join(spec["stage_a_real_cmd"]),
            "stage_b_command": shlex.join(spec["stage_b_cmd"]),
        })
    write_manifest(output_root, manifest_rows, dry_run=args.dry_run, patched_metadata_path=patched_metadata_path)

    command_log: Dict[str, Any] = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "preflight_root": str(preflight_root),
        "output_root": str(output_root),
        "big_disk_run_root": str(big_root),
        "candidate": args.candidate,
        "dry_run": bool(args.dry_run),
        "confirm_training": bool(args.confirm_training),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_datos_modified": False,
        "ledger_modified": False,
        "existing_model_outputs_modified": False,
        "patched_metadata_created": str(patched_metadata_path),
        "classifier_exclude_subject_ids": [CLASSIFIER_EXCLUDED_SUBJECT],
        "expected_classifier_pool": {"n": 396, "CN": 300, "AD": 96},
        "vae_required_metadata_cols": ["Manufacturer"],
        "full5x5_target_validation": FULL5X5_TARGET_VALIDATION_ROWS,
        "target_note": (
            "Focused FULL 5x5 targets ch1_0_2_decoder_only_manufacturer, not ch1_decoder_only_manufacturer, "
            "because the training-maturity audit found positive AUC/PR-AUC deltas, Manufacturer leakage "
            "reduction, and 2/3 right-censored or near-horizon folds for the [1,0,2] matched pair."
        ),
        "patch_audit": patch_audit_info,
        "candidates": [],
    }

    if args.dry_run:
        for spec in specs:
            print(f"\n## {spec['candidate_id']}")
            print("Stage A real command preview:")
            print(shlex.join(spec["stage_a_real_cmd"]))
            print("Stage B command preview:")
            print(shlex.join(spec["stage_b_cmd"]))
        (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return 0

    command_log["training_launched"] = True
    for spec in specs:
        run_record: Dict[str, Any] = {"candidate_id": spec["candidate_id"], "started_utc": now_utc()}
        if args.resume and stage_b_complete(spec["readout_dir"]):
            run_record["status"] = "skipped_complete"
            command_log["candidates"].append(run_record)
            continue

        quarantines = prepare_run_path(spec, force_clean=args.force_clean, no_external_symlink=args.no_external_symlink, resume=args.resume)
        run_record["quarantines"] = quarantines
        log_path = output_root / "logs" / f"mfrrecovered035_{spec['candidate_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        if not (args.resume and stage_a_complete(spec["local_run_dir"])):
            started = time.time()
            code = run_logged(spec["stage_a_real_cmd"], log_path)
            run_record["stage_a_exit_code"] = code
            if code != 0:
                run_record["status"] = "stage_a_failed"
                command_log["candidates"].append(run_record)
                (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
                raise SystemExit(f"Stage A failed for {spec['candidate_id']} with code {code}")
            verify_fresh_stage_a(spec["local_run_dir"], started)
        else:
            run_record["stage_a_exit_code"] = "skipped_resume_complete"

        audit_supervised_exclusion_for_specs(output_root, [spec])
        run_record["supervised_exclusion_audit"] = "pass"

        code = run_logged(spec["stage_b_cmd"], log_path)
        run_record["stage_b_exit_code"] = code
        if code != 0:
            run_record["status"] = "stage_b_failed"
            command_log["candidates"].append(run_record)
            (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
            raise SystemExit(f"Stage B failed for {spec['candidate_id']} with code {code}")
        verify_stage_b_classifier_pool_counts(spec["readout_dir"], spec["candidate_id"])
        run_record["stage_b_classifier_pool_counts"] = "pass_n396_cn300_ad96"
        run_record["status"] = "completed"
        run_record["finished_utc"] = now_utc()
        command_log["candidates"].append(run_record)
        (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    if not args.skip_aggregation:
        aggregate_cmd = [
            args.python_executable,
            str(AGGREGATOR_SCRIPT),
            "--output-root", str(output_root),
            "--preflight-root", str(preflight_root),
        ]
        print("Running read-only aggregation:")
        print(shlex.join(aggregate_cmd))
        code = run_logged(aggregate_cmd, output_root / "logs" / f"mfrrecovered035_aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        if code != 0:
            raise SystemExit(f"Aggregation failed with code {code}")
    integrity_cmd = [
        args.python_executable,
        str(INTEGRITY_SCRIPT),
        "--root", str(output_root),
    ]
    print("Run integrity audit:")
    print(shlex.join(integrity_cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
