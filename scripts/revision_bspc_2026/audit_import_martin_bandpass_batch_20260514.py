#!/usr/bin/env python3
"""Audit Martin's 2026-05-14 ADNI v5.1 bandpass ROISignals batch.

Read-only with respect to existing datasets:
- no training;
- no tensor construction;
- no Python bandpass in the final path;
- no modification of the live ledger.

This batch is audited against v5 base, the previous v5.1_batch20260513
candidate, and the 20260513 import decision so that already-imported subjects
are not proposed again.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_import_martin_bandpass_batch_20260513 import (  # noqa: E402
    EXPECTED_ROIS,
    TR_SECONDS,
    build_vs_ledger,
    candidate_rank,
    clean,
    discover_check_files,
    discover_roisignals,
    prepare_output_dir,
    qc_roisignals,
    read_csv,
)


BATCH_ROOT = PROJECT_ROOT / "data" / "OneDrive_1_14-5-2026"
LEDGER_CURRENT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
    / "ADNI_v5_1_DATA_LEDGER_CURRENT.csv"
)
PREVIOUS_BATCH_DECISION = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260513"
    / "batch_import_decision.csv"
)
V5_BASE_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_dparsf10000_no_pybandpass"
)
V5_BASE_TENSOR = (
    V5_BASE_ROOT
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
V51_BATCH20260513_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260513_no_pybandpass"
)
V51_BATCH20260513_TENSOR = (
    V51_BATCH20260513_ROOT
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260513_no_pybandpass.npz"
)
MASTER_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_master_subject_manifest_first_visit.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "martin_bandpass_batch_20260514"
LEDGER_CANDIDATE_NAME = "ADNI_v5_1_DATA_LEDGER_20260514_after_batch2_candidate.csv"

BATCH_LABEL = "20260514_bandpass_batch2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit/import-readiness for Martin bandpass ROISignals batch 20260514.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER_CURRENT)
    parser.add_argument("--previous-batch-decision", type=Path, default=PREVIOUS_BATCH_DECISION)
    parser.add_argument("--v5-base-tensor", type=Path, default=V5_BASE_TENSOR)
    parser.add_argument("--v51-batch20260513-tensor", type=Path, default=V51_BATCH20260513_TENSOR)
    parser.add_argument("--master-manifest", type=Path, default=MASTER_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_subject_ids_from_tensor(path: Path, label: str) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"{label} tensor not found: {path}")
    with np.load(path, allow_pickle=False) as zf:
        if "subject_ids" not in zf.files:
            raise RuntimeError(f"{label} tensor missing subject_ids: {path}")
        return [str(x) for x in zf["subject_ids"].astype(str).tolist()]


def index_by_subject(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df.empty or "SubjectID" not in df.columns:
        return {}
    return df.drop_duplicates("SubjectID", keep="first").set_index("SubjectID", drop=False).to_dict(orient="index")


def join_unique(values: Sequence[Any]) -> str:
    return "|".join(sorted({clean(v) for v in values if clean(v)}))


def normalize_yes_no(value: Any) -> str:
    text = clean(value).lower()
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    return ""


def first_visit_authorized(master_row: Dict[str, Any] | None) -> bool:
    if not master_row:
        return False
    selected = clean(master_row.get("is_first_visit_selected", ""))
    if selected == "" or selected.lower() in {"true", "1", "yes"}:
        return True
    return False


def subject_candidate_ambiguity(sub: pd.DataFrame) -> Tuple[bool, str]:
    if sub.empty:
        return True, "no_candidate_files"
    qc_pass = sub[sub["qc_signal_pass"].eq("yes")]
    stages = sorted({clean(x) for x in sub["stage_guess"].tolist() if clean(x)})
    suffixes = set(clean(x) for x in sub["suffix"].tolist() if clean(x))
    same_stage_mat_txt_pair = len(stages) == 1 and len(sub) <= 2 and suffixes.issubset({".mat", ".txt"})
    if len(qc_pass) > 2:
        return True, f"more_than_two_qc_pass_files:{len(qc_pass)}"
    if len(stages) > 1:
        return True, f"multiple_stage_guesses:{'|'.join(stages)}"
    if len(sub) > 2 and not same_stage_mat_txt_pair:
        return True, f"multiple_candidate_files:{len(sub)}"
    return False, ""


def build_previous_imports(
    subjects: Sequence[str],
    inventory: pd.DataFrame,
    ledger: pd.DataFrame,
    previous_decision: pd.DataFrame,
    master_manifest: pd.DataFrame,
    v5_ids: Sequence[str],
    v51_ids: Sequence[str],
) -> pd.DataFrame:
    ledger_by_sid = index_by_subject(ledger)
    previous_by_sid = index_by_subject(previous_decision)
    master_by_sid = index_by_subject(master_manifest)
    v5_set = set(v5_ids)
    v51_set = set(v51_ids)
    inv_counts = (
        inventory.groupby("SubjectID")
        .agg(
            roisignals_file_count=("path", "nunique"),
            stage_guesses=("stage_guess", join_unique),
            suffixes=("suffix", join_unique),
            paths=("path", join_unique),
        )
        .reset_index()
        .set_index("SubjectID", drop=False)
        .to_dict(orient="index")
    )
    rows: List[Dict[str, Any]] = []
    for sid in sorted(subjects):
        ledger_row = ledger_by_sid.get(sid)
        previous_row = previous_by_sid.get(sid)
        master_row = master_by_sid.get(sid)
        in_v5 = sid in v5_set
        in_v51 = sid in v51_set
        previous_import_ready = clean(previous_row.get("import_ready", "")).lower() == "yes" if previous_row else False
        already_imported_batch20260513 = previous_import_ready or (in_v51 and not in_v5)
        dicom_no = ledger_row is not None and clean(ledger_row.get("dicom_series_ok", "")).upper() == "NO"
        in_ledger = ledger_row is not None
        in_master = master_row is not None
        labels: List[str] = []
        if in_v5:
            labels.append("already_in_v5_base")
        if already_imported_batch20260513:
            labels.append("already_imported_batch20260513")
        if dicom_no:
            labels.append("dicom_no")
        if not in_v5 and not already_imported_batch20260513 and not dicom_no and in_ledger:
            labels.append("new_import_candidate")
        if not in_ledger and in_master:
            labels.append("outside_ledger_but_in_master")
        if not in_ledger and not in_master:
            labels.append("outside_all_manifests")
        primary = "new_import_candidate"
        for candidate in [
            "dicom_no",
            "already_in_v5_base",
            "already_imported_batch20260513",
            "new_import_candidate",
            "outside_ledger_but_in_master",
            "outside_all_manifests",
        ]:
            if candidate in labels:
                primary = candidate
                break
        inv = inv_counts.get(sid, {})
        rows.append(
            {
                "SubjectID": sid,
                "classification_primary": primary,
                "classification_labels": "|".join(labels),
                "in_ledger": "yes" if in_ledger else "no",
                "ledger_scope": clean(ledger_row.get("ledger_scope", "")) if in_ledger else "",
                "dicom_series_ok": clean(ledger_row.get("dicom_series_ok", "")) if in_ledger else "",
                "ledger_processing_status": clean(ledger_row.get("processing_status", "")) if in_ledger else "",
                "priority_batch": clean(ledger_row.get("priority_batch", "")) if in_ledger else "",
                "in_master_manifest": "yes" if in_master else "no",
                "master_first_visit_authorized": "yes" if first_visit_authorized(master_row) else "no",
                "master_diagnosis": clean(master_row.get("ResearchGroup_Mapped", "")) if in_master else "",
                "master_manufacturer": clean(master_row.get("Manufacturer", "")) if in_master else "",
                "master_image_id": clean(master_row.get("ImageID", "")) if in_master else "",
                "master_visit": clean(master_row.get("Visit", "")) if in_master else "",
                "in_v5_base": "yes" if in_v5 else "no",
                "in_v5_1_batch20260513": "yes" if in_v51 else "no",
                "in_previous_batch20260513_decision": "yes" if previous_row is not None else "no",
                "previous_batch20260513_import_ready": "yes" if previous_import_ready else "no",
                "already_imported_batch20260513": "yes" if already_imported_batch20260513 else "no",
                "roisignals_file_count": int(inv.get("roisignals_file_count", 0) or 0),
                "stage_guesses": clean(inv.get("stage_guesses", "")),
                "suffixes": clean(inv.get("suffixes", "")),
                "paths": clean(inv.get("paths", "")),
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)


def decide_subjects(
    qc: pd.DataFrame,
    previous_imports: pd.DataFrame,
    ledger: pd.DataFrame,
    master_manifest: pd.DataFrame,
) -> pd.DataFrame:
    previous_by_sid = index_by_subject(previous_imports)
    ledger_by_sid = index_by_subject(ledger)
    master_by_sid = index_by_subject(master_manifest)
    rows: List[Dict[str, Any]] = []
    for sid, sub in qc.groupby("SubjectID"):
        sub = sub.copy()
        sub["_rank"] = sub.apply(candidate_rank, axis=1)
        sub = sub.sort_values("_rank", kind="mergesort")
        best = sub.iloc[0]
        prev = previous_by_sid.get(sid, {})
        ledger_row = ledger_by_sid.get(sid)
        master_row = master_by_sid.get(sid)
        ambiguous, ambiguous_reason = subject_candidate_ambiguity(sub)
        labels = [x for x in clean(prev.get("classification_labels", "")).split("|") if x]
        if ambiguous and "duplicate_or_ambiguous" not in labels:
            labels.append("duplicate_or_ambiguous")
        reasons: List[str] = []
        if clean(prev.get("in_v5_base")) == "yes":
            reasons.append("already_in_v5_base")
        if clean(prev.get("in_v5_1_batch20260513")) == "yes":
            reasons.append("already_in_v5_1_batch20260513")
        if clean(prev.get("already_imported_batch20260513")) == "yes":
            reasons.append("already_imported_batch20260513")
        if ledger_row is not None and clean(ledger_row.get("dicom_series_ok", "")).upper() == "NO":
            reasons.append("dicom_series_ok_NO")
        if clean(best.get("qc_signal_pass")) != "yes":
            reasons.append(f"signal_qc_fail:{clean(best.get('qc_signal_reason'))}")
        if int(best.get("n_rois", 0) or 0) != EXPECTED_ROIS:
            reasons.append(f"n_rois={best.get('n_rois')}, expected 170")
        if clean(best.get("scale_label")) != "around_10000_global_scaled":
            reasons.append(f"scale_label={clean(best.get('scale_label'))}")
        if clean(best.get("stage_guess")) not in {"ARWSDCFN", "ARWSDCF"}:
            reasons.append(f"stage_not_explicit_F:{clean(best.get('stage_guess'))}")
        if clean(best.get("spectral_class")) == "unfiltered_like":
            reasons.append("spectral_unfiltered_like")
        if ambiguous:
            reasons.append(f"duplicate_or_ambiguous:{ambiguous_reason}")
        if not first_visit_authorized(master_row):
            reasons.append("not_authorized_by_first_visit_master_manifest")
        if ledger_row is not None and clean(ledger_row.get("python_bandpass_requested", "")) not in {"", "NO"}:
            reasons.append(f"python_bandpass_requested={clean(ledger_row.get('python_bandpass_requested'))}")
        import_ready_new = "yes" if not reasons else "no"
        if import_ready_new == "yes":
            decision_reason = "new_nonduplicate_qc_pass_dparsf_bandpass_first_visit_authorized_no_python_bandpass"
        else:
            decision_reason = "; ".join(reasons)
        if ambiguous:
            primary = "duplicate_or_ambiguous"
        else:
            primary = clean(prev.get("classification_primary", ""))
        rows.append(
            {
                "SubjectID": sid,
                "import_ready_new": import_ready_new,
                "decision_reason": decision_reason,
                "classification_primary": primary,
                "classification_labels": "|".join(labels),
                "selected_path": clean(best.get("path")),
                "selected_relative_path": clean(best.get("relative_path")),
                "selected_suffix": clean(best.get("suffix")),
                "stage_guess": clean(best.get("stage_guess")),
                "n_timepoints": best.get("n_timepoints"),
                "n_rois": best.get("n_rois"),
                "finite_fraction": best.get("finite_fraction"),
                "all_nan_columns": best.get("all_nan_columns"),
                "scale_label": clean(best.get("scale_label")),
                "qc_signal_pass": clean(best.get("qc_signal_pass")),
                "qc_signal_reason": clean(best.get("qc_signal_reason")),
                "qc_signal_warning": clean(best.get("qc_signal_warning")),
                "spectral_class": clean(best.get("spectral_class")),
                "energy_below_0p01": best.get("energy_below_0p01"),
                "energy_0p01_0p08": best.get("energy_0p01_0p08"),
                "energy_above_0p08": best.get("energy_above_0p08"),
                "in_ledger": clean(prev.get("in_ledger")),
                "ledger_scope": clean(prev.get("ledger_scope")),
                "priority_batch": clean(prev.get("priority_batch")),
                "dicom_series_ok": clean(prev.get("dicom_series_ok")),
                "in_master_manifest": clean(prev.get("in_master_manifest")),
                "master_first_visit_authorized": clean(prev.get("master_first_visit_authorized")),
                "master_diagnosis": clean(prev.get("master_diagnosis")),
                "master_manufacturer": clean(prev.get("master_manufacturer")),
                "master_image_id": clean(prev.get("master_image_id")),
                "master_visit": clean(prev.get("master_visit")),
                "in_v5_base": clean(prev.get("in_v5_base")),
                "in_v5_1_batch20260513": clean(prev.get("in_v5_1_batch20260513")),
                "already_imported_batch20260513": clean(prev.get("already_imported_batch20260513")),
                "matched_file_count": int(len(sub)),
                "duplicate_or_ambiguous": "yes" if ambiguous else "no",
                "duplicate_or_ambiguous_reason": ambiguous_reason,
                "python_bandpass_requested": clean(ledger_row.get("python_bandpass_requested", "NO")) if ledger_row is not None else "NO",
                "python_bandpass_final": "OFF",
                "all_candidate_paths": "|".join(clean(x) for x in sub["path"]),
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)


def append_note(existing: str, addition: str) -> str:
    existing = clean(existing)
    addition = clean(addition)
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing} | {addition}"


def update_candidate_ledger(ledger: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    for col in [
        "SubjectID",
        "ImageID",
        "Visit",
        "Diagnosis",
        "Manufacturer",
        "ledger_scope",
        "priority_batch",
        "dicom_series_ok",
        "dicom_issue_reason",
        "roisignals_status",
        "processing_status",
        "action_needed",
        "uploaded_batch",
        "uploaded_path",
        "included_in_dataset_version",
        "python_bandpass_requested",
        "notes_martin",
        "notes_santiago",
    ]:
        if col not in out.columns:
            out[col] = ""
    existing_ids = set(out["SubjectID"].astype(str).tolist())
    new_rows: List[Dict[str, Any]] = []
    for _, decision in decisions.iterrows():
        sid = clean(decision.get("SubjectID"))
        note = (
            f"{BATCH_LABEL}: import_ready_new={clean(decision.get('import_ready_new'))}; "
            f"class={clean(decision.get('classification_primary'))}; "
            f"stage={clean(decision.get('stage_guess'))}; n_rois={decision.get('n_rois')}; "
            f"finite={float(decision.get('finite_fraction', np.nan)):.4f}; "
            f"scale={clean(decision.get('scale_label'))}; spectral={clean(decision.get('spectral_class'))}; "
            f"reason={clean(decision.get('decision_reason'))}"
        )
        if sid in existing_ids:
            idx = out.index[out["SubjectID"].eq(sid)]
            out.loc[idx, "notes_santiago"] = out.loc[idx, "notes_santiago"].map(lambda x: append_note(x, note))
            if clean(decision.get("import_ready_new")) == "yes":
                out.loc[idx, "roisignals_status"] = "qc_pass"
                out.loc[idx, "processing_status"] = "uploaded"
                out.loc[idx, "uploaded_batch"] = BATCH_LABEL
                out.loc[idx, "uploaded_path"] = clean(decision.get("selected_path"))
                out.loc[idx, "python_bandpass_requested"] = "NO"
            continue
        if clean(decision.get("import_ready_new")) != "yes":
            continue
        new_rows.append(
            {
                "SubjectID": sid,
                "ImageID": clean(decision.get("master_image_id")),
                "Visit": clean(decision.get("master_visit")),
                "Diagnosis": clean(decision.get("master_diagnosis")),
                "Manufacturer": clean(decision.get("master_manufacturer")),
                "ledger_scope": "batch20260514_outside_ledger_master_authorized",
                "priority_batch": "batch20260514_extra",
                "dicom_series_ok": "UNKNOWN",
                "dicom_issue_reason": "",
                "roisignals_status": "qc_pass",
                "processing_status": "uploaded",
                "action_needed": "calculate_incremental_connectivity",
                "uploaded_batch": BATCH_LABEL,
                "uploaded_path": clean(decision.get("selected_path")),
                "included_in_dataset_version": "",
                "python_bandpass_requested": "NO",
                "notes_martin": "",
                "notes_santiago": note,
            }
        )
    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    return out


def write_readme(
    output_dir: Path,
    inventory: pd.DataFrame,
    check_df: pd.DataFrame,
    previous_imports: pd.DataFrame,
    vs_ledger: pd.DataFrame,
    decisions: pd.DataFrame,
    candidate_ledger: pd.DataFrame,
) -> None:
    unique_subjects = int(inventory["SubjectID"].nunique()) if not inventory.empty else 0
    already_prev = int(previous_imports["already_imported_batch20260513"].eq("yes").sum()) if not previous_imports.empty else 0
    already_v5 = int(previous_imports["in_v5_base"].eq("yes").sum()) if not previous_imports.empty else 0
    already_v51 = int(previous_imports["in_v5_1_batch20260513"].eq("yes").sum()) if not previous_imports.empty else 0
    not_prev = int(((previous_imports["in_v5_base"] != "yes") & (previous_imports["in_v5_1_batch20260513"] != "yes")).sum()) if not previous_imports.empty else 0
    ledger_matches = int(vs_ledger["in_ledger"].eq("yes").sum()) if not vs_ledger.empty else 0
    qc_pass = int(decisions["qc_signal_pass"].eq("yes").sum()) if not decisions.empty else 0
    import_ready_new = int(decisions["import_ready_new"].eq("yes").sum()) if not decisions.empty else 0
    outside_ledger = int(decisions["in_ledger"].eq("no").sum()) if not decisions.empty else 0
    outside_ledger_ready = int((decisions["in_ledger"].eq("no") & decisions["import_ready_new"].eq("yes")).sum()) if not decisions.empty else 0
    dicom_no = sorted(decisions.loc[decisions["dicom_series_ok"].eq("NO"), "SubjectID"].tolist()) if not decisions.empty else []
    ambiguous = int(decisions["duplicate_or_ambiguous"].eq("yes").sum()) if not decisions.empty else 0
    no_check_subjects = (
        sorted(set(inventory.loc[inventory["check_file_count"].astype(int).eq(0), "SubjectID"].astype(str).tolist()))
        if "check_file_count" in inventory.columns
        else []
    )
    no_check_ready = sorted(
        decisions.loc[decisions["SubjectID"].isin(no_check_subjects) & decisions["import_ready_new"].eq("yes"), "SubjectID"].tolist()
    )
    spectral_not_bandpassed = sorted(
        decisions.loc[~decisions["spectral_class"].eq("bandpassed_like"), "SubjectID"].astype(str).tolist()
    )
    spectral_not_bandpassed_ready = sorted(
        decisions.loc[
            ~decisions["spectral_class"].eq("bandpassed_like") & decisions["import_ready_new"].eq("yes"), "SubjectID"
        ].astype(str).tolist()
    )
    safe_incremental = "YES" if import_ready_new > 0 and ambiguous < unique_subjects else "NO"
    class_counts = (
        decisions["classification_primary"].value_counts(dropna=False).rename_axis("classification").reset_index(name="n")
        if not decisions.empty
        else pd.DataFrame(columns=["classification", "n"])
    )
    class_text = "```text\n" + class_counts.to_string(index=False) + "\n```"
    lines = [
        "# Martin Bandpass Batch 20260514 Audit",
        "",
        "Read-only audit. No training, no tensor construction, no final Python bandpass, and no live ledger modification were performed.",
        "",
        "## Answers",
        "",
        f"- Unique subjects in folder: `{unique_subjects}`.",
        f"- ROISignals files in folder: `{len(inventory)}`.",
        f"- Check files inventoried: `{len(check_df)}`.",
        f"- Already in batch20260513 import decision/final v5.1: `{already_prev}`.",
        f"- Already in v5 base: `{already_v5}`.",
        f"- Already in v5.1_batch20260513 tensor: `{already_v51}`.",
        f"- Subjects not already in v5/v5.1_batch20260513: `{not_prev}`.",
        f"- Subjects matching current ledger: `{ledger_matches}`.",
        f"- Subjects passing signal QC: `{qc_pass}`.",
        f"- import_ready_new subjects: `{import_ready_new}`.",
        f"- Subjects outside current ledger: `{outside_ledger}`.",
        f"- import_ready_new outside ledger but master-authorized: `{outside_ledger_ready}`.",
        f"- DICOM NO subjects in this batch: `{len(dicom_no)}`" + (f" ({', '.join(dicom_no)})" if dicom_no else "."),
        f"- Duplicate/ambiguous subjects: `{ambiguous}`.",
        f"- Subjects with ROISignals but no Check file: `{len(no_check_subjects)}`"
        + (f" ({', '.join(no_check_subjects)})" if no_check_subjects else "."),
        f"- import_ready_new subjects without Check file: `{len(no_check_ready)}`"
        + (f" ({', '.join(no_check_ready)})" if no_check_ready else "."),
        f"- Subjects with spectral class other than bandpassed_like: `{len(spectral_not_bandpassed)}`"
        + (f" ({', '.join(spectral_not_bandpassed)})" if spectral_not_bandpassed else "."),
        f"- import_ready_new subjects with spectral class other than bandpassed_like: `{len(spectral_not_bandpassed_ready)}`"
        + (f" ({', '.join(spectral_not_bandpassed_ready)})" if spectral_not_bandpassed_ready else "."),
        f"- Safe to calculate a new incremental tensor? `{safe_incremental}`. Use only `batch_import_decision.csv` rows with `import_ready_new=yes`.",
        "- Python bandpass final path: `OFF`. Spectral metrics are diagnostic only.",
        "- `v5.1_gecn9` used: `NO`; it remains quarantine.",
        "",
        "## Classification Counts",
        "",
        class_text,
        "",
        "## Candidate Ledger",
        "",
        f"- Candidate ledger rows: `{len(candidate_ledger)}`.",
        f"- Original live ledger was not modified: `{True}`.",
        f"- Candidate path: `{output_dir / LEDGER_CANDIDATE_NAME}`.",
        "",
        "## Outputs",
        "",
        "- `batch_roisignals_inventory.csv`",
        "- `batch_vs_previous_imports.csv`",
        "- `batch_vs_ledger_match.csv`",
        "- `batch_roisignals_qc.csv`",
        "- `batch_spectral_qc.csv`",
        "- `batch_import_decision.csv`",
        f"- `{LEDGER_CANDIDATE_NAME}`",
        "- `batch_check_inventory.csv`",
        "- `command_log.json`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.output_dir)

    ledger = read_csv(args.ledger)
    previous_decision = read_csv_optional(args.previous_batch_decision)
    master_manifest = read_csv_optional(args.master_manifest)
    v5_ids = load_subject_ids_from_tensor(args.v5_base_tensor, "v5_base")
    v51_ids = load_subject_ids_from_tensor(args.v51_batch20260513_tensor, "v5_1_batch20260513")

    check_df = discover_check_files(args.batch_root / "Check")
    inventory = discover_roisignals(args.batch_root, check_df)
    if inventory.empty:
        raise RuntimeError(f"No ROISignals files found under {args.batch_root / 'ResultsAAL3'}")
    subjects = sorted(inventory["SubjectID"].dropna().astype(str).unique().tolist())

    vs_ledger = build_vs_ledger(inventory, ledger, check_df)
    qc, spectral = qc_roisignals(inventory)
    previous_imports = build_previous_imports(
        subjects=subjects,
        inventory=inventory,
        ledger=ledger,
        previous_decision=previous_decision,
        master_manifest=master_manifest,
        v5_ids=v5_ids,
        v51_ids=v51_ids,
    )
    decisions = decide_subjects(qc, previous_imports, ledger, master_manifest)
    candidate_ledger = update_candidate_ledger(ledger, decisions)

    inventory.to_csv(args.output_dir / "batch_roisignals_inventory.csv", index=False)
    previous_imports.to_csv(args.output_dir / "batch_vs_previous_imports.csv", index=False)
    vs_ledger.to_csv(args.output_dir / "batch_vs_ledger_match.csv", index=False)
    qc.to_csv(args.output_dir / "batch_roisignals_qc.csv", index=False)
    spectral.to_csv(args.output_dir / "batch_spectral_qc.csv", index=False)
    decisions.to_csv(args.output_dir / "batch_import_decision.csv", index=False)
    candidate_ledger.to_csv(args.output_dir / LEDGER_CANDIDATE_NAME, index=False)
    check_df.to_csv(args.output_dir / "batch_check_inventory.csv", index=False)

    command = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "batch_root": str(args.batch_root.resolve()),
        "ledger": str(args.ledger.resolve()),
        "previous_batch_decision": str(args.previous_batch_decision.resolve()),
        "v5_base_tensor": str(args.v5_base_tensor.resolve()),
        "v51_batch20260513_tensor": str(args.v51_batch20260513_tensor.resolve()),
        "master_manifest": str(args.master_manifest.resolve()) if args.master_manifest.exists() else "",
        "output_dir": str(args.output_dir.resolve()),
        "batch_label": BATCH_LABEL,
        "tr_seconds": TR_SECONDS,
        "python_bandpass_final_pipeline": False,
        "training_run": False,
        "tensor_constructed": False,
        "ledger_original_modified": False,
        "uses_v5_1_gecn9": False,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(args.output_dir, inventory, check_df, previous_imports, vs_ledger, decisions, candidate_ledger)

    print(f"Wrote audit outputs to {args.output_dir}")
    print(f"roisignals_files={len(inventory)}")
    print(f"unique_subjects={inventory['SubjectID'].nunique()}")
    print(f"already_imported_batch20260513={previous_imports['already_imported_batch20260513'].eq('yes').sum()}")
    print(f"already_in_v5_base={previous_imports['in_v5_base'].eq('yes').sum()}")
    print(f"already_in_v5_1_batch20260513={previous_imports['in_v5_1_batch20260513'].eq('yes').sum()}")
    print(f"ledger_matches={vs_ledger['in_ledger'].eq('yes').sum()}")
    print(f"qc_signal_pass={decisions['qc_signal_pass'].eq('yes').sum()}")
    print(f"import_ready_new={decisions['import_ready_new'].eq('yes').sum()}")
    print(f"outside_ledger={decisions['in_ledger'].eq('no').sum()}")
    print(f"dicom_no={decisions['dicom_series_ok'].eq('NO').sum()}")
    print("No training. No tensor construction. No Python bandpass final. Original ledger not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
