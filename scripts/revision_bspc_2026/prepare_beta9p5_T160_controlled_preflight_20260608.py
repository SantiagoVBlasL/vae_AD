#!/usr/bin/env python3
"""Prepare/preflight beta9.5 and T0=160 controlled FULL branches.

Reference:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Candidates:
  recover035_latent384_beta9p5_T80_h10000_p560_full5x5
  recover035_latent384_beta3p75_T160_h10000_p560_full5x5

Default behavior is dry-run/preflight only. Real training is guarded by
--confirm-training and single-variant selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"

PROMOTED_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
PREFLIGHT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/beta9p5_T160_controlled_preflight_20260608"
OBJECTIVE_AUDIT_SUMMARY = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/promoted_loss_scheduler_deep_audit_chweighted_preflight_20260608/"
    / "vae_objective_scheduler_summary.csv"
)

STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
OOF_SCORE_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"

REFERENCE_RUN_ID = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REFERENCE_CHANNELS = [1, 0, 2]
REFERENCE_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
PRIMARY_MODEL = "logreg_l2"

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


@dataclass(frozen=True)
class Candidate:
    variant: str
    run_id: str
    config_name: str
    launcher_name: str
    allowed_parameter_diffs: Mapping[str, Tuple[Any, Any]]
    scientific_note: str

    @property
    def config_path(self) -> Path:
        return PROJECT_ROOT / "configs/runs" / self.config_name

    @property
    def launcher_path(self) -> Path:
        return PROJECT_ROOT / "scripts/revision_bspc_2026" / self.launcher_name


CANDIDATES: Dict[str, Candidate] = {
    "beta9p5": Candidate(
        variant="beta9p5",
        run_id="recover035_latent384_beta9p5_T80_h10000_p560_full5x5",
        config_name="adni_v5_1c_recover035_latent384_beta9p5_T80_h10000_p560_full5x5.json",
        launcher_name="run_adni_v5_1c_recover035_latent384_beta9p5_T80_h10000_p560_full5x5.py",
        allowed_parameter_diffs={"beta_vae": (3.75, 9.5)},
        scientific_note=(
            "Effective-regularization stress test. beta9.5 is the simple linear beta*KLD/D "
            "analogue of the ch1-only regime, but prior beta6.5 evidence showed beta scaling "
            "was not linear."
        ),
    ),
    "T160": Candidate(
        variant="T160",
        run_id="recover035_latent384_beta3p75_T160_h10000_p560_full5x5",
        config_name="adni_v5_1c_recover035_latent384_beta3p75_T160_h10000_p560_full5x5.json",
        launcher_name="run_adni_v5_1c_recover035_latent384_beta3p75_T160_h10000_p560_full5x5.py",
        allowed_parameter_diffs={"lr_scheduler_T0": (80, 160)},
        scientific_note=(
            "Scheduler sensitivity only. The objective/scheduler audit did not support T0=160 "
            "as a primary change because T0=80 is exactly beta-cycle aligned."
        ),
    ),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, max_rows: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    view = df.head(max_rows)
    try:
        md = view.to_markdown(index=False)
    except Exception:
        md = view.to_string(index=False)
    if len(df) > max_rows:
        md += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    (out_dir / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def selected_names_from_master(config: Dict[str, Any]) -> List[str]:
    master = list(config["channel_names_master_in_tensor_order"])
    return [master[int(i)] for i in config["parameters"]["channels_to_use"]]


def make_candidate_config(reference: Dict[str, Any], candidate: Candidate) -> Dict[str, Any]:
    cfg = copy.deepcopy(reference)
    cfg["run_name"] = candidate.run_id
    cfg["description"] = (
        f"Controlled post-final FULL 5x5 branch derived strictly from {REFERENCE_RUN_ID}. "
        f"Allowed scientific diff: {candidate.allowed_parameter_diffs}. {candidate.scientific_note} "
        "All tensor, metadata, channel, architecture, dropout, optimizer, fold, classifier, and "
        "readout conventions otherwise match the promoted recover035 reference. Prepared as "
        "preflight; real training requires --confirm-training."
    )
    cfg["paths"]["output_dir"] = f"results/revision_bspc_2026/{candidate.run_id}"
    cfg["paths"]["big_disk_output_dir"] = f"/media/diego/Datos/vae_AD_results/revision_bspc_2026/{candidate.run_id}"
    cfg["paths"]["split_preview_csv"] = f"results/revision_bspc_2026/{candidate.run_id}_split_preview.csv"
    cfg["paths"]["split_preview_summary_csv"] = f"results/revision_bspc_2026/{candidate.run_id}_split_preview_summary.csv"
    for key, (_, new_value) in candidate.allowed_parameter_diffs.items():
        cfg["parameters"][key] = new_value
    return cfg


def launcher_text(candidate: Candidate) -> str:
    return f'''#!/usr/bin/env python3
"""Guarded launcher for {candidate.run_id}.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026.prepare_beta9p5_T160_controlled_preflight_20260608 import main


if __name__ == "__main__":
    main(default_variant="{candidate.variant}")
'''


def ensure_configs_and_launchers(reference: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATES.values():
        cfg = make_candidate_config(reference, candidate)
        config_changed = write_text_if_changed(candidate.config_path, json.dumps(cfg, indent=2, sort_keys=False) + "\n")
        launcher_changed = write_text_if_changed(candidate.launcher_path, launcher_text(candidate))
        rows.append(
            {
                "variant": candidate.variant,
                "run_id": candidate.run_id,
                "config_path": rel(candidate.config_path),
                "config_written_or_updated": config_changed,
                "launcher_path": rel(candidate.launcher_path),
                "launcher_written_or_updated": launcher_changed,
            }
        )
    return pd.DataFrame(rows)


def config_diff(reference: Dict[str, Any], target: Dict[str, Any], candidate: Candidate) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for section in ["parameters", "paths", "split_strategy"]:
        src = reference.get(section, {})
        tgt = target.get(section, {})
        for key in sorted(set(src) | set(tgt)):
            old = src.get(key, "<MISSING>")
            new = tgt.get(key, "<MISSING>")
            if old != new:
                rows.append(
                    {
                        "variant": candidate.variant,
                        "section": section,
                        "key": key,
                        "reference_value": old,
                        "candidate_value": new,
                    }
                )
    for key in ["run_name", "description", "selected_channel_names"]:
        if reference.get(key) != target.get(key):
            rows.append(
                {
                    "variant": candidate.variant,
                    "section": "metadata",
                    "key": key,
                    "reference_value": reference.get(key),
                    "candidate_value": target.get(key),
                }
            )
    return pd.DataFrame(rows)


def validate_strict_diff(reference: Dict[str, Any], target: Dict[str, Any], candidate: Candidate) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ref_params = reference["parameters"]
    tgt_params = target["parameters"]
    if set(ref_params) != set(tgt_params):
        raise RuntimeError(
            f"{candidate.variant}: parameter key set changed: "
            f"missing={sorted(set(ref_params) - set(tgt_params))}, extra={sorted(set(tgt_params) - set(ref_params))}"
        )
    param_diffs = {k: (ref_params[k], tgt_params[k]) for k in ref_params if ref_params[k] != tgt_params[k]}
    if param_diffs != dict(candidate.allowed_parameter_diffs):
        raise RuntimeError(f"{candidate.variant}: unexpected parameter diffs: {param_diffs}")
    for key, (old, new) in candidate.allowed_parameter_diffs.items():
        rows.append(
            {
                "variant": candidate.variant,
                "diff_type": "allowed_scientific_parameter",
                "field": key,
                "reference_value": old,
                "candidate_value": new,
                "pass": True,
            }
        )
    for section in ["selected_channel_names", "channel_names_master_in_tensor_order", "split_strategy"]:
        if reference.get(section) != target.get(section):
            raise RuntimeError(f"{candidate.variant}: unexpected diff in {section}")
        rows.append(
            {
                "variant": candidate.variant,
                "diff_type": "unchanged_required_field",
                "field": section,
                "reference_value": reference.get(section),
                "candidate_value": target.get(section),
                "pass": True,
            }
        )
    allowed_path_diffs = {"output_dir", "big_disk_output_dir", "split_preview_csv", "split_preview_summary_csv"}
    path_diffs = {
        k: (reference["paths"].get(k), target["paths"].get(k))
        for k in set(reference["paths"]) | set(target["paths"])
        if reference["paths"].get(k) != target["paths"].get(k)
    }
    if set(path_diffs) != allowed_path_diffs:
        raise RuntimeError(f"{candidate.variant}: unexpected path diffs: {path_diffs}")
    for key, (_, new) in sorted(path_diffs.items()):
        if candidate.run_id not in str(new):
            raise RuntimeError(f"{candidate.variant}: path {key} lacks run_id provenance: {new}")
        rows.append(
            {
                "variant": candidate.variant,
                "diff_type": "allowed_provenance_path",
                "field": key,
                "reference_value": path_diffs[key][0],
                "candidate_value": new,
                "pass": True,
            }
        )
    for path_key in ["training_script", "global_tensor_path", "metadata_path"]:
        if reference["paths"].get(path_key) != target["paths"].get(path_key):
            raise RuntimeError(f"{candidate.variant}: {path_key} changed")
    return pd.DataFrame(rows)


def load_metadata(config: Dict[str, Any]) -> pd.DataFrame:
    meta = pd.read_csv(resolve(config["paths"]["metadata_path"]))
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Sex"] = meta["Sex"].astype(str)
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    bad = meta[required].isna().any(axis=1)
    if bad.any():
        raise RuntimeError("Metadata has missing required values:\n" + meta.loc[bad, required].head(20).to_string(index=False))
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID rows.")
    return meta


def validate_subject_pool(meta: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dx_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf_counts = clf["ResearchGroup_Mapped"].value_counts().to_dict()
    for dx, expected in EXPECTED_VAE_COUNTS.items():
        rows.append(
            {
                "variant": candidate.variant,
                "check": f"vae_{dx}",
                "observed": int(dx_counts.get(dx, 0)),
                "expected": expected,
                "pass": int(dx_counts.get(dx, 0)) == expected,
            }
        )
    for dx, expected in EXPECTED_CLF_COUNTS.items():
        rows.append(
            {
                "variant": candidate.variant,
                "check": f"classifier_{dx}",
                "observed": int(clf_counts.get(dx, 0)),
                "expected": expected,
                "pass": int(clf_counts.get(dx, 0)) == expected,
            }
        )
    rows.extend(
        [
            {
                "variant": candidate.variant,
                "check": "vae_pool_n",
                "observed": int(len(meta)),
                "expected": 647,
                "pass": int(len(meta)) == 647,
            },
            {
                "variant": candidate.variant,
                "check": "classifier_pool_n",
                "observed": int(len(clf)),
                "expected": 397,
                "pass": int(len(clf)) == 397,
            },
            {
                "variant": candidate.variant,
                "check": "035_S_6927_present",
                "observed": bool(meta["SubjectID"].eq(RECOVER_SUBJECT).any()),
                "expected": True,
                "pass": bool(meta["SubjectID"].eq(RECOVER_SUBJECT).any()),
            },
            {
                "variant": candidate.variant,
                "check": "128_S_2002_absent_from_metadata",
                "observed": bool(meta["SubjectID"].eq(EXCLUDED_SUBJECT).any()),
                "expected": False,
                "pass": not bool(meta["SubjectID"].eq(EXCLUDED_SUBJECT).any()),
            },
        ]
    )
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError(f"{candidate.variant}: subject pool validation failed:\n" + out.to_string(index=False))
    return out


def validate_tensor(config: Dict[str, Any], meta: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    tensor_path = resolve(config["paths"]["global_tensor_path"])
    with np.load(tensor_path, allow_pickle=True) as zf:
        tensor_shape = tuple(int(v) for v in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        subject_ids = [str(x) for x in zf["subject_ids"].astype(str)]
    channels = list(config["parameters"]["channels_to_use"])
    selected = [channel_names[int(i)] for i in channels]
    if channels != REFERENCE_CHANNELS or selected != REFERENCE_SELECTED_NAMES:
        raise RuntimeError(f"{candidate.variant}: selected channel mismatch: indices={channels}, names={selected}")
    if selected_names_from_master(config) != REFERENCE_SELECTED_NAMES:
        raise RuntimeError(f"{candidate.variant}: config master-channel lookup mismatch")
    if tensor_shape[1:] != (7, 131, 131):
        raise RuntimeError(f"{candidate.variant}: expected full tensor CxROI shape 7x131x131, got {tensor_shape[1:]}")
    if int(meta["tensor_idx"].max()) >= tensor_shape[0]:
        raise RuntimeError(f"{candidate.variant}: metadata tensor_idx exceeds tensor row count")
    return pd.DataFrame(
        [
            {
                "variant": candidate.variant,
                "tensor_path": str(tensor_path),
                "tensor_shape_full": str(tensor_shape),
                "selected_channels_to_use": str(channels),
                "selected_channel_names": " | ".join(selected),
                "selected_tensor_shape_if_loaded": str((len(meta), len(channels), tensor_shape[2], tensor_shape[3])),
                "n_metadata_rows": int(len(meta)),
                "n_tensor_subject_ids": int(len(subject_ids)),
                "035_S_6927_in_tensor": bool(RECOVER_SUBJECT in subject_ids),
                "128_S_2002_in_tensor": bool(EXCLUDED_SUBJECT in subject_ids),
                "pass": True,
            }
        ]
    )


def split_key(df: pd.DataFrame) -> pd.Series:
    return df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)


def count_dx_mfr(df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "n": int(len(df)),
        "CN": int((df["ResearchGroup_Mapped"] == "CN").sum()),
        "MCI": int((df["ResearchGroup_Mapped"] == "MCI").sum()),
        "AD": int((df["ResearchGroup_Mapped"] == "AD").sum()),
    }
    for mfr in ["GE", "Philips", "SIEMENS"]:
        row[f"Manufacturer_{mfr}"] = int((df["Manufacturer"].astype(str) == mfr).sum())
    return row


def validate_fold_feasibility(config: Dict[str, Any], meta: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    params = config["parameters"]
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    outer_key = split_key(clf)
    if int(outer_key.value_counts().min()) < int(params["outer_folds"]):
        raise RuntimeError(f"{candidate.variant}: outer stratification infeasible")
    outer = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=int(params["seed"]))
    rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(clf, outer_key), start=1):
        train_dev = clf.iloc[train_idx].copy()
        test = clf.iloc[test_idx].copy()
        inner_key = split_key(train_dev)
        inner_feasible = int(inner_key.value_counts().min()) >= int(params["inner_folds"])
        test_subjects = set(test["SubjectID"].astype(str))
        vae_pool = meta[~meta["SubjectID"].astype(str).isin(test_subjects)].copy()
        vae_key = split_key(vae_pool)
        val_ok = True
        val_error = ""
        try:
            train_test_split(
                np.arange(len(vae_pool)),
                test_size=float(params["vae_val_split_ratio"]),
                random_state=int(params["seed"]) + fold,
                stratify=vae_key if int(vae_key.value_counts().min()) >= 2 else None,
            )
        except Exception as exc:  # pragma: no cover
            val_ok = False
            val_error = str(exc)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test), ("vae_pool", vae_pool)]:
            row: Dict[str, Any] = {"variant": candidate.variant, "fold": fold, "split_component": split_name}
            row.update(count_dx_mfr(df))
            row["all_required_diagnoses_present"] = (
                bool(row["CN"] > 0 and row["AD"] > 0)
                if split_name.startswith("classifier")
                else bool(row["CN"] > 0 and row["MCI"] > 0 and row["AD"] > 0)
            )
            row["all_three_manufacturers_present"] = all(row.get(f"Manufacturer_{m}", 0) > 0 for m in ["GE", "Philips", "SIEMENS"])
            row["outer_test_overlap_in_vae_pool"] = int(len(test_subjects & set(vae_pool["SubjectID"].astype(str))))
            row["inner_cv_feasible"] = inner_feasible if split_name == "classifier_train_dev" else ""
            row["vae_internal_val_split_feasible"] = val_ok if split_name == "vae_pool" else ""
            row["vae_internal_val_split_error"] = val_error if split_name == "vae_pool" else ""
            rows.append(row)
    out = pd.DataFrame(rows)
    failed = out[
        (~out["all_required_diagnoses_present"])
        | (~out["all_three_manufacturers_present"])
        | (out["outer_test_overlap_in_vae_pool"] != 0)
        | ((out["split_component"] == "classifier_train_dev") & (out["inner_cv_feasible"] != True))
        | ((out["split_component"] == "vae_pool") & (out["vae_internal_val_split_feasible"] != True))
    ]
    if not failed.empty:
        raise RuntimeError(f"{candidate.variant}: fold feasibility failed:\n" + failed.to_string(index=False))
    return out


def stale_markers(path: Path) -> List[Path]:
    if not (path.exists() or path.is_symlink()):
        return []
    if path.is_symlink() and not path.exists():
        return []
    markers: List[Path] = []
    for child in path.iterdir():
        if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    for nested in path.rglob("latent_cache"):
        markers.append(nested)
    return sorted(set(markers), key=lambda p: str(p))


def output_audit(config: Dict[str, Any], candidate: Candidate) -> pd.DataFrame:
    local = resolve(config["paths"]["output_dir"])
    big = Path(config["paths"]["big_disk_output_dir"])
    local_exists = local.exists() or local.is_symlink()
    local_markers = stale_markers(local)
    big_markers = stale_markers(big)
    setup = f"mkdir -p {shlex.quote(str(big))} && ln -s {shlex.quote(str(big))} {shlex.quote(str(local))}"
    return pd.DataFrame(
        [
            {
                "variant": candidate.variant,
                "local_output_dir": str(local),
                "big_disk_output_dir": str(big),
                "big_disk_parent_exists": bool(big.parent.exists()),
                "big_disk_target_exists": bool(big.exists()),
                "local_exists": bool(local_exists),
                "local_is_symlink": bool(local.is_symlink()),
                "symlink_target": str(local.resolve()) if local_exists else "",
                "target_match": bool(local_exists and local.is_symlink() and local.resolve() == big.resolve()),
                "preflight_clean_missing_output_ok": bool((not local_exists) and (not big.exists()) and big.parent.exists()),
                "stale_marker_count": len(local_markers) + len(big_markers),
                "stale_markers": " | ".join(str(p) for p in local_markers + big_markers),
                "symlink_setup_command_if_needed": setup,
            }
        ]
    )


def append_arg(cmd: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    if value is None:
        return
    cmd.append(f"--{name}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    values: List[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def build_stage_a_command(config: Dict[str, Any], python_exe: str = PYTHON_EXE, dry_run: bool = False) -> List[str]:
    cmd = [
        python_exe,
        str(resolve(config["paths"]["training_script"])),
        "--global_tensor_path",
        str(resolve(config["paths"]["global_tensor_path"])),
        "--metadata_path",
        str(resolve(config["paths"]["metadata_path"])),
        "--output_dir",
        str(resolve(config["paths"]["output_dir"])),
    ]
    for key, value in config["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def build_stage_b_command(config: Dict[str, Any], python_exe: str = PYTHON_EXE) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(out),
        "--output-dir",
        str(out / "classifier_only_readout"),
        "--outer-folds",
        str(config["parameters"]["outer_folds"]),
        "--inner-folds",
        str(config["parameters"]["inner_folds"]),
        "--models",
        PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]


def build_oof_command(config: Dict[str, Any], candidate: Candidate, python_exe: str = PYTHON_EXE) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(OOF_SCORE_SCRIPT),
        "--run-dir",
        str(out),
        "--output-dir",
        str(PROJECT_ROOT / f"results/revision_bspc_2026/{candidate.run_id}_stageB_oof_score_calibration"),
    ]


def validate_commands(config: Dict[str, Any], candidate: Candidate, stage_a: Sequence[str], stage_b: Sequence[str], oof: Sequence[str]) -> pd.DataFrame:
    expected_beta = str(config["parameters"]["beta_vae"])
    expected_t0 = str(config["parameters"]["lr_scheduler_T0"])
    checks = [
        ("stage_a_channels", values_after_flag(stage_a, "--channels_to_use"), ["1", "0", "2"]),
        ("stage_a_latent_dim", values_after_flag(stage_a, "--latent_dim"), ["384"]),
        ("stage_a_beta", values_after_flag(stage_a, "--beta_vae"), [expected_beta]),
        ("stage_a_recon_loss", values_after_flag(stage_a, "--recon_loss_mode"), ["mse_sum_batchmean_current"]),
        ("stage_a_epochs", values_after_flag(stage_a, "--epochs_vae"), ["10000"]),
        ("stage_a_cycles", values_after_flag(stage_a, "--cyclical_beta_n_cycles"), ["125"]),
        ("stage_a_T0", values_after_flag(stage_a, "--lr_scheduler_T0"), [expected_t0]),
        ("stage_a_patience", values_after_flag(stage_a, "--early_stopping_patience_vae"), ["560"]),
        ("stage_a_dropout", values_after_flag(stage_a, "--dropout_rate_vae"), ["0.15"]),
        ("stage_a_dropout_scope", values_after_flag(stage_a, "--vae_dropout_scope"), ["legacy_all"]),
        ("stage_a_block_order", values_after_flag(stage_a, "--vae_block_order"), ["legacy_act_norm"]),
        ("stage_a_metadata_features", values_after_flag(stage_a, "--metadata_features"), ["Age", "Sex"]),
        ("stage_b_model", values_after_flag(stage_b, "--models"), [PRIMARY_MODEL]),
        ("stage_b_outer", values_after_flag(stage_b, "--outer-folds"), ["5"]),
        ("stage_b_inner", values_after_flag(stage_b, "--inner-folds"), ["5"]),
        ("oof_run_dir", values_after_flag(oof, "--run-dir"), [values_after_flag(stage_b, "--run-dir")[0]]),
    ]
    rows = [
        {
            "variant": candidate.variant,
            "check": name,
            "actual": " ".join(actual),
            "expected": " ".join(expected),
            "pass": actual == expected,
        }
        for name, actual, expected in checks
    ]
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError(f"{candidate.variant}: command validation failed:\n" + out.to_string(index=False))
    return out


def command_rows(config: Dict[str, Any], candidate: Candidate) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stage_a = build_stage_a_command(config)
    stage_b = build_stage_b_command(config)
    oof = build_oof_command(config, candidate)
    commands = pd.DataFrame(
        [
            {"variant": candidate.variant, "name": "stage_a_training_after_confirm", "command": shlex.join(stage_a)},
            {"variant": candidate.variant, "name": "stage_b_classifier_only_after_completion", "command": shlex.join(stage_b)},
            {"variant": candidate.variant, "name": "stageB_oof_score_calibration_after_completion", "command": shlex.join(oof)},
            {"variant": candidate.variant, "name": "launcher_dry_run", "command": shlex.join([PYTHON_EXE, str(candidate.launcher_path), "--dry-run"])},
            {
                "variant": candidate.variant,
                "name": "launcher_real_training_guarded",
                "command": shlex.join([PYTHON_EXE, str(candidate.launcher_path), "--confirm-training"]),
            },
        ]
    )
    return commands, validate_commands(config, candidate, stage_a, stage_b, oof)


def validate_candidate(reference: Dict[str, Any], candidate: Candidate) -> Dict[str, pd.DataFrame]:
    target = load_json(candidate.config_path)
    meta = load_metadata(target)
    commands, command_checks = command_rows(target, candidate)
    return {
        "config_diff": config_diff(reference, target, candidate),
        "strict_diff_guard": validate_strict_diff(reference, target, candidate),
        "subject_pool_validation": validate_subject_pool(meta, candidate),
        "tensor_validation": validate_tensor(target, meta, candidate),
        "fold_feasibility": validate_fold_feasibility(target, meta, candidate),
        "output_symlink_and_stale_audit": output_audit(target, candidate),
        "planned_commands": commands,
        "command_validation": command_checks,
    }


def concat_frames(results: Mapping[str, Mapping[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    return pd.concat([frames[key] for frames in results.values()], ignore_index=True)


def effective_beta_estimates() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not OBJECTIVE_AUDIT_SUMMARY.exists():
        return pd.DataFrame(
            [
                {
                    "variant": "beta9p5",
                    "status": "missing_objective_audit_summary",
                    "source": str(OBJECTIVE_AUDIT_SUMMARY),
                }
            ]
        )
    summary = pd.read_csv(OBJECTIVE_AUDIT_SUMMARY)

    label_col = "run_label" if "run_label" in summary.columns else "model_id"

    def pick(*labels: str) -> pd.Series:
        label_values = summary[label_col].astype(str)
        matches = summary[label_values.isin(labels)]
        if matches.empty:
            raise RuntimeError(f"Missing objective summary row for any of: {labels}")
        return matches.iloc[0]

    promoted = pick("promoted_ch102_beta3p75", "promoted_ch102_latent384_beta3p75")
    ch1 = pick("ch1only_beta3p75", "ch1only_latent384_beta3p75")
    beta6 = pick("beta6p5_ch102", "beta6p5_latent384_ch102", "latent384_beta6p5")
    promoted_ratio = float(promoted["beta_KLD_over_D_best_mean"])
    ch1_ratio = float(ch1["beta_KLD_over_D_best_mean"])
    beta6_ratio = float(beta6["beta_KLD_over_D_best_mean"])
    linear_beta9p5 = promoted_ratio * (9.5 / 3.75)
    observed_slope = (beta6_ratio - promoted_ratio) / (6.5 - 3.75)
    observed_beta9p5 = promoted_ratio + observed_slope * (9.5 - 3.75)
    implied_beta_to_ch1 = 3.75 + (ch1_ratio - promoted_ratio) / observed_slope if observed_slope else np.nan
    rows.append(
        {
            "variant": "beta9p5",
            "promoted_beta": 3.75,
            "candidate_beta": 9.5,
            "promoted_beta_KLD_over_D": promoted_ratio,
            "ch1only_beta_KLD_over_D": ch1_ratio,
            "beta6p5_beta_KLD_over_D": beta6_ratio,
            "linear_scaling_expected_beta_KLD_over_D": linear_beta9p5,
            "observed_beta3p75_to_beta6p5_slope_per_beta": observed_slope,
            "observed_slope_expected_beta9p5_beta_KLD_over_D": observed_beta9p5,
            "observed_slope_implied_beta_to_match_ch1": implied_beta_to_ch1,
            "interpretation": (
                "linear scaling puts beta9.5 near ch1-only, but observed beta6.5 response predicts "
                "a much lower effective ratio; beta9.5 is a stress-test, not a confirmed analogue"
            ),
        }
    )
    rows.append(
        {
            "variant": "T160",
            "promoted_beta": 3.75,
            "candidate_beta": 3.75,
            "promoted_beta_KLD_over_D": promoted_ratio,
            "ch1only_beta_KLD_over_D": ch1_ratio,
            "beta6p5_beta_KLD_over_D": beta6_ratio,
            "linear_scaling_expected_beta_KLD_over_D": promoted_ratio,
            "observed_beta3p75_to_beta6p5_slope_per_beta": observed_slope,
            "observed_slope_expected_beta9p5_beta_KLD_over_D": promoted_ratio,
            "observed_slope_implied_beta_to_match_ch1": implied_beta_to_ch1,
            "interpretation": "T0=160 is a scheduler-only sensitivity; no beta*KLD/D shift is assumed before training",
        }
    )
    return pd.DataFrame(rows)


def run_launcher_dryruns(variants: Sequence[str], out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        candidate = CANDIDATES[variant]
        cmd = [PYTHON_EXE, str(candidate.launcher_path), "--dry-run", "--preflight-dir", str(out_dir)]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        (out_dir / f"{variant}_launcher_dryrun_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (out_dir / f"{variant}_launcher_dryrun_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        rows.append(
            {
                "variant": variant,
                "command": shlex.join(cmd),
                "returncode": int(proc.returncode),
                "stdout_file": f"{variant}_launcher_dryrun_stdout.txt",
                "stderr_file": f"{variant}_launcher_dryrun_stderr.txt",
                "pass": int(proc.returncode) == 0,
            }
        )
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError("Launcher dry-run failed:\n" + out.to_string(index=False))
    return out


def run_core_dry_run(config: Dict[str, Any]) -> int:
    cmd = build_stage_a_command(config, dry_run=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return int(proc.returncode)


def write_package(
    out_dir: Path,
    reference: Dict[str, Any],
    creation: pd.DataFrame,
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    variants: Sequence[str],
    dry_run: bool,
    launcher_validation: pd.DataFrame | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(creation[creation["variant"].isin(variants)], out_dir, "generated_artifacts")
    for key in [
        "config_diff",
        "strict_diff_guard",
        "subject_pool_validation",
        "tensor_validation",
        "fold_feasibility",
        "output_symlink_and_stale_audit",
        "planned_commands",
        "command_validation",
    ]:
        write_table(concat_frames(results, key), out_dir, key)
    if launcher_validation is not None:
        write_table(launcher_validation, out_dir, "launcher_dryrun_validation")
    estimates = effective_beta_estimates()
    write_table(estimates[estimates["variant"].isin(variants)], out_dir, "beta9p5_effective_regularization_estimate")
    run_manifest = pd.DataFrame(
        [
            {
                "variant": c.variant,
                "run_id": c.run_id,
                "config": rel(c.config_path),
                "launcher": rel(c.launcher_path),
                "allowed_scientific_diffs": str(dict(c.allowed_parameter_diffs)),
                "channels_to_use": str(REFERENCE_CHANNELS),
                "selected_channel_names": " | ".join(REFERENCE_SELECTED_NAMES),
                "latent_dim": reference["parameters"]["latent_dim"],
                "beta_vae": load_json(c.config_path)["parameters"]["beta_vae"],
                "lr_scheduler_T0": load_json(c.config_path)["parameters"]["lr_scheduler_T0"],
                "epochs_vae": reference["parameters"]["epochs_vae"],
                "cycles": reference["parameters"]["cyclical_beta_n_cycles"],
                "patience": reference["parameters"]["early_stopping_patience_vae"],
                "status": "preflight_only",
                "scientific_note": c.scientific_note,
            }
            for c in CANDIDATES.values()
            if c.variant in variants
        ]
    )
    write_table(run_manifest, out_dir, "run_manifest")
    setup_rows: List[Dict[str, Any]] = []
    for c in CANDIDATES.values():
        if c.variant not in variants:
            continue
        target = load_json(c.config_path)
        local = resolve(target["paths"]["output_dir"])
        big = Path(target["paths"]["big_disk_output_dir"])
        setup_rows.append(
            {
                "variant": c.variant,
                "mkdir_command": shlex.join(["mkdir", "-p", str(big)]),
                "symlink_command": shlex.join(["ln", "-s", str(big), str(local)]),
                "note": "Run only if local output symlink is absent before real training.",
            }
        )
    write_table(pd.DataFrame(setup_rows), out_dir, "symlink_setup_commands")

    (out_dir / "T160_scheduler_sensitivity_note.md").write_text(
        "\n".join(
            [
                "# T0=160 Scheduler Sensitivity Note",
                "",
                "This branch is preflighted as a controlled scheduler sensitivity only.",
                "The prior objective/scheduler audit found that `T0=80` is exactly aligned to the beta-cycle length",
                "(`epochs_vae / cyclical_beta_n_cycles = 10000 / 125 = 80`) and did not identify a consistent",
                "history-based reason to make `T0=160` a primary model change.",
                "",
                "Therefore `recover035_latent384_beta3p75_T160_h10000_p560_full5x5` should be interpreted",
                "as a post-final sensitivity branch if launched, not as an audit-supported default.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stale = concat_frames(results, "output_symlink_and_stale_audit")
    checks = concat_frames(results, "strict_diff_guard")
    validation_ok = bool(checks["pass"].all()) and int(stale["stale_marker_count"].sum()) == 0
    if launcher_validation is not None:
        validation_ok = validation_ok and bool(launcher_validation["pass"].all())
    readme = [
        "# beta9.5 and T0=160 Controlled Preflight",
        "",
        "This package creates and validates two controlled FULL 5x5 branches based strictly on the promoted recover035 reference.",
        "No real training, OASIS scoring, tensor mutation, metadata mutation, ledger mutation, or model-artifact modification was performed.",
        "",
        "## Candidates",
        "",
        "- `recover035_latent384_beta9p5_T80_h10000_p560_full5x5`: beta_vae 3.75 -> 9.5 only.",
        "- `recover035_latent384_beta3p75_T160_h10000_p560_full5x5`: lr_scheduler_T0 80 -> 160 only.",
        "",
        "## Fixed Promoted Convention",
        "",
        "- channels [1,0,2] = Pearson Full, OMST, MI.",
        "- latent_dim 384, epochs 10000, beta cycles 125, patience 560, dropout 0.15 legacy_all.",
        "- convtranspose decoder, legacy_act_norm block order, mse_sum_batchmean_current reconstruction loss.",
        "- classifier readout plan: logreg_l2, z_plus_age_sex, post-completion OOF score calibration.",
        "- same tensor and patched recover035 metadata paths as promoted reference.",
        "",
        "## Validation Summary",
        "",
        f"- Strict diff guard rows: {len(checks)}; all pass = {bool(checks['pass'].all())}.",
        f"- Stale output markers across variants: {int(stale['stale_marker_count'].sum())}.",
        f"- Launcher dry-run validation pass = {bool(launcher_validation['pass'].all()) if launcher_validation is not None else 'not_run_in_child_context'}.",
        "- VAE pool: CN=300, MCI=250, AD=97.",
        "- Classifier pool: CN=300, AD=97.",
        "- 035_S_6927 included; 128_S_2002 absent from metadata pools.",
        "- All five outer folds and inner folds are feasible.",
        f"- Overall validation pass = {validation_ok}.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out_dir / "dry_run_report.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    write_json(
        out_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "dry_run": dry_run,
            "training_launched": False,
            "variants": list(variants),
            "promoted_reference_config": rel(PROMOTED_CONFIG),
            "preflight_root": rel(out_dir),
            "validation_pass": validation_ok,
            "guardrails": [
                "no real training",
                "no OASIS scoring",
                "no tensor modification",
                "no metadata modification",
                "no ledger modification",
                "no model artifact modification",
            ],
        },
    )


def parse_args(default_variant: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--variant", choices=["all", *CANDIDATES.keys()], default=default_variant or "all")
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    parser.add_argument("--confirm-training", action="store_true", help="Launch real Stage A training for one selected variant after guards pass.")
    parser.add_argument("--force-clean", action="store_true", help="Reserved guard for intentional stale-output handling.")
    return parser.parse_args()


def run_command(command: Iterable[str]) -> None:
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def main(default_variant: str | None = None) -> None:
    args = parse_args(default_variant=default_variant)
    reference = load_json(PROMOTED_CONFIG)
    creation = ensure_configs_and_launchers(reference)
    variants = list(CANDIDATES) if args.variant == "all" else [args.variant]
    results = {variant: validate_candidate(reference, CANDIDATES[variant]) for variant in variants}
    out_dir = resolve(args.preflight_dir)
    dry_run = args.dry_run or not args.confirm_training

    launcher_validation = None
    if default_variant is None and dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        launcher_validation = run_launcher_dryruns(variants, out_dir)

    write_package(out_dir, reference, creation, results, variants, dry_run=dry_run, launcher_validation=launcher_validation)

    print(f"Preflight package: {out_dir}")
    print("Validated variants: " + ", ".join(variants))
    print("\nStrict scientific diffs:")
    print(concat_frames(results, "strict_diff_guard").to_string(index=False))
    print("\nEffective beta*KLD/D estimates:")
    print(effective_beta_estimates()[effective_beta_estimates()["variant"].isin(variants)].to_string(index=False))

    if dry_run:
        if default_variant is not None and len(variants) == 1:
            target = load_json(CANDIDATES[variants[0]].config_path)
            rc = run_core_dry_run(target)
            if rc != 0:
                raise RuntimeError(f"Core training script --dry-run failed for {variants[0]} with return code {rc}")
        print("Dry-run OK. No training launched.")
        return

    if args.variant == "all":
        raise RuntimeError("Real training requires a single --variant, not --variant all.")
    output = concat_frames(results, "output_symlink_and_stale_audit")
    if int(output["stale_marker_count"].iloc[0]) and not args.force_clean:
        raise RuntimeError("Refusing real training with stale output markers.")
    if not bool(output["target_match"].iloc[0]):
        raise RuntimeError("Refusing real training because the local output symlink is not already pointed at the big-disk target.")
    target = load_json(CANDIDATES[args.variant].config_path)
    run_command(build_stage_a_command(target))


if __name__ == "__main__":
    main()
