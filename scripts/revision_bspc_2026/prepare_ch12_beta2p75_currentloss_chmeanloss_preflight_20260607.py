#!/usr/bin/env python3
"""Prepare/preflight ch12 beta2.75 FULL 5x5 post-final branches.

This script creates two controlled configs and launcher wrappers, then validates
that the only scientific changes versus the promoted recover035 reference are:

currentloss:
  channels_to_use [1,0,2] -> [1,2]
  beta_vae 3.75 -> 2.75

chmeanloss:
  channels_to_use [1,0,2] -> [1,2]
  beta_vae 3.75 -> 2.75
  recon_loss_mode mse_sum_batchmean_current -> mse_offdiag_channel_mean_sum

Default behavior is dry-run/preflight only. Real training requires
--confirm-training and a single --variant.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"

PROMOTED_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
PREFLIGHT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/ch12_beta2p75_currentloss_chmeanloss_preflight_20260607"
CH12_BETA375_RD = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/ch12_latent384_beta3p75_completion_promotion_gate_audit_20260605/rate_distortion_summary.csv"
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
TARGET_CHANNELS = [1, 2]
TARGET_SELECTED_NAMES = ["Pearson_Full_FisherZ_Signed", "MI_KNN_Symmetric"]
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
    user_label: str
    run_id: str
    config_name: str
    launcher_name: str
    beta_vae: float
    recon_loss_mode: str
    allowed_parameter_diffs: Mapping[str, Tuple[Any, Any]]

    @property
    def config_path(self) -> Path:
        return PROJECT_ROOT / "configs/runs" / self.config_name

    @property
    def launcher_path(self) -> Path:
        return PROJECT_ROOT / "scripts/revision_bspc_2026" / self.launcher_name


CANDIDATES: Dict[str, Candidate] = {
    "currentloss": Candidate(
        variant="currentloss",
        user_label="ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5",
        run_id="recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5",
        config_name="adni_v5_1c_recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5.json",
        launcher_name="run_adni_v5_1c_recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5.py",
        beta_vae=2.75,
        recon_loss_mode="mse_sum_batchmean_current",
        allowed_parameter_diffs={
            "channels_to_use": (REFERENCE_CHANNELS, TARGET_CHANNELS),
            "beta_vae": (3.75, 2.75),
        },
    ),
    "chmeanloss": Candidate(
        variant="chmeanloss",
        user_label="ch12_latent384_beta2p75_chmeanloss_T80_h10000_p560_full5x5",
        run_id="recover035_ch12_latent384_beta2p75_chmeanloss_T80_h10000_p560_full5x5",
        config_name="adni_v5_1c_recover035_ch12_latent384_beta2p75_chmeanloss_T80_h10000_p560_full5x5.json",
        launcher_name="run_adni_v5_1c_recover035_ch12_latent384_beta2p75_chmeanloss_T80_h10000_p560_full5x5.py",
        beta_vae=2.75,
        recon_loss_mode="mse_offdiag_channel_mean_sum",
        allowed_parameter_diffs={
            "channels_to_use": (REFERENCE_CHANNELS, TARGET_CHANNELS),
            "beta_vae": (3.75, 2.75),
            "recon_loss_mode": ("mse_sum_batchmean_current", "mse_offdiag_channel_mean_sum"),
        },
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
        f"Post-final exploratory reviewer-driven [1,2] channel-pair branch derived from "
        f"{REFERENCE_RUN_ID}. Controlled scientific diffs: channels_to_use [1,0,2] -> [1,2], "
        f"beta_vae 3.75 -> {candidate.beta_vae:g}, recon_loss_mode -> {candidate.recon_loss_mode}. "
        "All architecture, scheduler, dropout, fold, metadata, classifier, and recovered035 cohort "
        "settings are inherited from the promoted recover035 FULL convention. Prepared as preflight; "
        "real training requires --confirm-training."
    )
    cfg["selected_channel_names"] = list(TARGET_SELECTED_NAMES)
    cfg["paths"]["output_dir"] = f"results/revision_bspc_2026/{candidate.run_id}"
    cfg["paths"]["big_disk_output_dir"] = f"/media/diego/Datos/vae_AD_results/revision_bspc_2026/{candidate.run_id}"
    cfg["paths"]["split_preview_csv"] = f"results/revision_bspc_2026/{candidate.run_id}_split_preview.csv"
    cfg["paths"]["split_preview_summary_csv"] = f"results/revision_bspc_2026/{candidate.run_id}_split_preview_summary.csv"
    cfg["parameters"]["channels_to_use"] = list(TARGET_CHANNELS)
    cfg["parameters"]["beta_vae"] = candidate.beta_vae
    cfg["parameters"]["recon_loss_mode"] = candidate.recon_loss_mode
    return cfg


def launcher_text(candidate: Candidate) -> str:
    return f'''#!/usr/bin/env python3
"""Launcher/preflight wrapper for {candidate.run_id}.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training and passes through the guarded common preflight path.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026.prepare_ch12_beta2p75_currentloss_chmeanloss_preflight_20260607 import main


if __name__ == "__main__":
    main(default_variant="{candidate.variant}")
'''


def ensure_configs_and_launchers(reference: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATES.values():
        cfg = make_candidate_config(reference, candidate)
        config_text = json.dumps(cfg, indent=2, sort_keys=False) + "\n"
        config_changed = write_text_if_changed(candidate.config_path, config_text)
        launcher_changed = write_text_if_changed(candidate.launcher_path, launcher_text(candidate))
        rows.append(
            {
                "variant": candidate.variant,
                "user_label": candidate.user_label,
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

    if reference["selected_channel_names"] != REFERENCE_SELECTED_NAMES:
        raise RuntimeError(f"{candidate.variant}: promoted selected_channel_names did not match expected {REFERENCE_SELECTED_NAMES}")
    if target["selected_channel_names"] != TARGET_SELECTED_NAMES:
        raise RuntimeError(f"{candidate.variant}: target selected_channel_names did not match {TARGET_SELECTED_NAMES}")
    rows.append(
        {
            "variant": candidate.variant,
            "diff_type": "allowed_selected_channel_names",
            "field": "selected_channel_names",
            "reference_value": REFERENCE_SELECTED_NAMES,
            "candidate_value": target["selected_channel_names"],
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
    for key, (old, new) in sorted(path_diffs.items()):
        rows.append(
            {
                "variant": candidate.variant,
                "diff_type": "allowed_provenance_path",
                "field": key,
                "reference_value": old,
                "candidate_value": new,
                "pass": True,
            }
        )
    if reference["paths"]["global_tensor_path"] != target["paths"]["global_tensor_path"]:
        raise RuntimeError(f"{candidate.variant}: global tensor path changed")
    if reference["paths"]["metadata_path"] != target["paths"]["metadata_path"]:
        raise RuntimeError(f"{candidate.variant}: metadata path changed")
    return pd.DataFrame(rows)


def load_metadata(config: Dict[str, Any]) -> pd.DataFrame:
    meta = pd.read_csv(resolve(config["paths"]["metadata_path"]))
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    bad = meta[["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]].isna().any(axis=1)
    if bad.any():
        cols = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
        raise RuntimeError("Metadata has missing required values:\n" + meta.loc[bad, cols].head(30).to_string(index=False))
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
                "check": "035_S_6927_present",
                "observed": bool((meta["SubjectID"] == RECOVER_SUBJECT).any()),
                "expected": True,
                "pass": bool((meta["SubjectID"] == RECOVER_SUBJECT).any()),
            },
            {
                "variant": candidate.variant,
                "check": "128_S_2002_absent_from_metadata",
                "observed": bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any()),
                "expected": False,
                "pass": not bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any()),
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
                "check": "vae_pool_n",
                "observed": int(len(meta)),
                "expected": 647,
                "pass": int(len(meta)) == 647,
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
    if channels != TARGET_CHANNELS or selected != TARGET_SELECTED_NAMES:
        raise RuntimeError(f"{candidate.variant}: selected channel mismatch: indices={channels}, names={selected}")
    if selected_names_from_master(config) != TARGET_SELECTED_NAMES:
        raise RuntimeError(f"{candidate.variant}: config master-channel lookup mismatch")
    if tensor_shape[2:] != (131, 131):
        raise RuntimeError(f"{candidate.variant}: expected 131x131 matrices, got {tensor_shape}")
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
    setup = (
        f"mkdir -p {shlex.quote(str(big))} && "
        f"ln -s {shlex.quote(str(big))} {shlex.quote(str(local))}"
    )
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


def build_stage_a_command(config: Dict[str, Any], python_exe: str = PYTHON_EXE) -> List[str]:
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
    checks = [
        ("stage_a_channels", values_after_flag(stage_a, "--channels_to_use"), ["1", "2"]),
        ("stage_a_latent_dim", values_after_flag(stage_a, "--latent_dim"), ["384"]),
        ("stage_a_beta", values_after_flag(stage_a, "--beta_vae"), [str(candidate.beta_vae)]),
        ("stage_a_recon_loss", values_after_flag(stage_a, "--recon_loss_mode"), [candidate.recon_loss_mode]),
        ("stage_a_epochs", values_after_flag(stage_a, "--epochs_vae"), ["10000"]),
        ("stage_a_cycles", values_after_flag(stage_a, "--cyclical_beta_n_cycles"), ["125"]),
        ("stage_a_T0", values_after_flag(stage_a, "--lr_scheduler_T0"), ["80"]),
        ("stage_a_patience", values_after_flag(stage_a, "--early_stopping_patience_vae"), ["560"]),
        ("stage_a_dropout", values_after_flag(stage_a, "--dropout_rate_vae"), ["0.15"]),
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
            {
                "variant": candidate.variant,
                "name": "launcher_dry_run",
                "command": shlex.join([PYTHON_EXE, str(candidate.launcher_path), "--dry-run"]),
            },
            {
                "variant": candidate.variant,
                "name": "launcher_real_training_guarded",
                "command": shlex.join([PYTHON_EXE, str(candidate.launcher_path), "--confirm-training"]),
            },
        ]
    )
    return commands, validate_commands(config, candidate, stage_a, stage_b, oof)


def beta_regime_estimates() -> pd.DataFrame:
    if not CH12_BETA375_RD.exists():
        return pd.DataFrame(
            [
                {
                    "variant": c.variant,
                    "source": str(CH12_BETA375_RD),
                    "status": "missing_reference_rate_distortion",
                }
                for c in CANDIDATES.values()
            ]
        )
    rd = pd.read_csv(CH12_BETA375_RD)
    row = rd[rd["run_label"].astype(str).str.contains("ch12", case=False, na=False)]
    if row.empty:
        row = rd.head(1)
    ref = row.iloc[0].to_dict()
    kld_over_d = float(ref.get("KLD_over_D_best_mean", np.nan))
    beta_kld_over_d = float(ref.get("beta_KLD_over_D_best_mean", np.nan))
    ref_beta = 3.75
    rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATES.values():
        estimated = candidate.beta_vae * kld_over_d
        rows.append(
            {
                "variant": candidate.variant,
                "user_label": candidate.user_label,
                "source_run_label": ref.get("run_label", "unknown"),
                "source_beta": ref_beta,
                "candidate_beta": candidate.beta_vae,
                "source_KLD_over_D_best_mean": kld_over_d,
                "source_beta_KLD_over_D_best_mean": beta_kld_over_d,
                "estimated_candidate_beta_KLD_over_D_if_KLD_over_D_unchanged": estimated,
                "estimated_vs_source_ratio": estimated / beta_kld_over_d if beta_kld_over_d else np.nan,
                "interpretation": (
                    "current-loss estimate from existing ch12 beta3.75 KLD/D"
                    if candidate.variant == "currentloss"
                    else "approximate only: chmeanloss changes D scaling before retraining"
                ),
            }
        )
    return pd.DataFrame(rows)


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


def write_package(
    out_dir: Path,
    reference: Dict[str, Any],
    creation: pd.DataFrame,
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    variants: Sequence[str],
    dry_run: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(creation, out_dir, "generated_artifacts")
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
    beta_est = beta_regime_estimates()
    write_table(beta_est, out_dir, "effective_beta_kld_regime_estimate")
    run_manifest = pd.DataFrame(
        [
            {
                "variant": c.variant,
                "user_label": c.user_label,
                "run_id": c.run_id,
                "config": rel(c.config_path),
                "launcher": rel(c.launcher_path),
                "channels_to_use": str(TARGET_CHANNELS),
                "selected_channel_names": " | ".join(TARGET_SELECTED_NAMES),
                "beta_vae": c.beta_vae,
                "recon_loss_mode": c.recon_loss_mode,
                "outer_folds": reference["parameters"]["outer_folds"],
                "inner_folds": reference["parameters"]["inner_folds"],
                "epochs_vae": reference["parameters"]["epochs_vae"],
                "cycles": reference["parameters"]["cyclical_beta_n_cycles"],
                "T0": reference["parameters"]["lr_scheduler_T0"],
                "patience": reference["parameters"]["early_stopping_patience_vae"],
                "status": "preflight_only",
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

    checks = concat_frames(results, "strict_diff_guard")
    stale = concat_frames(results, "output_symlink_and_stale_audit")
    report = [
        "# ch12 beta2.75 current-loss/chmeanloss preflight",
        "",
        "This package prepares two post-final exploratory FULL 5x5 [1,2] branches. No real training was launched.",
        "",
        "## Candidates",
        "",
        "- `recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5`: current reconstruction loss, beta 2.75.",
        "- `recover035_ch12_latent384_beta2p75_chmeanloss_T80_h10000_p560_full5x5`: off-diagonal channel-normalized reconstruction loss, beta 2.75.",
        "",
        "## Shared Fixed Convention",
        "",
        "- channels [1,2] = Pearson_Full_FisherZ_Signed + MI_KNN_Symmetric.",
        "- latent_dim 384, epochs 10000, cycles 125, T0 80, patience 560.",
        "- dropout 0.15, legacy_all, convtranspose decoder, tanh final activation.",
        "- classifier readout plan: logreg_l2, z_plus_age_sex, OOF score calibration after completion.",
        "- metadata/tensor paths are unchanged from promoted recover035 reference.",
        "",
        "## Validation Summary",
        "",
        f"- Strict diff guard rows: {len(checks)}; all pass = {bool(checks['pass'].all())}.",
        f"- Stale output markers across selected variants: {int(stale['stale_marker_count'].sum())}.",
        "- VAE pool: CN=300, MCI=250, AD=97.",
        "- Classifier pool: CN=300, AD=97.",
        "- 035_S_6927 included; 128_S_2002 absent from metadata pools.",
        "- All five outer folds and inner folds are feasible.",
        "",
        "Real training remains blocked unless a launcher is run with `--confirm-training`.",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out_dir / "dry_run_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        out_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "dry_run": dry_run,
            "training_launched": False,
            "variants": list(variants),
            "promoted_reference_config": rel(PROMOTED_CONFIG),
            "preflight_root": rel(out_dir),
            "guardrails": [
                "no real training",
                "no tensor modification",
                "no metadata modification",
                "no ledger modification",
                "no existing model artifact modification",
                "no OASIS scoring",
                "no threshold/calibration fitting",
            ],
        },
    )


def run_command(command: Iterable[str]) -> None:
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def parse_args(default_variant: str | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--variant", choices=["all", *CANDIDATES.keys()], default=default_variant or "all")
    p.add_argument("--preflight-dir", type=Path, default=PREFLIGHT_ROOT)
    p.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    p.add_argument("--confirm-training", action="store_true", help="Launch real Stage A training for one selected variant after all guards pass.")
    p.add_argument("--force-clean", action="store_true", help="Allow real launch only after caller has intentionally handled stale outputs.")
    return p.parse_args()


def main(default_variant: str | None = None) -> None:
    args = parse_args(default_variant=default_variant)
    reference = load_json(PROMOTED_CONFIG)
    creation = ensure_configs_and_launchers(reference)
    variants = list(CANDIDATES) if args.variant == "all" else [args.variant]
    results = {variant: validate_candidate(reference, CANDIDATES[variant]) for variant in variants}
    dry_run = args.dry_run or not args.confirm_training
    out_dir = resolve(args.preflight_dir)
    write_package(out_dir, reference, creation, results, variants, dry_run=dry_run)

    print(f"Preflight package: {out_dir}")
    print("Validated variants: " + ", ".join(variants))
    print("Generated configs/launchers:")
    print(creation[creation["variant"].isin(variants)].to_string(index=False))
    print("\nStrict scientific diffs:")
    print(concat_frames(results, "strict_diff_guard").to_string(index=False))
    print("\nEffective beta*KLD/D estimate:")
    print(beta_regime_estimates()[beta_regime_estimates()["variant"].isin(variants)].to_string(index=False))

    if dry_run:
        print("Dry-run OK. No training launched.")
        return

    if args.variant == "all":
        raise RuntimeError("Real training requires a single --variant, not --variant all.")
    output = concat_frames(results, "output_symlink_and_stale_audit")
    stale_count = int(output["stale_marker_count"].iloc[0])
    if stale_count and not args.force_clean:
        raise RuntimeError("Refusing real training with stale output markers.")
    if not bool(output["target_match"].iloc[0]):
        raise RuntimeError("Refusing real training because the local output symlink is not already pointed at the big-disk target.")
    target = load_json(CANDIDATES[args.variant].config_path)
    run_command(build_stage_a_command(target))


if __name__ == "__main__":
    main()
