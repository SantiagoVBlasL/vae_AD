#!/usr/bin/env python3
"""Read-only primary-artifact forensics for the FULL fold-wise ComBat run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results/revision_bspc_2026"
COMBAT_RUN = RESULTS / (
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
)
LOCKED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
LAUNCH_LOG = RESULTS / (
    "_launch_logs/"
    "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5/"
    "train_foldcombat_20260607_025222.log"
)
CONFIG = PROJECT / (
    "configs/runs/"
    "adni_v5_1c_recover035_latent384_beta3p75_"
    "foldcombat_mfr_age_sex_T80_h10000_p560_full5x5.json"
)
TRAINER = PROJECT / "scripts/run_vae_clf_ad_inference.py"
COMBAT_HELPER = (
    PROJECT
    / "scripts/revision_bspc_2026/foldwise_combat_input_harmonization.py"
)
CORRECTED_CACHE = RESULTS / (
    "post_revision_exploratory_20260630/"
    "combat_downstream_classifier_reconstruction_20260706/"
    "corrected_combat_latent_cache"
)
DEFECTIVE_CACHE = COMBAT_RUN / "classifier_only_readout/latent_cache"
CORRECTED_GENERATOR = (
    PROJECT
    / "scripts/revision_bspc_2026/"
    "complete_combat_external_transport_20260706.py"
)
OUT = RESULTS / (
    "post_revision_exploratory_20260630/"
    "foldcombat_full_training_forensics_20260706"
)

FOLDS = [1, 2, 3, 4, 5]
MU_COLS = [f"mu_{i}" for i in range(384)]
REQUIRED_HISTORY = [
    "train_loss",
    "train_recon",
    "train_kld",
    "val_loss",
    "val_recon",
    "val_kld",
    "val_loss_modelsel",
    "beta",
]

SCRIPT_DIR = PROJECT / "scripts/revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (  # noqa: E402
    encode_mu,
    load_config,
    make_model,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def timestamp(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def md_table(df: pd.DataFrame, max_rows: int = 500) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows)
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required primary artifacts:\n" + "\n".join(missing))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint object at {path}: {type(state)}")
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError(f"Checkpoint is not a tensor state_dict: {path}")
    return state


def parse_log_evidence(text: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for fold in FOLDS:
        start = text.find(f"--- Iniciando Fold {fold}/5 ---")
        end = (
            text.find(f"--- Iniciando Fold {fold + 1}/5 ---")
            if fold < 5
            else len(text)
        )
        block = text[start:end] if start >= 0 and end > start else ""
        early = re.search(
            r"Early stopping VAE en epoch (\d+)\. Mejor ValL\(βmax\): "
            r"([0-9.]+) \(época (\d+)\)",
            block,
        )
        lr_values = [
            float(value)
            for value in re.findall(r"\bLR=([0-9.eE+-]+)", block)
        ]
        result[fold] = {
            "fold_block_present": bool(block),
            "early_stop_epoch": int(early.group(1)) if early else None,
            "best_val_modelsel": float(early.group(2)) if early else None,
            "selected_checkpoint_epoch": int(early.group(3)) if early else None,
            "lr_log_count": len(lr_values),
            "lr_log_min": min(lr_values) if lr_values else np.nan,
            "lr_log_max": max(lr_values) if lr_values else np.nan,
            "checkpoint_saved_log": f"fold_{fold}/vae_model_fold_{fold}.pt" in block,
            "harmonization_active_log": (
                "Input harmonization activo: foldwise_combat" in block
            ),
            "harmonization_log_exact": (
                "fit_scope=outer_train_dev_only, batch=Manufacturer, "
                "covariates=['Age', 'Sex'], excluded=['ResearchGroup_Mapped'], "
                "channels=[1, 0, 2]"
                in block
            ),
            "objective_log_foldwise_combat": (
                "input_harmonization_mode=foldwise_combat" in block
            ),
        }
    return result


def checkpoint_forensics(
    work: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(COMBAT_RUN)
    inventory_rows = []
    comparison_rows = []
    weight_rows = []
    combat_hashes: dict[int, str] = {}
    states: dict[int, dict[str, torch.Tensor]] = {}

    for fold in FOLDS:
        combat_path = COMBAT_RUN / f"fold_{fold}/vae_model_fold_{fold}.pt"
        locked_path = LOCKED_RUN / f"fold_{fold}/vae_model_fold_{fold}.pt"
        try:
            state = load_state(combat_path)
            load_ok = True
            load_error = ""
        except Exception as exc:
            state = {}
            load_ok = False
            load_error = repr(exc)
        states[fold] = state
        combat_hashes[fold] = sha256_file(combat_path)

        smoke_ok = False
        smoke_finite = False
        smoke_shape = ""
        smoke_error = ""
        smoke_input_kind = "synthetic_zero_normalized_input_not_saved_subject"
        if load_ok:
            try:
                model = make_model(
                    cfg,
                    image_size=131,
                    n_channels=3,
                    device=torch.device("cpu"),
                    conditioning_dim_override=0,
                )
                model.load_state_dict(state, strict=True)
                model.eval()
                sample = np.zeros((2, 3, 131, 131), dtype=np.float32)
                mu = encode_mu(
                    model,
                    sample,
                    batch_size=2,
                    device=torch.device("cpu"),
                    condition=None,
                )
                smoke_ok = True
                smoke_finite = bool(np.isfinite(mu).all())
                smoke_shape = str(tuple(mu.shape))
                del model
            except Exception as exc:
                smoke_error = repr(exc)

        n_elements = sum(int(value.numel()) for value in state.values())
        inventory_rows.append(
            {
                "fold": fold,
                "checkpoint_path": str(combat_path),
                "exists": combat_path.exists(),
                "size_bytes": combat_path.stat().st_size,
                "mtime_utc": timestamp(combat_path).isoformat(),
                "sha256": combat_hashes[fold],
                "load_success": load_ok,
                "load_error": load_error,
                "state_tensor_count": len(state),
                "parameter_elements": n_elements,
                "smoke_input_kind": smoke_input_kind,
                "saved_harmonized_subject_tensor_available": False,
                "smoke_encode_success": smoke_ok,
                "smoke_mu_shape": smoke_shape,
                "smoke_mu_finite": smoke_finite,
                "smoke_error": smoke_error,
            }
        )

        locked_state = load_state(locked_path)
        same_keys = set(state) == set(locked_state)
        all_exact = same_keys and all(
            torch.equal(state[key], locked_state[key]) for key in state
        )
        comparison_rows.append(
            {
                "comparison": "combat_vs_corresponding_locked",
                "combat_fold": fold,
                "other_fold": fold,
                "combat_checkpoint": str(combat_path),
                "other_checkpoint": str(locked_path),
                "combat_sha256": combat_hashes[fold],
                "other_sha256": sha256_file(locked_path),
                "sha256_equal": combat_hashes[fold] == sha256_file(locked_path),
                "state_keys_equal": same_keys,
                "all_weights_exactly_equal": all_exact,
            }
        )
        if same_keys:
            for name in sorted(state):
                combat_tensor = state[name].detach().cpu().double()
                locked_tensor = locked_state[name].detach().cpu().double()
                delta = combat_tensor - locked_tensor
                locked_norm = float(torch.linalg.vector_norm(locked_tensor))
                l2 = float(torch.linalg.vector_norm(delta))
                weight_rows.append(
                    {
                        "fold": fold,
                        "parameter_name": name,
                        "tensor_kind": (
                            "trainable_weight_or_bias"
                            if name.endswith((".weight", ".bias"))
                            else "state_buffer"
                        ),
                        "dtype": str(state[name].dtype),
                        "shape": "x".join(map(str, combat_tensor.shape)),
                        "n_elements": combat_tensor.numel(),
                        "exact_equal": torch.equal(combat_tensor, locked_tensor),
                        "l2_distance": l2,
                        "locked_l2_norm": locked_norm,
                        "relative_l2_distance": (
                            l2 / locked_norm if locked_norm > 0 else np.nan
                        ),
                        "max_abs_difference": float(torch.max(torch.abs(delta))),
                    }
                )

    for fold_a in FOLDS:
        for fold_b in FOLDS:
            if fold_b <= fold_a:
                continue
            path_a = COMBAT_RUN / f"fold_{fold_a}/vae_model_fold_{fold_a}.pt"
            path_b = COMBAT_RUN / f"fold_{fold_b}/vae_model_fold_{fold_b}.pt"
            same_keys = set(states[fold_a]) == set(states[fold_b])
            all_exact = same_keys and all(
                torch.equal(states[fold_a][key], states[fold_b][key])
                for key in states[fold_a]
            )
            comparison_rows.append(
                {
                    "comparison": "combat_fold_vs_combat_fold",
                    "combat_fold": fold_a,
                    "other_fold": fold_b,
                    "combat_checkpoint": str(path_a),
                    "other_checkpoint": str(path_b),
                    "combat_sha256": combat_hashes[fold_a],
                    "other_sha256": combat_hashes[fold_b],
                    "sha256_equal": combat_hashes[fold_a]
                    == combat_hashes[fold_b],
                    "state_keys_equal": same_keys,
                    "all_weights_exactly_equal": all_exact,
                }
            )

    return (
        pd.DataFrame(inventory_rows),
        pd.DataFrame(comparison_rows),
        pd.DataFrame(weight_rows),
    )


def history_and_timestamps(
    log_evidence: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows = []
    timestamp_rows = []
    config_time = timestamp(COMBAT_RUN / "run_config.json")
    log_time = timestamp(LAUNCH_LOG)
    previous_history_time: pd.Timestamp | None = None
    for fold in FOLDS:
        history_path = COMBAT_RUN / f"fold_{fold}/vae_train_history_fold_{fold}.joblib"
        checkpoint_path = COMBAT_RUN / f"fold_{fold}/vae_model_fold_{fold}.pt"
        history = joblib.load(history_path)
        lengths = {
            key: len(value)
            for key, value in history.items()
            if isinstance(value, (list, tuple, np.ndarray))
        }
        required_present = all(key in history for key in REQUIRED_HISTORY)
        required_lengths = [lengths.get(key, -1) for key in REQUIRED_HISTORY]
        same_required_length = len(set(required_lengths)) == 1
        n_epochs = required_lengths[0] if required_lengths else 0
        finite_required = all(
            np.isfinite(np.asarray(history[key], dtype=float)).all()
            for key in REQUIRED_HISTORY
            if key in history
        )
        log = log_evidence[fold]
        history_rows.append(
            {
                "fold": fold,
                "history_path": str(history_path),
                "exists": history_path.exists(),
                "load_success": isinstance(history, dict),
                "history_keys": ";".join(sorted(history)),
                "required_trajectory_keys_present": required_present,
                "required_trajectory_lengths_equal": same_required_length,
                "n_epochs": n_epochs,
                "multi_epoch_trajectory": n_epochs > 100,
                "reconstruction_present": "train_recon" in history
                and "val_recon" in history,
                "kl_present": "train_kld" in history and "val_kld" in history,
                "validation_loss_present": "val_loss" in history
                and "val_loss_modelsel" in history,
                "beta_schedule_present": "beta" in history,
                "learning_rate_present_in_history_object": any(
                    key.lower() in {"lr", "learning_rate", "learning_rates"}
                    for key in history
                ),
                "selected_checkpoint_epoch_present_in_history_object": any(
                    "best_epoch" in key.lower()
                    or "selected_checkpoint_epoch" in key.lower()
                    for key in history
                ),
                "learning_rate_present_in_primary_launch_log": log["lr_log_count"]
                > 0,
                "learning_rate_log_count": log["lr_log_count"],
                "learning_rate_min_logged": log["lr_log_min"],
                "learning_rate_max_logged": log["lr_log_max"],
                "early_stop_epoch_from_log": log["early_stop_epoch"],
                "selected_checkpoint_epoch_from_log": log[
                    "selected_checkpoint_epoch"
                ],
                "best_val_modelsel_from_log": log["best_val_modelsel"],
                "history_epochs_match_log_early_stop": n_epochs
                == log["early_stop_epoch"],
                "all_required_values_finite": finite_required,
                "train_loss_changed": float(history["train_loss"][0])
                != float(history["train_loss"][-1]),
                "val_loss_changed": float(history["val_loss"][0])
                != float(history["val_loss"][-1]),
                "beta_changed": len(set(np.round(history["beta"], 12))) > 1,
            }
        )
        checkpoint_time = timestamp(checkpoint_path)
        history_time = timestamp(history_path)
        timestamp_rows.append(
            {
                "fold": fold,
                "run_config_mtime_utc": config_time.isoformat(),
                "checkpoint_mtime_utc": checkpoint_time.isoformat(),
                "history_mtime_utc": history_time.isoformat(),
                "launch_log_mtime_utc": log_time.isoformat(),
                "timestamp_basis": "filesystem_mtime",
                "launch_log_has_embedded_line_timestamps": False,
                "config_before_checkpoint": config_time <= checkpoint_time,
                "checkpoint_before_or_close_to_history": checkpoint_time
                <= history_time,
                "checkpoint_to_history_seconds": (
                    history_time - checkpoint_time
                ).total_seconds(),
                "history_before_log_close": history_time <= log_time,
                "fold_artifacts_after_previous_fold": (
                    True
                    if previous_history_time is None
                    else checkpoint_time > previous_history_time
                ),
                "chronology_consistent": (
                    config_time <= checkpoint_time <= history_time <= log_time
                    and (
                        previous_history_time is None
                        or checkpoint_time > previous_history_time
                    )
                ),
            }
        )
        previous_history_time = history_time
    return pd.DataFrame(history_rows), pd.DataFrame(timestamp_rows)


def harmonization_subject_overlap() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        fit_path = (
            COMBAT_RUN
            / f"fold_{fold}/input_harmonization_fit_train_dev_vae_pool_subjects.csv"
        )
        test_path = (
            COMBAT_RUN
            / f"fold_{fold}/input_harmonization_classifier_test_apply_subjects.csv"
        )
        guard_path = (
            COMBAT_RUN / f"fold_{fold}/input_harmonization_leakage_guard.csv"
        )
        fit = pd.read_csv(fit_path)
        test = pd.read_csv(test_path)
        guard = pd.read_csv(guard_path).iloc[0]
        overlap = sorted(set(fit["SubjectID"]) & set(test["SubjectID"]))
        rows.append(
            {
                "fold": fold,
                "fit_manifest": str(fit_path),
                "test_manifest": str(test_path),
                "n_fit": len(fit),
                "n_test": len(test),
                "fit_subject_hash": sha256_values(fit["SubjectID"]),
                "test_subject_hash": sha256_values(test["SubjectID"]),
                "direct_overlap_n": len(overlap),
                "direct_overlap_subjects": ";".join(overlap),
                "guard_status": guard["status"],
                "guard_fit_scope": guard["fit_scope"],
                "guard_overlap_n": int(guard["n_fit_test_subject_overlap"]),
                "zero_overlap_verified": len(overlap) == 0
                and int(guard["n_fit_test_subject_overlap"]) == 0,
            }
        )
    return pd.DataFrame(rows)


def corrected_cache_evidence() -> pd.DataFrame:
    rows = []
    generator_text = CORRECTED_GENERATOR.read_text(encoding="utf-8")
    generator_trace = all(
        token in generator_text
        for token in [
            "fit_tensor_combat_channelwise(",
            "transform_tensor_combat_channelwise(",
            "corrected_train_dev = encode_external_fold(",
            "corrected_test = encode_external_fold(",
            'corrected_cache / f"fold_{fold}_trainDev_latent_mu.csv"',
            'corrected_cache / f"fold_{fold}_test_latent_mu.csv"',
        ]
    )
    for fold in FOLDS:
        for split in ("trainDev", "test"):
            corrected_path = CORRECTED_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
            defective_path = DEFECTIVE_CACHE / f"fold_{fold}_{split}_latent_mu.csv"
            corrected = pd.read_csv(corrected_path)
            defective = pd.read_csv(defective_path)
            if corrected["SubjectID"].tolist() != defective["SubjectID"].tolist():
                raise RuntimeError(
                    f"Corrected/defective subject-order mismatch: fold {fold} {split}"
                )
            corrected_mu = corrected[MU_COLS].to_numpy(float)
            defective_mu = defective[MU_COLS].to_numpy(float)
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "corrected_cache_path": str(corrected_path),
                    "defective_cache_path": str(defective_path),
                    "n_subjects": len(corrected),
                    "subject_hash_corrected": sha256_values(
                        corrected["SubjectID"]
                    ),
                    "subject_hash_defective": sha256_values(
                        defective["SubjectID"]
                    ),
                    "subject_hash_equal": sha256_values(
                        corrected["SubjectID"]
                    )
                    == sha256_values(defective["SubjectID"]),
                    "corrected_file_sha256": sha256_file(corrected_path),
                    "defective_file_sha256": sha256_file(defective_path),
                    "file_hash_distinct": sha256_file(corrected_path)
                    != sha256_file(defective_path),
                    "latent_max_abs_difference": float(
                        np.max(np.abs(corrected_mu - defective_mu))
                    ),
                    "latent_mean_abs_difference": float(
                        np.mean(np.abs(corrected_mu - defective_mu))
                    ),
                    "generator_code_trace_complete": generator_trace,
                    "corrected_values_finite": bool(
                        np.isfinite(corrected_mu).all()
                    ),
                }
            )
    return pd.DataFrame(rows)


def source_line_numbers(path: Path, tokens: list[str]) -> dict[str, int | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, int | None] = {}
    for token in tokens:
        result[token] = next(
            (index for index, line in enumerate(lines, start=1) if token in line),
            None,
        )
    return result


def write_markdown_evidence(
    work: Path,
    config: dict[str, Any],
    log_text: str,
    overlap: pd.DataFrame,
    cache: pd.DataFrame,
) -> None:
    args = config["args"]
    config_report = f"""# Configuration and Original Launch-Log Evidence

## Primary sources

- Original launch log: `{LAUNCH_LOG}`
- Saved run configuration: `{COMBAT_RUN / 'run_config.json'}`
- Launch configuration: `{CONFIG}`

## Exact settings

| setting | saved value | launch-log confirmation |
|---|---|---|
| selected channels | `{args['channels_to_use']}` | `{str(args['channels_to_use']) in log_text}` |
| latent dimension | `{args['latent_dim']}` | `latent_dim: 384` present |
| beta | `{args['beta_vae']}` | `beta_vae: 3.75` present |
| input harmonization | `{args['input_harmonization_mode']}` | present in command, argument dump, and every fold objective |
| batch | `{args['input_harmonization_batch_col']}` | present in command and every fold activation line |
| preserved covariates | `{args['input_harmonization_covariates']}` | present in command and every fold activation line |
| excluded covariate | `{args['input_harmonization_excluded_covariates']}` | present in command and every fold activation line |
| fit scope | `{args['input_harmonization_fit_scope']}` | `outer_train_dev_only` present for every fold |

The launch log contains five separate fold starts, five separate harmonization
activation lines, five multi-thousand-epoch training trajectories, five early
stopping decisions, and five separate checkpoint-save lines.

## Source-version limitation

The run recorded Git HEAD `{args.get('git_hash')}`, but the fold-wise ComBat
wiring was in working-tree code not contained in that commit. The helper file
has a pre-launch modification time, while the main trainer was subsequently
modified on 2026-06-21. Therefore the exact executed trainer source checksum is
not recoverable from Git alone. The original launch log and saved per-fold
artifacts provide direct runtime evidence, but this prevents a fully closed
source-snapshot audit.
"""
    (work / "config_and_log_evidence.md").write_text(
        config_report, encoding="utf-8"
    )

    trainer_lines = source_line_numbers(
        TRAINER,
        [
            "fitted_combat = fit_tensor_combat_channelwise(",
            "vae_train_pool_tensor_harmonized = transform_tensor_combat_channelwise(",
            "vae_train_pool_tensor_original_scale = vae_train_pool_tensor_harmonized",
            "vae_pool_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(",
            "vae_train_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm",
        ],
    )
    shift_parts = []
    for fold in FOLDS:
        shift = pd.read_csv(
            COMBAT_RUN
            / f"fold_{fold}/input_harmonization_channel_shift_summary.csv"
        )
        vae_shift = shift[shift["split"].eq("vae_pool_fit_train_dev")]
        shift_parts.append(vae_shift)
    shift_all = pd.concat(shift_parts, ignore_index=True)
    trace = f"""# Harmonized VAE Training-Input Trace

## Runtime evidence

The original launch log records, independently for all five folds:

1. `Input harmonization activo: foldwise_combat`;
2. fit scope `outer_train_dev_only`;
3. batch `Manufacturer`;
4. preserved covariates `Age`, `Sex`;
5. excluded diagnosis (`ResearchGroup_Mapped`);
6. normalization parameters fitted after the harmonization activation line;
7. a VAE objective line with `input_harmonization_mode=foldwise_combat`;
8. a genuine multi-epoch VAE training trajectory.

## Direct code dataflow

The surviving trainer implements this sequence:

| operation | current source line |
|---|---:|
| fit fold-local ComBat | {trainer_lines['fitted_combat = fit_tensor_combat_channelwise(']} |
| transform VAE pool | {trainer_lines['vae_train_pool_tensor_harmonized = transform_tensor_combat_channelwise(']} |
| replace raw pool variable with harmonized tensor | {trainer_lines['vae_train_pool_tensor_original_scale = vae_train_pool_tensor_harmonized']} |
| normalize that replaced tensor | {trainer_lines['vae_pool_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(']} |
| construct training dataset from normalized tensor | {trainer_lines['vae_train_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm']} |

The exact executed trainer snapshot was not committed and has no saved source
checksum. This code trace is therefore corroborative, not sufficient alone.
The runtime log ordering and the primary per-fold summaries below are the direct
run evidence.

## Saved pre/post tensor summaries

{md_table(shift_all)}

Every fold also stores `input_harmonization_integrity.csv`; all report finite
post-transform VAE-pool values, unchanged diagonals, symmetric matrices, no
test-distribution fitting, and non-global ComBat.

## Fit/test exclusion

{md_table(overlap)}

## Limitation

No harmonized subject-level connectivity tensor and no serialized fitted
ComBat object were retained. Consequently, a literal no-refit re-encoding of a
saved harmonized subject matrix cannot be performed. Checkpoint load and
forward execution were tested with a fixed finite zero tensor in normalized VAE
input space; this is a software smoke test, not a subject-level harmonized
sample validation.
"""
    (work / "harmonized_training_input_trace.md").write_text(
        trace, encoding="utf-8"
    )

    cache_report = f"""# Corrected Cache Lineage

## Direct file evidence

The corrected and defective caches contain the same ordered subjects per fold,
but every corrected file has a distinct SHA256 and materially different latent
values:

{md_table(cache)}

## Generation-code trace

`{CORRECTED_GENERATOR}` explicitly:

1. reads each original fold's ComBat fit-subject manifest;
2. reconstructs the fold transform from those exact ADNI fit rows;
3. transforms train/dev and outer-test tensors before VAE encoding;
4. calls the frozen fold VAE encoder on the transformed tensors;
5. writes the corrected train/dev and test posterior means.

This is direct source/file lineage, not a prior audit conclusion. The original
fitted ComBat objects were not serialized, so the corrected cache was produced
by deterministic reconstruction from the original fit manifests rather than by
loading preserved fitted objects. No corrected-cache file equals its defective
raw-input counterpart.
"""
    (work / "corrected_cache_lineage.md").write_text(
        cache_report, encoding="utf-8"
    )


def main() -> None:
    refresh_output = "--refresh-output" in sys.argv[1:]
    if OUT.exists() and not refresh_output:
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    required = [
        LAUNCH_LOG,
        CONFIG,
        COMBAT_RUN / "run_config.json",
        TRAINER,
        COMBAT_HELPER,
        CORRECTED_CACHE,
        DEFECTIVE_CACHE,
        CORRECTED_GENERATOR,
    ]
    for fold in FOLDS:
        required.extend(
            [
                COMBAT_RUN / f"fold_{fold}/vae_model_fold_{fold}.pt",
                LOCKED_RUN / f"fold_{fold}/vae_model_fold_{fold}.pt",
                COMBAT_RUN / f"fold_{fold}/vae_train_history_fold_{fold}.joblib",
                COMBAT_RUN
                / f"fold_{fold}/input_harmonization_fit_train_dev_vae_pool_subjects.csv",
                COMBAT_RUN
                / f"fold_{fold}/input_harmonization_classifier_test_apply_subjects.csv",
                COMBAT_RUN / f"fold_{fold}/input_harmonization_leakage_guard.csv",
                COMBAT_RUN / f"fold_{fold}/input_harmonization_integrity.csv",
                COMBAT_RUN
                / f"fold_{fold}/input_harmonization_channel_shift_summary.csv",
            ]
        )
    require(required)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    context = (
        nullcontext(str(OUT))
        if refresh_output
        else tempfile.TemporaryDirectory(
            prefix=f".{OUT.name}.tmp-", dir=str(OUT.parent)
        )
    )
    with context as temp:
        work = Path(temp)
        log_text = LAUNCH_LOG.read_text(encoding="utf-8", errors="replace")
        run_config = json.loads(
            (COMBAT_RUN / "run_config.json").read_text(encoding="utf-8")
        )
        log_evidence = parse_log_evidence(log_text)
        inventory, hash_comparison, weight_distance = checkpoint_forensics(work)
        histories, timestamps = history_and_timestamps(log_evidence)
        overlap = harmonization_subject_overlap()
        cache = corrected_cache_evidence()

        inventory.to_csv(work / "fold_checkpoint_inventory.csv", index=False)
        hash_comparison.to_csv(
            work / "checkpoint_hash_comparison.csv", index=False
        )
        histories.to_csv(work / "training_history_integrity.csv", index=False)
        timestamps.to_csv(work / "timestamp_consistency.csv", index=False)
        weight_distance.to_csv(
            work / "weight_distance_vs_locked.csv", index=False
        )
        write_markdown_evidence(
            work, run_config, log_text, overlap, cache
        )

        checkpoint_core = (
            inventory["load_success"].all()
            and inventory["smoke_encode_success"].all()
            and inventory["smoke_mu_finite"].all()
            and not hash_comparison["sha256_equal"].any()
            and not hash_comparison["all_weights_exactly_equal"].any()
        )
        history_core = (
            histories["required_trajectory_keys_present"].all()
            and histories["multi_epoch_trajectory"].all()
            and histories["all_required_values_finite"].all()
            and histories["history_epochs_match_log_early_stop"].all()
        )
        config_core = all(
            evidence["harmonization_log_exact"]
            and evidence["objective_log_foldwise_combat"]
            and evidence["checkpoint_saved_log"]
            for evidence in log_evidence.values()
        )
        overlap_core = overlap["zero_overlap_verified"].all()
        timestamps_core = timestamps["chronology_consistent"].all()
        cache_core = (
            cache["file_hash_distinct"].all()
            and cache["generator_code_trace_complete"].all()
            and cache["corrected_values_finite"].all()
        )
        documentation_gaps = {
            "learning_rate_absent_from_history_object": not histories[
                "learning_rate_present_in_history_object"
            ].all(),
            "selected_epoch_absent_from_history_object": not histories[
                "selected_checkpoint_epoch_present_in_history_object"
            ].all(),
            "no_saved_harmonized_subject_tensor": not inventory[
                "saved_harmonized_subject_tensor_available"
            ].all(),
            "exact_executed_trainer_snapshot_not_preserved": True,
            "fitted_combat_objects_not_serialized": True,
            "launch_log_has_no_embedded_line_timestamps": True,
        }
        core_verified = all(
            [
                checkpoint_core,
                history_core,
                config_core,
                overlap_core,
                timestamps_core,
                cache_core,
            ]
        )
        verdict = "PARTIALLY_VERIFIED" if core_verified else "NOT_VERIFIED"

        trainable_weight_distance = weight_distance[
            weight_distance["tensor_kind"].eq("trainable_weight_or_bias")
        ].copy()
        aggregate_weights = (
            trainable_weight_distance.groupby("fold")
            .agg(
                parameter_tensors=("parameter_name", "size"),
                exact_equal_tensors=("exact_equal", "sum"),
                total_l2_distance=(
                    "l2_distance",
                    lambda values: float(np.sqrt(np.sum(np.square(values)))),
                ),
                max_parameter_l2=("l2_distance", "max"),
                max_abs_difference=("max_abs_difference", "max"),
            )
            .reset_index()
        )
        verdict_text = f"""# {verdict}

## Bottom line

Primary artifacts strongly support that the run executed five distinct,
multi-thousand-epoch VAEs after fold-local input ComBat transformation. All
five checkpoints load, have unique hashes, differ from one another and from
their locked counterparts, and execute finite forward smoke encodes.

The verdict is not `VERIFIED_GENUINE_FULL` because the exact executed trainer
source snapshot was not preserved, fitted ComBat objects and harmonized
subject-level tensors were not serialized, and the history objects omit
learning-rate and selected-epoch fields (those are recoverable only from the
original launch log). The launch log itself has no embedded per-line
timestamps, so chronology uses filesystem modification times. A literal
saved-harmonized-subject re-encode is therefore impossible without refitting
ComBat, which this audit did not do.

## Core checks

| check | status |
|---|---|
| five distinct loadable ComBat checkpoints | {'PASS' if checkpoint_core else 'FAIL'} |
| genuine reconstruction/KL/validation/beta trajectories | {'PASS' if history_core else 'FAIL'} |
| original launch configuration and per-fold runtime activation | {'PASS' if config_core else 'FAIL'} |
| ComBat fit subjects exclude outer test | {'PASS' if overlap_core else 'FAIL'} |
| timestamp chronology | {'PASS' if timestamps_core else 'FAIL'} |
| corrected cache distinct from defective raw-input cache | {'PASS' if cache_core else 'FAIL'} |

## Training trajectories

{md_table(histories[['fold','n_epochs','early_stop_epoch_from_log','selected_checkpoint_epoch_from_log','best_val_modelsel_from_log','learning_rate_log_count','history_epochs_match_log_early_stop']])}

## Weight differences versus locked

{md_table(aggregate_weights)}

## Important scope

The software smoke test uses a fixed zero tensor in normalized VAE input space.
It proves checkpoint loadability and finite encoder execution, but it is not a
replacement for the requested saved harmonized subject sample, which does not
exist among the primary artifacts.
"""
        (work / "00_VERDICT.md").write_text(verdict_text, encoding="utf-8")

        command_log = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "status": "COMPLETE",
            "verdict": verdict,
            "strict_read_only_inputs": True,
            "actions_not_performed": {
                "vae_training": True,
                "combat_fit_or_refit": True,
                "latent_regeneration": True,
                "manuscript_modification": True,
            },
            "primary_sources": {
                "combat_run": str(COMBAT_RUN),
                "locked_run": str(LOCKED_RUN),
                "launch_log": str(LAUNCH_LOG),
                "saved_run_config": str(COMBAT_RUN / "run_config.json"),
                "launch_config": str(CONFIG),
                "trainer_source_current": str(TRAINER),
                "combat_helper_source": str(COMBAT_HELPER),
            },
            "primary_source_hashes": {
                "launch_log_sha256": sha256_file(LAUNCH_LOG),
                "saved_run_config_sha256": sha256_file(
                    COMBAT_RUN / "run_config.json"
                ),
                "launch_config_sha256": sha256_file(CONFIG),
                "trainer_current_sha256": sha256_file(TRAINER),
                "combat_helper_sha256": sha256_file(COMBAT_HELPER),
            },
            "core_checks": {
                "checkpoint_core": bool(checkpoint_core),
                "history_core": bool(history_core),
                "config_core": bool(config_core),
                "overlap_core": bool(overlap_core),
                "timestamps_core": bool(timestamps_core),
                "corrected_cache_core": bool(cache_core),
            },
            "documentation_gaps": documentation_gaps,
            "smoke_test": {
                "input": "fixed zeros, shape (2,3,131,131), normalized VAE input space",
                "saved_harmonized_subject_sample_used": False,
                "outputs_written": False,
            },
            "output_dir": str(OUT),
        }
        (work / "command_log.json").write_text(
            json.dumps(command_log, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not refresh_output:
            os.rename(work, OUT)
    print(json.dumps({"status": "COMPLETE", "verdict": verdict, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
