#!/usr/bin/env python3
"""Preflight real training wiring for Branch B foldwise ComBat input harmonization.

This script does not train, score OASIS, or write replacement tensors. It mirrors
the fold construction used by scripts/run_vae_clf_ad_inference.py, validates the
leakage guards, and probes fit-on-train/apply-to-test ComBat behavior.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
REVISION_DIR = PROJECT_ROOT / "scripts/revision_bspc_2026"
for p in [SRC_DIR, REVISION_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from betavae_xai.data.preprocessing import load_data  # noqa: E402
from foldwise_combat_input_harmonization import (  # noqa: E402
    channel_shift_summary,
    dependency_status,
    fit_tensor_combat_channelwise,
    manufacturer_centroid_separability_proxy,
    normalize_manufacturer,
    transform_tensor_combat_channelwise,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5.json"
REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
OUT_DEFAULT = PROJECT_ROOT / "results/revision_bspc_2026/foldcombat_real_training_wiring_preflight_20260607"
BRANCH_B_LAUNCHER = PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5.py"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"

RUN_NAME = "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"
REQUIRED_METADATA_COLS = ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]
EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--skip-launcher-dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def append_arg(command: list[str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(f"--{key}")
        return
    if value is None:
        return
    command.append(f"--{key}")
    if isinstance(value, list):
        command.extend(str(v) for v in value)
    else:
        command.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> list[str]:
    if flag not in tokens:
        return []
    out: list[str] = []
    items = list(tokens)
    for token in items[items.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def validate_config(ref: dict[str, Any], cand: dict[str, Any]) -> None:
    if cand["run_name"] != RUN_NAME:
        raise RuntimeError(f"Unexpected run_name: {cand['run_name']}")
    ref_params = dict(ref["parameters"])
    cand_params = dict(cand["parameters"])
    harmonization_expected = {
        "input_harmonization_mode": "foldwise_combat",
        "input_harmonization_batch_col": "Manufacturer",
        "input_harmonization_covariates": ["Age", "Sex"],
        "input_harmonization_excluded_covariates": ["ResearchGroup_Mapped"],
        "input_harmonization_fit_scope": "outer_train_dev_only",
        "input_harmonization_vectorization": "upper_offdiag_by_channel",
    }
    for key, expected in harmonization_expected.items():
        if cand_params.get(key) != expected:
            raise RuntimeError(f"{key}: expected {expected!r}, got {cand_params.get(key)!r}")
        cand_params.pop(key, None)
    diffs = {
        key: (ref_params.get(key), cand_params.get(key))
        for key in sorted(set(ref_params) | set(cand_params))
        if ref_params.get(key) != cand_params.get(key)
    }
    if diffs:
        raise RuntimeError(f"Unexpected non-harmonization parameter diffs: {diffs}")
    for key in ["global_tensor_path", "metadata_path", "training_script"]:
        if ref["paths"].get(key) != cand["paths"].get(key):
            raise RuntimeError(f"Unexpected path diff {key}: {ref['paths'].get(key)} -> {cand['paths'].get(key)}")
    if cand["parameters"]["channels_to_use"] != EXPECTED_CHANNELS:
        raise RuntimeError(f"channels_to_use must be {EXPECTED_CHANNELS}")
    if cand["selected_channel_names"] != EXPECTED_CHANNEL_NAMES:
        raise RuntimeError(f"selected_channel_names must be {EXPECTED_CHANNEL_NAMES}")


def build_training_command(config: dict[str, Any], python_exe: str, *, dry_run: bool) -> list[str]:
    command = [
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
        append_arg(command, key, value)
    command.extend(["--vae_required_metadata_cols", *REQUIRED_METADATA_COLS])
    command.append("--vae_abort_if_val_split_fails")
    if dry_run:
        command.append("--dry-run")
    return command


def normalize_meta_for_counts(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].astype(str)
    out["Manufacturer"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Sex"] = out["Sex"].astype(str)
    out["tensor_idx"] = out["tensor_idx"].astype(int)
    return out


def missing_required_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in REQUIRED_METADATA_COLS:
        mask |= (
            df[col].isna()
            | df[col].astype(str).str.strip().isin(["", "nan", "NaN", "None", "none", "NA", "N/A"])
        )
    return mask


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def component_counts(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(df))}
    for dx in ["CN", "MCI", "AD"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "Philips", "SIEMENS"]:
        out[f"Manufacturer_{mfr}"] = int(df["Manufacturer"].eq(mfr).sum())
    return out


def validate_harmonization_guards(fit_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> dict[str, Any]:
    fit_subjects = set(fit_df["SubjectID"].astype(str))
    test_subjects = set(test_df["SubjectID"].astype(str))
    overlap = sorted(fit_subjects.intersection(test_subjects))
    fit_levels = sorted(set(fit_df["Manufacturer"].map(normalize_manufacturer)))
    test_levels = sorted(set(test_df["Manufacturer"].map(normalize_manufacturer)))
    required_levels = {"GE", "Philips", "SIEMENS"}
    missing_levels = sorted(required_levels.difference(set(fit_levels)))
    missing_age_fit = int(pd.to_numeric(fit_df["Age"], errors="coerce").isna().sum())
    missing_age_test = int(pd.to_numeric(test_df["Age"], errors="coerce").isna().sum())
    missing_sex_fit = int(fit_df["Sex"].isna().sum() + fit_df["Sex"].astype(str).str.strip().isin(["", "nan", "NaN", "NA", "None"]).sum())
    missing_sex_test = int(test_df["Sex"].isna().sum() + test_df["Sex"].astype(str).str.strip().isin(["", "nan", "NaN", "NA", "None"]).sum())
    status = "PASS"
    errors: list[str] = []
    if overlap:
        status = "FAIL"
        errors.append(f"fit/test overlap n={len(overlap)}")
    if missing_levels:
        status = "FAIL"
        errors.append(f"missing fit Manufacturer levels {missing_levels}")
    if missing_age_fit or missing_age_test or missing_sex_fit or missing_sex_test:
        status = "FAIL"
        errors.append("missing Age/Sex in fit or test")
    return {
        "fold": int(fold),
        "status": status,
        "fit_scope": "outer_train_dev_only",
        "batch_col": "Manufacturer",
        "covariates_preserved": "Age+Sex",
        "excluded_covariates": "ResearchGroup_Mapped",
        "diagnosis_used_in_harmonizer": False,
        "global_combat": False,
        "oasis_used": False,
        "n_fit_subjects": int(len(fit_df)),
        "n_test_subjects": int(len(test_df)),
        "n_fit_test_subject_overlap": int(len(overlap)),
        "fit_manufacturer_levels": ";".join(fit_levels),
        "test_manufacturer_levels": ";".join(test_levels),
        "missing_age_fit": missing_age_fit,
        "missing_age_test": missing_age_test,
        "missing_sex_fit": missing_sex_fit,
        "missing_sex_test": missing_sex_test,
        "errors": " | ".join(errors),
    }


def synthetic_smoke_test() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n, c, r = 12, 3, 6
    base = rng.normal(size=(n, c, r, r))
    tensor = (base + np.swapaxes(base, 2, 3)) / 2.0
    idx = np.arange(r)
    tensor[:, :, idx, idx] = 0.0
    meta = pd.DataFrame(
        {
            "SubjectID": [f"S{i:02d}" for i in range(n)],
            "Manufacturer": ["GE", "Philips", "SIEMENS"] * 4,
            "Age": np.linspace(65, 80, n),
            "Sex": ["F", "M"] * 6,
            "ResearchGroup_Mapped": ["CN", "AD", "MCI"] * 4,
        }
    )
    train_idx = np.arange(9)
    test_idx = np.arange(9, 12)
    fitted = fit_tensor_combat_channelwise(
        tensor[train_idx],
        meta.iloc[train_idx].copy(),
        channel_indices=[1, 0, 2],
        channel_names=EXPECTED_CHANNEL_NAMES,
    )
    test_out = transform_tensor_combat_channelwise(fitted, tensor[test_idx], meta.iloc[test_idx].copy())
    diag_delta = float(np.max(np.abs(np.diagonal(test_out, axis1=2, axis2=3) - np.diagonal(tensor[test_idx], axis1=2, axis2=3))))
    sym_error = float(np.max(np.abs(test_out - np.swapaxes(test_out, 2, 3))))
    return {
        "status": "PASS" if test_out.shape == tensor[test_idx].shape and diag_delta <= 1e-10 and sym_error <= 1e-8 else "FAIL",
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "shape": "x".join(map(str, test_out.shape)),
        "diag_delta_max": diag_delta,
        "symmetry_error_max": sym_error,
        "finite_fraction": float(np.isfinite(test_out).mean()),
    }


def run_foldwise_probe(
    tensor: np.ndarray,
    meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    selected_tensor = tensor[:, EXPECTED_CHANNELS, :, :]
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    y = strat_key(cn_ad, ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]])
    if (y.value_counts() < int(params["outer_folds"])).any():
        raise RuntimeError("Manufacturer-aware 5-fold split is not feasible.")
    splitter = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=int(params["seed"]))
    all_valid_idx = meta.loc[meta["tensor_idx"] <= selected_tensor.shape[0] - 1, "tensor_idx"].astype(int).to_numpy()
    by_tensor = meta.set_index("tensor_idx", drop=False)

    plan_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []

    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_valid_idx, test["tensor_idx"].to_numpy(dtype=int), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        n_before_filter = int(len(vae_pool))
        removed = missing_required_mask(vae_pool)
        vae_pool = vae_pool.loc[~removed].reset_index(drop=True)
        vae_pool_idx = vae_pool["tensor_idx"].to_numpy(dtype=int)
        guard = validate_harmonization_guards(vae_pool, test, fold)
        guard_rows.append(guard)
        if guard["status"] != "PASS":
            continue
        for split_name, df in [("fit_train_dev_vae_pool", vae_pool), ("classifier_train_dev_apply", train_dev), ("classifier_test_apply", test)]:
            row = {
                "fold": int(fold),
                "split": split_name,
                "n_before_required_metadata_filter": n_before_filter if split_name == "fit_train_dev_vae_pool" else "",
                "n_removed_required_metadata_filter": int(removed.sum()) if split_name == "fit_train_dev_vae_pool" else "",
                "selected_channels": ",".join(map(str, EXPECTED_CHANNELS)),
                "selected_channel_names": ";".join(EXPECTED_CHANNEL_NAMES),
                "would_generate_harmonized_input": True,
            }
            row.update(component_counts(df))
            plan_rows.append(row)
        fit_tensor = selected_tensor[vae_pool_idx]
        fitted = fit_tensor_combat_channelwise(
            fit_tensor,
            vae_pool,
            channel_indices=EXPECTED_CHANNELS,
            channel_names=EXPECTED_CHANNEL_NAMES,
        )
        fit_h = transform_tensor_combat_channelwise(fitted, fit_tensor, vae_pool)
        train_dev_tensor = selected_tensor[train_dev["tensor_idx"].to_numpy(dtype=int)]
        test_tensor = selected_tensor[test["tensor_idx"].to_numpy(dtype=int)]
        train_dev_h = transform_tensor_combat_channelwise(fitted, train_dev_tensor, train_dev)
        test_h = transform_tensor_combat_channelwise(fitted, test_tensor, test)
        for split_name, before, after in [
            ("fit_train_dev_vae_pool", fit_tensor, fit_h),
            ("classifier_train_dev_apply", train_dev_tensor, train_dev_h),
            ("classifier_test_apply", test_tensor, test_h),
        ]:
            for row in channel_shift_summary(before, after, split_name=split_name, channel_names=EXPECTED_CHANNEL_NAMES):
                row["fold"] = int(fold)
                shift_rows.append(row)
        pre_sep = manufacturer_centroid_separability_proxy(fit_tensor, vae_pool, channel_names=EXPECTED_CHANNEL_NAMES)
        post_sep = manufacturer_centroid_separability_proxy(fit_h, vae_pool, channel_names=EXPECTED_CHANNEL_NAMES)
        for pre, post in zip(pre_sep, post_sep):
            shift_rows.append(
                {
                    "fold": int(fold),
                    "split": "fit_train_dev_vae_pool",
                    "channel_position": int(pre["channel_position"]),
                    "channel_name": str(pre["channel_name"]),
                    "pre_mean": pre["mean_pairwise_centroid_distance_per_edge"],
                    "post_mean": post["mean_pairwise_centroid_distance_per_edge"],
                    "delta_mean": post["mean_pairwise_centroid_distance_per_edge"] - pre["mean_pairwise_centroid_distance_per_edge"],
                    "pre_std": np.nan,
                    "post_std": np.nan,
                    "delta_std": np.nan,
                    "mean_abs_delta": np.nan,
                    "max_abs_delta": np.nan,
                    "finite_fraction_post": float(np.isfinite(fit_h).mean()),
                    "metric": "manufacturer_centroid_separability_proxy",
                }
            )

    return pd.DataFrame(plan_rows), pd.DataFrame(guard_rows), pd.DataFrame(shift_rows)


def stale_markers(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    names = []
    for child in output_dir.iterdir():
        if child.name.startswith("fold_") or child.name in {"classifier_only_readout", "latent_cache", "run_manifest.json"}:
            names.append(str(child))
    return sorted(names)


def run_py_compile_checks(python_exe: str) -> list[dict[str, Any]]:
    targets = [
        TRAINING_SCRIPT,
        REVISION_DIR / "foldwise_combat_input_harmonization.py",
        BRANCH_B_LAUNCHER,
        Path(__file__).resolve(),
    ]
    rows: list[dict[str, Any]] = []
    for target in targets:
        command = [python_exe, "-m", "py_compile", str(target)]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
        rows.append(
            {
                "script": str(target.relative_to(PROJECT_ROOT)),
                "command": shlex.join(command),
                "returncode": int(completed.returncode),
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "stderr": completed.stderr.strip(),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(resolve(args.config))
    ref = load_json(resolve(args.reference_config))
    validate_config(ref, config)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable

    dep = dependency_status()
    if not dep.get("neurocombat_sklearn_CombatModel_available"):
        raise RuntimeError(f"ComBat dependency unavailable: {dep}")

    tensor, meta, roi_names, network_labels = load_data(
        resolve(config["paths"]["global_tensor_path"]),
        resolve(config["paths"]["metadata_path"]),
    )
    if tensor is None or meta is None:
        raise RuntimeError("Could not load tensor/metadata for preflight.")
    meta = normalize_meta_for_counts(meta)
    selected_names = [config["channel_names_master_in_tensor_order"][i] for i in EXPECTED_CHANNELS]
    if selected_names != EXPECTED_CHANNEL_NAMES:
        raise RuntimeError(f"Selected channel names mismatch: {selected_names}")
    full_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    clf_counts = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]["ResearchGroup_Mapped"].value_counts().to_dict()
    for dx, expected in EXPECTED_VAE_COUNTS.items():
        if int(full_counts.get(dx, 0)) != expected:
            raise RuntimeError(f"VAE {dx} count expected {expected}, got {full_counts.get(dx, 0)}")
    for dx, expected in EXPECTED_CLF_COUNTS.items():
        if int(clf_counts.get(dx, 0)) != expected:
            raise RuntimeError(f"Classifier {dx} count expected {expected}, got {clf_counts.get(dx, 0)}")

    smoke = synthetic_smoke_test()
    if smoke["status"] != "PASS":
        raise RuntimeError(f"Synthetic smoke test failed: {smoke}")
    py_compile_rows = run_py_compile_checks(python_exe)
    if any(row["status"] != "PASS" for row in py_compile_rows):
        raise RuntimeError(f"py_compile validation failed: {py_compile_rows}")

    plan, guards, shifts = run_foldwise_probe(tensor, meta, config)
    if guards.empty or not guards["status"].eq("PASS").all():
        raise RuntimeError("One or more foldwise harmonization guards failed.")
    write_table(plan, out_dir, "foldwise_harmonized_input_plan")
    write_table(guards, out_dir, "harmonization_leakage_guard_report")
    write_table(shifts, out_dir, "harmonization_scale_shift_probe")

    training_dry = build_training_command(config, python_exe, dry_run=True)
    training_real = build_training_command(config, python_exe, dry_run=False)
    stale = stale_markers(resolve(config["paths"]["output_dir"]))
    launcher_dry = [
        python_exe,
        str(BRANCH_B_LAUNCHER),
        "--dry-run",
        "--config",
        str(resolve(args.config)),
        "--reference-config",
        str(resolve(args.reference_config)),
        "--preflight-output-dir",
        str(out_dir),
    ]
    dryrun_report = f"""# Branch B Launcher Dry-Run Report

Launcher: `{BRANCH_B_LAUNCHER.relative_to(PROJECT_ROOT)}`

Trainer dry-run command:

```bash
{shlex.join(training_dry)}
```

Real training command, still blocked unless `--confirm-training` is passed to the launcher:

```bash
{shlex.join(training_real)}
```

Stale output markers: `{len(stale)}`

No real training was launched by this preflight.
"""
    (out_dir / "branch_b_launcher_dryrun_report.md").write_text(dryrun_report, encoding="utf-8")

    code_audit = f"""# Branch B Real Wiring Code Audit

Implemented wiring:
- `scripts/run_vae_clf_ad_inference.py` now supports default-off `--input_harmonization_mode none|foldwise_combat`.
- Default `none` leaves existing configs unchanged.
- `foldwise_combat` fits one ComBat model per selected channel on the fold-local VAE train/dev pool after outer-test exclusion and required-metadata filtering.
- The same frozen channel-wise models are applied to classifier train/dev and outer-test tensors.
- Diagnosis/`ResearchGroup_Mapped` is explicitly excluded from the harmonization covariate model.
- Downstream normalization, VAE training, latent extraction, and classifier readout reuse existing pipeline code.

Helper:
- `scripts/revision_bspc_2026/foldwise_combat_input_harmonization.py`
- dependency status: `{json.dumps(dep, sort_keys=True)}`

Leakage guards:
- no global ComBat
- no outer-test fitting
- no diagnosis covariate
- no OASIS data
- train/test SubjectID overlap abort
- all Manufacturer levels required in fit train/dev
- diagonal and symmetry preservation checked after transform
"""
    (out_dir / "branch_b_real_wiring_code_audit.md").write_text(code_audit, encoding="utf-8")

    run_manifest = pd.DataFrame(
        [
            {
                "run_name": config["run_name"],
                "config": str(resolve(args.config)),
                "reference_config": str(resolve(args.reference_config)),
                "output_dir": str(resolve(config["paths"]["output_dir"])),
                "big_disk_output_dir": str(resolve(config["paths"]["big_disk_output_dir"])),
                "selected_channels": ",".join(map(str, EXPECTED_CHANNELS)),
                "selected_channel_names": ";".join(EXPECTED_CHANNEL_NAMES),
                "input_harmonization_mode": "foldwise_combat",
                "batch": "Manufacturer",
                "covariates_preserved": "Age+Sex",
                "diagnosis_used": False,
                "dry_run_training_command": shlex.join(training_dry),
                "real_training_command": shlex.join(training_real),
                "stale_output_markers": len(stale),
                "synthetic_smoke_status": smoke["status"],
            }
        ]
    )
    write_table(run_manifest, out_dir, "run_manifest")

    readme = f"""# Foldwise ComBat Real Training Wiring Preflight

Target run: `{RUN_NAME}`

This package validates real training wiring for foldwise input ComBat harmonization by Manufacturer while preserving Age/Sex and excluding diagnosis.

Validation summary:
- ComBat dependency available: `{dep.get('neurocombat_sklearn_CombatModel_available')}`
- Synthetic fit/apply smoke test: `{smoke['status']}`
- py_compile checks: `{sum(row['status'] == 'PASS' for row in py_compile_rows)}/{len(py_compile_rows)} PASS`
- Foldwise leakage guards: `{int(guards['status'].eq('PASS').sum())}/{len(guards)} PASS`
- Selected channels: `{EXPECTED_CHANNELS}` / `{'; '.join(EXPECTED_CHANNEL_NAMES)}`
- VAE pool counts: CN={EXPECTED_VAE_COUNTS['CN']}, MCI={EXPECTED_VAE_COUNTS['MCI']}, AD={EXPECTED_VAE_COUNTS['AD']}
- Classifier pool counts: CN={EXPECTED_CLF_COUNTS['CN']}, AD={EXPECTED_CLF_COUNTS['AD']}
- Real training launched: `False`

Generated files:
- `branch_b_real_wiring_code_audit.md`
- `foldwise_harmonized_input_plan.csv/.md`
- `harmonization_leakage_guard_report.csv/.md`
- `harmonization_scale_shift_probe.csv/.md`
- `branch_b_launcher_dryrun_report.md`
- `run_manifest.csv/.md`
- `command_log.json`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    command_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(resolve(args.config)),
        "reference_config": str(resolve(args.reference_config)),
        "output_dir": str(out_dir),
        "dependency_status": dep,
        "synthetic_smoke_test": smoke,
        "py_compile_checks": py_compile_rows,
        "launcher_dry_run_command": shlex.join(launcher_dry),
        "trainer_dry_run_command": shlex.join(training_dry),
        "trainer_real_command": shlex.join(training_real),
        "guardrails": {
            "real_training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "model_artifact_modified": False,
            "oasis_scoring": False,
            "threshold_or_calibration_fitting": False,
        },
    }
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir}")
    print("Foldwise ComBat wiring preflight PASS. No training launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
