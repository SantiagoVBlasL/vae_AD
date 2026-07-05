#!/usr/bin/env python3
"""Create the Branch A/Branch B post-final method preflight package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "scripts/revision_bspc_2026") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/revision_bspc_2026"))

from foldwise_combat_input_harmonization import dependency_status  # noqa: E402
from run_vae_clf_ad_inference import (  # noqa: E402
    RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM,
    RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN,
    RECON_LOSS_MODES,
    describe_recon_loss_mode,
    vae_reconstruction_loss,
)

RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
OUT_DEFAULT = RESULTS / "postfinal_method_branch_A_chmeanloss_B_foldcombat_preflight_20260606"
REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
BRANCH_A_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5.json"
BRANCH_B_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5.json"
BRANCH_A_LAUNCHER = PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_v5_1c_recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5.py"
BRANCH_B_LAUNCHER = PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5.py"
SCALE_PROBE = PROJECT_ROOT / "scripts/revision_bspc_2026/probe_postfinal_chmeanloss_scale_20260606.py"
SMOKE_TEST = PROJECT_ROOT / "scripts/revision_bspc_2026/smoke_test_vae_recon_loss_modes.py"
SELECTED_CHANNELS = [1, 0, 2]
SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
STALE_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--python-executable", default="/home/diego/anaconda3/envs/vae_ad/bin/python")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def run_cmd(cmd: Sequence[str]) -> dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {"cmd": list(cmd), "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return view.to_markdown(index=False) + "\n"


def write_table(path_stem: Path, df: pd.DataFrame, max_rows: int = 200) -> None:
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    path_stem.with_suffix(".md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def normalize_dx(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text in {"CN", "CONTROL", "NORMAL", "0"}:
        return "CN"
    if text in {"AD", "AD_DEMENTIA", "DEMENTIA", "1"}:
        return "AD"
    if text == "MCI":
        return "MCI"
    return str(value)


def normalize_mfr(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    low = text.lower()
    if "philips" in low:
        return "Philips"
    if "siemens" in low:
        return "SIEMENS"
    if low in {"ge", "general electric"} or "general electric" in low:
        return "GE"
    return text or "UNKNOWN"


def normalize_sex(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text in {"F", "FEMALE", "0"}:
        return "F"
    if text in {"M", "MALE", "1"}:
        return "M"
    return "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "tensor_index" in meta.columns and "tensor_idx" not in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].map(normalize_dx)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    meta["Sex"] = meta["Sex"].map(normalize_sex)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = pd.to_numeric(meta["tensor_idx"], errors="coerce").astype(int)
    return meta


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in tmp.columns:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_dx(df: pd.DataFrame) -> dict[str, int]:
    return {dx: int(df["ResearchGroup_Mapped"].eq(dx).sum()) for dx in ["CN", "MCI", "AD"]}


def subject_pool_audit(meta: pd.DataFrame) -> pd.DataFrame:
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    rows = [
        {"check": "vae_CN", "actual": count_dx(meta)["CN"], "expected": EXPECTED_VAE_COUNTS["CN"]},
        {"check": "vae_MCI", "actual": count_dx(meta)["MCI"], "expected": EXPECTED_VAE_COUNTS["MCI"]},
        {"check": "vae_AD", "actual": count_dx(meta)["AD"], "expected": EXPECTED_VAE_COUNTS["AD"]},
        {"check": "classifier_CN", "actual": count_dx(clf)["CN"], "expected": EXPECTED_CLF_COUNTS["CN"]},
        {"check": "classifier_AD", "actual": count_dx(clf)["AD"], "expected": EXPECTED_CLF_COUNTS["AD"]},
        {"check": "035_S_6927_present", "actual": int(meta["SubjectID"].eq(RECOVER_SUBJECT).any()), "expected": 1},
        {"check": "128_S_2002_absent", "actual": int(~meta["SubjectID"].eq(EXCLUDED_SUBJECT).any()), "expected": 1},
    ]
    out = pd.DataFrame(rows)
    out["pass"] = out["actual"].eq(out["expected"])
    if not out["pass"].all():
        raise RuntimeError("Subject pool audit failed:\n" + out.to_string(index=False))
    return out


def tensor_audit(config: dict[str, Any]) -> pd.DataFrame:
    tensor_path = resolve(config["paths"]["global_tensor_path"])
    with np.load(tensor_path, allow_pickle=True) as zf:
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in np.asarray(zf["channel_names"]).tolist()]
        subject_ids = np.asarray(zf["subject_ids"]).astype(str).tolist()
    selected = [channel_names[i] for i in SELECTED_CHANNELS]
    if selected != SELECTED_NAMES:
        raise RuntimeError(f"Selected channels mismatch: {selected}")
    return pd.DataFrame(
        [
            {
                "tensor_path": str(tensor_path),
                "exists": tensor_path.exists(),
                "shape": str(shape),
                "selected_channels": str(SELECTED_CHANNELS),
                "selected_channel_names": " | ".join(selected),
                "n_subject_ids": len(subject_ids),
                "035_S_6927_in_tensor": RECOVER_SUBJECT in subject_ids,
                "128_S_2002_in_tensor": EXCLUDED_SUBJECT in subject_ids,
            }
        ]
    )


def fold_feasibility(meta: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    params = config["parameters"]
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    outer_key = strat_key(cn_ad, ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]])
    if (outer_key.value_counts() < int(params["outer_folds"])).any():
        raise RuntimeError("Outer split infeasible.")
    all_tensor_idx = meta["tensor_idx"].to_numpy(dtype=int)
    by_idx = meta.set_index("tensor_idx", drop=False)
    rows: list[dict[str, Any]] = []
    outer = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=int(params["seed"]))
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(cn_ad)), outer_key), start=1):
        train_dev = cn_ad.iloc[train_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(dtype=int), assume_unique=False)
        vae_pool = by_idx.loc[vae_pool_idx].reset_index(drop=True)
        vae_key = strat_key(vae_pool, ["ResearchGroup_Mapped", *params["vae_stratify_cols"]])
        vae_split_ok = bool((vae_key.value_counts() >= 2).all())
        if vae_split_ok:
            train_local, val_local = train_test_split(
                np.arange(len(vae_pool)),
                test_size=float(params["vae_val_split_ratio"]),
                stratify=vae_key,
                random_state=int(params["seed"]) + fold + 9,
                shuffle=True,
            )
            vae_train = vae_pool.iloc[train_local]
            vae_val = vae_pool.iloc[val_local]
        else:
            vae_train = vae_pool
            vae_val = vae_pool.iloc[[]]
        for name, df in [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_train),
            ("vae_internal_val", vae_val),
        ]:
            row: dict[str, Any] = {"fold": fold, "split_component": name, "n": int(len(df))}
            row.update({f"dx_{k}": v for k, v in count_dx(df).items()})
            for mfr in ["GE", "Philips", "SIEMENS"]:
                row[f"mfr_{mfr}"] = int(df["Manufacturer"].eq(mfr).sum())
            row["age_missing"] = int(pd.to_numeric(df["Age"], errors="coerce").isna().sum())
            row["sex_missing_or_unknown"] = int(df["Sex"].map(normalize_sex).eq("UNKNOWN").sum())
            row["manufacturer_levels"] = ";".join(sorted(df["Manufacturer"].dropna().astype(str).unique()))
            row["contains_035_S_6927"] = bool(df["SubjectID"].eq(RECOVER_SUBJECT).any())
            row["contains_128_S_2002"] = bool(df["SubjectID"].eq(EXCLUDED_SUBJECT).any())
            row["vae_split_feasible"] = vae_split_ok
            row["passes"] = (
                row["age_missing"] == 0
                and row["sex_missing_or_unknown"] == 0
                and not row["contains_128_S_2002"]
                and all(row[f"mfr_{mfr}"] > 0 for mfr in ["GE", "Philips", "SIEMENS"])
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out["passes"].all():
        raise RuntimeError("Fold feasibility failed:\n" + out.to_string(index=False))
    return out


def config_diff(ref: dict[str, Any], cand: dict[str, Any]) -> pd.DataFrame:
    r = dict(ref["parameters"])
    c = dict(cand["parameters"])
    rows = []
    for key in sorted(set(r) | set(c)):
        if r.get(key) != c.get(key):
            rows.append({"scope": "parameters", "field": key, "reference": r.get(key), "candidate": c.get(key)})
    for key, value in cand["paths"].items():
        if ref["paths"].get(key) != value:
            rows.append({"scope": "paths", "field": key, "reference": ref["paths"].get(key), "candidate": value})
    if ref["run_name"] != cand["run_name"]:
        rows.append({"scope": "top", "field": "run_name", "reference": ref["run_name"], "candidate": cand["run_name"]})
    return pd.DataFrame(rows)


def output_status(config: dict[str, Any]) -> dict[str, Any]:
    local = resolve(config["paths"]["output_dir"])
    target = Path(config["paths"]["big_disk_output_dir"])
    stale: list[Path] = []
    if local.exists():
        for child in local.iterdir():
            if child.name in STALE_NAMES or child.name.startswith(STALE_PREFIXES):
                stale.append(child)
    return {
        "local_output_dir": str(local),
        "big_disk_output_dir": str(target),
        "local_exists": local.exists() or local.is_symlink(),
        "local_is_symlink": local.is_symlink(),
        "big_disk_exists": target.exists(),
        "target_match": bool(local.is_symlink() and target.exists() and local.resolve() == target.resolve()),
        "stale_marker_count": len(stale),
        "stale_markers": " | ".join(str(p) for p in stale),
        "setup_command_if_needed": (
            f"mkdir -p {target} && ln -s {target} {local}"
            if not (local.exists() or local.is_symlink())
            else ""
        ),
    }


def write_reports(
    outdir: Path,
    ref: dict[str, Any],
    branch_a: dict[str, Any],
    branch_b: dict[str, Any],
    pool: pd.DataFrame,
    tensor: pd.DataFrame,
    folds: pd.DataFrame,
    launcher_results: list[dict[str, Any]],
    command_log: dict[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_table(outdir / "subject_pool_audit", pool)
    write_table(outdir / "tensor_compatibility_audit", tensor)
    write_table(outdir / "branch_b_foldwise_harmonization_feasibility", folds, max_rows=500)

    a_diff = config_diff(ref, branch_a)
    b_diff = config_diff(ref, branch_b)
    a_status = pd.DataFrame([output_status(branch_a)])
    b_status = pd.DataFrame([output_status(branch_b)])
    manifest = pd.DataFrame(
        [
            {
                "branch": "A_channel_normalized_loss",
                "run_name": branch_a["run_name"],
                "config": str(BRANCH_A_CONFIG),
                "launcher": str(BRANCH_A_LAUNCHER),
                "allowed_scientific_diff": "recon_loss_mode mse_sum_batchmean_current -> mse_offdiag_channel_mean_sum",
                **output_status(branch_a),
            },
            {
                "branch": "B_foldwise_combat_mfr_age_sex",
                "run_name": branch_b["run_name"],
                "config": str(BRANCH_B_CONFIG),
                "launcher": str(BRANCH_B_LAUNCHER),
                "allowed_scientific_diff": "foldwise input harmonization: batch Manufacturer, preserve Age/Sex, exclude Diagnosis",
                **output_status(branch_b),
            },
        ]
    )
    write_table(outdir / "run_manifest", manifest)

    loss_probe_summary = outdir / "branch_a_loss_scale_probe_summary.json"
    loss_probe_text = ""
    if loss_probe_summary.exists():
        loss_probe = json.loads(loss_probe_summary.read_text(encoding="utf-8"))
        loss_probe_text = (
            f"- Mean D ratio chmean/current on VAE pools: `{loss_probe.get('mean_chmean_over_current_ratio_vae_pool_all'):.6g}`.\n"
            f"- Mean estimated beta*KLD/D under chmean scaling: `{loss_probe.get('mean_estimated_beta_kld_over_D_chmean_vae_pool_all'):.6g}`.\n"
        )
    (outdir / "branch_a_loss_code_audit.md").write_text(
        "\n".join(
            [
                "# Branch A Loss Code Audit",
                "",
                f"- Requested mode: `{RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM}`.",
                f"- Existing equivalent mode: `{RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN}`.",
                f"- Available modes now include: `{', '.join(RECON_LOSS_MODES)}`.",
                "- Implementation: `scripts/run_vae_clf_ad_inference.py::vae_reconstruction_loss`.",
                "- Semantics: off-diagonal squared error is summed within each selected channel, averaged across channels, then averaged across batch.",
                "- Synthetic invariant test: `scripts/revision_bspc_2026/smoke_test_vae_recon_loss_modes.py` checks one-channel equivalence and three-identical-channel non-tripling.",
                f"- Scale description: `{describe_recon_loss_mode(RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM, 3, 131)}`.",
                "",
                "## Config Diff",
                "",
                md_table(a_diff),
                "## Scale Probe",
                "",
                loss_probe_text or "- Scale probe CSV is expected at `branch_a_loss_scale_probe.csv` after running the probe script.",
            ]
        ),
        encoding="utf-8",
    )
    (outdir / "branch_a_preflight_report.md").write_text(
        "\n".join(
            [
                "# Branch A Preflight Report",
                "",
                "- Status: prepared, dry-run only.",
                "- Real training launched: no.",
                "- Tensor/metadata/ledger/model-output modification: no.",
                "- Strict scientific diff: `recon_loss_mode` only.",
                "- Selected channels: `[1,0,2]` = Pearson Full, OMST, MI.",
                "- VAE pool: CN=300, MCI=250, AD=97.",
                "- Classifier pool: CN=300, AD=97.",
                "- 035_S_6927 included; 128_S_2002 excluded from metadata pools.",
                f"- Stale markers: `{int(a_status['stale_marker_count'].iloc[0])}`.",
                f"- Symlink target valid: `{bool(a_status['target_match'].iloc[0])}`.",
                f"- Setup if needed: `{a_status['setup_command_if_needed'].iloc[0]}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    deps = dependency_status()
    (outdir / "branch_b_harmonization_code_audit.md").write_text(
        "\n".join(
            [
                "# Branch B Harmonization Code Audit",
                "",
                "- Existing foldwise ComBat audit code found: `scripts/revision_bspc_2026/audit_combat_train_apply_validation_classifier_only_sensitivity_20260603.py`.",
                "- Minimal helper module prepared: `scripts/revision_bspc_2026/foldwise_combat_input_harmonization.py`.",
                f"- Dependency status: `{deps}`.",
                "- Vectorization: upper-triangle off-diagonal features, channel-wise.",
                "- Reconstruction: mirrored symmetric matrices with original diagonal preserved.",
                "- Leakage guard: fit on outer train/dev only; apply frozen transform to outer test.",
                "- Batch: Manufacturer.",
                "- Preserved covariates: Age, Sex.",
                "- Excluded covariates: diagnosis / ResearchGroup_Mapped.",
                "",
                "## Config Diff",
                "",
                md_table(b_diff),
                "",
                "## Training-Path Status",
                "",
                "The transform primitive is available for validation, but the main VAE trainer is not yet wired to consume foldwise harmonized tensors. The Branch B launcher therefore refuses real training and passes dry-run/preflight only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (outdir / "branch_b_preflight_report.md").write_text(
        "\n".join(
            [
                "# Branch B Preflight Report",
                "",
                "- Status: prepared as leakage-feasibility branch, dry-run only.",
                "- Real training launched: no.",
                "- Tensor/metadata/ledger/model-output modification: no.",
                "- Strict scientific diff: harmonization settings only.",
                "- Selected channels: `[1,0,2]` = Pearson Full, OMST, MI.",
                "- Age/Sex missingness: zero in all audited fold components.",
                "- Manufacturer levels: GE, Philips, SIEMENS present in every audited fold component.",
                "- Diagnosis is excluded from the harmonization model.",
                f"- Stale markers: `{int(b_status['stale_marker_count'].iloc[0])}`.",
                f"- Symlink target valid: `{bool(b_status['target_match'].iloc[0])}`.",
                f"- Setup if needed: `{b_status['setup_command_if_needed'].iloc[0]}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Post-Final Method Branch A/B Preflight",
        "",
        "Prepared two controlled exploratory FULL-method branches based on the promoted reference `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`.",
        "",
        "## Branches",
        "",
        "- Branch A: `recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5`, changing only `recon_loss_mode` to `mse_offdiag_channel_mean_sum`.",
        "- Branch B: `recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5`, enabling foldwise Manufacturer ComBat feasibility settings only.",
        "",
        "## Guardrails",
        "",
        "- No real training was launched.",
        "- No tensor, metadata, ledger, or model artifacts were modified.",
        "- No OASIS scoring, threshold fitting, or calibration fitting was performed.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log["launcher_results"] = launcher_results
    command_log["end_utc"] = now_utc()
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    command_log: dict[str, Any] = {
        "start_utc": now_utc(),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "model_artifact_modified": False,
        "commands": [],
    }
    ref = load_json(REFERENCE_CONFIG)
    branch_a = load_json(BRANCH_A_CONFIG)
    branch_b = load_json(BRANCH_B_CONFIG)
    meta = load_metadata(resolve(ref["paths"]["metadata_path"]))
    pool = subject_pool_audit(meta)
    tensor = tensor_audit(ref)
    folds = fold_feasibility(meta, ref)

    # Run the synthetic loss smoke test and the Branch A scale probe.
    for cmd in [
        [args.python_executable, str(SMOKE_TEST)],
        [args.python_executable, str(SCALE_PROBE), "--output-dir", str(outdir)],
    ]:
        result = run_cmd(cmd)
        command_log["commands"].append(result)
        if result["returncode"] != 0:
            raise RuntimeError(f"Command failed: {cmd}\nSTDOUT:\n{result['stdout']}\nSTDERR:\n{result['stderr']}")

    launcher_results: list[dict[str, Any]] = []
    for cmd in [
        [args.python_executable, str(BRANCH_A_LAUNCHER), "--dry-run"],
        [args.python_executable, str(BRANCH_B_LAUNCHER), "--dry-run"],
    ]:
        result = run_cmd(cmd)
        launcher_results.append(result)
        command_log["commands"].append(result)
        if result["returncode"] != 0:
            raise RuntimeError(f"Launcher dry-run failed: {cmd}\nSTDOUT:\n{result['stdout']}\nSTDERR:\n{result['stderr']}")

    write_reports(outdir, ref, branch_a, branch_b, pool, tensor, folds, launcher_results, command_log)
    print(f"Wrote preflight package: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
