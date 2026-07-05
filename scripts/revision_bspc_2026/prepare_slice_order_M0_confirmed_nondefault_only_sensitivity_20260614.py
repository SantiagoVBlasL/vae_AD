#!/usr/bin/env python3
"""Preflight the M0 confirmed-nondefault slice-order sensitivity run.

This is a preflight-only generator. It creates a derived metadata CSV that
excludes exactly the five confirmed Philips Site31 reverse/non-default
slice-order subjects from all relevant VAE and StageB pools, writes a guarded
launcher, runs launcher --dry-run, and records feasibility/score-only retained
set summaries. It does not train and does not modify promoted artifacts,
source tensors, original metadata, predictions, thresholds, or fold files.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "slice_order_M0_confirmed_nondefault_only_sensitivity_20260614"

FULL_DB = (
    RESULTS
    / "full_database_for_martin_and_validity_preflight_20260612"
    / "promoted_model_full_database_for_martin_20260612.csv"
)
REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
PREVIOUS_PREFLIGHT = RESULTS / "slice_order_curated_valid_only_sensitivity_20260612"

RUN_NAME = (
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_"
    "sliceorderM0_confirmed_nondefault_only_sensitivity_20260614"
)
EXPECTED_M0_SUBJECTS = [
    "031_S_4021",
    "031_S_4032",
    "031_S_4218",
    "031_S_4474",
    "031_S_4496",
]

DERIVED_METADATA = OUT / "M0_derived_metadata.csv"
EXCLUSION_MANIFEST = OUT / "M0_exclusion_manifest.csv"
GENERATED_CONFIG = OUT / "generated_config.json"
GUARDED_LAUNCHER = OUT / "guarded_launcher.py"
DRY_RUN_PASS = OUT / "DRY_RUN_PASS.txt"

SEED = 42
OUTER_FOLDS = 5
VAE_VAL_SPLIT = 0.2


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_df(stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def norm_str(s: pd.Series) -> pd.Series:
    return s.fillna("MISSING").astype(str).str.strip().replace(
        {"": "MISSING", "nan": "MISSING", "NaN": "MISSING"}
    )


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return norm_str(s).str.lower().isin({"true", "1", "yes", "y"})


def site_norm(df: pd.DataFrame) -> pd.Series:
    return norm_str(df["Site3_final"]).str.replace(r"\.0$", "", regex=True)


def add_m0_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SubjectID"] = out["SubjectID"].astype(str).str.strip()
    out["_diagnosis_norm"] = norm_str(out["diagnosis_group"])
    out["_manufacturer_norm"] = norm_str(out["Manufacturer_final"])
    out["_site3_norm"] = site_norm(out)
    out["_slice_norm"] = norm_str(out["slice_order_class"])
    out["_is_philips"] = out["_manufacturer_norm"].str.lower().eq("philips")
    expected = set(EXPECTED_M0_SUBJECTS)
    out["exclude_M0_confirmed_nondefault_only"] = out["SubjectID"].isin(expected)
    out["retained_M0_confirmed_nondefault_only"] = ~out["exclude_M0_confirmed_nondefault_only"]
    out["M0_exclusion_reason"] = np.where(
        out["exclude_M0_confirmed_nondefault_only"],
        "Philips_Site31_reverse_even_odd_48_confirmed_nondefault",
        "",
    )
    actual = sorted(out.loc[out["exclude_M0_confirmed_nondefault_only"], "SubjectID"].tolist())
    if actual != sorted(EXPECTED_M0_SUBJECTS):
        raise RuntimeError(f"M0 exclusion subject mismatch. expected={sorted(EXPECTED_M0_SUBJECTS)}, actual={actual}")
    checks = out[out["exclude_M0_confirmed_nondefault_only"]].copy()
    bad = checks[
        ~(
            checks["_is_philips"]
            & checks["_site3_norm"].eq("31")
            & checks["_slice_norm"].eq("reverse_even_odd_48")
        )
    ]
    if not bad.empty:
        raise RuntimeError("M0 excluded subjects are not all Philips Site31 reverse_even_odd_48 subjects.")
    return out


def exclusion_manifest(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "SubjectID",
        "ImageID",
        "RID",
        "diagnosis_group",
        "Manufacturer_final",
        "Site3_final",
        "Age_final",
        "Sex_final",
        "raw_tp_group_final",
        "slice_order_class",
        "matches_dparsf_default",
        "high_confidence_slice_timing_match",
        "match_method",
        "match_confidence",
        "y_score_final",
        "y_pred",
        "confusion_label",
        "M0_exclusion_reason",
    ]
    keep = [c for c in cols if c in df.columns]
    return df.loc[df["exclude_M0_confirmed_nondefault_only"], keep].copy()


def retained_counts_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status, sub in [
        ("retained", df[df["retained_M0_confirmed_nondefault_only"]]),
        ("excluded", df[~df["retained_M0_confirmed_nondefault_only"]]),
    ]:
        rows.append({"status": status, "stratification": "total", "group": "ALL", "N": len(sub)})
        for strat, col in [
            ("diagnosis", "_diagnosis_norm"),
            ("Manufacturer", "_manufacturer_norm"),
            ("Site3", "_site3_norm"),
            ("raw_tp_group", "raw_tp_group_final"),
        ]:
            for group, n in norm_str(sub[col]).value_counts(dropna=False).sort_index().items():
                rows.append({"status": status, "stratification": strat, "group": str(group), "N": int(n)})
        ph = sub[sub["_manufacturer_norm"].eq("Philips")]
        for group, n in ph["_diagnosis_norm"].value_counts(dropna=False).sort_index().items():
            rows.append({"status": status, "stratification": "Philips_diagnosis", "group": str(group), "N": int(n)})
    return pd.DataFrame(rows)


def choose_stratification(clf: pd.DataFrame) -> tuple[np.ndarray, str, int]:
    y_label = clf["_diagnosis_norm"].map({"CN": 0, "AD": 1}).to_numpy()
    key = clf["_diagnosis_norm"].astype(str) + "_" + clf["_manufacturer_norm"].astype(str)
    vc = key.value_counts()
    if vc.empty or int(vc.min()) < OUTER_FOLDS:
        return y_label, "diagnosis_only_fallback", int(vc.min()) if not vc.empty else 0
    return key.to_numpy(), "diagnosis_plus_manufacturer", int(vc.min())


def split_train_val(vae_pool: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    key = vae_pool["_diagnosis_norm"].astype(str) + "_" + vae_pool["_manufacturer_norm"].astype(str)
    note = "diagnosis_plus_manufacturer"
    if int(key.value_counts().min()) < 2:
        key = vae_pool["_diagnosis_norm"].astype(str)
        note = "diagnosis_only"
    train_idx, val_idx = train_test_split(
        np.arange(len(vae_pool)),
        test_size=VAE_VAL_SPLIT,
        random_state=SEED + (fold - 1) + 10,
        shuffle=True,
        stratify=key,
    )
    return vae_pool.iloc[train_idx].copy(), vae_pool.iloc[val_idx].copy(), note


def retained_counts_by_fold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    feas_rows: list[dict[str, Any]] = []
    ret = df[
        df["retained_M0_confirmed_nondefault_only"]
        & df["_diagnosis_norm"].isin(["CN", "AD", "MCI"])
    ].copy()
    clf = ret[ret["_diagnosis_norm"].isin(["CN", "AD"])].copy()
    strat_y, strat_note, min_stratum = choose_stratification(clf)
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    all_ok = True
    reasons: list[str] = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(np.arange(len(clf)), strat_y), start=1):
        train = clf.iloc[tr_idx].copy()
        test = clf.iloc[te_idx].copy()
        test_ids = set(test["SubjectID"].astype(str))
        vae_pool = ret[~ret["SubjectID"].astype(str).isin(test_ids)].copy()
        try:
            vae_train, vae_val, vae_note = split_train_val(vae_pool, fold)
            val_failed = False
        except Exception as exc:
            vae_train, vae_val, vae_note = vae_pool, vae_pool.iloc[0:0], f"failed:{exc}"
            val_failed = True
        for split_name, sub in [
            ("stageB_train", train),
            ("stageB_test_oof", test),
            ("vae_train_dev_pool", vae_pool),
            ("vae_actual_train", vae_train),
            ("vae_internal_val", vae_val),
        ]:
            rows.append(
                {
                    "fold": fold,
                    "split": split_name,
                    "N": len(sub),
                    "N_CN": int((sub["_diagnosis_norm"] == "CN").sum()),
                    "N_AD": int((sub["_diagnosis_norm"] == "AD").sum()),
                    "N_MCI": int((sub["_diagnosis_norm"] == "MCI").sum()),
                    "N_GE": int((sub["_manufacturer_norm"] == "GE").sum()),
                    "N_Philips": int((sub["_manufacturer_norm"] == "Philips").sum()),
                    "N_SIEMENS": int((sub["_manufacturer_norm"] == "SIEMENS").sum()),
                    "stratification_note": strat_note if split_name.startswith("stageB") else vae_note,
                    "min_outer_stratum": min_stratum,
                    "vae_val_split_failed": val_failed if split_name.startswith("vae") else False,
                }
            )
        checks = {
            "stageB_train_has_CN_AD": all((train["_diagnosis_norm"] == dx).any() for dx in ["CN", "AD"]),
            "stageB_test_has_CN_AD": all((test["_diagnosis_norm"] == dx).any() for dx in ["CN", "AD"]),
            "vae_pool_has_CN_AD_MCI": all((vae_pool["_diagnosis_norm"] == dx).any() for dx in ["CN", "AD", "MCI"]),
            "train_has_all_mfr": set(train["_manufacturer_norm"]) >= {"GE", "Philips", "SIEMENS"},
            "test_has_all_mfr": set(test["_manufacturer_norm"]) >= {"GE", "Philips", "SIEMENS"},
        }
        bad = [k for k, v in checks.items() if not bool(v)]
        if bad or val_failed:
            all_ok = False
            reasons.extend([f"fold_{fold}_{b}" for b in bad])
            if val_failed:
                reasons.append(f"fold_{fold}_vae_val_split_failed")
        feas_rows.append(
            {
                "fold": fold,
                "status": "PASS" if not bad and not val_failed else "FAIL",
                "reason": "PASS" if not bad and not val_failed else ";".join(bad + (["vae_val_split_failed"] if val_failed else [])),
                **{k: bool(v) for k, v in checks.items()},
            }
        )
    feas_rows.append({"fold": "ALL", "status": "PASS" if all_ok else "FAIL", "reason": "PASS" if all_ok else ";".join(sorted(set(reasons)))})
    return pd.DataFrame(rows), pd.DataFrame(feas_rows)


def score_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "N": int(len(y)),
        "TN": int(((y == 0) & (pred == 0)).sum()),
        "FP": int(((y == 0) & (pred == 1)).sum()),
        "FN": int(((y == 1) & (pred == 0)).sum()),
        "TP": int(((y == 1) & (pred == 1)).sum()),
        "BA": float(balanced_accuracy_score(y, pred)) if len(y) else np.nan,
        "Sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)) if len(y) else np.nan,
        "Specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)) if len(y) else np.nan,
        "F1": float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "Brier": float(brier_score_loss(y, score)) if len(y) else np.nan,
    }
    if len(np.unique(y)) == 2:
        out["AUC"] = float(roc_auc_score(y, score))
        out["PR_AUC"] = float(average_precision_score(y, score))
    else:
        out["AUC"] = np.nan
        out["PR_AUC"] = np.nan
    return out


def locked_score_only_m0(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    sub = df[
        df["retained_M0_confirmed_nondefault_only"]
        & df["_diagnosis_norm"].isin(["CN", "AD"])
        & df["y_score_final"].notna()
        & df["y_pred"].notna()
    ].copy()
    y = sub["_diagnosis_norm"].map({"CN": 0, "AD": 1}).to_numpy(dtype=int)
    score = pd.to_numeric(sub["y_score_final"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(sub["y_pred"], errors="coerce").to_numpy(dtype=int)
    row = {"comparison": "locked_promoted_score_only_on_M0_retained_subjects", **score_metrics(y, score, pred)}
    cn = sub[sub["_diagnosis_norm"].eq("CN")]
    phil_cn = cn[cn["_manufacturer_norm"].eq("Philips")]
    den = len(phil_cn)
    fp = int((pd.to_numeric(phil_cn["y_pred"], errors="coerce") == 1).sum())
    row["Philips_CN_N"] = den
    row["Philips_CN_FP"] = fp
    row["Philips_CN_FPR"] = fp / den if den else np.nan
    row["note"] = "Locked promoted OOF scores recomputed on M0 retained subjects only; not a new model."
    return pd.DataFrame([row]), row["Philips_CN_FPR"]


def create_derived_metadata(df: pd.DataFrame, ref_cfg: dict[str, Any]) -> pd.DataFrame:
    metadata_path = Path(ref_cfg["paths"]["metadata_path"])
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path
    meta = pd.read_csv(metadata_path)
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    excluded = set(EXPECTED_M0_SUBJECTS)
    reason_map = dict(zip(df["SubjectID"].astype(str), df["M0_exclusion_reason"].astype(str)))
    meta["sliceorder_m0_excluded"] = meta["SubjectID"].isin(excluded)
    meta["sliceorder_m0_exclusion_reason"] = meta["SubjectID"].map(reason_map).where(meta["sliceorder_m0_excluded"], "")
    meta["sliceorder_m0_vae_eligible"] = np.where(meta["sliceorder_m0_excluded"], "", "eligible")
    meta["ResearchGroup_Mapped_original"] = meta["ResearchGroup_Mapped"]
    meta.loc[meta["sliceorder_m0_excluded"], "ResearchGroup_Mapped"] = "EXCLUDED_SLICEORDER_M0"
    if "training_ready" in meta.columns:
        meta.loc[meta["sliceorder_m0_excluded"], "training_ready"] = False
    if "exclude_from_supervised" in meta.columns:
        meta.loc[meta["sliceorder_m0_excluded"], "exclude_from_supervised"] = True
    else:
        meta["exclude_from_supervised"] = meta["sliceorder_m0_excluded"]
    if "supervised_exclusion_reason" not in meta.columns:
        meta["supervised_exclusion_reason"] = ""
    meta["supervised_exclusion_reason"] = meta["supervised_exclusion_reason"].fillna("").astype(str)
    meta.loc[meta["sliceorder_m0_excluded"], "supervised_exclusion_reason"] = (
        meta.loc[meta["sliceorder_m0_excluded"], "supervised_exclusion_reason"]
        + ";sliceorderM0_confirmed_nondefault_only_sensitivity_exclusion"
    ).str.strip(";")
    actual = sorted(meta.loc[meta["sliceorder_m0_excluded"], "SubjectID"].astype(str).tolist())
    if actual != sorted(EXPECTED_M0_SUBJECTS):
        raise RuntimeError(f"Derived metadata exclusion mismatch: {actual}")
    return meta


def create_config(ref_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(ref_cfg))
    cfg["run_name"] = RUN_NAME
    cfg["description"] = (
        "Exploratory M0 QC sensitivity: exact promoted [1,0,2] latent384 beta3.75 T80 p560 FULL 5x5 "
        "configuration, using a derived metadata CSV that excludes only five confirmed Philips Site31 "
        "reverse/non-default slice-order subjects from all relevant VAE and StageB pools."
    )
    cfg["paths"]["metadata_path"] = str(DERIVED_METADATA.relative_to(PROJECT_ROOT))
    cfg["paths"]["output_dir"] = f"results/revision_bspc_2026/{RUN_NAME}"
    cfg["paths"]["big_disk_output_dir"] = f"/media/diego/Datos/vae_AD_results/revision_bspc_2026/{RUN_NAME}"
    cfg["paths"]["split_preview_csv"] = f"results/revision_bspc_2026/{RUN_NAME}_split_preview.csv"
    cfg["paths"]["split_preview_summary_csv"] = f"results/revision_bspc_2026/{RUN_NAME}_split_preview_summary.csv"
    cfg["slice_order_M0_confirmed_nondefault_only_sensitivity"] = {
        "mask": "M0_confirmed_nondefault_only",
        "excluded_subjects": EXPECTED_M0_SUBJECTS,
        "derived_metadata": str(DERIVED_METADATA.relative_to(PROJECT_ROOT)),
        "exclusion_manifest": str(EXCLUSION_MANIFEST.relative_to(PROJECT_ROOT)),
        "exploratory_qc_sensitivity": True,
        "not_primary_replacement_without_independent_validation": True,
    }
    return cfg


def launcher_source() -> str:
    return f'''#!/usr/bin/env python3
"""Guarded launcher for {RUN_NAME}.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training and an existing DRY_RUN_PASS marker from the preflight
package. This launcher uses a derived metadata CSV and does not modify source
metadata, tensors, promoted artifacts, predictions, thresholds, or fold files.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
REFERENCE_CONFIG = PROJECT_ROOT / {str(REFERENCE_CONFIG.relative_to(PROJECT_ROOT))!r}
CONFIG = PROJECT_ROOT / {str(GENERATED_CONFIG.relative_to(PROJECT_ROOT))!r}
DRY_RUN_PASS = PROJECT_ROOT / {str(DRY_RUN_PASS.relative_to(PROJECT_ROOT))!r}
RUN_NAME = {RUN_NAME!r}
STALE_NAMES = {{"run_config.json", "classifier_only_readout", "latent_cache"}}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def append_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{{name}}")
        return
    cmd.append(f"--{{name}}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def validate_config(ref: dict[str, Any], cfg: dict[str, Any]) -> None:
    if cfg.get("run_name") != RUN_NAME:
        raise RuntimeError(f"Unexpected run_name: {{cfg.get('run_name')}}")
    ref_params = dict(ref["parameters"])
    cfg_params = dict(cfg["parameters"])
    diffs = {{k: (ref_params.get(k), cfg_params.get(k)) for k in sorted(set(ref_params) | set(cfg_params)) if ref_params.get(k) != cfg_params.get(k)}}
    if diffs:
        raise RuntimeError(f"Scientific parameter diffs are not allowed for M0 sensitivity: {{diffs}}")
    for key in ["global_tensor_path", "training_script"]:
        if ref["paths"].get(key) != cfg["paths"].get(key):
            raise RuntimeError(f"Unexpected path diff for {{key}}")
    metadata_path = str(cfg["paths"].get("metadata_path", ""))
    if "M0_derived_metadata.csv" not in metadata_path:
        raise RuntimeError("metadata_path must point to M0_derived_metadata.csv.")
    if RUN_NAME not in str(cfg["paths"].get("output_dir", "")):
        raise RuntimeError("output_dir must contain the M0 run name.")
    if cfg.get("selected_channel_names") != ref.get("selected_channel_names"):
        raise RuntimeError("selected_channel_names changed unexpectedly.")
    if cfg["parameters"].get("channels_to_use") != [1, 0, 2]:
        raise RuntimeError("channels_to_use must remain [1,0,2].")


def stage_a_command(cfg: dict[str, Any], dry_run: bool) -> list[str]:
    python_exe = cfg.get("python_executable") or sys.executable
    cmd = [
        python_exe,
        str(resolve(cfg["paths"]["training_script"])),
        "--global_tensor_path", str(resolve(cfg["paths"]["global_tensor_path"])),
        "--metadata_path", str(resolve(cfg["paths"]["metadata_path"])),
        "--output_dir", str(resolve(cfg["paths"]["output_dir"])),
    ]
    for key, value in cfg["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend([
        "--vae_required_metadata_cols",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "sliceorder_m0_vae_eligible",
    ])
    cmd.append("--vae_abort_if_val_split_fails")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def stale_markers(outdir: Path) -> list[str]:
    if not outdir.exists():
        return []
    return sorted(p.name for p in outdir.iterdir() if p.name in STALE_NAMES or any(p.name.startswith(x) for x in STALE_PREFIXES))


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; no training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real training.")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight.")
    if args.confirm_training and not DRY_RUN_PASS.exists():
        raise SystemExit(f"Refusing real training: dry-run PASS marker is missing: {{DRY_RUN_PASS}}")
    ref = load_json(REFERENCE_CONFIG)
    cfg = load_json(CONFIG)
    validate_config(ref, cfg)
    outdir = resolve(cfg["paths"]["output_dir"])
    stale = stale_markers(outdir)
    if stale:
        raise SystemExit(f"Refusing to continue because stale output markers exist in {{outdir}}: {{stale}}")
    cmd = stage_a_command(cfg, dry_run=args.dry_run)
    print(shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def training_feasibility_text(counts_fold: pd.DataFrame, feasibility: pd.DataFrame) -> str:
    all_row = feasibility[feasibility["fold"].astype(str).eq("ALL")]
    status = all_row["status"].iloc[0] if not all_row.empty else "NOT_EVALUATED"
    reason = all_row["reason"].iloc[0] if not all_row.empty else ""
    return f"""# Training Feasibility M0

Status: `{status}`

Reason: `{reason}`

The M0 mask excludes exactly five confirmed Philips Site31
`reverse_even_odd_48` subjects and is fold-feasible if all fold rows are PASS.
The retained-count table records StageB CN/AD counts and VAE CN/AD/MCI counts
by outer fold.

This remains an exploratory QC sensitivity. It is narrower than M1 and does
not remove the Site18/Site301 missing-slice-order Philips CN subjects.
"""


def executive_summary(
    df: pd.DataFrame,
    score_df: pd.DataFrame,
    feasibility: pd.DataFrame,
    dry_returncode: int,
) -> str:
    ret = df[df["retained_M0_confirmed_nondefault_only"]]
    ex = df[~df["retained_M0_confirmed_nondefault_only"]]
    all_row = feasibility[feasibility["fold"].astype(str).eq("ALL")]
    status = all_row["status"].iloc[0] if not all_row.empty else "NOT_EVALUATED"
    score = score_df.iloc[0].to_dict() if not score_df.empty else {}
    return f"""# Executive Summary

This package preflights a guarded M0 confirmed-nondefault slice-order
sensitivity branch. No training was launched.

M0 excludes exactly `{len(ex)}` subjects:

`{', '.join(ex['SubjectID'].astype(str).tolist())}`

Retained rows: `{len(ret)}` of `{len(df)}` full database rows.

Retained diagnosis counts:

`{ret['_diagnosis_norm'].value_counts().to_dict()}`

Retained manufacturer counts:

`{ret['_manufacturer_norm'].value_counts().to_dict()}`

Fold feasibility: `{status}`.

Locked promoted score-only M0 retained-set metrics, not a new model:

- AUC: `{score.get('AUC', np.nan):.6f}`
- PR-AUC: `{score.get('PR_AUC', np.nan):.6f}`
- BA: `{score.get('BA', np.nan):.6f}`
- Sensitivity: `{score.get('Sensitivity', np.nan):.6f}`
- Specificity: `{score.get('Specificity', np.nan):.6f}`
- F1: `{score.get('F1', np.nan):.6f}`
- Brier: `{score.get('Brier', np.nan):.6f}`
- Philips CN FPR: `{score.get('Philips_CN_FPR', np.nan):.6f}`

Launcher dry-run return code: `{dry_returncode}`.
Dry-run PASS marker: `{'created' if dry_returncode == 0 and DRY_RUN_PASS.exists() else 'not_created'}`.
"""


def final_recommendation_text(df: pd.DataFrame, feasibility: pd.DataFrame, dry_returncode: int) -> str:
    all_row = feasibility[feasibility["fold"].astype(str).eq("ALL")]
    status = all_row["status"].iloc[0] if not all_row.empty else "NOT_EVALUATED"
    ex = df[~df["retained_M0_confirmed_nondefault_only"]]
    return f"""# Final Recommendation

M0 is scientifically narrower and more defensible than M1 because it excludes
only the five Philips Site31 subjects with confirmed reverse/non-default slice
order:

`{', '.join(ex['SubjectID'].astype(str).tolist())}`

Fold feasibility status: `{status}`.
Launcher dry-run return code: `{dry_returncode}`.

Recommendation: M0 is ready as an exploratory QC sensitivity launcher if and
only if `DRY_RUN_PASS.txt` exists and the user explicitly launches
`guarded_launcher.py --confirm-training`.

The promoted model remains locked. Any future M0 retrain must be reported as
a slice-order QC sensitivity, not as an automatic replacement of the promoted
primary model. Score-only retained-set metrics are mechanical retained-subset
metrics using promoted predictions and are not a new model.
"""


def launch_commands_text(py: str) -> str:
    dry_cmd = [py, str(GUARDED_LAUNCHER), "--dry-run"]
    train_cmd = [py, str(GUARDED_LAUNCHER), "--confirm-training"]
    return "\n".join(
        [
            "# M0 confirmed-nondefault slice-order sensitivity commands",
            "",
            "# Dry-run validation:",
            shlex.join(dry_cmd),
            "",
            "# Real training remains guarded and requires DRY_RUN_PASS.txt plus explicit user approval:",
            shlex.join(train_cmd),
            "",
            f"# Expected output_dir: results/revision_bspc_2026/{RUN_NAME}",
            f"# Derived metadata: {DERIVED_METADATA.relative_to(PROJECT_ROOT)}",
            f"# Exclusion manifest: {EXCLUSION_MANIFEST.relative_to(PROJECT_ROOT)}",
        ]
    ) + "\n"


def create_outputs(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_cfg = read_json(REFERENCE_CONFIG)

    manifest = exclusion_manifest(df)
    counts_group = retained_counts_by_group(df)
    counts_fold, feasibility = retained_counts_by_fold(df)
    score_df, _ = locked_score_only_m0(df)
    derived = create_derived_metadata(df, ref_cfg)
    cfg = create_config(ref_cfg)

    write_df("M0_exclusion_manifest", manifest, max_rows=50)
    derived.to_csv(DERIVED_METADATA, index=False)
    write_df("retained_counts_by_group", counts_group, max_rows=240)
    write_df("retained_counts_by_fold", counts_fold, max_rows=120)
    write_df("locked_model_score_only_M0", score_df, max_rows=20)
    (OUT / "training_feasibility_M0.md").write_text(training_feasibility_text(counts_fold, feasibility), encoding="utf-8")

    write_json(GENERATED_CONFIG, cfg)
    GUARDED_LAUNCHER.write_text(launcher_source(), encoding="utf-8")
    GUARDED_LAUNCHER.chmod(0o755)

    py = cfg.get("python_executable") or sys.executable
    compile_proc = subprocess.run(
        [py, "-m", "py_compile", str(GUARDED_LAUNCHER)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if compile_proc.returncode != 0:
        raise RuntimeError(f"Generated launcher py_compile failed:\n{compile_proc.stderr}")

    dry_cmd = [py, str(GUARDED_LAUNCHER), "--dry-run"]
    dry_proc = subprocess.run(dry_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (OUT / "launcher_dryrun_stdout.txt").write_text(dry_proc.stdout, encoding="utf-8")
    (OUT / "launcher_dryrun_stderr.txt").write_text(dry_proc.stderr, encoding="utf-8")
    if dry_proc.returncode == 0:
        DRY_RUN_PASS.write_text(
            f"PASS created_utc={datetime.now(timezone.utc).isoformat()}\ncommand={shlex.join(dry_cmd)}\n",
            encoding="utf-8",
        )

    (OUT / "launch_commands.txt").write_text(launch_commands_text(py), encoding="utf-8")
    (OUT / "00_EXECUTIVE_SUMMARY.md").write_text(
        executive_summary(df, score_df, feasibility, dry_proc.returncode),
        encoding="utf-8",
    )
    (OUT / "final_recommendation.md").write_text(
        final_recommendation_text(df, feasibility, dry_proc.returncode),
        encoding="utf-8",
    )

    write_json(
        OUT / "command_log.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "argv": sys.argv,
            "guardrails": {
                "preflight_only": True,
                "model_training": False,
                "original_metadata_edits": False,
                "tensor_edits": False,
                "prediction_edits": False,
                "threshold_refitting": False,
                "promoted_artifact_edits": False,
                "fold_file_edits": False,
            },
            "inputs": {
                "full_database": str(FULL_DB),
                "reference_config": str(REFERENCE_CONFIG),
                "previous_preflight": str(PREVIOUS_PREFLIGHT),
            },
            "outputs": str(OUT.relative_to(PROJECT_ROOT)),
            "generated_launcher": str(GUARDED_LAUNCHER.relative_to(PROJECT_ROOT)),
            "generated_config": str(GENERATED_CONFIG.relative_to(PROJECT_ROOT)),
            "derived_metadata": str(DERIVED_METADATA.relative_to(PROJECT_ROOT)),
            "dry_run_command": shlex.join(dry_cmd),
            "dry_run_returncode": int(dry_proc.returncode),
            "dry_run_pass_file": str(DRY_RUN_PASS.relative_to(PROJECT_ROOT)) if DRY_RUN_PASS.exists() else None,
            "m0_excluded_subjects": EXPECTED_M0_SUBJECTS,
        },
    )
    if dry_proc.returncode != 0:
        raise RuntimeError(f"Launcher dry-run failed with return code {dry_proc.returncode}. See launcher_dryrun_stderr.txt")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Explicit preflight flag; this script never trains.")
    _ = parser.parse_args()
    if not FULL_DB.exists():
        raise FileNotFoundError(FULL_DB)
    if not REFERENCE_CONFIG.exists():
        raise FileNotFoundError(REFERENCE_CONFIG)
    df = pd.read_csv(FULL_DB)
    required = [
        "SubjectID",
        "diagnosis_group",
        "Manufacturer_final",
        "Site3_final",
        "slice_order_class",
        "matches_dparsf_default",
        "y_score_final",
        "y_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Full database is missing required columns: {missing}")
    df = add_m0_flags(df)
    create_outputs(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
