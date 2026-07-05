#!/usr/bin/env python3
"""Read-only subject-pool audit for the recover035_scheduler90_sync FULL 5x5 run.

Verifies:
  - 035_S_6927 is present in patched metadata with correct fields (Age, Sex, Manufacturer, DX, tensor_idx).
  - 128_S_2002 is absent from classifier pool and not training-ready.
  - VAE pool n=647 (CN=300, MCI=250, AD=97).
  - Classifier pool n=397 (CN=300, AD=97).
  - Diagnosis × Manufacturer crosstab.

This is identical in scope to audit_adni_recover035_subject_pool.py but targets the
recover035_scheduler90_sync run output directory.

Does not train, modify tensor, metadata, ledger, configs, or model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_METADATA = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
)
DEFAULT_OUTPUT = RESULTS / "recover035_scheduler90_sync_full5x5_subject_pool_audit"

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
EXPECTED_RESCUE_AGE = 59.6
EXPECTED_RESCUE_SEX = "F"
EXPECTED_RESCUE_MANUFACTURER = "SIEMENS"
EXPECTED_RESCUE_DX = "AD"
EXPECTED_RESCUE_TENSOR_IDX = 256

EXPECTED_VAE_N = 647
EXPECTED_VAE_CN = 300
EXPECTED_VAE_MCI = 250
EXPECTED_VAE_AD = 97
EXPECTED_CLF_N = 397
EXPECTED_CLF_CN = 300
EXPECTED_CLF_AD = 97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    return df.to_markdown(index=False) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tensor_idx" not in df.columns and "tensor_index" in df.columns:
        df = df.rename(columns={"tensor_index": "tensor_idx"})
    df["Manufacturer"] = df["Manufacturer"].map(normalize_manufacturer)
    return df


def verify_subject_035(df: pd.DataFrame) -> Dict[str, Any]:
    rows = df[df["SubjectID"].astype(str) == RECOVER_SUBJECT]
    result: Dict[str, Any] = {
        "SubjectID": RECOVER_SUBJECT,
        "present_in_metadata": not rows.empty,
    }
    checks: Dict[str, bool] = {}
    if rows.empty:
        checks["present"] = False
        result["checks"] = checks
        result["all_checks_pass"] = False
        return result
    r = rows.iloc[0]
    age = float(r.get("Age", float("nan")))
    sex = str(r.get("Sex", ""))
    mfr = str(r.get("Manufacturer", ""))
    dx = str(r.get("ResearchGroup_Mapped", ""))
    tidx = int(r.get("tensor_idx", -1))
    tr = str(r.get("training_ready", "")).lower()
    excl = str(r.get("exclude_from_supervised", "")).lower()
    checks["age_59_6"] = abs(age - EXPECTED_RESCUE_AGE) <= 0.01
    checks["sex_F"] = sex == EXPECTED_RESCUE_SEX
    checks["manufacturer_SIEMENS"] = mfr == EXPECTED_RESCUE_MANUFACTURER
    checks["dx_AD"] = dx == EXPECTED_RESCUE_DX
    checks["tensor_idx_256"] = tidx == EXPECTED_RESCUE_TENSOR_IDX
    checks["training_ready_True"] = tr in {"true", "1"}
    checks["exclude_from_supervised_False"] = excl in {"false", "0"}
    result.update({
        "Age": age, "Sex": sex, "Manufacturer": mfr,
        "ResearchGroup_Mapped": dx, "tensor_idx": tidx,
        "training_ready": r.get("training_ready"),
        "exclude_from_supervised": r.get("exclude_from_supervised"),
        "metadata_source": str(r.get("metadata_source", "")),
    })
    result["checks"] = checks
    result["all_checks_pass"] = all(checks.values())
    return result


def verify_subject_128(df: pd.DataFrame) -> Dict[str, Any]:
    rows = df[df["SubjectID"].astype(str) == EXCLUDED_SUBJECT]
    clf_rows = df[
        df["SubjectID"].astype(str).eq(EXCLUDED_SUBJECT)
        & df["ResearchGroup_Mapped"].astype(str).isin(["CN", "AD"])
    ]
    tr_col = df.columns.intersection(["training_ready"])
    in_tr = False
    if not rows.empty and tr_col.size:
        in_tr = bool(str(rows.iloc[0].get("training_ready", "")).lower() in {"true", "1"})
    return {
        "SubjectID": EXCLUDED_SUBJECT,
        "present_in_metadata": not rows.empty,
        "training_ready": in_tr,
        "in_classifier_pool": not clf_rows.empty,
        "correctly_excluded": not in_tr and clf_rows.empty,
    }


def build_pool_summary(df: pd.DataFrame) -> pd.DataFrame:
    rg = df["ResearchGroup_Mapped"].astype(str)
    cn = int((rg == "CN").sum())
    mci = int((rg == "MCI").sum())
    ad = int((rg == "AD").sum())
    total_vae = len(df)
    total_clf = cn + ad
    rows = [
        {
            "pool": "VAE pool (all training_ready)",
            "CN": cn, "MCI": mci, "AD": ad, "total": total_vae,
            "expected": EXPECTED_VAE_N,
            "ok": total_vae == EXPECTED_VAE_N and cn == EXPECTED_VAE_CN and mci == EXPECTED_VAE_MCI and ad == EXPECTED_VAE_AD,
        },
        {
            "pool": "Classifier pool (CN+AD only)",
            "CN": cn, "MCI": 0, "AD": ad, "total": total_clf,
            "expected": EXPECTED_CLF_N,
            "ok": total_clf == EXPECTED_CLF_N and cn == EXPECTED_CLF_CN and ad == EXPECTED_CLF_AD,
        },
    ]
    return pd.DataFrame(rows)


def build_dx_mfr_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab(df["ResearchGroup_Mapped"], df["Manufacturer"])
    tab.index.name = "Diagnosis"
    return tab.reset_index()


def build_035_verification_table(v035: Dict[str, Any]) -> pd.DataFrame:
    checks = v035.get("checks", {})
    rows: List[Dict[str, Any]] = []
    field_map = {
        "age_59_6": ("Age", EXPECTED_RESCUE_AGE, v035.get("Age", "")),
        "sex_F": ("Sex", EXPECTED_RESCUE_SEX, v035.get("Sex", "")),
        "manufacturer_SIEMENS": ("Manufacturer", EXPECTED_RESCUE_MANUFACTURER, v035.get("Manufacturer", "")),
        "dx_AD": ("ResearchGroup_Mapped", EXPECTED_RESCUE_DX, v035.get("ResearchGroup_Mapped", "")),
        "tensor_idx_256": ("tensor_idx", EXPECTED_RESCUE_TENSOR_IDX, v035.get("tensor_idx", "")),
        "training_ready_True": ("training_ready", True, v035.get("training_ready", "")),
        "exclude_from_supervised_False": ("exclude_from_supervised", False, v035.get("exclude_from_supervised", "")),
    }
    for check_key, (field, expected, actual) in field_map.items():
        rows.append({
            "field": field,
            "expected": str(expected),
            "actual": str(actual),
            "pass": checks.get(check_key, False),
        })
    return pd.DataFrame(rows)


def write_report(
    outdir: Path,
    pool: pd.DataFrame,
    v035: Dict[str, Any],
    v128: Dict[str, Any],
    crosstab: pd.DataFrame,
) -> None:
    pool_ok = bool(pool["ok"].all())
    sub035_ok = bool(v035.get("all_checks_pass", False))
    sub128_ok = bool(v128.get("correctly_excluded", False))
    overall = pool_ok and sub035_ok and sub128_ok

    lines = [
        "# recover035_scheduler90_sync Subject Pool Audit",
        "",
        "Read-only. No original metadata, tensor, ledger, configs, or model outputs modified.",
        "Metadata: patched_metadata_candidate.csv (includes 035_S_6927 as AD, excludes 128_S_2002).",
        "",
        f"## Overall result: {'PASS' if overall else 'FAIL'}",
        "",
        "## Pool sizes",
        "",
        md_table(pool),
        "",
        f"## {RECOVER_SUBJECT} verification: {'PASS' if sub035_ok else 'FAIL'}",
        "",
        md_table(build_035_verification_table(v035)),
        f"metadata_source: `{v035.get('metadata_source', '')}`",
        "",
        f"## {EXCLUDED_SUBJECT} exclusion: {'PASS' if sub128_ok else 'FAIL'}",
        "",
        f"- Present in metadata: {v128['present_in_metadata']}",
        f"- training_ready: {v128['training_ready']}",
        f"- In classifier pool (should be False): {v128['in_classifier_pool']}",
        f"- Correctly excluded: {v128['correctly_excluded']}",
        "",
        "## Diagnosis × Manufacturer crosstab",
        "",
        md_table(crosstab),
    ]
    (outdir / "pool_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    meta_path = resolve(args.metadata)
    outdir = resolve(args.output_dir)
    if not meta_path.exists():
        print(f"ERROR: metadata not found: {meta_path}", flush=True)
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    df = load_metadata(meta_path)
    pool = build_pool_summary(df)
    v035 = verify_subject_035(df)
    v128 = verify_subject_128(df)
    crosstab = build_dx_mfr_crosstab(df)
    ver_table = build_035_verification_table(v035)

    write_table(outdir, "pool_summary", pool)
    write_table(outdir, "subject_035_verification", ver_table)
    v128_df = pd.DataFrame([v128])
    write_table(outdir, "subject_128_exclusion", v128_df)
    write_table(outdir, "diagnosis_manufacturer_crosstab", crosstab)
    write_report(outdir, pool, v035, v128, crosstab)

    overall = (
        bool(pool["ok"].all())
        and bool(v035.get("all_checks_pass", False))
        and bool(v128.get("correctly_excluded", False))
    )

    print(f"Pool audit: {'PASS' if overall else 'FAIL'}")
    print(pool.to_string(index=False))
    print(f"035_S_6927 all checks: {'PASS' if v035.get('all_checks_pass') else 'FAIL'}")
    print(f"128_S_2002 correctly excluded: {v128['correctly_excluded']}")

    command_log = {
        "created_utc": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "metadata": str(meta_path),
        "output_dir": str(outdir),
        "overall_pass": overall,
        "pool_ok": bool(pool["ok"].all()),
        "subject_035_ok": bool(v035.get("all_checks_pass", False)),
        "subject_128_excluded": bool(v128.get("correctly_excluded", False)),
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
