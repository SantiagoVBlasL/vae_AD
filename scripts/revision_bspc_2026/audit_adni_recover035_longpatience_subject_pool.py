#!/usr/bin/env python3
"""Read-only subject-pool audit for the recover035_longpatience_T80_h10000_p560 FULL 5x5 run.

Verifies:
  - 035_S_6927 is present in patched metadata with correct fields (Age, Sex, Manufacturer, DX, tensor_idx).
  - 128_S_2002 is absent from classifier pool and not training-ready.
  - VAE pool n=647 (CN=300, MCI=250, AD=97).
  - Classifier pool n=397 (CN=300, AD=97).
  - Diagnosis × Manufacturer crosstab.

Identical pool as recover035_full5x5 and recover035_scheduler90_sync_full5x5 (same metadata).

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
DEFAULT_OUTPUT = RESULTS / "recover035_longpatience_T80_h10000_p560_subject_pool_audit"

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
        "present": False,
        "age_ok": None, "sex_ok": None, "manufacturer_ok": None,
        "dx_ok": None, "tensor_idx_ok": None, "errors": [],
    }
    if rows.empty:
        result["errors"].append(f"{RECOVER_SUBJECT} not found in metadata")
        return result
    result["present"] = True
    r = rows.iloc[0]
    age = float(r.get("Age", float("nan")))
    sex = str(r.get("Sex", ""))
    mfr = normalize_manufacturer(r.get("Manufacturer", ""))
    dx = str(r.get("ResearchGroup_Mapped", ""))
    try:
        tidx = int(r.get("tensor_idx", -1))
    except (ValueError, TypeError):
        tidx = -1
    result["age"] = age
    result["sex"] = sex
    result["manufacturer"] = mfr
    result["dx"] = dx
    result["tensor_idx"] = tidx
    result["age_ok"] = bool(abs(age - EXPECTED_RESCUE_AGE) < 0.01)
    result["sex_ok"] = bool(sex == EXPECTED_RESCUE_SEX)
    result["manufacturer_ok"] = bool(mfr == EXPECTED_RESCUE_MANUFACTURER)
    result["dx_ok"] = bool(dx == EXPECTED_RESCUE_DX)
    result["tensor_idx_ok"] = bool(tidx == EXPECTED_RESCUE_TENSOR_IDX)
    for key, ok, expected in [
        ("age", result["age_ok"], EXPECTED_RESCUE_AGE),
        ("sex", result["sex_ok"], EXPECTED_RESCUE_SEX),
        ("manufacturer", result["manufacturer_ok"], EXPECTED_RESCUE_MANUFACTURER),
        ("dx", result["dx_ok"], EXPECTED_RESCUE_DX),
        ("tensor_idx", result["tensor_idx_ok"], EXPECTED_RESCUE_TENSOR_IDX),
    ]:
        if not ok:
            result["errors"].append(f"{RECOVER_SUBJECT} {key}={result[key]!r}, expected {expected!r}")
    return result


def verify_subject_128(df: pd.DataFrame) -> Dict[str, Any]:
    clf_pool = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    in_clf = bool((clf_pool["SubjectID"].astype(str) == EXCLUDED_SUBJECT).any())
    in_full = bool((df["SubjectID"].astype(str) == EXCLUDED_SUBJECT).any())
    result: Dict[str, Any] = {
        "SubjectID": EXCLUDED_SUBJECT,
        "in_full_metadata": in_full,
        "in_classifier_pool": in_clf,
        "errors": [],
    }
    if in_clf:
        result["errors"].append(f"{EXCLUDED_SUBJECT} appears in classifier pool (must be excluded)")
    return result


def pool_size_check(df: pd.DataFrame) -> Dict[str, Any]:
    rg = df["ResearchGroup_Mapped"]
    cn = int((rg == "CN").sum())
    mci = int((rg == "MCI").sum())
    ad = int((rg == "AD").sum())
    total_vae = len(df)
    total_clf = cn + ad
    errors: List[str] = []
    if total_vae != EXPECTED_VAE_N:
        errors.append(f"VAE pool n={total_vae}, expected {EXPECTED_VAE_N}")
    if cn != EXPECTED_VAE_CN:
        errors.append(f"VAE CN={cn}, expected {EXPECTED_VAE_CN}")
    if mci != EXPECTED_VAE_MCI:
        errors.append(f"VAE MCI={mci}, expected {EXPECTED_VAE_MCI}")
    if ad != EXPECTED_VAE_AD:
        errors.append(f"VAE AD={ad}, expected {EXPECTED_VAE_AD}")
    if total_clf != EXPECTED_CLF_N:
        errors.append(f"CLF pool n={total_clf}, expected {EXPECTED_CLF_N}")
    return {
        "vae_n": total_vae, "vae_cn": cn, "vae_mci": mci, "vae_ad": ad,
        "clf_n": total_clf, "clf_cn": cn, "clf_ad": ad,
        "errors": errors,
    }


def crosstab_dx_mfr(df: pd.DataFrame) -> pd.DataFrame:
    clf = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf["Manufacturer"] = clf["Manufacturer"].map(normalize_manufacturer)
    ct = pd.crosstab(clf["ResearchGroup_Mapped"], clf["Manufacturer"], margins=True)
    return ct.reset_index()


def age_sex_summary(df: pd.DataFrame) -> pd.DataFrame:
    clf = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    rows: List[Dict[str, Any]] = []
    for dx, sub in clf.groupby("ResearchGroup_Mapped"):
        rows.append({
            "ResearchGroup_Mapped": dx,
            "n": int(len(sub)),
            "age_mean": float(sub["Age"].mean()),
            "age_std": float(sub["Age"].std()),
            "age_min": float(sub["Age"].min()),
            "age_max": float(sub["Age"].max()),
            "sex_F": int((sub["Sex"] == "F").sum()),
            "sex_M": int((sub["Sex"] == "M").sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    meta_path = resolve(args.metadata)
    outdir = resolve(args.output_dir)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    df = load_metadata(meta_path)
    r035 = verify_subject_035(df)
    r128 = verify_subject_128(df)
    pool = pool_size_check(df)

    all_errors = r035["errors"] + r128["errors"] + pool["errors"]

    print(f"Metadata path : {meta_path}")
    print(f"Pool size     : VAE n={pool['vae_n']} (CN={pool['vae_cn']}, MCI={pool['vae_mci']}, AD={pool['vae_ad']})")
    print(f"CLF pool      : n={pool['clf_n']} (CN={pool['clf_cn']}, AD={pool['clf_ad']})")
    print(f"035_S_6927    : present={r035['present']}, age_ok={r035['age_ok']}, sex_ok={r035['sex_ok']}, manufacturer_ok={r035['manufacturer_ok']}, dx_ok={r035['dx_ok']}, tensor_idx_ok={r035['tensor_idx_ok']}")
    print(f"128_S_2002    : in_full_metadata={r128['in_full_metadata']}, in_classifier_pool={r128['in_classifier_pool']}")
    if all_errors:
        for e in all_errors:
            print(f"  [ERROR] {e}")
        print(f"Subject-pool audit: FAIL ({len(all_errors)} error(s))")
    else:
        print("Subject-pool audit: PASS")

    if outdir.exists() and not args.overwrite:
        raise FileExistsError(f"{outdir} exists; pass --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {"check": "vae_pool_n", "expected": EXPECTED_VAE_N, "actual": pool["vae_n"], "pass": pool["vae_n"] == EXPECTED_VAE_N},
        {"check": "vae_cn", "expected": EXPECTED_VAE_CN, "actual": pool["vae_cn"], "pass": pool["vae_cn"] == EXPECTED_VAE_CN},
        {"check": "vae_mci", "expected": EXPECTED_VAE_MCI, "actual": pool["vae_mci"], "pass": pool["vae_mci"] == EXPECTED_VAE_MCI},
        {"check": "vae_ad", "expected": EXPECTED_VAE_AD, "actual": pool["vae_ad"], "pass": pool["vae_ad"] == EXPECTED_VAE_AD},
        {"check": "clf_pool_n", "expected": EXPECTED_CLF_N, "actual": pool["clf_n"], "pass": pool["clf_n"] == EXPECTED_CLF_N},
        {"check": "035_present", "expected": True, "actual": r035["present"], "pass": r035["present"]},
        {"check": "035_age", "expected": EXPECTED_RESCUE_AGE, "actual": r035.get("age"), "pass": r035.get("age_ok")},
        {"check": "035_sex", "expected": EXPECTED_RESCUE_SEX, "actual": r035.get("sex"), "pass": r035.get("sex_ok")},
        {"check": "035_manufacturer", "expected": EXPECTED_RESCUE_MANUFACTURER, "actual": r035.get("manufacturer"), "pass": r035.get("manufacturer_ok")},
        {"check": "035_dx", "expected": EXPECTED_RESCUE_DX, "actual": r035.get("dx"), "pass": r035.get("dx_ok")},
        {"check": "035_tensor_idx", "expected": EXPECTED_RESCUE_TENSOR_IDX, "actual": r035.get("tensor_idx"), "pass": r035.get("tensor_idx_ok")},
        {"check": "128_not_in_clf_pool", "expected": False, "actual": r128["in_classifier_pool"], "pass": not r128["in_classifier_pool"]},
    ]
    write_table(outdir, "subject_pool_checks", pd.DataFrame(summary_rows))
    write_table(outdir, "dx_manufacturer_crosstab", crosstab_dx_mfr(df))
    write_table(outdir, "age_sex_summary", age_sex_summary(df))

    cl = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "metadata_path": str(meta_path),
        "output_dir": str(outdir),
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
        "all_checks_pass": not all_errors,
        "n_errors": len(all_errors),
    }
    (outdir / "command_log.json").write_text(json.dumps(cl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
