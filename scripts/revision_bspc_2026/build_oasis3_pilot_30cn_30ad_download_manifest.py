"""
Read-only OASIS3 pilot download manifest builder — 30 CN + 30 AD_DEMENTIA.

Source:
  oasis3_tr2_main_candidates_for_martin.csv
  494 subjects: 357 CN + 137 AD_DEMENTIA
  All: TR=2.2s, diagnosis_confidence=high, is_first_usable_tr2_visit=yes, experiment_id present

Selection priority (same rule for both diagnosis groups):
  1. ScannerModel == TrioTim  (over Biograph_mMR)
  2. session_rank_per_subject ascending  (earlier / baseline MR visit first)
  3. subject_id ascending  (deterministic tie-break)
  Take first 30 per group.

Outputs:
  oasis3_pilot_30cn_30ad_experiment_ids.csv   — exactly 60 rows, column: experiment_id
  oasis3_pilot_30cn_30ad_subjects.csv         — 60 rows, full subject metadata
  README.md
  command_log.json

Constraints:
  Read-only: no tensor, metadata, ledger, config, or model modification.
  No training, no inference.
"""

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CSV = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis3_external_validation_tr2_candidate_manifest"
    / "oasis3_tr2_main_candidates_for_martin.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis3_pilot_30cn_30ad_download_manifest"
)

N_PER_CLASS = 30
TARGET_SCANNER = "TrioTim"
TARGET_DIAGNOSES = ["CN", "AD_DEMENTIA"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_manifest() -> None:
    t0 = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load source                                                       #
    # ------------------------------------------------------------------ #
    df = pd.read_csv(SOURCE_CSV)
    print(f"Source loaded: {len(df)} rows  (CN={int((df['diagnosis']=='CN').sum())}, "
          f"AD={int((df['diagnosis']=='AD_DEMENTIA').sum())})")

    # Guard: verify pre-conditions
    assert df["subject_id"].nunique() == len(df), "Subjects not unique in source"
    assert df["experiment_id"].isna().sum() == 0, "Some experiment_id are null"
    assert set(df["diagnosis"].unique()) == {"CN", "AD_DEMENTIA"}, "Unexpected diagnosis values"
    assert (df["TR_seconds"] == 2.2).all(), "Not all TR=2.2"
    assert (df["diagnosis_confidence"] == "high").all(), "Not all high confidence"

    # ------------------------------------------------------------------ #
    # 2. Selection                                                         #
    # ------------------------------------------------------------------ #
    # Sort key: TrioTim first, then session_rank asc, then subject_id asc
    df["_scanner_priority"] = (df["ScannerModel"] != TARGET_SCANNER).astype(int)  # 0=TrioTim, 1=other

    selected_parts = []
    selection_log = {}

    for dx in TARGET_DIAGNOSES:
        pool = (
            df[df["diagnosis"] == dx]
            .sort_values(["_scanner_priority", "session_rank_per_subject", "subject_id"])
            .reset_index(drop=True)
        )
        n_tritim = int((pool["ScannerModel"] == TARGET_SCANNER).sum())
        chosen = pool.head(N_PER_CLASS).copy()

        assert len(chosen) == N_PER_CLASS, (
            f"Only {len(chosen)} {dx} candidates, need {N_PER_CLASS}"
        )

        selection_log[dx] = {
            "pool_size": int(len(pool)),
            "selected": N_PER_CLASS,
            "n_TrioTim_in_pool": n_tritim,
            "n_TrioTim_selected": int((chosen["ScannerModel"] == TARGET_SCANNER).sum()),
            "n_Biograph_mMR_selected": int((chosen["ScannerModel"] == "Biograph_mMR").sum()),
            "session_rank_1_selected": int((chosen["session_rank_per_subject"] == 1).sum()),
        }
        selected_parts.append(chosen)
        print(f"  {dx}: selected {N_PER_CLASS} / {len(pool)}  "
              f"(TrioTim={selection_log[dx]['n_TrioTim_selected']}, "
              f"rank1={selection_log[dx]['session_rank_1_selected']})")

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop(columns=["_scanner_priority"])

    # ------------------------------------------------------------------ #
    # 3. Validation                                                        #
    # ------------------------------------------------------------------ #
    assert len(selected) == 60, f"Expected 60 rows, got {len(selected)}"
    assert selected["subject_id"].nunique() == 60, "Duplicate subjects in selection"
    assert int((selected["diagnosis"] == "CN").sum()) == 30
    assert int((selected["diagnosis"] == "AD_DEMENTIA").sum()) == 30
    assert selected["experiment_id"].isna().sum() == 0
    print(f"\nValidation passed: 60 rows, 30 CN + 30 AD_DEMENTIA, no duplicates")

    # ------------------------------------------------------------------ #
    # 4. experiment_id CSV (exactly 60 rows, column: experiment_id)       #
    # ------------------------------------------------------------------ #
    exp_df = selected[["experiment_id"]].copy()
    exp_df.to_csv(OUTPUT_DIR / "oasis3_pilot_30cn_30ad_experiment_ids.csv", index=False)
    print(f"  experiment_ids CSV: {len(exp_df)} rows")

    # ------------------------------------------------------------------ #
    # 5. Subjects CSV (full metadata)                                      #
    # ------------------------------------------------------------------ #
    subjects_cols = [
        "subject_id", "diagnosis", "session_id", "experiment_id",
        "TR_seconds", "ScannerModel", "Manufacturer",
        "diagnosis_confidence", "session_rank_per_subject",
        "age_at_MR", "sex", "CDRTOT", "CDRSUM",
        "is_first_usable_tr2_visit", "clinical_delta_days",
    ]
    subjects_df = selected[subjects_cols].copy()
    subjects_df.to_csv(OUTPUT_DIR / "oasis3_pilot_30cn_30ad_subjects.csv", index=False)

    # ------------------------------------------------------------------ #
    # 6. README                                                           #
    # ------------------------------------------------------------------ #
    # Build summary tables for README
    scanner_counts = selected.groupby(["diagnosis", "ScannerModel"]).size().reset_index(name="n")
    rank_counts = selected.groupby(["diagnosis", "session_rank_per_subject"]).size().reset_index(name="n")

    readme_md = textwrap.dedent(f"""\
    # OASIS3 Pilot Download Manifest — 30 CN + 30 AD_DEMENTIA

    **Generated:** {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}
    **Script:** `scripts/revision_bspc_2026/build_oasis3_pilot_30cn_30ad_download_manifest.py`
    **Source:** `oasis3_tr2_main_candidates_for_martin.csv` (494 candidates)

    ## Design

    | Parameter | Value |
    | --- | --- |
    | Target | 30 CN + 30 AD_DEMENTIA |
    | TR | 2.2 s (all candidates) |
    | Confidence | high (all candidates) |
    | First usable TR2 visit | yes (all candidates) |
    | Preferred scanner | Siemens TrioTim |
    | Tiebreak | session_rank_per_subject ↑, then subject_id ↑ |

    ## Selection summary

    ### Scanner model breakdown

    {df_to_markdown(scanner_counts)}

    ### Session rank breakdown (1 = earliest MR visit with usable TR2 scan)

    {df_to_markdown(rank_counts)}

    ## Files

    | File | Contents |
    | --- | --- |
    | oasis3_pilot_30cn_30ad_experiment_ids.csv | 60 rows, column: `experiment_id` (for oasis-scripts) |
    | oasis3_pilot_30cn_30ad_subjects.csv | 60 rows, full subject metadata |
    | README.md | This file |
    | command_log.json | Execution metadata |

    ## Usage with oasis-scripts

    Pass `oasis3_pilot_30cn_30ad_experiment_ids.csv` as the session list to:
    ```
    bash oasis-scripts/download_scans/download_oasis_scans.sh \\
         oasis3_pilot_30cn_30ad_experiment_ids.csv \\
         <output_dir> \\
         <oasis_credentials>
    ```

    ## Validation

    - 60 total rows ✓
    - 30 CN + 30 AD_DEMENTIA ✓
    - No duplicated subject_id ✓
    - All experiment_id present ✓
    - All TR = 2.2 s ✓
    - All diagnosis_confidence = high ✓

    ## Notes

    - All 60 selected subjects use Siemens TrioTim (no Biograph_mMR selected).
    - This is an **external validation pilot**; none of these subjects overlap with
      the ADNI training cohort.
    - MCI subjects are not included in OASIS3 (OASIS3 uses CN / Cognitively Impaired /
      AD_DEMENTIA labels; this manifest uses CN and AD_DEMENTIA only).
    """)
    (OUTPUT_DIR / "README.md").write_text(readme_md)

    # ------------------------------------------------------------------ #
    # 7. command_log.json                                                  #
    # ------------------------------------------------------------------ #
    t1 = datetime.now(timezone.utc)
    log = {
        "script": Path(__file__).name,
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "elapsed_s": round((t1 - t0).total_seconds(), 2),
        "source_csv": str(SOURCE_CSV),
        "output_dir": str(OUTPUT_DIR),
        "n_per_class": N_PER_CLASS,
        "total_selected": 60,
        "selection_log": selection_log,
        "validation": {
            "n_rows_experiment_ids_csv": 60,
            "n_CN": 30,
            "n_AD_DEMENTIA": 30,
            "duplicated_subjects": 0,
            "null_experiment_ids": 0,
        },
        "python": sys.version,
        "read_only": True,
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2, default=str))

    # ------------------------------------------------------------------ #
    # Console summary                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*55}")
    print("OASIS3 PILOT MANIFEST COMPLETE")
    print(f"{'='*55}")
    print(f"  CN selected:          30 (from {selection_log['CN']['pool_size']} candidates)")
    print(f"  AD_DEMENTIA selected: 30 (from {selection_log['AD_DEMENTIA']['pool_size']} candidates)")
    print(f"  TrioTim:              {int((selected['ScannerModel']=='TrioTim').sum())} / 60")
    print(f"  session_rank=1:       {int((selected['session_rank_per_subject']==1).sum())} / 60")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*55)


if __name__ == "__main__":
    build_manifest()
