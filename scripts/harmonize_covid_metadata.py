"""
harmonize_covid_metadata.py
===========================
Canonical metadata harmonization for the Long-COVID connectome study.

Merges the updated clinical metadata file (ResumenRespuestasBasico.csv) with
the legacy file (SubjectsData_AAL3_COVID.csv) to produce a single harmonized
DataFrame suitable for use in notebook 03_a.

Usage
-----
    from scripts.harmonize_covid_metadata import load_harmonized_metadata

    meta, audit_df = load_harmonized_metadata(project_root)

Returns
-------
    meta     : pd.DataFrame — 214 rows, harmonized metadata
    audit_df : pd.DataFrame — row-level discrepancy table (old vs new)

Design decisions
----------------
- New file (ResumenRespuestasBasico.csv) is the CANONICAL source for all
  variables it contains (Age, Sex, CategoriaFAS, MOCA, cognitive battery, etc.)
- Old file (SubjectsData_AAL3_COVID.csv) contributes: SubjectID, ResearchGroup
  (absent from new file), and serves as integrity cross-check.
- MOCA.1 / MOCA_perc.1 are preserved as MOCA_v2 / MOCA_perc_v2 columns with
  explicit documentation of discrepancies. Use MOCA as primary; consult
  MOCA_v2 only after provenance is confirmed with the data custodian.
- Percentile columns with strings "<2", "<5", ">95" are coerced to numeric
  sentinel values (1, 2.5, 97.5 respectively) and flagged in a _clipped column.
- Symptom columns (DolorDeCabeza, Fatiga, etc.) use 1=persistent /
  2=had-but-resolved / 3=never-had coding; NaN = CONTROL (not applicable).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import warnings


# ──────────────────────────────────────────────────────────────────────────────
# Sentinel mapping for clipped percentile strings
# ──────────────────────────────────────────────────────────────────────────────
_PERC_CLIP_MAP = {"<2": 1.0, "<3": 1.5, "<5": 2.5, ">95": 97.5}


def _coerce_percentile_col(series: pd.Series) -> tuple:
    """
    Coerce a percentile column that may contain strings like '<5' or '>95'.
    Returns (numeric_series, clipped_flag_series).
    The flag is 1 where a sentinel was applied, 0 otherwise.
    """
    clipped = pd.Series(0, index=series.index, dtype=int)
    result = series.copy()
    for raw, sentinel in _PERC_CLIP_MAP.items():
        mask = result == raw
        if mask.any():
            result[mask] = sentinel
            clipped[mask] = 1
    # Remaining non-numeric → NaN
    result = pd.to_numeric(result, errors="coerce")
    return result, clipped


# ──────────────────────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────────────────────

def load_harmonized_metadata(project_root: Path) -> tuple:
    """
    Load, merge, and harmonize COVID clinical metadata.

    Parameters
    ----------
    project_root : Path
        Root of the repository (must contain data/).

    Returns
    -------
    meta : pd.DataFrame
        Harmonized metadata, 214 rows.
        Key columns guaranteed to exist:
          ID, SubjectID, ResearchGroup, Age, Sex,
          Altura, Peso, BMI, CategoríaCOVID, Recuperado,
          EQ-VAS, FAS, CategoriaFAS, NivelEducativo, Ocupacion,
          DiasEntrePrimeraInfeccionYCuestionario,
          DiasEntreUltimaInfeccionYCuestionario,
          MOCA, MOCA_perc, MOCA_v2, MOCA_perc_v2,
          sGMV, Ventricles, WMHTotal, TotalGM_B_Perfusion,
          TMT-A, TMT-A_perc, TMT-B, TMT-B_perc,
          WMS-R-DIR, WMS-R-DIR_perc, WMS-R-INV, WMS-R-INV_perc,
          STROOP_P, STROOP_P_perc, STROOP_C, STROOP_C_perc,
          STROOP_P/C, STROOP_P/C_perc, STROOP_P/C_INTERF, STROOP_P/C_INTERF_perc,
          MOCA-rendimiento,
          DolorDeCabeza, Fatiga_sym, Olfato, Gusto, Disnea,
          DebilidadMuscular, DolorMuscular, Confusion, Comunicacion,
          Dormir, Memoria, Atencion,
          _merge_id, _source
    audit_df : pd.DataFrame
        Row-level discrepancy table between old and new sources.
    """
    project_root = Path(project_root)
    path_old = project_root / "data" / "SubjectsData_AAL3_COVID.csv"
    path_new = project_root / "data" / "ResumenRespuestasBasico.csv"

    assert path_old.exists(), f"Old metadata not found: {path_old}"
    assert path_new.exists(), f"New metadata not found: {path_new}"

    # ── Load files ────────────────────────────────────────────────────────────
    old = pd.read_csv(path_old)
    new = pd.read_csv(path_new)

    # Strip whitespace from column names (new file has trailing spaces)
    new.columns = [c.strip() for c in new.columns]
    old.columns = [c.strip() for c in old.columns]

    # ── Verify perfect ID alignment ───────────────────────────────────────────
    ids_old = set(old["ID"].astype(str))
    ids_new = set(new["ID"].astype(str))
    only_old = ids_old - ids_new
    only_new = ids_new - ids_old
    if only_old or only_new:
        raise ValueError(
            f"ID mismatch between old and new metadata!\n"
            f"  Only in old: {only_old}\n"
            f"  Only in new: {only_new}"
        )
    assert len(old) == len(new) == 214, f"Expected 214 rows; got old={len(old)}, new={len(new)}"

    # ── Build audit table ─────────────────────────────────────────────────────
    audit_rows = []
    # Columns to compare; new file uses Edad/Genero instead of Age/Sex
    compare_specs = [
        ("Age", "Edad"),
        ("Sex", "Genero"),
        ("EQ-VAS", "EQ-VAS"),
        ("CategoriaFAS", "CategoriaFAS"),
        ("MOCA", "MOCA"),
        ("MOCA_perc", "MOCA_perc"),
        ("Recuperado", "Recuperado"),
    ]
    old_idx = old.set_index("ID")
    new_idx = new.set_index("ID")

    for subject_id in old["ID"].astype(str):
        for col_old, col_new in compare_specs:
            old_val = old_idx.at[subject_id, col_old] if col_old in old_idx.columns else np.nan
            new_val = new_idx.at[subject_id, col_new] if col_new in new_idx.columns else np.nan
            if pd.isna(old_val) and pd.isna(new_val):
                continue
            if str(old_val) != str(new_val):
                audit_rows.append({
                    "ID": subject_id,
                    "variable": col_old,
                    "value_old": old_val,
                    "value_new": new_val,
                })

    # MOCA.1 vs MOCA comparison
    moca_merged = new[["ID", "MOCA", "MOCA.1", "MOCA_perc", "MOCA_perc.1"]].copy()
    for _, row in moca_merged.iterrows():
        if row["MOCA"] != row["MOCA.1"]:
            audit_rows.append({
                "ID": row["ID"],
                "variable": "MOCA_v1_vs_v2",
                "value_old": row["MOCA"],
                "value_new": row["MOCA.1"],
            })

    audit_df = pd.DataFrame(audit_rows)

    # ── Rename new columns ────────────────────────────────────────────────────
    new = new.rename(columns={
        "Edad": "Age",
        "Genero": "Sex",
        "TotalGM_B Perfusion": "TotalGM_B_Perfusion",
        "Fatiga": "Fatiga_sym",   # avoid confusion with FAS
        "MOCA.1": "MOCA_v2",
        "MOCA_perc.1": "MOCA_perc_v2",
    })
    # Drop the junk Unnamed:34 column
    new = new.drop(columns=["Unnamed: 34"], errors="ignore")

    # ── Coerce percentile columns ─────────────────────────────────────────────
    perc_cols = [c for c in new.columns if "_perc" in c and c not in ("MOCA_perc", "MOCA_perc_v2")]
    for col in perc_cols:
        num, flag = _coerce_percentile_col(new[col].astype(str).where(new[col].notna(), np.nan))
        new[col] = num
        new[f"{col}_clipped"] = flag

    # ── Coerce DiasEntre columns (object → numeric) ───────────────────────────
    for col in ["DiasEntrePrimeraInfeccionYCuestionario",
                "DiasEntreUltimaInfeccionYCuestionario"]:
        if col in new.columns:
            new[col] = pd.to_numeric(new[col], errors="coerce")

    # ── Coerce WMHTotal (may be object due to rounding artifacts) ────────────
    if "WMHTotal" in new.columns:
        new["WMHTotal"] = pd.to_numeric(new["WMHTotal"], errors="coerce")

    # ── Bring in SubjectID and ResearchGroup from old file ────────────────────
    bridge = old[["ID", "SubjectID", "ResearchGroup"]].copy()
    bridge["ID"] = bridge["ID"].astype(str)
    new["ID"] = new["ID"].astype(str)

    meta = new.merge(bridge, on="ID", how="left")

    # Verify SubjectID coverage
    n_with_subjectid = meta["SubjectID"].notna().sum()
    if n_with_subjectid != 214:
        warnings.warn(
            f"SubjectID coverage: {n_with_subjectid}/214 — check merge.",
            stacklevel=2,
        )

    # ── Add ResearchGroup normalization ───────────────────────────────────────
    meta["ResearchGroup"] = (
        meta["ResearchGroup"]
        .astype(str).str.strip().str.upper()
        .replace({"CTRL": "CONTROL", "HC": "CONTROL", "HEALTHY": "CONTROL"})
    )

    # ── Add merge key for downstream notebook join ────────────────────────────
    meta["_merge_id"] = meta["ID"].astype(str)
    meta["_source"] = "ResumenRespuestasBasico_v1"

    # ── Validate ──────────────────────────────────────────────────────────────
    assert len(meta) == 214
    assert meta["SubjectID"].notna().all(), "Some rows missing SubjectID after merge"
    assert "ResearchGroup" in meta.columns

    print(f"[harmonize] Loaded harmonized metadata: {meta.shape}")
    print(f"[harmonize] Groups: {meta['ResearchGroup'].value_counts().to_dict()}")
    print(f"[harmonize] MOCA coverage: {meta['MOCA'].notna().sum()}/214")
    print(f"[harmonize] TMT-A coverage: {meta['TMT-A'].notna().sum()}/214")
    print(f"[harmonize] FAS (continuous) coverage: {meta['FAS'].notna().sum()}/214")
    if len(audit_df) > 0:
        print(f"[harmonize] Discrepancies old→new: {len(audit_df)} rows "
              f"({audit_df['variable'].value_counts().to_dict()})")

    return meta, audit_df


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: save audit CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_audit(audit_df: pd.DataFrame, output_dir: Path) -> None:
    """Save audit table to Tables directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metadata_old_vs_new_audit.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"[harmonize] Audit saved → {out_path}")


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
    meta, audit = load_harmonized_metadata(root)
    print(meta.columns.tolist())
    print(audit)
