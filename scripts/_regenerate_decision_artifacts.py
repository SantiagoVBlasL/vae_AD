#!/usr/bin/env python3
"""
Regenerate stop_go_decision_table.csv and cross_target_interpretation.txt
from existing all_baseline_results.csv using the CORRECTED decision logic
(verdicts lead with INDETERMINATE, never with SIGNAL).

Run after editing run_fatigue_connectome_baselines.py decision_table():
    python3 scripts/_regenerate_decision_artifacts.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT    = Path(__file__).resolve().parents[1] / "results" / "fatigue_connectome_baselines"
TABLES = OUT / "Tables"

df_base = pd.read_csv(TABLES / "all_baseline_results.csv")
rows = []

for tgt in sorted(df_base["target"].unique()):
    t = df_base[df_base["target"] == tgt]

    meta = t[t["label"].str.startswith("metadata")]
    meta_auc = float(meta["auc_cv"].iloc[0]) if len(meta) else float("nan")
    meta_p   = meta["perm_p"].iloc[0] if len(meta) else None

    raw = t[t["label"].str.startswith("raw_")]
    if len(raw):
        best_raw = raw.loc[raw["auc_cv"].idxmax()]
        raw_auc  = float(best_raw["auc_cv"])
        raw_p    = best_raw["perm_p"]
        raw_ch   = best_raw.get("channel_name", "")
    else:
        raw_auc, raw_p, raw_ch = float("nan"), None, ""

    resid = t[t["label"].str.startswith("resid_")]
    if len(resid):
        best_resid = resid.loc[resid["auc_cv"].idxmax()]
        resid_auc  = float(best_resid["auc_cv"])
        resid_p    = best_resid["perm_p"]
        resid_ch   = best_resid.get("channel_name", "")
    else:
        resid_auc, resid_p, resid_ch = float("nan"), None, ""

    fem = t[t["label"].str.startswith("female_")]
    if len(fem):
        best_fem = fem.loc[fem["auc_cv"].idxmax()]
        fem_auc  = float(best_fem["auc_cv"])
        fem_p    = best_fem["perm_p"]
        fem_ch   = best_fem.get("channel_name", "")
    else:
        fem_auc, fem_p, fem_ch = float("nan"), None, ""

    def _sig(p):
        if p is None:
            return False
        try:
            v = float(p)
            return (not np.isnan(v)) and v < 0.05
        except (TypeError, ValueError):
            return False

    raw_sig   = _sig(raw_p)
    resid_sig = _sig(resid_p)
    fem_sig   = _sig(fem_p)
    meta_sig  = _sig(meta_p)

    # ── Nominal layer ─────────────────────────────────────────────
    if resid_sig:
        nominal_layer = "nominal-pass (single ch, pre-correction)"
    elif raw_sig and not resid_sig:
        nominal_layer = "confound (raw passes, resid collapses)"
    else:
        nominal_layer = "nominal-null (does not beat perm null)"

    # ── Verdict: ALWAYS leads with INDETERMINATE ──────────────────
    if resid_sig and fem_sig and not meta_sig:
        verdict = (
            "INDETERMINATE (NOMINAL) — "
            + nominal_layer
            + " — requires multiplicity correction via hardening script"
        )
    elif resid_sig and not fem_sig:
        verdict = (
            "INDETERMINATE — "
            + nominal_layer
            + "; female-only NOT confirmed;"
            + " multiplicity correction and incremental-value test required"
        )
    elif resid_sig and meta_sig:
        verdict = (
            "INDETERMINATE — "
            + nominal_layer
            + "; metadata predicts target (incremental value untested);"
            + " multiplicity correction required"
        )
    else:
        verdict = "INDETERMINATE / NULL — " + nominal_layer

    rows.append(dict(
        target=tgt,
        meta_auc=meta_auc, meta_p=meta_p, meta_sig=meta_sig,
        best_raw_ch=raw_ch,   raw_auc=raw_auc,   raw_p=raw_p,   raw_sig=raw_sig,
        best_resid_ch=resid_ch, resid_auc=resid_auc, resid_p=resid_p, resid_sig=resid_sig,
        best_fem_ch=fem_ch,   fem_auc=fem_auc,   fem_p=fem_p,   fem_sig=fem_sig,
        verdict=verdict,
    ))

dec = pd.DataFrame(rows)

# ── Cross-target interpretation ───────────────────────────────────────
d = dec[dec["target"] == "D"]
e = dec[dec["target"] == "E"]
d_sig = bool(d["resid_sig"].iloc[0]) if len(d) else False
e_sig = bool(e["resid_sig"].iloc[0]) if len(e) else False

if e_sig and not d_sig:
    cross = (
        "⚠️  SHORTCUT LEARNING / MIXED PHENOTYPE — "
        "Target E succeeds but Target D fails. "
        "The most parsimonious explanation is that the classifier exploits "
        "COVID-vs-CONTROL group differences rather than fatigue biology. "
        "Do NOT claim a fatigue biomarker."
    )
elif d_sig and e_sig:
    d_fem  = bool(d["fem_sig"].iloc[0]) if len(d) else False
    d_meta = bool(d["meta_sig"].iloc[0]) if len(d) else False
    if d_fem and not d_meta:
        cross = (
            "⚠️  INDETERMINATE (nominal pass, female-only confirms, "
            "metadata clean) — Both targets survive nominal test and "
            "female-only direction is consistent. "
            "STILL requires multiplicity correction via hardening script "
            "before any 'GO' verdict. Do NOT claim fatigue biomarker yet."
        )
    else:
        caveats = []
        if not d_fem:
            caveats.append("female-only NOT confirmed")
        if d_meta:
            caveats.append("metadata predicts target (incremental value untested)")
        cross = (
            "⚠️  INDETERMINATE (nominal signal, caveats unresolved) — "
            "Both targets show nominal signal at best single channel, "
            f"but: {'; '.join(caveats)}. "
            "Run hardening script for multiplicity correction. "
            "Do NOT claim fatigue biomarker without resolving caveats."
        )
elif d_sig and not e_sig:
    cross = (
        "⚠️  INDETERMINATE (nominally fatigue-specific) — "
        "Target D passes nominal test but Target E fails. "
        "Pattern is consistent with a within-COVID fatigue signal, "
        "but requires multiplicity correction and incremental-value "
        "test before any fatigue-biomarker claim."
    )
else:
    cross = (
        "❌  GLOBAL NULL — Neither target beats the permutation null "
        "after residualization. No evidence for a fatigue-related "
        "connectomic signal. Recommend publishing as a negative result."
    )

dec.to_csv(TABLES / "stop_go_decision_table.csv", index=False)
(TABLES / "cross_target_interpretation.txt").write_text(cross + "\n", encoding="utf-8")

print("=== stop_go_decision_table.csv regenerated ===")
for _, row in dec.iterrows():
    print(f"  TARGET {row['target']}: {row['verdict']}")
print()
print("=== cross_target_interpretation.txt regenerated ===")
print(cross)
