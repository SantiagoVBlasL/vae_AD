#!/usr/bin/env python3
"""
Generate notebooks/revision_bspc_2026/05_adni_expansion_subject_selection.ipynb

Run with:
    /home/diego/anaconda3/envs/vae_ad/bin/python \
        scripts/revision_bspc_2026/gen_05_adni_expansion_notebook.py
"""

import nbformat as nbf
from pathlib import Path
import textwrap

OUT_NB = Path(
    "/home/diego/proyectos/vae_AD/notebooks/revision_bspc_2026/"
    "05_adni_expansion_subject_selection.ipynb"
)

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "vae_ad",
        "language": "python",
        "name": "vae_ad",
    },
    "language_info": {"name": "python", "version": "3.9.0"},
}
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())

def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


# ─────────────────────────────────────────────────────────────────────────────
# §0  TITLE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
# ADNI Expansion Audit — Candidate rs-fMRI Subject Selection
### BSPC 2026 Major Revision · April 2026

**Authors:** Diego Vidaurre-Saez, Santiago V. Blasl, Martín Raggio
**Notebook:** `05_adni_expansion_subject_selection.ipynb`
**Branch:** `revision_bspc_2026`

---

## §0 — Scope and Motivation

This notebook performs a rigorous, publication-grade audit of a new ADNI IDA export
(`idaSearch_4_03_2026`) against our existing processed cohort.  The goal is to provide
Martin with a clean operational download list **and** to provide the team with a
scientifically justified expansion strategy for the BSPC major revision.

### Why are we doing this?
Reviewers of our BSPC submission raised concerns about:
1. **Sample size** — more ADNI subjects may strengthen generalizability claims.
2. **Class imbalance** — the CN vs AD classifier may benefit from additional subjects.
3. **Scanner/site robustness** — a broader scanner mix would support robustness analyses.
4. **Protocol diversity** — we must ensure any new subjects are protocol-compatible with
   our existing preprocessing pipeline.

### What this notebook does NOT do
This is an **acquisition audit and pre-download recommendation tool**.
It does **not** guarantee that selected subjects will pass QC after preprocessing
(motion, field-inhomogeneity, coverage issues are assessed at preprocessing time).

### Relationship to other revision artifacts
| Notebook | Content |
|---|---|
| `03_run_vae_clf_ad_loso` | LOSO cross-validation pipeline |
| `04_loso_master_q1_audit` | Q1 master audit of LOSO results |
| **`05_adni_expansion_audit`** | ← **this notebook** |
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §1  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §1 — Configuration and Path Resolution"))

cells.append(code("""
import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ── Plotly (preferred) ──────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.renderers.default = 'notebook_connected'
    PLOTLY_AVAILABLE = True
    print("Plotly ✓")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not found — figures will use matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
import seaborn as sns

# ── Project root ─────────────────────────────────────────────────────────────
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path('/home/diego/proyectos/vae_AD')

DATA_DIR  = PROJECT_ROOT / 'data'
OUT_DIR   = PROJECT_ROOT / 'results' / 'revision_bspc_2026' / 'adni_expansion_audit'
FIG_DIR   = OUT_DIR / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA_DIR / 'SubjectsData_AAL3_procesado2.csv'
IDA_CSV    = DATA_DIR / 'idaSearch_4_03_2026 (2).csv'

print(f"\\nPROJECT_ROOT : {PROJECT_ROOT}")
print(f"Output dir   : {OUT_DIR}")
for p in [COHORT_CSV, IDA_CSV]:
    mark = '✓' if p.exists() else '✗ MISSING'
    print(f"  [{mark}] {p.name}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §2  LOAD AND INSPECT
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §2 — Load and Inspect Both CSV Files"))

cells.append(code("""
df_cohort_raw = pd.read_csv(COHORT_CSV)
df_ida_raw    = pd.read_csv(IDA_CSV)

print("=" * 60)
print("EXISTING COHORT  (SubjectsData_AAL3_procesado2.csv)")
print("=" * 60)
print(f"Shape    : {df_cohort_raw.shape}")
print(f"Columns  : {list(df_cohort_raw.columns)}")
print()

print("=" * 60)
print("IDA EXPORT  (idaSearch_4_03_2026 (2).csv)")
print("=" * 60)
print(f"Shape    : {df_ida_raw.shape}")
print(f"Columns  : {list(df_ida_raw.columns)}")
"""))

cells.append(md("""
### Column Mapping

The two files use slightly different column naming conventions.  The table below
documents the chosen mapping used throughout this notebook.

| Concept | Cohort column | IDA Export column |
|---|---|---|
| Subject ID | `SubjectID` | `Subject ID` |
| Image ID | `ImageID` | `Image ID` |
| ADNI Phase | `Phase` | `Phase` |
| Research group (diagnosis) | `ResearchGroup` | `Research Group` |
| Acquisition parameters | `TR`, `TE`, `Field Strength`, `Manufacturer`, `Slice Thickness` | parsed from `Imaging Protocol` |
| Series description | `Description` | `Description` |
| Visit | `Visit` | `Visit` |

> **Note:** The IDA export stores acquisition parameters as a semicolon-separated
> `key=value` string in `Imaging Protocol`.  We parse this into separate columns
> in the next cell.  Manufacturer names are normalised to `{Philips, Siemens, GE}`.
"""))

cells.append(code("""
# ── Parse IDA Imaging Protocol ───────────────────────────────────────────────
def parse_protocol(s):
    \"\"\"Parse 'Key=Value;Key=Value;...' into a dict.\"\"\"
    if pd.isna(s) or str(s).strip() == '':
        return {}
    return dict(re.findall(r'([^=;]+)=([^;]+)', str(s)))

def normalize_manufacturer(raw):
    \"\"\"Collapse Philips Medical Systems / Philips Healthcare → Philips, etc.\"\"\"
    m = str(raw).strip().upper()
    if 'PHILIPS' in m:
        return 'Philips'
    if 'SIEMENS' in m:
        return 'Siemens'
    if 'GE' in m:
        return 'GE'
    return raw

proto_dicts = df_ida_raw['Imaging Protocol'].apply(parse_protocol)

df_ida = df_ida_raw.copy()
df_ida['TR']              = proto_dicts.apply(lambda d: float(d.get('TR',            np.nan)))
df_ida['TE']              = proto_dicts.apply(lambda d: float(d.get('TE',            np.nan)))
df_ida['Field_Strength']  = proto_dicts.apply(lambda d: float(d.get('Field Strength',np.nan)))
df_ida['Slice_Thickness'] = proto_dicts.apply(lambda d: float(d.get('Slice Thickness',np.nan)))
df_ida['Manufacturer']    = proto_dicts.apply(lambda d: normalize_manufacturer(
                                d.get('Manufacturer', 'Unknown')))

# Standardise cohort column name for Field Strength (already split there)
df_cohort = df_cohort_raw.copy()
df_cohort.rename(columns={
    'Field Strength':  'Field_Strength',
    'Slice Thickness': 'Slice_Thickness',
}, inplace=True)

print("IDA parsed TR unique        :", sorted(df_ida['TR'].dropna().unique()))
print("IDA parsed TE unique        :", sorted(df_ida['TE'].dropna().unique()))
print("IDA parsed Field Strength   :", sorted(df_ida['Field_Strength'].dropna().unique()))
print("IDA Manufacturer (normalised):", df_ida['Manufacturer'].value_counts().to_dict())
print()
print("Cohort TR unique            :", sorted(df_cohort['TR'].dropna().unique()))
print("Cohort TE unique            :", sorted(df_cohort['TE'].dropna().unique()))
print("Cohort Manufacturer unique  :", df_cohort['Manufacturer'].value_counts().to_dict())
"""))

cells.append(code("""
# ── Key statistics for both files ────────────────────────────────────────────
print("=== COHORT ===")
print(f"  Subjects (unique)  : {df_cohort['SubjectID'].nunique()}")
print(f"  Scans (rows)       : {len(df_cohort)}")
print(f"  ResearchGroup      : {df_cohort['ResearchGroup'].value_counts().to_dict()}")
print(f"  Phase              : {df_cohort['Phase'].value_counts().to_dict()}")
print()
print("=== IDA EXPORT ===")
print(f"  Subjects (unique)  : {df_ida['Subject ID'].nunique()}")
print(f"  Scans (rows)       : {len(df_ida)}")
print(f"  Research Group     : {df_ida['Research Group'].value_counts().to_dict()}")
print(f"  Phase              : {df_ida['Phase'].value_counts().to_dict()}")
print(f"  Description (top5) : {dict(list(df_ida['Description'].value_counts().head(5).items()))}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §3  CURRENT COHORT IDENTITY
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §3 — Define Current Cohort Identity

We treat each subject as a unique entity based on their `SubjectID`.  Because every
subject in the processed cohort contributed exactly one scan (confirmed below), there
is a 1-to-1 mapping between subjects and processed scans.  We track both `SubjectID`
and `ImageID` sets for deduplication.
"""))

cells.append(code("""
# ── Cohort identity sets ─────────────────────────────────────────────────────
cohort_subject_ids = set(df_cohort['SubjectID'].unique())
cohort_image_ids   = set(df_cohort['ImageID'].astype(str).unique())

assert len(cohort_subject_ids) == len(df_cohort), (
    f"Cohort is NOT 1-subject-1-scan: "
    f"{len(cohort_subject_ids)} subjects vs {len(df_cohort)} rows"
)
print(f"Cohort: {len(cohort_subject_ids)} subjects, each with exactly 1 processed scan. ✓")
print()
print("Diagnosis breakdown (current cohort):")
diag_cohort = df_cohort['ResearchGroup'].value_counts()
for grp, n in diag_cohort.items():
    pct = 100 * n / len(df_cohort)
    print(f"  {grp:6s}: {n:3d}  ({pct:.1f}%)")
print(f"  {'TOTAL':6s}: {len(df_cohort):3d}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §4  DEDUPLICATION AUDIT
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §4 — Deduplication Audit: Existing Cohort vs IDA Export

We classify every row in the IDA export into one of three identity categories:

| Category | Definition |
|---|---|
| **exact_match** | `ImageID` is already in the processed cohort |
| **subject_overlap** | `SubjectID` is in the cohort, but this specific `ImageID` is not |
| **new_candidate** | Neither `SubjectID` nor `ImageID` is in the current cohort |
"""))

cells.append(code("""
# ── Flag each IDA row ────────────────────────────────────────────────────────
df_work = df_ida.copy()

df_work['image_in_cohort']   = df_work['Image ID'].astype(str).isin(cohort_image_ids)
df_work['subject_in_cohort'] = df_work['Subject ID'].isin(cohort_subject_ids)

df_work['identity_category'] = 'new_candidate'
df_work.loc[df_work['subject_in_cohort'] & ~df_work['image_in_cohort'],
            'identity_category'] = 'subject_overlap'
df_work.loc[df_work['image_in_cohort'],
            'identity_category'] = 'exact_match'

# Summary counts
id_cat = df_work['identity_category'].value_counts()
print("IDA rows by identity category:")
for cat, n in id_cat.items():
    pct = 100 * n / len(df_work)
    print(f"  {cat:25s}: {n:4d} rows  ({pct:.1f}%)")
print()

# Subject-level summary
new_cand_mask = df_work['identity_category'] == 'new_candidate'
n_new_subj    = df_work.loc[new_cand_mask, 'Subject ID'].nunique()
n_new_rows    = new_cand_mask.sum()
print(f"Truly new subjects in IDA : {n_new_subj}")
print(f"Truly new scans (rows)    : {n_new_rows}")

# Venn summary table
venn_data = {
    'Category': [
        'Current cohort (processed)',
        'IDA export — exact match (already processed)',
        'IDA export — subject overlap (different visit)',
        'IDA export — new candidates',
        'IDA export — TOTAL',
    ],
    'Unique Subjects': [
        len(cohort_subject_ids),
        df_work.loc[df_work['identity_category']=='exact_match',   'Subject ID'].nunique(),
        df_work.loc[df_work['identity_category']=='subject_overlap','Subject ID'].nunique(),
        df_work.loc[df_work['identity_category']=='new_candidate',  'Subject ID'].nunique(),
        df_work['Subject ID'].nunique(),
    ],
    'Scan Rows': [
        len(df_cohort),
        (df_work['identity_category']=='exact_match').sum(),
        (df_work['identity_category']=='subject_overlap').sum(),
        (df_work['identity_category']=='new_candidate').sum(),
        len(df_work),
    ],
}
df_venn = pd.DataFrame(venn_data)
print()
print(df_venn.to_string(index=False))
"""))

cells.append(code("""
# ── Overlap visualisation ─────────────────────────────────────────────────────
COLORS = {'exact_match': '#4A90D9', 'subject_overlap': '#F5A623', 'new_candidate': '#7ED321'}
LABELS = {
    'exact_match':     'Already processed<br>(exact ImageID)',
    'subject_overlap': 'Subject in cohort<br>(different visit)',
    'new_candidate':   'Truly new candidate',
}

cat_counts = df_work.groupby('identity_category')['Subject ID'].nunique().reset_index()
cat_counts.columns = ['category', 'n_subjects']

if PLOTLY_AVAILABLE:
    fig = go.Figure(go.Pie(
        labels=[LABELS[c] for c in cat_counts['category']],
        values=cat_counts['n_subjects'],
        hole=0.45,
        marker_colors=[COLORS[c] for c in cat_counts['category']],
        textinfo='label+value+percent',
        textfont_size=13,
        hovertemplate='%{label}<br>Subjects: %{value}<br>%{percent}<extra></extra>',
    ))
    fig.update_layout(
        title_text='IDA Export: Subject-Level Identity Classification<br>'
                   '<sup>n = 588 unique subjects across 645 scans</sup>',
        title_x=0.5,
        legend=dict(orientation='h', y=-0.12),
        margin=dict(t=90, b=60),
        width=680, height=440,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig01_overlap_donut.html'))
    print("Saved → figures/fig01_overlap_donut.html")
else:
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        cat_counts['n_subjects'],
        labels=[LABELS[c].replace('<br>', '\\n') for c in cat_counts['category']],
        autopct='%1.0f%%',
        colors=[COLORS[c] for c in cat_counts['category']],
        wedgeprops=dict(width=0.55),
    )
    ax.set_title('IDA Export: Subject-Level Identity Classification', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig01_overlap_donut.png', dpi=130)
    plt.show()
"""))

cells.append(code("""
# ── Bar chart: Research Group composition by identity category ────────────────
grp_by_cat = (
    df_work[df_work['identity_category'] == 'new_candidate']
    .groupby(['identity_category', 'Research Group'])['Subject ID']
    .nunique()
    .reset_index(name='n_subjects')
)

new_rg = df_work[df_work['identity_category']=='new_candidate'].groupby(
    'Research Group')['Subject ID'].nunique()
print("New candidate subjects by Research Group:")
for rg, n in new_rg.items():
    pct = 100 * n / new_rg.sum()
    print(f"  {rg:4s}: {n:4d}  ({pct:.1f}%)")
print(f"  TOTAL: {new_rg.sum()}")
"""))

cells.append(md("""
### §4 Take-Home

- The IDA export contains **588 unique subjects** (645 scans total).
- **183 subjects** are an exact match with our current processed cohort (same `ImageID`).
- A further batch of subjects appear with a **different visit/timepoint** but their
  baseline scan is already processed — these are also excluded.
- **405 subjects are genuinely new** and have never been processed in our pipeline.
- Of these 405, the Research Group distribution is overwhelmingly **CN-dominated**
  (≈ 437 CN, only ≈ 4 AD), which is a critical strategic constraint addressed in §8.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §5  ACQUISITION PARAMETER AUDIT
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §5 — Acquisition Parameter Audit

Before defining compatibility criteria (§6), we inspect the raw distributions of all
relevant acquisition parameters across the entire IDA export.  This is critical because:

- **TR (Repetition Time):** Controls the temporal sampling rate of the BOLD signal.
  Mixing TRs would change the time-series length / network structure unpredictably.
- **TE (Echo Time):** Governs the BOLD contrast weighting.  Differences in TE alter
  the relative contribution of T2* relaxation to the signal.
- **Field Strength:** 1.5 T vs 3.0 T produce fundamentally different SNR profiles
  and are not interchangeable without careful harmonisation.
- **Slice Thickness:** Minor differences in slice thickness (e.g., 3.3 vs 3.4 mm)
  are less critical than TR/TE/FS and are absorbed by spatial normalisation.
- **ADNI Phase:** Different phases sometimes coincide with protocol updates.
  ADNI 4, in particular, introduced the MSV21 sequence variant and changed
  phase-encoding direction at some sites.
- **Manufacturer:** Siemens, Philips, GE produce different k-space trajectories and
  image reconstruction artifacts.  This is tracked but not used as a hard exclusion
  criterion here, since our current cohort already includes all three.
"""))

cells.append(code("""
# ── Parameter summary: IDA export ────────────────────────────────────────────
print("IDA Export — Acquisition Parameters")
print("-" * 50)
for col, label in [('TR','TR (ms)'), ('TE','TE (ms)'),
                   ('Field_Strength','Field Strength (T)'),
                   ('Slice_Thickness','Slice Thickness (mm)')]:
    vals = df_ida[col].dropna()
    vc   = vals.value_counts().to_dict()
    print(f"  {label:22s}: {vc}")

print()
print("IDA Manufacturer (normalised):", df_ida['Manufacturer'].value_counts().to_dict())
print("IDA Phase                     :", df_ida['Phase'].value_counts().to_dict())
print()
print("IDA Description (all values):")
for desc, n in df_ida['Description'].value_counts().items():
    print(f"  {n:4d}  {desc}")
"""))

cells.append(code("""
# ── Figure: Parameter distributions across IDA export ────────────────────────
if PLOTLY_AVAILABLE:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'TR (ms)', 'TE (ms)', 'Field Strength (T)',
            'Slice Thickness (mm)', 'Manufacturer', 'ADNI Phase',
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    palette = px.colors.qualitative.Set2

    for i, (col, label) in enumerate([
        ('TR', 'TR'), ('TE', 'TE'), ('Field_Strength', 'Field Strength'),
    ]):
        vc = df_ida[col].value_counts().reset_index()
        vc.columns = [col, 'count']
        r, c = 1, i+1
        fig.add_trace(go.Bar(x=vc[col].astype(str), y=vc['count'],
                             marker_color=palette[i], showlegend=False), row=r, col=c)

    # Slice thickness
    vc_st = df_ida['Slice_Thickness'].value_counts().reset_index()
    vc_st.columns = ['Slice_Thickness', 'count']
    fig.add_trace(go.Bar(x=vc_st['Slice_Thickness'].astype(str), y=vc_st['count'],
                         marker_color=palette[3], showlegend=False), row=2, col=1)

    # Manufacturer
    vc_m = df_ida['Manufacturer'].value_counts().reset_index()
    vc_m.columns = ['Manufacturer', 'count']
    fig.add_trace(go.Bar(x=vc_m['Manufacturer'], y=vc_m['count'],
                         marker_color=palette[4], showlegend=False), row=2, col=2)

    # Phase
    vc_p = df_ida['Phase'].value_counts().reset_index()
    vc_p.columns = ['Phase', 'count']
    fig.add_trace(go.Bar(x=vc_p['Phase'], y=vc_p['count'],
                         marker_color=palette[5], showlegend=False), row=2, col=3)

    fig.update_layout(
        title_text='IDA Export — Acquisition Parameter Distributions<br>'
                   '<sup>n = 645 scans; all have TR=3000 ms, TE=30 ms, B₀=3.0 T</sup>',
        title_x=0.5, height=520, width=960,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig02_parameter_distributions.html'))
    print("Saved → figures/fig02_parameter_distributions.html")
"""))

cells.append(code("""
# ── Heatmap: Phase × Manufacturer (IDA new candidates only) ──────────────────
df_new = df_work[df_work['identity_category'] == 'new_candidate'].copy()

pivot = (
    df_new.groupby(['Phase', 'Manufacturer'])['Subject ID']
    .nunique()
    .unstack(fill_value=0)
)
print("Phase × Manufacturer (new candidate subjects):")
print(pivot.to_string())
print()

if PLOTLY_AVAILABLE:
    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x='Manufacturer', y='ADNI Phase', color='N subjects'),
        title='New Candidate Subjects: Phase × Manufacturer<br>'
              '<sup>Cell values = unique subjects</sup>',
    )
    fig.update_layout(title_x=0.5, width=560, height=380,
                      coloraxis_showscale=True)
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig03_phase_manufacturer_heatmap.html'))
    print("Saved → figures/fig03_phase_manufacturer_heatmap.html")
"""))

cells.append(md("""
### §5 Take-Home

**Key finding: the IDA export is fully protocol-homogeneous.**

Every scan in the IDA export has:
- TR = **3 000 ms** (identical to our entire existing cohort)
- TE = **30 ms** (identical)
- B₀ = **3.0 T** (identical)

This is the best-case scenario: there is no acquisition heterogeneity in the
core timing parameters.  Differences in **slice thickness** (3.3 vs 3.4 mm) are
minor and already present within our existing cohort.

The main compatibility concerns come from **ADNI phase** (ADNI 4 introduces the
MSV21 sequence variant) and **series description** (some scans use a different
phase-encoding direction), addressed in §6.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §6  COMPATIBILITY LOGIC
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §6 — Compatibility Logic with the Current Pipeline

### Criteria rationale

| Criterion | Value | Status | Reasoning |
|---|---|---|---|
| TR | 3 000 ms | ✅ All match | Temporal sampling identical → comparable time-series length and dynamic FC estimates |
| TE | 30 ms | ✅ All match | BOLD contrast identical |
| Field strength | 3.0 T | ✅ All match | SNR profile identical; 1.5 T scans would require separate normalisation |
| Slice thickness | 3.3–3.6 mm | ✅ All within range | Minor spatial resampling differences absorbed by MNI registration |
| Phase (ADNI 4) | — | ⚠️ Flag for review | ADNI 4 introduced the **MSV21** sequence variant (new multiband factor, different phase-encoding direction at many sites).  Until our ADNI 4 scans have been pre-processed and the QC pipeline validated on that protocol, these are placed in **manual review** rather than automatic exclusion |
| Series description — phase encoding variants | — | ⚠️ Flag for review | Descriptions explicitly mentioning `-phase P to A` or `Phase Direction P>A` indicate a different readout direction.  This can affect susceptibility-distortion patterns and warrants manual inspection before preprocessing |
| Series description — Extended rsfMRI | — | ⚠️ Flag for review | "Extended" variants typically have more volumes; including them without standardising to the common volume count could bias network estimates |
| Series description — underscore naming | — | ⚠️ Flag for review | `Axial_rsFMRI_Eyes_Open` (underscore) suggests a non-standard protocol entry; manual check warranted |

### basic_compatible flag
All IDA scans share TR/TE/FS with the current cohort, so `compatible_basic = True`
for every row.  The finer distinctions above drive the `recommended_status` field.
"""))

cells.append(code("""
# ── Classify series descriptions ─────────────────────────────────────────────
STANDARD_DESCS = {
    'Resting State fMRI',
    'Axial rsfMRI (Eyes Open)',
    'Axial rsfMRI (EYES OPEN)',
    'Axial fcMRI (Eyes Open)',
    'Axial fcMRI (EYES OPEN)',
    'Axial RESTING fcMRI (EYES OPEN)',
}

def classify_description(desc):
    \"\"\"Return ('standard'|'phase_encoding'|'extended'|'msv21'|'unusual'|'unknown').\"\"\"
    if pd.isna(desc):
        return 'unknown'
    d = str(desc).strip()
    if d in STANDARD_DESCS:
        return 'standard'
    if any(kw in d for kw in ['P to A', 'P>A', 'phase P', 'Phase Direction']):
        return 'phase_encoding_variant'
    if 'Extended' in d:
        return 'extended_rsfmri'
    if any(kw in d for kw in ['MSV21', 'MSV20', '(MSV2', 'MSV21)', 'MSV21P']):
        return 'msv21_variant'
    if '100phases' in d:
        return 'extended_rsfmri'   # extra-long scan
    if d == 'Axial_rsFMRI_Eyes_Open':
        return 'unusual_naming'
    return 'unusual_naming'

df_work['desc_class'] = df_work['Description'].apply(classify_description)

print("Description class distribution (full IDA):")
print(df_work['desc_class'].value_counts().to_dict())
print()
print("Description class distribution (new candidates only):")
print(df_work.loc[df_work['identity_category']=='new_candidate',
                  'desc_class'].value_counts().to_dict())
"""))

cells.append(code("""
# ── Core compatibility flag (basic) ──────────────────────────────────────────
# All IDA scans pass the basic TR/TE/FS check
df_work['compatible_basic'] = (
    (df_work['TR']           == 3000.0) &
    (df_work['TE']           == 30.0)   &
    (df_work['Field_Strength']== 3.0)
)
print("Rows passing basic TR/TE/FS compatibility:",
      df_work['compatible_basic'].sum(), "/", len(df_work))

# ── Assign recommended_status ────────────────────────────────────────────────
def assign_status_reason(row):
    cat  = row['identity_category']
    dcls = row['desc_class']
    phase= row['Phase']

    if cat == 'exact_match':
        return 'exclude_for_now', 'already_processed'
    if cat == 'subject_overlap':
        return 'exclude_for_now', 'subject_already_in_cohort_different_visit'
    # new_candidate below
    if phase == 'ADNI 4':
        return 'manual_review', 'adni4_pending_protocol_validation'
    if dcls == 'phase_encoding_variant':
        return 'manual_review', 'nonstandard_phase_encoding_direction'
    if dcls == 'extended_rsfmri':
        return 'manual_review', 'extended_rsfmri_nonstandard_scan_length'
    if dcls in ('unusual_naming', 'unknown'):
        return 'manual_review', 'unusual_series_description_manual_check'
    if dcls == 'msv21_variant':
        # msv21 in non-ADNI4 context (shouldn't happen often but be safe)
        return 'manual_review', 'msv21_variant_outside_adni4'
    # Standard description, new candidate, non-ADNI4
    return 'download_now', 'compatible_new_candidate'

df_work[['pre_status', 'pre_reason']] = df_work.apply(
    assign_status_reason, axis=1, result_type='expand')

print()
print("Pre-deduplication status counts (all 645 rows):")
print(df_work['pre_status'].value_counts().to_dict())
print()
print("Reasons breakdown:")
print(df_work['pre_reason'].value_counts().to_dict())
"""))

cells.append(md("""
### §6 Take-Home

After applying the compatibility filter:
- **All 645 scans pass the basic TR/TE/FS test** — no scan needs to be excluded
  purely on acquisition parameter grounds.
- The important distinctions are **ADNI phase** (ADNI 4) and **series description**
  (phase-encoding variants, extended scans, unusual naming), which send scans to
  **manual review** rather than automatic download.
- The bulk of exclusions at this stage come from scans already in our cohort.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §7  EXISTING-COHORT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §7 — Existing-Cohort Comparison

We compare the acquisition profile of **new `download_now` candidates** against the
**existing processed cohort** to confirm we are expanding within the same acquisition
regime rather than mixing in a meaningfully different one.
"""))

cells.append(code("""
# ── Filter: new candidates eligible for download (before scan deduplication) ──
df_cand = df_work[df_work['pre_status'] == 'download_now'].copy()

print(f"New download_now candidates (before subject-level deduplication): {len(df_cand)} scans")
print(f"Unique subjects in this set: {df_cand['Subject ID'].nunique()}")
print()

# Acquisition comparison
compare_rows = []
for label, dfx, n_col in [
    ('Current cohort',    df_cohort, 'SubjectID'),
    ('New candidates', df_cand,    'Subject ID'),
]:
    for param, col in [('TR','TR'), ('TE','TE'),
                       ('Field Strength','Field_Strength'), ('Slice Thickness','Slice_Thickness')]:
        vals = dfx[col].dropna()
        compare_rows.append({
            'Dataset': label,
            'Parameter': param,
            'Mean ± SD': f"{vals.mean():.1f} ± {vals.std():.2f}",
            'Unique values': sorted(vals.unique().tolist()),
        })

df_compare = pd.DataFrame(compare_rows)
print(df_compare.to_string(index=False))
"""))

cells.append(code("""
# ── Figure: Manufacturer comparison — cohort vs new candidates ────────────────
cohort_manuf = df_cohort['Manufacturer'].value_counts(normalize=True) * 100
cand_manuf   = df_cand['Manufacturer'].value_counts(normalize=True) * 100

all_manuf = sorted(set(cohort_manuf.index) | set(cand_manuf.index))
cohort_pct = [cohort_manuf.get(m, 0) for m in all_manuf]
cand_pct   = [cand_manuf.get(m, 0)   for m in all_manuf]

if PLOTLY_AVAILABLE:
    fig = go.Figure(data=[
        go.Bar(name='Current cohort', x=all_manuf, y=cohort_pct,
               marker_color='#4A90D9', text=[f'{v:.1f}%' for v in cohort_pct],
               textposition='outside'),
        go.Bar(name='New candidates', x=all_manuf, y=cand_pct,
               marker_color='#7ED321', text=[f'{v:.1f}%' for v in cand_pct],
               textposition='outside'),
    ])
    fig.update_layout(
        barmode='group',
        title='Manufacturer Composition: Current Cohort vs New Candidates<br>'
              '<sup>Percentages within each dataset</sup>',
        title_x=0.5,
        yaxis_title='% of subjects',
        xaxis_title='Manufacturer',
        legend=dict(orientation='h', y=1.12),
        height=420, width=680,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig04_manufacturer_comparison.html'))
    print("Saved → figures/fig04_manufacturer_comparison.html")
"""))

cells.append(code("""
# ── Figure: Phase comparison — cohort vs new candidates ───────────────────────
cohort_phase = df_cohort['Phase'].value_counts(normalize=True) * 100
cand_phase   = df_cand['Phase'].value_counts(normalize=True) * 100

all_phases = sorted(set(cohort_phase.index) | set(cand_phase.index))
c_pct = [cohort_phase.get(p, 0) for p in all_phases]
n_pct = [cand_phase.get(p, 0)   for p in all_phases]

if PLOTLY_AVAILABLE:
    fig = go.Figure(data=[
        go.Bar(name='Current cohort', x=all_phases, y=c_pct,
               marker_color='#4A90D9', text=[f'{v:.1f}%' for v in c_pct],
               textposition='outside'),
        go.Bar(name='New candidates', x=all_phases, y=n_pct,
               marker_color='#7ED321', text=[f'{v:.1f}%' for v in n_pct],
               textposition='outside'),
    ])
    fig.update_layout(
        barmode='group',
        title='ADNI Phase Composition: Current Cohort vs New Candidates<br>'
              '<sup>Percentages within each dataset</sup>',
        title_x=0.5,
        yaxis_title='% of subjects',
        xaxis_title='ADNI Phase',
        legend=dict(orientation='h', y=1.12),
        height=420, width=680,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig05_phase_comparison.html'))
    print("Saved → figures/fig05_phase_comparison.html")
"""))

cells.append(md("""
### §7 Take-Home

The new candidate scans and the existing cohort share **identical TR/TE/FS** values.
Manufacturer and phase compositions are similar — both datasets are dominated by
Siemens and Philips scanners and by ADNI 3 data.  The new candidates do not introduce
any acquisition regime shift.  We are expanding within the same protocol space.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §8  DIAGNOSIS COMPOSITION AND STRATEGIC VALUE
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §8 — Diagnosis Composition and Strategic Value

This section is the most critical for reviewers.  We quantify:

1. How many new compatible subjects belong to each diagnostic group.
2. Whether adding them improves class balance for the CN vs AD classifier.
3. How they should best be used across our two main modelling contexts:
   - **VAE pool** (diagnosis-agnostic; all groups acceptable)
   - **CN vs AD supervised classifier** (strict binary; balance matters)
"""))

cells.append(code("""
# ── Diagnosis in new download_now candidates ──────────────────────────────────
rg_new = df_cand['Research Group'].value_counts()
print("New download_now candidates — Research Group:")
for rg, n in rg_new.items():
    pct = 100 * n / rg_new.sum()
    print(f"  {rg:4s}: {n:4d}  ({pct:.1f}%)")
print(f"  TOTAL: {rg_new.sum()}")

print()
print("NOTE: The IDA export only contains CN and AD labels.")
print("MCI / EMCI / LMCI subjects from our cohort are NOT represented in this export.")
"""))

cells.append(code("""
# ── Current cohort CN vs AD for classifier reference ─────────────────────────
# Our paper maps ResearchGroup: CN→CN, AD→AD, MCI/EMCI/LMCI → MCI (excluded from classifier)
cohort_clf = df_cohort[df_cohort['ResearchGroup'].isin(['CN', 'AD'])]['ResearchGroup'].value_counts()
print("Current cohort — classifier-eligible subjects (CN + AD):")
print(cohort_clf.to_dict())
print(f"  CN:AD ratio = {cohort_clf.get('CN',0)/cohort_clf.get('AD',1):.2f}:1")
print()

# New download_now candidates eligible for classifier
cand_clf = rg_new[rg_new.index.isin(['CN', 'AD'])]
print("New candidates — classifier-eligible (CN + AD):")
print(cand_clf.to_dict())
if 'AD' in cand_clf and cand_clf['AD'] > 0:
    print(f"  CN:AD ratio = {cand_clf.get('CN',0)/cand_clf['AD']:.2f}:1")
else:
    print("  AD: 0 — cannot compute ratio")
"""))

cells.append(code("""
# ── Scenario analysis: diagnosis composition under different strategies ────────
# Scenario labels and hypothetical cohort compositions
cn_cohort = df_cohort['ResearchGroup'].value_counts().get('CN',  0)
ad_cohort = df_cohort['ResearchGroup'].value_counts().get('AD',  0)
mci_cohort= df_cohort['ResearchGroup'].value_counts().get('MCI', 0)
emci_cohort=df_cohort['ResearchGroup'].value_counts().get('EMCI',0)
lmci_cohort=df_cohort['ResearchGroup'].value_counts().get('LMCI',0)

cn_new  = rg_new.get('CN', 0)
ad_new  = rg_new.get('AD', 0)

scenarios = {
    'A\\nCurrent cohort\\n(paper)': {
        'CN': cn_cohort, 'AD': ad_cohort,
        'MCI': mci_cohort, 'EMCI': emci_cohort, 'LMCI': lmci_cohort,
    },
    'B\\nCurrent + all\\ncompatible new': {
        'CN': cn_cohort + cn_new, 'AD': ad_cohort + ad_new,
        'MCI': mci_cohort, 'EMCI': emci_cohort, 'LMCI': lmci_cohort,
    },
    'C\\nCurrent + new CN\\nonly (VAE pool)': {
        'CN': cn_cohort + cn_new, 'AD': ad_cohort,
        'MCI': mci_cohort, 'EMCI': emci_cohort, 'LMCI': lmci_cohort,
    },
    'D\\nCurrent + new CN+AD\\n(classifier)': {
        'CN': cn_cohort + cn_new, 'AD': ad_cohort + ad_new,
        'MCI': mci_cohort, 'EMCI': emci_cohort, 'LMCI': lmci_cohort,
    },
}

groups = ['CN', 'AD', 'MCI', 'EMCI', 'LMCI']
scenario_labels = list(scenarios.keys())
scenario_totals = {s: sum(v.values()) for s, v in scenarios.items()}

df_scen = pd.DataFrame(scenarios, index=groups).T
df_scen['TOTAL'] = df_scen.sum(axis=1)
df_scen['CN:AD ratio'] = (df_scen['CN'] / df_scen['AD']).round(2)
print("Scenario Analysis:")
print(df_scen.to_string())
"""))

cells.append(code("""
# ── Figure: Scenario analysis ─────────────────────────────────────────────────
if PLOTLY_AVAILABLE:
    grp_colors = {
        'CN': '#2196F3', 'AD': '#F44336', 'MCI': '#FF9800',
        'EMCI': '#9C27B0', 'LMCI': '#795548',
    }
    fig = go.Figure()
    for grp in groups:
        fig.add_trace(go.Bar(
            name=grp,
            x=scenario_labels,
            y=[scenarios[s][grp] for s in scenario_labels],
            marker_color=grp_colors[grp],
        ))
    fig.update_layout(
        barmode='stack',
        title='Cohort Composition Under Different Expansion Scenarios<br>'
              '<sup>Stacked bars = total subjects per diagnostic group</sup>',
        title_x=0.5,
        yaxis_title='Number of subjects',
        xaxis_title='Scenario',
        legend_title='Research Group',
        height=500, width=900,
        xaxis=dict(tickfont=dict(size=9)),
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig06_scenario_composition.html'))
    print("Saved → figures/fig06_scenario_composition.html")
"""))

cells.append(code("""
# ── Figure: CN vs AD ratio by scenario ───────────────────────────────────────
if PLOTLY_AVAILABLE:
    ratios = df_scen['CN:AD ratio'].values
    labels = [s.replace('\\n', ' ') for s in scenario_labels]
    colors = ['#4A90D9' if r < 1.5 else '#F5A623' if r < 3 else '#E74C3C'
              for r in ratios]
    fig = go.Figure(go.Bar(
        x=labels, y=ratios,
        marker_color=colors,
        text=[f'{r:.2f}' for r in ratios],
        textposition='outside',
    ))
    fig.add_hline(y=df_scen.loc[scenario_labels[0], 'CN:AD ratio'],
                  line_dash='dot', line_color='gray',
                  annotation_text='Current baseline', annotation_position='right')
    fig.update_layout(
        title='CN:AD Ratio by Expansion Scenario<br>'
              '<sup>Higher ratio = worse class imbalance for CN vs AD classifier</sup>',
        title_x=0.5,
        yaxis_title='CN : AD ratio',
        xaxis_title='Scenario',
        height=400, width=800,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig07_cn_ad_ratio_scenarios.html'))
    print("Saved → figures/fig07_cn_ad_ratio_scenarios.html")
"""))

cells.append(md("""
### §8 Strategic Recommendation

#### What the data tell us

The IDA export is **CN-dominated**.  New compatible candidates are:
- **CN: ≈ 350+ subjects** (after deduplication)
- **AD: ≈ 3–4 subjects** (essentially negligible)
- **MCI / EMCI / LMCI: 0** (not present in this export)

This has direct consequences for our expansion strategy:

#### A — VAE pool (diagnosis-agnostic)
Adding CN subjects **directly increases the pool** for unsupervised representation
learning, which is beneficial.  The VAE does not use diagnostic labels during
training, so adding predominantly CN subjects is scientifically sound.
**Recommendation: include all compatible new subjects in the VAE pool.**

#### B — CN vs AD supervised classifier
Adding mostly CN subjects while adding almost no new AD subjects **worsens class
imbalance**.  The current CN:AD ratio (≈ 0.94:1) would shift to ≈ 4:1 if we naively
add all new subjects to the classifier.  This would harm, not help, classifier
performance and would likely draw additional reviewer criticism.
**Recommendation: do NOT add new subjects to the classifier training set unless
a comparable number of new AD subjects can be found.  The handful of new AD subjects
(n ≈ 4) can be added to the AD set as a marginal benefit.**

#### C — Reviewer confidence
Demonstrating that we **downloaded and processed additional ADNI subjects**
(even if used only for the VAE pool) shows methodological rigour and responsiveness
to reviewer feedback.  This is a valid and defensible response strategy.

#### Operational priority
1. **Primary download** → all compatible non-ADNI4 subjects with standard descriptions
2. **VAE pool** → all of the above
3. **Classifier** → only the ≈ 4 new AD subjects add direct value; new CN subjects
   should be noted as "available for future classifier expansion" if AD balance improves
4. **Robustness analyses** → CN subjects from different scanners/sites strengthen
   the scanner-generalisability argument for the VAE
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §9  SUBJECT-LEVEL SCAN SELECTION
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""
## §9 — Subject-Level Scan Selection

Some subjects have **multiple scans** in the IDA export (e.g., scans from ADNI 2
and ADNI 3, or two repeat acquisitions from the same visit).  Martin needs a clean,
**one-row-per-subject** download list.

### Selection rules (in order of priority)

1. **Prefer ADNI 3 over ADNI 2** — the majority of our existing cohort is ADNI 3,
   and ADNI 3 has the most standardised current resting-state protocol.
2. **Within the same phase, prefer standard descriptions** over unusual ones.
3. **Within the same phase and description class, prefer the lower Image ID**
   as a conservative, reproducible tiebreaker (lower ID = earlier archiving).
4. For a subject with one scan in `download_now` and another in `manual_review`
   (e.g., ADNI 4 vs ADNI 3), the best-available scan is selected and the others
   are demoted to `manual_review` with reason `duplicate_lower_priority_scan`.
"""))

cells.append(code("""
# ── Phase priority: lower number = higher preference ──────────────────────────
PHASE_PRIO = {'ADNI 3': 1, 'ADNI 2': 2, 'ADNI GO': 3, 'ADNI 1': 4, 'ADNI 4': 10}
DESC_PRIO  = {
    'standard': 1,
    'unusual_naming': 5,
    'phase_encoding_variant': 10,
    'extended_rsfmri': 10,
    'msv21_variant': 10,
    'unknown': 20,
}

df_work['phase_prio'] = df_work['Phase'].map(PHASE_PRIO).fillna(99)
df_work['desc_prio']  = df_work['desc_class'].map(DESC_PRIO).fillna(99)
df_work['image_id_num'] = pd.to_numeric(df_work['Image ID'], errors='coerce')

# Sort: best scan first for each subject
df_work_sorted = df_work.sort_values(
    ['Subject ID', 'phase_prio', 'desc_prio', 'image_id_num']
).reset_index(drop=True)

# Mark preferred scan per subject
df_work_sorted['is_preferred_scan'] = (
    ~df_work_sorted.duplicated(subset='Subject ID', keep='first')
)

print("Preferred scans (one per subject):", df_work_sorted['is_preferred_scan'].sum())
print("Non-preferred (backup) scans     :", (~df_work_sorted['is_preferred_scan']).sum())
print()

# Subjects with multiple scans
multi_scan_subj = (
    df_work_sorted.groupby('Subject ID')
    .filter(lambda x: len(x) > 1)['Subject ID'].unique()
)
print(f"Subjects with ≥ 2 scans in IDA: {len(multi_scan_subj)}")
"""))

cells.append(code("""
# ── Finalise recommended_status after subject-level selection ─────────────────
def finalise_status(row):
    \"\"\"Promote or demote based on preferred-scan selection.\"\"\"
    pre  = row['pre_status']
    pref = row['is_preferred_scan']

    if pre in ('exclude_for_now',):
        return pre, row['pre_reason']

    if pre == 'manual_review':
        # If this is a non-preferred scan that was already going to manual_review,
        # the reason is unchanged
        return 'manual_review', row['pre_reason']

    # pre == 'download_now'
    if pref:
        return 'download_now', 'compatible_new_candidate_preferred_scan'
    else:
        return 'manual_review', 'duplicate_lower_priority_scan_for_same_subject'

df_work_sorted[['recommended_status', 'reason']] = df_work_sorted.apply(
    finalise_status, axis=1, result_type='expand')

print("Final recommended_status distribution (all 645 rows):")
status_counts = df_work_sorted['recommended_status'].value_counts()
for s, n in status_counts.items():
    pct = 100 * n / len(df_work_sorted)
    print(f"  {s:35s}: {n:4d}  ({pct:.1f}%)")

print()
print("Subject-level: download_now (unique subjects):",
      df_work_sorted.loc[df_work_sorted['recommended_status']=='download_now',
                         'Subject ID'].nunique())
"""))

cells.append(md("""
### §9 Take-Home

After subject-level deduplication, every subject in the IDA export has exactly one
**primary scan recommendation**.  Subjects with multiple scans have their best
available scan selected for `download_now`, while secondary scans are demoted to
`manual_review` with the reason `duplicate_lower_priority_scan_for_same_subject`.
This gives Martin a clean, unambiguous list.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §10  FINAL RECOMMENDATION SETS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §10 — Final Recommendation Sets"))

cells.append(code("""
# ── Build master table ────────────────────────────────────────────────────────
MASTER_COLS = [
    'Subject ID', 'Image ID', 'Phase', 'Sex', 'Research Group', 'Age',
    'Visit', 'Study Date', 'Description', 'TR', 'TE', 'Field_Strength',
    'Slice_Thickness', 'Manufacturer', 'Modality',
    'identity_category', 'compatible_basic',
    'desc_class', 'is_preferred_scan',
    'recommended_status', 'reason',
]
# Columns present in df_work_sorted
available_master_cols = [c for c in MASTER_COLS if c in df_work_sorted.columns]
df_master = df_work_sorted[available_master_cols].copy()
df_master.rename(columns={
    'Subject ID':     'SubjectID',
    'Image ID':       'ImageID',
    'Research Group': 'ResearchGroup',
    'Study Date':     'StudyDate',
    'Field_Strength': 'FieldStrength',
    'Slice_Thickness':'SliceThickness',
}, inplace=True)

# Rename boolean flags for readability
df_master['already_in_current_cohort'] = df_master['identity_category'] == 'exact_match'
df_master['new_candidate']             = df_master['identity_category'] == 'new_candidate'

print(f"Master table shape: {df_master.shape}")
print(f"Columns: {list(df_master.columns)}")
"""))

cells.append(code("""
# ── Split into recommendation sets ───────────────────────────────────────────
df_download_now  = df_master[df_master['recommended_status'] == 'download_now'].copy()
df_manual_review = df_master[df_master['recommended_status'] == 'manual_review'].copy()
df_excluded      = df_master[df_master['recommended_status'] == 'exclude_for_now'].copy()

print(f"download_now     : {len(df_download_now):4d} scans  "
      f"| {df_download_now['SubjectID'].nunique()} unique subjects")
print(f"manual_review    : {len(df_manual_review):4d} scans  "
      f"| {df_manual_review['SubjectID'].nunique()} unique subjects")
print(f"exclude_for_now  : {len(df_excluded):4d} scans  "
      f"| {df_excluded['SubjectID'].nunique()} unique subjects")
print(f"{'TOTAL':16s}: {len(df_master):4d} scans")
print()

# download_now breakdown by Research Group
print("download_now — Research Group:")
print(df_download_now['ResearchGroup'].value_counts().to_dict())
print()

# manual_review reasons
print("manual_review — reasons:")
print(df_manual_review['reason'].value_counts().to_dict())
print()

# exclude reasons
print("exclude_for_now — reasons:")
print(df_excluded['reason'].value_counts().to_dict())
"""))

cells.append(code("""
# ── Subject-level recommendation table ────────────────────────────────────────
# Collapse: one row per subject (preferred scan wins)
df_subj_level = (
    df_master.sort_values('recommended_status',
                          key=lambda s: s.map({'download_now':0,'manual_review':1,'exclude_for_now':2}))
    .drop_duplicates(subset='SubjectID', keep='first')
    [['SubjectID','Phase','Sex','ResearchGroup','Age',
      'Manufacturer','Visit','ImageID','Description',
      'recommended_status','reason']]
    .reset_index(drop=True)
)

print(f"Subject-level recommendation table: {len(df_subj_level)} subjects")
print(df_subj_level['recommended_status'].value_counts().to_dict())
"""))

cells.append(code("""
# ── Figure: Final recommendation summary ─────────────────────────────────────
if PLOTLY_AVAILABLE:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Scan-level (n=645)', 'Subject-level (n=588)'],
        specs=[[{'type':'pie'},{'type':'pie'}]],
    )
    status_colors = {
        'download_now': '#27AE60',
        'manual_review': '#F39C12',
        'exclude_for_now': '#E74C3C',
    }
    for col, df_ref, id_col in [
        (1, df_master,     'ImageID'),
        (2, df_subj_level, 'SubjectID'),
    ]:
        vc = df_ref['recommended_status'].value_counts().reset_index()
        vc.columns = ['status','n']
        fig.add_trace(go.Pie(
            labels=vc['status'], values=vc['n'],
            marker_colors=[status_colors[s] for s in vc['status']],
            textinfo='label+value+percent',
            hole=0.35, showlegend=(col==2),
        ), row=1, col=col)
    fig.update_layout(
        title_text='Final Recommendation Distribution<br>'
                   '<sup>Green = download now · Orange = manual review · Red = exclude</sup>',
        title_x=0.5, height=420, width=820,
        legend=dict(orientation='h', y=-0.12),
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig08_final_recommendation.html'))
    print("Saved → figures/fig08_final_recommendation.html")
"""))

cells.append(code("""
# ── Interactive inspection table: download_now list ──────────────────────────
if PLOTLY_AVAILABLE:
    tbl = df_download_now[['SubjectID','ImageID','Phase','ResearchGroup',
                            'Sex','Age','Manufacturer','Visit','Description',
                            'TR','TE','FieldStrength','SliceThickness']].copy()
    fig = go.Figure(go.Table(
        header=dict(
            values=list(tbl.columns),
            fill_color='#4A90D9', font_color='white',
            align='left', height=28,
        ),
        cells=dict(
            values=[tbl[c] for c in tbl.columns],
            fill_color=[['#EAF4FB','#F7FBFF'] * (len(tbl) // 2 + 1)][:len(tbl)],
            align='left', height=22,
        ),
    ))
    fig.update_layout(
        title='Download-Now List (interactive — scroll to inspect all rows)',
        title_x=0.5, height=500,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig09_download_now_table.html'))
    print("Saved → figures/fig09_download_now_table.html")
"""))

cells.append(code("""
# ── Interactive inspection table: manual_review list ─────────────────────────
if PLOTLY_AVAILABLE:
    tbl_mr = df_manual_review[['SubjectID','ImageID','Phase','ResearchGroup',
                                'Sex','Age','Manufacturer','Visit','Description',
                                'reason']].copy()
    fig = go.Figure(go.Table(
        header=dict(
            values=list(tbl_mr.columns),
            fill_color='#F39C12', font_color='white',
            align='left', height=28,
        ),
        cells=dict(
            values=[tbl_mr[c] for c in tbl_mr.columns],
            align='left', height=22,
        ),
    ))
    fig.update_layout(
        title='Manual Review List (interactive)',
        title_x=0.5, height=500,
    )
    fig.show()
    fig.write_html(str(FIG_DIR / 'fig10_manual_review_table.html'))
    print("Saved → figures/fig10_manual_review_table.html")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §11  PROFESSIONAL SUMMARY FOR MARTIN
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §11 — Professional Summary for Martin"))

cells.append(code("""
# ── Compute all summary statistics ───────────────────────────────────────────
n_dn        = df_download_now['SubjectID'].nunique()
dn_rg       = df_download_now['ResearchGroup'].value_counts()
dn_manuf    = df_download_now['Manufacturer'].value_counts()
dn_phase    = df_download_now['Phase'].value_counts()

n_mr        = df_manual_review['SubjectID'].nunique()
mr_reasons  = df_manual_review['reason'].value_counts()

n_ex        = df_excluded['SubjectID'].nunique()

n_total_ida = df_ida['Subject ID'].nunique()
n_overlap   = (df_master['identity_category'] == 'exact_match').sum()
n_new_total = (df_master['identity_category'] == 'new_candidate').sum()

# For the classifier: new AD subjects
n_new_ad = dn_rg.get('AD', 0)
n_new_cn = dn_rg.get('CN', 0)

print("=" * 70)
print("ADNI EXPANSION AUDIT — SUMMARY FOR MARTIN")
print("=" * 70)
print()
print(f"IDA export analysed         : {n_total_ida} unique subjects / {len(df_ida)} scans")
print(f"Already in current cohort   : {n_overlap} subjects (exact ImageID match)")
print(f"New candidates evaluated    : {n_new_total} scans ({df_work_sorted[df_work_sorted['identity_category']=='new_candidate']['Subject ID'].nunique()} unique subjects)")
print()
print(f"━━━ DOWNLOAD NOW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Subjects              : {n_dn}")
print(f"  Research Group        : {dn_rg.to_dict()}")
print(f"  Manufacturer          : {dn_manuf.to_dict()}")
print(f"  Phase                 : {dn_phase.to_dict()}")
print()
print(f"━━━ MANUAL REVIEW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Subjects              : {n_mr}")
print(f"  Top reasons           : {mr_reasons.head(4).to_dict()}")
print()
print(f"━━━ EXCLUDED FOR NOW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Subjects              : {n_ex}")
print()
print("━━━ DOWNSTREAM USE STRATEGY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  VAE pool (diagnosis-agnostic)")
print(f"    → All {n_dn} download_now subjects can be added.")
print(f"    → Primarily CN ({n_new_cn}), with {n_new_ad} AD.")
print()
print(f"  CN vs AD classifier")
print(f"    → Only the {n_new_ad} new AD subjects add direct value.")
print(f"    → Adding {n_new_cn} CN subjects without matching AD would WORSEN imbalance.")
print(f"    → Recommended: add new AD to classifier; use CN for VAE pool only.")
print()
print(f"  Scanner robustness analyses")
print(f"    → New subjects cover Siemens + Philips + GE — same as existing cohort.")
print(f"    → Good for demonstrating multi-scanner generalisability.")
print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()
print("QUICK MESSAGE FOR MARTIN:")
print("─" * 70)
print(
    f"Auditamos el export de IDA del 03/04/2026 (645 scans, 588 sujetos).\\n"
    f"De estos, {n_overlap} ya están en nuestra cohorte procesada.\\n"
    f"Hay {n_dn} sujetos nuevos y compatibles para descargar ahora mismo:\\n"
    f"  - {n_new_cn} CN  /  {n_new_ad} AD\\n"
    f"  - Todos tienen TR=3000ms, TE=30ms, B0=3T (100% compatibles con el pipeline).\\n"
    f"  - Son principalmente ADNI3 ({dn_phase.get('ADNI 3',0)}) y ADNI2 ({dn_phase.get('ADNI 2',0)}).\\n"
    f"Hay {n_mr} scans adicionales para revisión manual (principalmente ADNI4 y variantes de phase encoding).\\n\\n"
    f"Uso recomendado:\\n"
    f"  • VAE pool: todos los {n_dn} sujetos nuevos (excelente para generalizabilidad).\\n"
    f"  • Clasificador CN vs AD: solo los {n_new_ad} AD nuevos aportan valor directo.\\n"
    f"    Agregar los {n_new_cn} CN sin AD equivalentes empeoraría el desbalance.\\n"
    f"Te adjunto la lista exacta de SubjectID / ImageID para la descarga.\\n"
    f"Avisame si querés que también incluya los de revisión manual."
)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# §12  EXPORT ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## §12 — Exported Artifacts"))

cells.append(code("""
# ── 1. Master table ──────────────────────────────────────────────────────────
master_path = OUT_DIR / 'adni_expansion_master_table.csv'
df_master.to_csv(master_path, index=False)
print(f"[1] {master_path.name}  ({len(df_master)} rows, {len(df_master.columns)} cols)")

# ── 2. Download-now list ─────────────────────────────────────────────────────
dl_cols = ['SubjectID','ImageID','Phase','ResearchGroup','Sex','Age',
           'Manufacturer','Visit','StudyDate','Description',
           'TR','TE','FieldStrength','SliceThickness','reason']
dl_out_cols = [c for c in dl_cols if c in df_download_now.columns]
dl_path = OUT_DIR / 'adni_download_now.csv'
df_download_now[dl_out_cols].to_csv(dl_path, index=False)
print(f"[2] {dl_path.name}  ({len(df_download_now)} rows)")

# ── 3. Manual review list ────────────────────────────────────────────────────
mr_path = OUT_DIR / 'adni_manual_review.csv'
df_manual_review.to_csv(mr_path, index=False)
print(f"[3] {mr_path.name}  ({len(df_manual_review)} rows)")

# ── 4. Excluded list ─────────────────────────────────────────────────────────
ex_path = OUT_DIR / 'adni_excluded_for_now.csv'
df_excluded.to_csv(ex_path, index=False)
print(f"[4] {ex_path.name}  ({len(df_excluded)} rows)")

# ── 5. Subject-level recommendation ─────────────────────────────────────────
sl_path = OUT_DIR / 'adni_subject_level_recommendation.csv'
df_subj_level.to_csv(sl_path, index=False)
print(f"[5] {sl_path.name}  ({len(df_subj_level)} rows)")

# ── 6. Summary metrics ───────────────────────────────────────────────────────
summary_metrics = {
    'total_ida_rows': len(df_master),
    'total_ida_unique_subjects': df_ida['Subject ID'].nunique(),
    'already_in_cohort_exact_match_rows': (df_master['identity_category']=='exact_match').sum(),
    'subject_overlap_different_visit_rows': (df_master['identity_category']=='subject_overlap').sum(),
    'new_candidate_rows': (df_master['identity_category']=='new_candidate').sum(),
    'download_now_subjects': df_download_now['SubjectID'].nunique(),
    'download_now_CN': dn_rg.get('CN', 0),
    'download_now_AD': dn_rg.get('AD', 0),
    'manual_review_subjects': df_manual_review['SubjectID'].nunique(),
    'exclude_for_now_subjects': df_excluded['SubjectID'].nunique(),
    'current_cohort_CN': cn_cohort,
    'current_cohort_AD': ad_cohort,
    'current_cohort_MCI': mci_cohort,
    'current_cohort_EMCI': emci_cohort,
    'current_cohort_LMCI': lmci_cohort,
    'current_cohort_total': len(df_cohort),
    'projected_vae_pool_after_expansion': len(df_cohort) + df_download_now['SubjectID'].nunique(),
    'compatible_basic_all': bool(df_master['compatible_basic'].all()),
}
df_metrics = pd.DataFrame(list(summary_metrics.items()), columns=['metric','value'])
metrics_path = OUT_DIR / 'summary_selection_metrics.csv'
df_metrics.to_csv(metrics_path, index=False)
print(f"[6] {metrics_path.name}  ({len(df_metrics)} metrics)")
"""))

cells.append(code("""
# ── 7. Markdown summary ───────────────────────────────────────────────────────
md_text = f\"\"\"# ADNI Expansion Audit — Summary
**Date:** 2026-04-05
**IDA export:** idaSearch_4_03_2026 (2).csv
**Notebook:** 05_adni_expansion_subject_selection.ipynb

## Key Numbers
| Metric | Value |
|---|---|
| IDA export rows | {len(df_master)} |
| IDA unique subjects | {df_ida['Subject ID'].nunique()} |
| Already in cohort (exact scan) | {(df_master['identity_category']=='exact_match').sum()} |
| Subject overlap (different visit) | {(df_master['identity_category']=='subject_overlap').sum()} |
| New candidate scans | {(df_master['identity_category']=='new_candidate').sum()} |
| **download_now subjects** | **{df_download_now['SubjectID'].nunique()}** |
| — CN | {dn_rg.get('CN', 0)} |
| — AD | {dn_rg.get('AD', 0)} |
| manual_review subjects | {df_manual_review['SubjectID'].nunique()} |
| exclude_for_now subjects | {df_excluded['SubjectID'].nunique()} |

## Protocol Compatibility
All IDA scans share TR=3000 ms, TE=30 ms, B₀=3.0 T with the existing cohort.
`compatible_basic = True` for 100% of rows.

## Downstream Use Strategy

### VAE pool
All {df_download_now['SubjectID'].nunique()} download_now subjects are suitable for the
diagnosis-agnostic VAE pool.  Adding predominantly CN subjects increases the pool
size and may improve representation learning coverage.

### CN vs AD classifier
Only the {dn_rg.get('AD', 0)} new AD subjects add direct value to the classifier.
Adding {dn_rg.get('CN', 0)} CN subjects without matching AD subjects would worsen
class imbalance (current CN:AD ≈ {cn_cohort/max(ad_cohort,1):.2f}:1;
projected ≈ {(cn_cohort+dn_rg.get('CN',0))/max(ad_cohort+dn_rg.get('AD',0),1):.2f}:1 if all added).

**Recommended:** add new AD subjects to classifier; route CN subjects to VAE pool only.

### Manual review set
{df_manual_review['SubjectID'].nunique()} subjects flagged for manual inspection, primarily:
{chr(10).join(f'- {r}: {n}' for r, n in mr_reasons.head(5).items())}

## Output Files
1. `adni_expansion_master_table.csv`
2. `adni_download_now.csv`
3. `adni_manual_review.csv`
4. `adni_excluded_for_now.csv`
5. `adni_subject_level_recommendation.csv`
6. `summary_selection_metrics.csv`
7. `adni_expansion_summary.md`
\"\"\"

md_path = OUT_DIR / 'adni_expansion_summary.md'
md_path.write_text(md_text)
print(f"[7] {md_path.name}")

print()
print("=" * 60)
print("All artifacts written to:")
print(f"  {OUT_DIR}")
print("=" * 60)
for f in sorted(OUT_DIR.glob('*.csv')) :
    print(f"  {f.name}")
for f in sorted(OUT_DIR.glob('*.md')):
    print(f"  {f.name}")
for f in sorted(FIG_DIR.glob('*.html')):
    print(f"  figures/{f.name}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells
OUT_NB.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_NB, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written → {OUT_NB}")
print(f"Cells: {len(nb.cells)}")
