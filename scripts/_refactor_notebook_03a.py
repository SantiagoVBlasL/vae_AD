#!/usr/bin/env python3
"""
Programmatic refactoring of 03_a_inference_covid_from_adcn.ipynb.

This script applies all mandatory fixes and inserts the fatigue analysis.
Run once, then delete this script.
"""
import json, copy, uuid, sys
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "03_a_inference_covid_from_adcn.ipynb"

def make_cell(cell_type, source_str, cell_id=None):
    """Create a notebook cell dict."""
    cid = cell_id or uuid.uuid4().hex[:8]
    lines = (source_str + "\n").splitlines(True)
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "id": cid,
            "metadata": {},
            "source": lines,
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": cid,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }

# Load
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

# ── Build ID→index map ────────────────────────────────────────────
id_map = {}
for i, c in enumerate(cells):
    cid = c.get("id", "")
    # The ids in the notebook may have prefixes like #VSC-
    id_map[cid] = i

def find_cell(partial_id):
    """Find cell index by partial id match."""
    for cid, idx in id_map.items():
        if partial_id in cid:
            return idx
    return None

def get_source(cell):
    return "".join(cell["source"])

def set_source(cell, txt):
    cell["source"] = (txt + "\n").splitlines(True)
    # Clear outputs for code cells
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# ══════════════════════════════════════════════════════════════════
# FIX 1: Rewrite intro markdown (cell 2)
# ══════════════════════════════════════════════════════════════════
idx = find_cell("eed99c1f")
if idx is not None:
    set_source(cells[idx], r"""# COVID → AD Transfer Inference — Paper-Grade Analysis

**Objective.** Apply a β-VAE + classical-classifier pipeline trained on
ADNI (AD vs CN, 5-fold nested CV) to a Long-COVID cohort (N ≈ 194)
and evaluate AD-likeness through clinical, out-of-distribution (OOD),
and connectomic lenses.

## Scientific stance

The transferred score is an **AD-likeness index** — not a diagnostic
probability in the COVID domain.  It is difficult to robustly
discriminate "had COVID ever" vs control from rs-fMRI alone in this
cohort, so we do **not** oversell COVID-vs-control group separation.
A more plausible signal may exist for **fatigue phenotype**
(CategoriaFAS), especially FATIGA EXTREMA vs NO HAY FATIGA.
Null findings are scientifically valuable and are reported honestly.

## Key scientific questions

| # | Question | Sections |
|---|----------|----------|
| 1 | Do Long-COVID subjects show elevated AD-like connectivity? | §4, §6 |
| 2 | Is AD-likeness associated with cognitive decline (MOCA)? | §7 |
| 3 | Are AD-like subjects in-distribution relative to ADNI? | §5 |
| 4 | Is AD-likeness associated with fatigue burden (CategoriaFAS)? | **§10 (new)** |
| 5 | Which connectivity pathways drive AD-likeness? | §8, §9 |

## Notebook sections

| § | Section | Type |
|---|---------|------|
| 0 | Configuration | Infrastructure |
| 1 | Setup & imports | Infrastructure |
| 2 | Artifact discovery & stale-output guard | Data loading |
| 3 | Data integrity / physics QC | QC |
| 4 | Core inference results (scores, stability) | Descriptive |
| 5 | OOD diagnostics (ADNI P95 recon-error primary gate) | Methodological |
| 6 | Threshold policy (ADNI-derived) | Methodological |
| 7 | Clinical validation (MOCA, severity, recovery) | **Primary** |
| 8 | Signature decomposition by Yeo-17 networks | Mechanistic |
| 9 | Edge-level connectomic characterisation | Exploratory |
| 10 | **Fatigue phenotype analysis (CategoriaFAS)** | **Primary (new)** |
| 11 | Subject selection for interpretability | Selection |
| QC | Calibration drift & sensitivity | QC |
| F | Outputs index & provenance | Reproducibility |

> **Methodological notes.**
> - §10 and §14 from the prior notebook version (latent-space UMAP and
>   K-Means clustering on averaged latent μ) have been **removed** because
>   element-wise averaging of latent vectors across independently trained
>   fold-specific VAEs is not valid (latent axes are not identifiable).
> - Network connectivity summaries are now **channel-specific** (OMST as
>   primary channel), removing the prior invalid cross-channel average.
> - The primary OOD gate is ADNI P95 reconstruction error.
> - "Risk categories" have been renamed to **AD-Likeness categories**
>   to avoid clinical misinterpretation.""")

# ══════════════════════════════════════════════════════════════════
# FIX 2: Update §0 config — add NOTEBOOK_VERSION
# ══════════════════════════════════════════════════════════════════
idx0 = find_cell("7109a128")
if idx0 is not None:
    src0 = get_source(cells[idx0])
    # Add version and primary OOD channel at the top
    if "NOTEBOOK_VERSION" not in src0:
        src0 = (
            '# Notebook version (used for output namespacing)\n'
            'NOTEBOOK_VERSION = "v5.0"\n\n'
            '# Primary connectivity channel for network summaries\n'
            'PRIMARY_CHANNEL_IDX = 0  # OMST (Pearson_OMST_GCE_Signed_Weighted)\n\n'
        ) + src0
        set_source(cells[idx0], src0)

# ══════════════════════════════════════════════════════════════════
# FIX 3: §2 — add stale output guard
# ══════════════════════════════════════════════════════════════════
idx2 = find_cell("b0c84e07")
if idx2 is not None:
    src2 = get_source(cells[idx2])
    # Replace OUTPUT_DIR definition to include version namespacing
    src2 = src2.replace(
        'OUTPUT_DIR  = RESULTS_DIR / "inference_covid_paper_output"',
        'OUTPUT_DIR  = RESULTS_DIR / "inference_covid_paper_output"\n'
        '# Version-namespaced subdir for this notebook run\n'
        '_VERSION_TAG = NOTEBOOK_VERSION if "NOTEBOOK_VERSION" in dir() else "v5.0"\n'
    )
    set_source(cells[idx2], src2)

# ══════════════════════════════════════════════════════════════════
# FIX 4: Rewrite §5 OOD markdown — simple ADNI P95 gate
# ══════════════════════════════════════════════════════════════════
idx5md = find_cell("d24a1968")
if idx5md is not None:
    set_source(cells[idx5md], r"""## §5 — OOD / Domain Shift Diagnostics

How "in-distribution" are COVID subjects relative to ADNI?

**Primary OOD gate: ADNI P95 reconstruction error.**
A subject is flagged as OOD if its off-diagonal MAE reconstruction error
exceeds the 95th percentile of reconstruction errors observed in the
ADNI training-development set.  This is a simple, robust, and
interpretable gate.

**Supplementary diagnostics** (descriptive only, not used for gating):
- **PCA-Mahalanobis distance** in the fold-specific latent space.
- **Cross-fold σ** (score disagreement across folds).

> Mahalanobis distance and the former joint recon+Mahalanobis gate are
> retained only as supplementary diagnostics because the joint gate
> collapsed to near-zero OOD rates and was not useful as a primary filter.""")

# ══════════════════════════════════════════════════════════════════
# FIX 5: §5 code — simplify OOD to ADNI P95 recon primary
# ══════════════════════════════════════════════════════════════════
idx5c = find_cell("4812b9da")
if idx5c is not None:
    set_source(cells[idx5c], r"""# ═══════════════════════════════════════════════════════════════════════
# §5  OOD DIAGNOSTICS — ADNI P95 recon error as primary gate
# ═══════════════════════════════════════════════════════════════════════

# ── Derive t_primary (Youden) from ADNI OOF ──
if "threshold_policies" in dir() and threshold_policies:
    t_primary = threshold_policies.get("Youden", {}).get("threshold", 0.5)
    print(f"  → Using Youden from threshold_policies: {t_primary:.4f}")
elif adni_oof_all is not None:
    from sklearn.metrics import roc_curve as _roc_curve
    _oof = adni_oof_all[adni_oof_all["classifier_type"] == TARGET_CLF]
    _fpr, _tpr, _thr = _roc_curve(_oof["y_true"], _oof["y_score_final"])
    t_primary = float(_thr[np.argmax(_tpr - _fpr)])
    print(f"  → Youden threshold (inline from ADNI OOF): {t_primary:.4f}")
else:
    t_primary = 0.5
    print("  ⚠ ADNI OOF unavailable. Fallback 0.5.")

# ── Load ADNI-derived OOD distribution ──
_ood_dist_path = TABLES_DIR / "adni_ood_distribution.csv"
_ADNI_THR = {}

if _ood_dist_path.exists():
    _adni_dist = pd.read_csv(_ood_dist_path)
    _adni_ens = _adni_dist[_adni_dist["fold"] == "ensemble_median"]
    for _, row in _adni_ens.iterrows():
        m = row["metric"]
        _ADNI_THR[m] = {
            "p90": float(row["p90"]), "p95": float(row["p95"]),
            "p99": float(row["p99"]), "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
    print(f"  → ADNI OOD thresholds loaded:")
    for m, v in _ADNI_THR.items():
        print(f"    {m:20s}: P95={v['p95']:.4f}")
else:
    print("  ⚠ ADNI OOD distribution not found. Run: python scripts/ood_policy_adni.py")

# Primary gate threshold
_adni_ood_p95 = _ADNI_THR.get("recon_error", {}).get("p95")
_adni_ood_p90 = _ADNI_THR.get("recon_error", {}).get("p90")

if _adni_ood_p95 is not None:
    print(f"\n  ★ PRIMARY OOD GATE: ADNI P95 recon error = {_adni_ood_p95:.4f}")
else:
    print("  ⚠ Cannot define primary OOD gate (ADNI recon P95 not available).")

for clf in CLASSIFIER_TYPES:
    sub = ensemble[ensemble["classifier"] == clf].copy()
    sub = sub.merge(recon_ens, on="SubjectID", how="left")
    sub = sub.merge(dist_ens, on="SubjectID", how="left")

    _score_col = "y_score_ensemble"
    _ood_col = "recon_error_ensemble"

    # ── PRIMARY OOD flag: ADNI P95 recon error ──
    if _adni_ood_p95 is not None:
        sub["ood_flag_primary"] = (sub[_ood_col] >= _adni_ood_p95).astype(int)
        _n_ood = sub["ood_flag_primary"].sum()
        print(f"\n  PRIMARY OOD (ADNI P95): {_n_ood}/{len(sub)} "
              f"({100*_n_ood/len(sub):.1f}%) flagged")
    else:
        # Fallback: within-cohort P95
        _p95_fallback = sub[_ood_col].quantile(0.95)
        sub["ood_flag_primary"] = (sub[_ood_col] >= _p95_fallback).astype(int)
        print(f"  Fallback OOD (within-cohort P95={_p95_fallback:.4f})")

    # ── Quadrant assignment (primary gate) ──
    sub["quadrant"] = "CN-like_InDist"
    sub.loc[(sub[_score_col] >= t_primary) & (sub["ood_flag_primary"] == 0),
            "quadrant"] = "AD-like_InDist"
    sub.loc[(sub[_score_col] >= t_primary) & (sub["ood_flag_primary"] == 1),
            "quadrant"] = "AD-like_OOD"
    sub.loc[(sub[_score_col] < t_primary) & (sub["ood_flag_primary"] == 1),
            "quadrant"] = "CN-like_OOD"

    # ── Supplementary: ADNI P90 recon ──
    if _adni_ood_p90 is not None:
        sub["ood_flag_adni_p90"] = (sub[_ood_col] >= _adni_ood_p90).astype(int)

    # ── Supplementary: Mahalanobis (descriptive only) ──
    has_dist = "latent_distance_ensemble" in sub.columns
    _mdist_col = "latent_distance_ensemble"

    # ── Plot ──
    ncols = 3 if has_dist else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5*ncols, 5))

    # Panel 1: Score × Recon Error
    ax = axes[0]
    quad_colors = {"AD-like_InDist": "#E74C3C", "AD-like_OOD": "#F39C12",
                   "CN-like_InDist": "#3498DB", "CN-like_OOD": "#95A5A6"}
    for q, qc in quad_colors.items():
        mask = sub["quadrant"] == q
        if mask.sum() == 0: continue
        ax.scatter(sub.loc[mask, _score_col], sub.loc[mask, _ood_col],
                  c=qc, s=18, alpha=0.6, label=f"{q} (n={mask.sum()})",
                  edgecolors="none")
    if _adni_ood_p95 is not None:
        ax.axhline(_adni_ood_p95, color="darkgreen", ls="-.", lw=1.4,
                   label=f"★ ADNI-P95={_adni_ood_p95:.4f} (primary)")
    if _adni_ood_p90 is not None:
        ax.axhline(_adni_ood_p90, color="green", ls=":", lw=1.0, alpha=0.7,
                   label=f"ADNI-P90={_adni_ood_p90:.4f} (suppl.)")
    ax.axvline(t_primary, color="red", ls=":", lw=1.0, alpha=0.7,
               label=f"Youden={t_primary:.3f}")
    ax.set_xlabel("AD-Likeness Score"); ax.set_ylabel("Recon Error")
    ax.set_title(f"{clf.upper()} — Score × Recon (ADNI P95 primary gate)")
    ax.legend(fontsize=6, loc="upper left")

    # Panel 2: Recon error distribution
    ax = axes[1]
    ax.hist(sub[_ood_col], bins=30, alpha=0.7, color="#1ABC9C", edgecolor="white")
    if _adni_ood_p95 is not None:
        ax.axvline(_adni_ood_p95, color="darkgreen", ls="-.", lw=1.4,
                   label=f"ADNI-P95={_adni_ood_p95:.4f}")
    if _adni_ood_p90 is not None:
        ax.axvline(_adni_ood_p90, color="green", ls=":", lw=1.0,
                   label=f"ADNI-P90={_adni_ood_p90:.4f}")
    ax.set_xlabel("Recon Error"); ax.set_ylabel("Count")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend(fontsize=8)

    # Panel 3: Latent distance (supplementary)
    if has_dist:
        ax = axes[2]
        sc2 = ax.scatter(sub[_score_col], sub[_mdist_col],
                        c=sub["y_score_std"], cmap="magma", s=18, alpha=0.6,
                        edgecolors="none")
        plt.colorbar(sc2, ax=ax, label="Cross-fold σ")
        ax.set_xlabel("AD-Likeness Score")
        ax.set_ylabel("Latent Distance (supplementary)")
        ax.set_title(f"{clf.upper()} — Score × Latent Distance")
        ax.axvline(t_primary, color="red", ls=":", lw=1.0, alpha=0.7)

    plt.tight_layout()
    save_fig(fig, f"fig3_ood_diagnostics_{clf}.png", close=False)
    if SHOW_FIGURES: plt.show()
    else: plt.close()

    # Summary
    r_re, p_re = spearmanr(sub[_score_col], sub[_ood_col])
    print(f"\n{clf.upper()} OOD Summary:")
    print(f"  Primary gate: ADNI P95 recon error")
    print(f"  Recon error: {sub[_ood_col].mean():.4f} ± {sub[_ood_col].std():.4f}")
    print(f"  Spearman(score, recon) = {r_re:.4f}, p = {p_re:.4f}")
    print(f"\n  Quadrant counts (ADNI P95 primary gate):")
    for q in ["AD-like_InDist","AD-like_OOD","CN-like_InDist","CN-like_OOD"]:
        n = (sub["quadrant"] == q).sum()
        print(f"    {q:20s}: {n:4d} ({100*n/len(sub):.1f}%)")

    sub.to_csv(TABLES_DIR / f"covid_ood_quadrants_{clf}.csv", index=False)""")

# ══════════════════════════════════════════════════════════════════
# FIX 6: §6b — rename risk_category → adlikeness_category
# ══════════════════════════════════════════════════════════════════
idx6b = find_cell("46bd296d")
if idx6b is not None:
    src = get_source(cells[idx6b])
    # Replace risk_category with adlikeness_category and rename labels
    src = src.replace('"risk_category"', '"adlikeness_category"')
    src = src.replace("'risk_category'", "'adlikeness_category'")
    src = src.replace('risk_category', 'adlikeness_category')
    src = src.replace('"Low"', '"Low AD-Likeness"')
    src = src.replace('"Moderate"', '"Intermediate AD-Likeness"')
    src = src.replace('"High"', '"High AD-Likeness"')
    src = src.replace("'Low'", "'Low AD-Likeness'")
    src = src.replace("'Moderate'", "'Intermediate AD-Likeness'")
    src = src.replace("'High'", "'High AD-Likeness'")
    # Fix display labels
    src = src.replace("Risk Categories", "AD-Likeness Categories")
    src = src.replace("Risk categories", "AD-Likeness categories")
    src = src.replace("risk categories", "AD-Likeness categories")
    set_source(cells[idx6b], src)

# ══════════════════════════════════════════════════════════════════
# FIX 7: §7c — also rename risk references
# ══════════════════════════════════════════════════════════════════
idx7c = find_cell("fa3e136a")
if idx7c is not None:
    src = get_source(cells[idx7c])
    src = src.replace('"risk_category"', '"adlikeness_category"')
    src = src.replace("'risk_category'", "'adlikeness_category'")
    src = src.replace('risk_category', 'adlikeness_category')
    # Update plot order
    src = src.replace('["Low","Moderate","High"]', '["Low AD-Likeness","Intermediate AD-Likeness","High AD-Likeness"]')
    src = src.replace('"Low":"#2ECC71","Moderate":"#F39C12","High":"#E74C3C"',
                     '"Low AD-Likeness":"#2ECC71","Intermediate AD-Likeness":"#F39C12","High AD-Likeness":"#E74C3C"')
    set_source(cells[idx7c], src)

# ══════════════════════════════════════════════════════════════════
# FIX 7b: §7d — rename risk references
# ══════════════════════════════════════════════════════════════════
idx7d = find_cell("b804c945")
if idx7d is not None:
    src = get_source(cells[idx7d])
    src = src.replace('"risk_category"', '"adlikeness_category"')
    src = src.replace("'risk_category'", "'adlikeness_category'")
    src = src.replace('risk_category', 'adlikeness_category')
    set_source(cells[idx7d], src)

# ══════════════════════════════════════════════════════════════════
# FIX 8: §7a.1 — improve tensor-restricted cohort audit
# ══════════════════════════════════════════════════════════════════
idx7a1 = find_cell("e943347c")
if idx7a1 is not None:
    set_source(cells[idx7a1], r"""# ═══════════════════════════════════════════════════════════════════════
# §7a.1  COHORT DEFINITIONS — TENSOR-RESTRICTED (paper-critical)
# ═══════════════════════════════════════════════════════════════════════

clin_all = _clin_df.copy()

# Normalize ResearchGroup
clin_all["ResearchGroup"] = (
    clin_all["ResearchGroup"]
    .astype(str).str.strip().str.upper()
    .replace({"CTRL": "CONTROL", "HC": "CONTROL", "HEALTHY": "CONTROL"})
)

expected_groups = {"COVID", "CONTROL"}
observed_groups = set(clin_all["ResearchGroup"].dropna().unique())
bad = observed_groups - expected_groups
if bad:
    print(f"⚠ Unexpected ResearchGroup labels: {bad}")

clin_covid   = clin_all[clin_all["ResearchGroup"] == "COVID"].copy()
clin_control = clin_all[clin_all["ResearchGroup"] == "CONTROL"].copy()

# ── TENSOR-RESTRICTED COHORT AUDIT ──
_n_meta_raw = len(covid_meta_raw)
_n_tensor = len(covid_ids)
_n_merged = len(clin_all)
_n_meta_matched = covid_meta_raw["_merge_id"].isin(clin_all["SubjectID"]).sum()
_n_meta_unmatched = _n_meta_raw - _n_meta_matched

print("╔══════════════════════════════════════════════════════╗")
print("║  TENSOR-RESTRICTED COHORT AUDIT                     ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Raw metadata rows          : {_n_meta_raw:>5d}                 ║")
print(f"║  Tensor subjects (IDs)      : {_n_tensor:>5d}                 ║")
print(f"║  Matched after merge        : {_n_merged:>5d}                 ║")
print(f"║  Metadata rows NOT in tensor: {_n_meta_unmatched:>5d}                 ║")
print(f"║                                                      ║")
print(f"║  COVID (tensor-restricted)  : {len(clin_covid):>5d}                 ║")
print(f"║  CONTROL (tensor-restricted): {len(clin_control):>5d}                 ║")
print("╚══════════════════════════════════════════════════════╝")

# Missingness summary for key clinical variables
clinical_cols = ["MOCA", "MOCA_perc", "Age", "Sex", "CategoriaFAS",
                 "EQ-VAS", "BMI", "NivelEducativo"]
print("\nMissingness by variable (tensor-restricted, COVID only):")
for col in clinical_cols:
    if col in clin_covid.columns:
        n_miss = clin_covid[col].isna().sum()
        n_avail = len(clin_covid) - n_miss
        print(f"  {col:25s}: {n_avail:3d}/{len(clin_covid)} "
              f"(missing {n_miss})")

# Cross-tab: ResearchGroup × CategoríaCOVID
catcovid_col = None
for c in ["CategoríaCOVID", "CategoriaCOVID"]:
    if c in clin_all.columns:
        catcovid_col = c
        break
if catcovid_col:
    print(f"\nCross-tab: ResearchGroup × {catcovid_col}")
    print(pd.crosstab(clin_all["ResearchGroup"], clin_all[catcovid_col], dropna=False))

# CategoriaFAS cross-tab
if "CategoriaFAS" in clin_all.columns:
    print(f"\nCross-tab: ResearchGroup × CategoriaFAS (tensor-restricted)")
    print(pd.crosstab(clin_all["ResearchGroup"],
                      clin_all["CategoriaFAS"], dropna=False, margins=True))

print(f"\n⚠ All subsequent analyses use the TENSOR-RESTRICTED cohort "
      f"(N={_n_merged}), not raw metadata counts.")""")

# ══════════════════════════════════════════════════════════════════
# FIX 9: §8b — cardinality-normalized network decomposition
# ══════════════════════════════════════════════════════════════════
idx8b = find_cell("edef10a8")
if idx8b is not None:
    set_source(cells[idx8b], r"""# ═══════════════════════════════════════════════════════════════════════
# §8b  Network-pair decomposition of S_sig  (cardinality-normalized)
# ═══════════════════════════════════════════════════════════════════════
if consensus_edges is not None and "src_Yeo17_Network" in consensus_edges.columns:
    ce = consensus_edges.copy()

    # Network pair
    ce["net_pair"] = ce.apply(
        lambda r: tuple(sorted([r["src_Yeo17_Network"], r["dst_Yeo17_Network"]])),
        axis=1)
    ce["net_pair_str"] = ce["net_pair"].apply(lambda x: f"{x[0]} — {x[1]}")

    # Compute total possible edges per network pair for normalization
    _roi_net = roi_info[_net_col].values if "_net_col" in dir() else None
    _net_sizes = {}
    if _roi_net is not None:
        for n in pd.unique(ce["src_Yeo17_Network"].tolist() + ce["dst_Yeo17_Network"].tolist()):
            _net_sizes[n] = int((pd.Series(_roi_net) == n).sum())

    def _possible_edges(na, nb):
        sa = _net_sizes.get(na, 1)
        sb = _net_sizes.get(nb, 1)
        if na == nb:
            return max(sa * (sa - 1) // 2, 1)
        return max(sa * sb, 1)

    net_agg = (ce.groupby("net_pair_str")
               .agg(n_edges=("w_signed", "count"),
                    w_abs_sum=("w_signed", lambda x: np.abs(x).sum()),
                    w_signed_sum=("w_signed", "sum"),
                    mean_abs_attr=("mean_abs_topk", "mean"))
               .reset_index())

    # Add cardinality normalization
    _pair_possible = {}
    for _, r in ce.drop_duplicates("net_pair_str").iterrows():
        na, nb = r["net_pair"]
        _pair_possible[r["net_pair_str"]] = _possible_edges(na, nb)

    net_agg["n_possible_edges"] = net_agg["net_pair_str"].map(_pair_possible)
    net_agg["w_abs_sum_normalized"] = (
        net_agg["w_abs_sum"] / net_agg["n_possible_edges"])
    net_agg["edge_density"] = net_agg["n_edges"] / net_agg["n_possible_edges"]
    net_agg = net_agg.sort_values("w_abs_sum_normalized", ascending=False)

    top_net = net_agg.head(15)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Raw weight (for reference)
    ax = axes[0]
    _top_raw = net_agg.sort_values("w_abs_sum", ascending=False).head(15)
    colors_raw = ["#E74C3C" if v > 0 else "#3498DB"
                  for v in _top_raw["w_signed_sum"]]
    ax.barh(range(len(_top_raw)), _top_raw["w_abs_sum"], color=colors_raw, alpha=0.8)
    ax.set_yticks(range(len(_top_raw)))
    ax.set_yticklabels(_top_raw["net_pair_str"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|w_signed| sum (raw)")
    ax.set_title("Raw aggregate weight\n(biased by network-pair size)")
    for i, (_, row) in enumerate(_top_raw.iterrows()):
        ax.text(row["w_abs_sum"] + 0.002, i,
                f" n={int(row['n_edges'])}/{int(row['n_possible_edges'])}",
                va="center", fontsize=6.5)

    # Panel 2: Cardinality-normalized
    ax = axes[1]
    colors_norm = ["#E74C3C" if v > 0 else "#3498DB"
                   for v in top_net["w_signed_sum"]]
    ax.barh(range(len(top_net)), top_net["w_abs_sum_normalized"],
            color=colors_norm, alpha=0.8)
    ax.set_yticks(range(len(top_net)))
    ax.set_yticklabels(top_net["net_pair_str"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|w_signed| sum / n_possible_edges")
    ax.set_title("Cardinality-normalized weight\n(corrected for network-pair size)")
    for i, (_, row) in enumerate(top_net.iterrows()):
        ax.text(row["w_abs_sum_normalized"] + 0.0001, i,
                f" d={row['edge_density']:.2f}", va="center", fontsize=6.5)

    plt.suptitle("Top-15 Network Pairs by Signature Weight\n"
                 "(red = net positive, blue = net negative)", fontsize=11)
    plt.tight_layout()
    save_fig(fig, "fig8_network_decomposition.png", close=False)
    if SHOW_FIGURES: plt.show()
    else: plt.close()

    print(f"Network pairs with consensus edges: {len(net_agg)}")
    print(f"\nTop-5 by normalized |w_signed|:")
    for _, r in net_agg.head(5).iterrows():
        print(f"  {r['net_pair_str']:50s} n={int(r['n_edges']):3d}/"
              f"{int(r['n_possible_edges'])}  |w|_norm={r['w_abs_sum_normalized']:.5f}")

    net_agg.to_csv(TABLES_DIR / "signature_network_decomposition.csv", index=False)
else:
    if consensus_edges is None:
        print("⚠ No consensus edges. Skipping network decomposition.")
    else:
        print("⚠ No Yeo17 network labels. Skipping.")""")

# ══════════════════════════════════════════════════════════════════
# FIX 10: Reframe §12 markdown
# ══════════════════════════════════════════════════════════════════
idx12md = find_cell("38c07167")
if idx12md is not None:
    set_source(cells[idx12md], r"""## §9 — Connectomic Characterisation of the AD-Likeness Gradient

**Important methodological note.** The within-COVID AD-like vs CN-like
edge comparison (Tier 1) is a **post-selection characterisation**: the two
groups are defined by the model score itself (Youden threshold). Therefore
this analysis characterises what the score captures at the edge level — it
is **not** independent mechanistic confirmation of AD biology.

| Tier | Comparison | Rationale |
|------|-----------|-----------|
| **Primary** | Within Long-COVID: AD-like vs CN-like (Youden) | Score-stratified connectomic characterisation |
| **Secondary** | Long-COVID vs Study controls | Robustness check (weaker design) |

**Per-channel analysis**: each selected channel is analysed independently
(no cross-channel averaging) to preserve mechanistic interpretability.

**Statistical approach**: Mann–Whitney U per edge → BH-FDR correction.
Effect sizes: Cliff's δ.""")

# ══════════════════════════════════════════════════════════════════
# FIX 10b: Reframe §12a code — update prints
# ══════════════════════════════════════════════════════════════════
idx12a = find_cell("480dcce9")
if idx12a is not None:
    src = get_source(cells[idx12a])
    src = src.replace(
        '§12a — PRIMARY: Within Long-COVID  (AD-like vs CN-like by Youden)',
        '§9a — Connectomic characterisation of AD-likeness gradient (within Long-COVID)'
    )
    src = src.replace(
        '§12a PRIMARY — Within Long-COVID',
        '§9a — Score-stratified characterisation'
    )
    set_source(cells[idx12a], src)

# ══════════════════════════════════════════════════════════════════
# FIX 10c: Reframe §12b code
# ══════════════════════════════════════════════════════════════════
idx12b = find_cell("0b029fc8")
if idx12b is not None:
    src = get_source(cells[idx12b])
    src = src.replace("§12b", "§9b")
    set_source(cells[idx12b], src)

# ══════════════════════════════════════════════════════════════════
# FIX 10d: Update §12 interpretation notes
# ══════════════════════════════════════════════════════════════════
idx12notes = find_cell("98f39b35")
if idx12notes is not None:
    set_source(cells[idx12notes], r"""### §9 — Interpretation Notes

**Tier 1 — Within-COVID score-stratified characterisation.**
Groups are defined by the model's own Youden threshold, so this is
a **post-selection characterisation** of what the AD-likeness gradient
captures at the edge level — not an independent mechanistic test.
Cliff's δ (|δ| < 0.11 negligible, 0.11–0.28 small, 0.28–0.43 medium,
> 0.43 large; Romano et al., 2006).

**Tier 2 — Long-COVID vs Controls.**
Given modest control-group size (n ≈ 43), power for edge-level
group differences is limited and FDR-null results are expected.

**Full-universe enrichment (v3.2).**
Cliff's δ computed over all 8 515 upper-triangle edges; top 5 %
by |δ| are the hit set; Fisher's exact test measures overlap
with 1 026 AD-signature edges.

**Sign agreement.**
~50 % agreement across all channels (null by binomial test),
indicating the within-COVID gradient is distributed, not edge-specific.

**Null results are informative.**
The AD-likeness gradient appears driven by distributed, subtle
connectivity patterns rather than by individual strong edges.""")

# ══════════════════════════════════════════════════════════════════
# FIX 11: Rewrite §13 markdown — channel-specific
# ══════════════════════════════════════════════════════════════════
idx13md = find_cell("1fb79101")
if idx13md is not None:
    set_source(cells[idx13md], r"""## §QC-Network — Network Connectivity Summary (Channel-Specific)

Aggregate edge-level functional connectivity into **Yeo-17 network pairs**
and compare mean connectivity between Long-COVID and study controls.

**Channel policy**: each selected channel is summarised independently.
The primary channel for reporting is **OMST** (channel 0), based on prior
enrichment results. Cross-channel averaging is **not used** because the
channels have different semantics, scales, and sparsity structures.""")

# ══════════════════════════════════════════════════════════════════
# FIX 12: §13 code — channel-specific, no averaging
# ══════════════════════════════════════════════════════════════════
idx13c = find_cell("76daf2ee")
if idx13c is not None:
    set_source(cells[idx13c], r"""# ═══════════════════════════════════════════════════════════════════════
# §QC-Network  Network Connectivity Summary (channel-specific)
# ═══════════════════════════════════════════════════════════════════════

roi_info_path = RESULTS_DIR / "roi_info_from_tensor.csv"
roi_info = pd.read_csv(roi_info_path)

_net_col = None
for _c in ["network_label_in_tensor", "Yeo17_Network", "network_label"]:
    if _c in roi_info.columns:
        _net_col = _c; break
assert _net_col is not None

yeo = roi_info[_net_col].astype(str).str.strip().values
R = len(yeo)
nets = sorted(pd.unique(yeo))
net_to_idx = {n: i for i, n in enumerate(nets)}
K = len(nets)
print(f"Network labels from '{_net_col}': {K} networks, {R} ROIs")

_covid_set = set(clin_covid["SubjectID"].astype(str))
_ctrl_set  = set(clin_control["SubjectID"].astype(str))
_mask_covid = np.array([sid in _covid_set for sid in covid_ids])
_mask_ctrl  = np.array([sid in _ctrl_set  for sid in covid_ids])

def _netpair_matrix(conn_matrix, yeo_labels, net_list, net2idx):
    K = len(net_list)
    M = np.full((K, K), np.nan)
    for a_name in net_list:
        ia = np.where(yeo_labels == a_name)[0]
        for b_name in net_list:
            ib = np.where(yeo_labels == b_name)[0]
            if a_name == b_name:
                if len(ia) < 2: continue
                ii, jj = np.triu_indices(len(ia), k=1)
                vals = conn_matrix[np.ix_(ia, ia)][ii, jj]
            else:
                vals = conn_matrix[np.ix_(ia, ib)].ravel()
            if vals.size > 0:
                M[net2idx[a_name], net2idx[b_name]] = float(np.nanmean(vals))
    return M

# ── Per-channel analysis (no cross-channel averaging) ──
_primary_ch = PRIMARY_CHANNEL_IDX if "PRIMARY_CHANNEL_IDX" in dir() else 0
_short = {n: n.replace("_", " ") for n in nets}
_short_labels = [_short[n] for n in nets]

for ch_idx in channels_to_use:
    _ch_name = covid_chs[ch_idx] if ch_idx < len(covid_chs) else f"ch{ch_idx}"
    _is_primary = (ch_idx == _primary_ch)
    _tag = " ★ PRIMARY" if _is_primary else " (supplementary)"
    print(f"\n{'='*60}")
    print(f"Channel {ch_idx}: {_ch_name[:50]}{_tag}")
    print(f"{'='*60}")

    Xch = covid_tensor[:, ch_idx, :, :]
    X_covid = Xch[_mask_covid].mean(axis=0)
    X_ctrl  = Xch[_mask_ctrl].mean(axis=0)

    M_covid = _netpair_matrix(X_covid, yeo, nets, net_to_idx)
    M_ctrl  = _netpair_matrix(X_ctrl, yeo, nets, net_to_idx)
    M_diff  = M_covid - M_ctrl

    # Save per-channel table
    rows = []
    for i_n, a in enumerate(nets):
        for j_n, b in enumerate(nets):
            if j_n < i_n: continue
            rows.append({"net_a": a, "net_b": b, "channel": ch_idx,
                         "mean_COVID": M_covid[i_n, j_n],
                         "mean_CONTROL": M_ctrl[i_n, j_n],
                         "diff": M_diff[i_n, j_n]})
    _df = pd.DataFrame(rows)
    _df.to_csv(TABLES_DIR / f"network_pair_connectivity_ch{ch_idx}.csv",
               index=False)

    # Plot only primary channel (keep notebook concise)
    if _is_primary:
        _vmax = np.nanmax(np.abs(M_diff)) * 1.05
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        for ax_i, (M, title, cmap) in enumerate([
            (M_covid, f"Long-COVID (ch{ch_idx})", "YlOrRd"),
            (M_ctrl,  f"Controls (ch{ch_idx})", "YlOrRd"),
            (M_diff,  f"Diff COVID−Controls (ch{ch_idx})", "RdBu_r"),
        ]):
            ax = axes[ax_i]
            _ddf = pd.DataFrame(M, index=_short_labels, columns=_short_labels)
            _kw = {"center": 0, "vmin": -_vmax, "vmax": _vmax} if "Diff" in title else {}
            sns.heatmap(_ddf, ax=ax, cmap=cmap, linewidths=0.15, linecolor="white",
                        square=True, cbar_kws={"shrink": 0.7}, **_kw)
            ax.set_title(title, fontsize=10, pad=8)
            ax.tick_params(axis="both", labelsize=6.5)

        fig.suptitle(
            f"Network-pair connectivity — {_ch_name[:40]} (PRIMARY)\n"
            f"COVID (n={_mask_covid.sum()}) vs Controls (n={_mask_ctrl.sum()})",
            fontsize=12, y=1.04)
        plt.tight_layout()
        save_fig(fig, f"fig14_network_connectivity_ch{ch_idx}.png", close=False)
        if SHOW_FIGURES: plt.show()
        else: plt.close()

    # A-priori focus
    _APRIORI = ["Default", "Limbic", "Salience", "VentralAttention", "Control"]
    _frows = []
    for i_n, a in enumerate(nets):
        for j_n, b in enumerate(nets):
            if j_n < i_n: continue
            if any(k.lower() in a.lower() or k.lower() in b.lower() for k in _APRIORI):
                _frows.append({"net_a": a, "net_b": b, "ch": ch_idx,
                               "diff": M_diff[i_n, j_n]})
    if _frows and _is_primary:
        _fdf = pd.DataFrame(_frows).sort_values("diff", key=abs, ascending=False)
        print(f"  A-priori focus (top-5 by |diff|):")
        for _, r in _fdf.head(5).iterrows():
            print(f"    {r['net_a']:20s} — {r['net_b']:20s}: Δ={r['diff']:+.5f}")

print("\n✓ Network summary complete (channel-specific, no cross-channel averaging).")""")

# ══════════════════════════════════════════════════════════════════
# FIX 13: Update §13 interpretation notes
# ══════════════════════════════════════════════════════════════════
idx13int = find_cell("6ad2800b")
if idx13int is not None:
    set_source(cells[idx13int], r"""### Network Summary — Interpretation

**Channel-specific analysis.**
Network-pair summaries are computed per channel. No cross-channel
averaging is performed because channels have different semantics
and scales (OMST = sparse/thresholded, Pearson Full = dense,
MI = information-theoretic).

**OMST as primary channel.**
Channel 0 (Pearson_OMST_GCE_Signed_Weighted) is the primary
reporting channel based on prior enrichment results showing
genuine overlap between COVID-vs-control top edges and the
AD signature in OMST but not in other channels.

**Aggregation caveat.**
Network-pair means collapse many edges into single values.
Interpret the heatmap as a descriptive spatial summary, not
as a formal statistical test.""")

# ══════════════════════════════════════════════════════════════════
# FIX 14: DELETE §10 (UMAP latent) + §14 (clustering)
# Replace with explanation markdown
# ══════════════════════════════════════════════════════════════════

# Find §10 markdown (0c8b8b62) and code (de97bdf0)
# Find §14 markdown (f086730a) and code (06869b8a)
cells_to_delete = []
for partial_id in ["0c8b8b62", "de97bdf0", "f086730a", "06869b8a"]:
    idx = find_cell(partial_id)
    if idx is not None:
        cells_to_delete.append(idx)

# Sort in reverse to not invalidate indices
cells_to_delete.sort(reverse=True)
for idx in cells_to_delete:
    del cells[idx]

# Rebuild id_map after deletions
id_map = {}
for i, c in enumerate(cells):
    id_map[c.get("id", "")] = i

# Insert a markdown cell explaining the removal
# Find QC markdown (after old §10 position)
_insert_pos = find_cell("e5bb1635")
if _insert_pos is None:
    _insert_pos = find_cell("8fe7f1c9")  # §11 markdown
if _insert_pos is None:
    _insert_pos = len(cells) - 5  # fallback

removal_md = make_cell("markdown", r"""## Removed Sections: Latent UMAP (former §10) and Clustering (former §14)

**Latent-space UMAP** and **K-Means clustering** sections from the prior
notebook version have been removed for the following methodological reason:

> **Latent non-identifiability across folds.**  Each of the 5 outer-fold
> VAEs is trained independently.  Independently trained VAEs produce
> latent spaces that are rotation/reflection-equivalent but not aligned.
> Element-wise averaging of μ vectors across non-aligned latent spaces
> produces a meaningless "consensus" embedding.  Any UMAP or clustering
> applied to such averaged latents would inherit this fundamental flaw.

**Valid alternatives** (not pursued here due to limited interpretive value):
- Single-fold latent visualisation (arbitrary fold choice).
- Procrustes alignment across folds (requires a reference).

Neither alternative was deemed sufficiently informative to justify
inclusion, so these sections are omitted entirely.
""")
cells.insert(_insert_pos, removal_md)

# Rebuild id_map
id_map = {}
for i, c in enumerate(cells):
    id_map[c.get("id", "")] = i

# ══════════════════════════════════════════════════════════════════
# FIX 15: INSERT FATIGUE ANALYSIS as §10
# ══════════════════════════════════════════════════════════════════

# Find position: after the removed-sections markdown, before subject selection
_subj_sel_md = find_cell("fc142a5f")
if _subj_sel_md is None:
    _subj_sel_md = len(cells) - 3
_fat_insert = _subj_sel_md

fatigue_cells = []

# §10 Intro markdown
fatigue_cells.append(make_cell("markdown", r"""## §10 — Fatigue Phenotype Analysis (CategoriaFAS)

**Primary research question.** Does the transferred AD-likeness signal,
its associated connectomic signature score (S_sig), or its OOD behaviour
show a meaningful relationship with fatigue burden, especially
**FATIGA EXTREMA vs NO HAY FATIGA**?

**Fatigue levels** (ordinal):
- `NO HAY FATIGA` (no fatigue) — coded 0
- `FATIGA` (fatigue) — coded 1
- `FATIGA EXTREMA` (extreme fatigue) — coded 2

**Cohort policy.** Primary analyses are within the **tensor-restricted
COVID cohort** only.  Controls are used secondarily for descriptive
context (fatigue is not COVID-specific).

**Analysis plan:**
- A) Cohort audit for fatigue
- B) Descriptive table by CategoriaFAS
- C) Nonparametric group comparisons (Kruskal-Wallis, post-hoc Mann-Whitney, BH-FDR)
- D) Ordinal trend analysis (Spearman, ordinal regression)
- E) Multivariable models controlling for confounders
- F) Extreme-vs-none focused contrast
- G) Sensitivity analyses
- H) Publication-quality figures
- I) Interpretation summary

**Important.** Null findings are explicitly stated.  We do not force
positive narratives.
"""))

# §10a Fatigue cohort audit
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10a  FATIGUE COHORT AUDIT
# ═══════════════════════════════════════════════════════════════════════

# Build fatigue analysis DataFrame from tensor-restricted COVID cohort
fat_covid = clin_covid.copy()
fat_all   = clin_all.copy()

# Ensure CategoriaFAS is present
assert "CategoriaFAS" in fat_covid.columns, "CategoriaFAS not in metadata"

# ── FAS ordinal coding ──
_fas_map = {"NO HAY FATIGA": 0, "FATIGA": 1, "FATIGA EXTREMA": 2}
_fas_order = ["NO HAY FATIGA", "FATIGA", "FATIGA EXTREMA"]
fat_covid["FAS_ordinal"] = fat_covid["CategoriaFAS"].map(_fas_map)
fat_all["FAS_ordinal"] = fat_all["CategoriaFAS"].map(_fas_map)

# ── Counts ──
print("╔══════════════════════════════════════════════════════════╗")
print("║  FATIGUE COHORT AUDIT (tensor-restricted)               ║")
print("╠══════════════════════════════════════════════════════════╣")

print(f"\n  COVID subjects with CategoriaFAS:")
_vc_covid = fat_covid["CategoriaFAS"].value_counts(dropna=False)
for cat in _fas_order + [np.nan]:
    _label = cat if isinstance(cat, str) else "NaN/missing"
    _n = _vc_covid.get(cat, 0) if isinstance(cat, str) else fat_covid["CategoriaFAS"].isna().sum()
    print(f"    {_label:25s}: {_n:3d}")
print(f"    {'TOTAL':25s}: {len(fat_covid):3d}")

_fas_missing_covid = fat_covid["CategoriaFAS"].isna().sum()
print(f"\n  Missingness: {_fas_missing_covid}/{len(fat_covid)} COVID subjects missing CategoriaFAS")

print(f"\n  CONTROL subjects with CategoriaFAS (secondary context):")
_vc_ctrl = fat_all[fat_all["ResearchGroup"]=="CONTROL"]["CategoriaFAS"].value_counts(dropna=False)
for cat in _fas_order:
    print(f"    {cat:25s}: {_vc_ctrl.get(cat, 0):3d}")

# ── Overlap with severity and recovery ──
catcovid_col = None
for c in ["CategoríaCOVID", "CategoriaCOVID"]:
    if c in fat_covid.columns:
        catcovid_col = c; break

if catcovid_col:
    print(f"\n  Cross-tab: CategoriaFAS × {catcovid_col} (COVID only)")
    print(pd.crosstab(fat_covid["CategoriaFAS"], fat_covid[catcovid_col],
                      dropna=False, margins=True))

if "Recuperado" in fat_covid.columns:
    print(f"\n  Cross-tab: CategoriaFAS × Recuperado (COVID only)")
    print(pd.crosstab(fat_covid["CategoriaFAS"], fat_covid["Recuperado"],
                      dropna=False, margins=True))

if "Sex" in fat_covid.columns:
    print(f"\n  Cross-tab: CategoriaFAS × Sex (COVID only)")
    print(pd.crosstab(fat_covid["CategoriaFAS"], fat_covid["Sex"],
                      dropna=False, margins=True))

print("\n╚══════════════════════════════════════════════════════════╝")

# Filter to COVID with valid FAS for all subsequent analyses
fat_valid = fat_covid[fat_covid["CategoriaFAS"].isin(_fas_order)].copy()
print(f"\nFatigue analysis cohort (COVID, valid FAS): N = {len(fat_valid)}")
"""))

# §10b Descriptive tables
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10b  DESCRIPTIVE TABLE BY CategoriaFAS
# ═══════════════════════════════════════════════════════════════════════

# Merge inference data if not already merged
if "y_score_ensemble" not in fat_valid.columns:
    _ens = ensemble[ensemble["classifier"] == TARGET_CLF].copy()
    _ens = _ens.merge(recon_ens, on="SubjectID", how="left")
    _ens = _ens.merge(sig_df, on="SubjectID", how="left")
    fat_valid = fat_valid.merge(
        _ens[["SubjectID", "y_score_ensemble", "y_score_std",
              "recon_error_ensemble", "S_sig"]],
        on="SubjectID", how="left")

_desc_cols = ["Age", "Sex", "MOCA", "MOCA_perc", "EQ-VAS", "BMI",
              "NivelEducativo", "y_score_ensemble", "y_score_std",
              "recon_error_ensemble", "S_sig"]
if catcovid_col:
    _desc_cols.append(catcovid_col)
if "Recuperado" in fat_valid.columns:
    _desc_cols.append("Recuperado")

_desc_rows = []
for grp in _fas_order:
    g = fat_valid[fat_valid["CategoriaFAS"] == grp]
    row = {"CategoriaFAS": grp, "n": len(g)}
    for col in _desc_cols:
        if col not in g.columns:
            continue
        if g[col].dtype in [np.float64, np.int64, float, int]:
            vals = g[col].dropna()
            if len(vals) > 0:
                row[f"{col}_median"] = f"{vals.median():.2f}"
                row[f"{col}_IQR"] = f"[{vals.quantile(0.25):.2f}, {vals.quantile(0.75):.2f}]"
                row[f"{col}_n"] = len(vals)
            else:
                row[f"{col}_median"] = "—"
                row[f"{col}_IQR"] = "—"
                row[f"{col}_n"] = 0
        elif col == "Sex":
            _nm = (g["Sex"] == "M").sum()
            row["Sex_M_n"] = _nm
            row["Sex_M_pct"] = f"{100*_nm/len(g):.1f}%" if len(g) > 0 else "—"
        else:
            # Categorical: show mode
            _mode = g[col].mode()
            row[f"{col}_mode"] = _mode.iloc[0] if len(_mode) > 0 else "—"
    _desc_rows.append(row)

desc_table = pd.DataFrame(_desc_rows)
desc_table.to_csv(TABLES_DIR / "fatigue_descriptive_table.csv", index=False)

# Display
print("Descriptive table by CategoriaFAS (COVID, tensor-restricted):\n")
print(desc_table.T.to_string())
"""))

# §10c Nonparametric comparisons
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10c  NONPARAMETRIC GROUP COMPARISONS
# ═══════════════════════════════════════════════════════════════════════

from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

_fas_order = ["NO HAY FATIGA", "FATIGA", "FATIGA EXTREMA"]
_test_vars = ["y_score_ensemble", "y_score_std", "recon_error_ensemble",
              "S_sig", "MOCA", "MOCA_perc", "EQ-VAS"]
_test_vars = [v for v in _test_vars if v in fat_valid.columns]

# ── Omnibus: Kruskal-Wallis ──
kw_rows = []
for var in _test_vars:
    groups = [fat_valid.loc[fat_valid["CategoriaFAS"] == g, var].dropna().values
              for g in _fas_order]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        continue
    H, p = kruskal(*groups)
    kw_rows.append({"variable": var, "H_stat": H, "p_raw": p,
                    "n_groups": len(groups),
                    "group_ns": [len(g) for g in groups]})

kw_df = pd.DataFrame(kw_rows)
if len(kw_df) > 0:
    _, q_vals, _, _ = multipletests(kw_df["p_raw"], method="fdr_bh")
    kw_df["q_fdr"] = q_vals
    kw_df["sig_fdr05"] = q_vals < 0.05

print("Kruskal-Wallis omnibus tests (COVID only, 3-group):\n")
print(kw_df.to_string(index=False, float_format="{:.4f}".format))

# ── Post-hoc pairwise: Mann-Whitney with BH-FDR ──
_pairs = [("NO HAY FATIGA", "FATIGA"),
          ("NO HAY FATIGA", "FATIGA EXTREMA"),
          ("FATIGA", "FATIGA EXTREMA")]

ph_rows = []
for var in _test_vars:
    for g1_name, g2_name in _pairs:
        g1 = fat_valid.loc[fat_valid["CategoriaFAS"] == g1_name, var].dropna().values
        g2 = fat_valid.loc[fat_valid["CategoriaFAS"] == g2_name, var].dropna().values
        if len(g1) < 3 or len(g2) < 3:
            continue
        U, p = mannwhitneyu(g1, g2, alternative="two-sided")
        # Cliff's delta
        n1, n2 = len(g1), len(g2)
        delta = 2 * U / (n1 * n2) - 1
        ph_rows.append({
            "variable": var,
            "group1": g1_name, "group2": g2_name,
            "n1": n1, "n2": n2,
            "U_stat": U, "p_raw": p,
            "cliffs_delta": delta,
            "median_g1": np.median(g1),
            "median_g2": np.median(g2),
        })

ph_df = pd.DataFrame(ph_rows)
if len(ph_df) > 0:
    _, q_vals, _, _ = multipletests(ph_df["p_raw"], method="fdr_bh")
    ph_df["q_fdr"] = q_vals
    ph_df["sig_fdr05"] = q_vals < 0.05

    print("\n\nPost-hoc pairwise comparisons (Mann-Whitney, BH-FDR):\n")
    _disp_cols = ["variable", "group1", "group2", "n1", "n2",
                  "cliffs_delta", "p_raw", "q_fdr", "sig_fdr05"]
    print(ph_df[_disp_cols].to_string(index=False, float_format="{:.4f}".format))

    ph_df.to_csv(TABLES_DIR / "fatigue_pairwise_comparisons.csv", index=False)
kw_df.to_csv(TABLES_DIR / "fatigue_kruskal_wallis.csv", index=False)
"""))

# §10d Ordinal trend analysis
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10d  ORDINAL TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

from scipy.stats import spearmanr

_trend_vars = ["y_score_ensemble", "y_score_std", "recon_error_ensemble",
               "S_sig", "MOCA", "MOCA_perc", "EQ-VAS"]
_trend_vars = [v for v in _trend_vars if v in fat_valid.columns]

print("Ordinal trend analysis: NO HAY FATIGA (0) < FATIGA (1) < FATIGA EXTREMA (2)\n")
trend_rows = []
for var in _trend_vars:
    valid = fat_valid[["FAS_ordinal", var]].dropna()
    if len(valid) < 10:
        continue
    rho, p = spearmanr(valid["FAS_ordinal"], valid[var])

    # Bootstrap CI for Spearman
    rng = np.random.RandomState(SEED_GLOBAL)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(valid), len(valid), replace=True)
        r_b, _ = spearmanr(valid.iloc[idx]["FAS_ordinal"], valid.iloc[idx][var])
        boots.append(r_b)
    lo = np.percentile(boots, 2.5)
    hi = np.percentile(boots, 97.5)

    trend_rows.append({
        "variable": var, "n": len(valid),
        "spearman_rho": rho, "p_value": p,
        "ci_lo": lo, "ci_hi": hi,
        "direction": "↑ with fatigue" if rho > 0 else "↓ with fatigue",
    })

trend_df = pd.DataFrame(trend_rows)
if len(trend_df) > 0:
    _, q_vals, _, _ = multipletests(trend_df["p_value"], method="fdr_bh")
    trend_df["q_fdr"] = q_vals
    trend_df["sig_fdr05"] = q_vals < 0.05

    print(trend_df.to_string(index=False, float_format="{:.4f}".format))
    trend_df.to_csv(TABLES_DIR / "fatigue_ordinal_trend.csv", index=False)

    # Ordinal regression: FAS_ordinal ~ y_score_ensemble (exploratory)
    try:
        import statsmodels.api as sm
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        _ord_data = fat_valid[["FAS_ordinal", "y_score_ensemble", "Age"]].dropna()
        if "Sex" in fat_valid.columns:
            _ord_data["Sex_M"] = (fat_valid.loc[_ord_data.index, "Sex"] == "M").astype(float)

        if len(_ord_data) > 20:
            _X_cols = [c for c in _ord_data.columns if c != "FAS_ordinal"]
            mod = OrderedModel(_ord_data["FAS_ordinal"],
                              _ord_data[_X_cols],
                              distr="logit")
            res = mod.fit(method="bfgs", disp=False)
            print("\nOrdinal logit: FAS_ordinal ~ y_score_ensemble + Age [+ Sex]")
            print(f"N = {len(_ord_data)}")
            print(res.summary().tables[1])
    except Exception as e:
        print(f"\n⚠ Ordinal regression skipped: {e}")
"""))

# §10e Multivariable fatigue models
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10e  MULTIVARIABLE FATIGUE MODELS
# ═══════════════════════════════════════════════════════════════════════
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Outcome: y_score_ensemble  (AD-likeness)
# Primary predictor: FAS_ordinal
# Confounders: Age, Sex, COVID severity, recovery

_model_data = fat_valid.copy()

# Encode confounders
if "Sex" in _model_data.columns:
    _model_data["Sex_M"] = (_model_data["Sex"] == "M").astype(float)

catcovid_col_local = None
for c in ["CategoríaCOVID", "CategoriaCOVID"]:
    if c in _model_data.columns:
        catcovid_col_local = c; break
if catcovid_col_local:
    _sev_map = {"No infectado": 0, "Paciente ambulatorio": 1,
                "Internación moderada": 2, "Internación severa (UCI)": 3}
    _model_data["severity_ord"] = _model_data[catcovid_col_local].map(_sev_map)

if "Recuperado" in _model_data.columns:
    _rec_map = {"completamente": 0, "parcialmente": 1, "No mucho": 2, "para nada": 3}
    _model_data["recovery_ord"] = _model_data["Recuperado"].map(_rec_map)

# Build predictor list
_pred_cols = ["FAS_ordinal", "Age"]
if "Sex_M" in _model_data.columns:
    _pred_cols.append("Sex_M")
if "severity_ord" in _model_data.columns and _model_data["severity_ord"].notna().sum() > 20:
    _pred_cols.append("severity_ord")
if "recovery_ord" in _model_data.columns and _model_data["recovery_ord"].notna().sum() > 20:
    _pred_cols.append("recovery_ord")

# Complete cases
_reg_cols = ["y_score_ensemble"] + _pred_cols
_avail = [c for c in _reg_cols if c in _model_data.columns]
_cc = _model_data[_avail].dropna()

print(f"§10e — Multivariable model: y_score_ensemble ~ {' + '.join(_pred_cols)}")
print(f"Complete cases: N = {len(_cc)}\n")

if len(_cc) < 20:
    print("⚠ Too few complete cases. Skipping multivariable model.")
else:
    y = _cc["y_score_ensemble"]
    X = _cc[[c for c in _pred_cols if c in _cc.columns]]

    # Standardise for interpretable coefficients
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std)

    model = sm.OLS(y, X_std).fit(cov_type="HC3")
    print(f"R² = {model.rsquared:.4f},  adj-R² = {model.rsquared_adj:.4f}\n")
    print(model.summary2().tables[1].to_string())

    # VIF
    _vif_data = sm.add_constant(X)
    if _vif_data.shape[1] > 2:
        print("\nVariance Inflation Factors:")
        for i, col in enumerate(_vif_data.columns):
            if col == "const": continue
            vif = variance_inflation_factor(_vif_data.values, i)
            print(f"  {col:20s}: VIF = {vif:.2f}")

    # Also model S_sig as outcome
    if "S_sig" in _model_data.columns:
        _cc2 = _model_data[["S_sig"] + _pred_cols].dropna()
        if len(_cc2) > 20:
            y2 = _cc2["S_sig"]
            X2 = _cc2[[c for c in _pred_cols if c in _cc2.columns]]
            X2_std = (X2 - X2.mean()) / X2.std()
            X2_std = sm.add_constant(X2_std)
            m2 = sm.OLS(y2, X2_std).fit(cov_type="HC3")
            print(f"\n--- S_sig ~ {' + '.join(_pred_cols)} ---")
            print(f"N = {len(_cc2)}, R² = {m2.rsquared:.4f}")
            if "FAS_ordinal" in m2.params.index:
                _b = m2.params["FAS_ordinal"]
                _p = m2.pvalues["FAS_ordinal"]
                print(f"FAS_ordinal: β = {_b:.4f}, p = {_p:.4f}")

    # Also model recon_error_ensemble
    if "recon_error_ensemble" in _model_data.columns:
        _cc3 = _model_data[["recon_error_ensemble"] + _pred_cols].dropna()
        if len(_cc3) > 20:
            y3 = _cc3["recon_error_ensemble"]
            X3 = _cc3[[c for c in _pred_cols if c in _cc3.columns]]
            X3_std = (X3 - X3.mean()) / X3.std()
            X3_std = sm.add_constant(X3_std)
            m3 = sm.OLS(y3, X3_std).fit(cov_type="HC3")
            print(f"\n--- recon_error ~ {' + '.join(_pred_cols)} ---")
            print(f"N = {len(_cc3)}, R² = {m3.rsquared:.4f}")
            if "FAS_ordinal" in m3.params.index:
                _b = m3.params["FAS_ordinal"]
                _p = m3.pvalues["FAS_ordinal"]
                print(f"FAS_ordinal: β = {_b:.4f}, p = {_p:.4f}")
"""))

# §10f Extreme-vs-none contrast
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10f  EXTREME-VS-NONE FOCUSED CONTRAST
# ═══════════════════════════════════════════════════════════════════════

from scipy.stats import mannwhitneyu

_g_none = fat_valid[fat_valid["CategoriaFAS"] == "NO HAY FATIGA"]
_g_ext  = fat_valid[fat_valid["CategoriaFAS"] == "FATIGA EXTREMA"]

print(f"§10f — FATIGA EXTREMA (n={len(_g_ext)}) vs NO HAY FATIGA (n={len(_g_none)})")
print("=" * 60)

_evn_vars = ["y_score_ensemble", "y_score_std", "recon_error_ensemble",
             "S_sig", "MOCA", "MOCA_perc", "EQ-VAS"]
_evn_vars = [v for v in _evn_vars if v in fat_valid.columns]

evn_rows = []
for var in _evn_vars:
    v1 = _g_ext[var].dropna().values
    v2 = _g_none[var].dropna().values
    if len(v1) < 3 or len(v2) < 3:
        continue
    U, p = mannwhitneyu(v1, v2, alternative="two-sided")
    n1, n2 = len(v1), len(v2)
    delta = 2 * U / (n1 * n2) - 1  # Cliff's delta

    # Bootstrap CI for Cliff's delta
    rng = np.random.RandomState(SEED_GLOBAL)
    boot_deltas = []
    for _ in range(N_BOOTSTRAP):
        _b1 = rng.choice(v1, len(v1), replace=True)
        _b2 = rng.choice(v2, len(v2), replace=True)
        _bU, _ = mannwhitneyu(_b1, _b2, alternative="two-sided")
        boot_deltas.append(2 * _bU / (n1 * n2) - 1)
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)

    evn_rows.append({
        "variable": var,
        "median_EXTREMA": np.median(v1),
        "median_NONE": np.median(v2),
        "n_EXTREMA": n1, "n_NONE": n2,
        "cliffs_delta": delta,
        "delta_CI_lo": ci_lo, "delta_CI_hi": ci_hi,
        "p_raw": p,
    })

evn_df = pd.DataFrame(evn_rows)
if len(evn_df) > 0:
    _, q_vals, _, _ = multipletests(evn_df["p_raw"], method="fdr_bh")
    evn_df["q_fdr"] = q_vals
    evn_df["sig_fdr05"] = q_vals < 0.05

    print("\nExtreme-vs-None contrast (BH-FDR corrected):\n")
    print(evn_df.to_string(index=False, float_format="{:.4f}".format))
    evn_df.to_csv(TABLES_DIR / "fatigue_extreme_vs_none.csv", index=False)

    # Effect-size interpretation
    print("\nEffect-size interpretation (Cliff's δ):")
    for _, r in evn_df.iterrows():
        d = abs(r["cliffs_delta"])
        if d < 0.11:
            mag = "negligible"
        elif d < 0.28:
            mag = "small"
        elif d < 0.43:
            mag = "medium"
        else:
            mag = "large"
        sig = "★" if r["sig_fdr05"] else ""
        print(f"  {r['variable']:25s}: δ={r['cliffs_delta']:+.3f} "
              f"[{r['delta_CI_lo']:+.3f}, {r['delta_CI_hi']:+.3f}] "
              f"({mag}) {sig}")
"""))

# §10g Sensitivity analyses
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10g  SENSITIVITY ANALYSES  (fatigue)
# ═══════════════════════════════════════════════════════════════════════

from scipy.stats import spearmanr, mannwhitneyu, kruskal

print("§10g — Sensitivity / robustness checks\n")

_fas_order = ["NO HAY FATIGA", "FATIGA", "FATIGA EXTREMA"]
_main_var = "y_score_ensemble"
_sens_results = []

# ── 1. Ambulatory-only subset ──
if catcovid_col_local and "severity_ord" in fat_valid.columns:
    _amb = fat_valid[fat_valid[catcovid_col_local] == "Paciente ambulatorio"]
    if len(_amb) > 10 and _amb["CategoriaFAS"].nunique() >= 2:
        _grps = [_amb.loc[_amb["CategoriaFAS"] == g, _main_var].dropna().values
                 for g in _fas_order]
        _grps = [g for g in _grps if len(g) >= 3]
        if len(_grps) >= 2:
            H, p = kruskal(*_grps) if len(_grps) >= 3 else mannwhitneyu(_grps[0], _grps[1])
            print(f"  1. Ambulatory-only (n={len(_amb)}): "
                  f"{'KW' if len(_grps)>=3 else 'MWU'} p = {p:.4f}")
            _sens_results.append({"check": "ambulatory_only",
                                  "n": len(_amb), "p": p})

# ── 2. Excluding ICU (tiny stratum) ──
if catcovid_col_local:
    _no_icu = fat_valid[fat_valid[catcovid_col_local] != "Internación severa (UCI)"]
    if len(_no_icu) > 10 and _no_icu["CategoriaFAS"].nunique() >= 2:
        rho, p = spearmanr(_no_icu["FAS_ordinal"].dropna(),
                           _no_icu.loc[_no_icu["FAS_ordinal"].notna(), _main_var])
        print(f"  2. Excluding ICU (n={len(_no_icu)}): "
              f"Spearman ρ = {rho:.4f}, p = {p:.4f}")
        _sens_results.append({"check": "excl_ICU", "n": len(_no_icu),
                              "rho": rho, "p": p})

# ── 3. Sex-stratified ──
if "Sex" in fat_valid.columns:
    for sex in ["M", "F"]:
        _ss = fat_valid[fat_valid["Sex"] == sex]
        if len(_ss) > 10 and _ss["FAS_ordinal"].notna().sum() > 10:
            rho, p = spearmanr(_ss["FAS_ordinal"].dropna(),
                               _ss.loc[_ss["FAS_ordinal"].notna(), _main_var])
            print(f"  3. Sex={sex} only (n={len(_ss)}): "
                  f"Spearman ρ = {rho:.4f}, p = {p:.4f}")
            _sens_results.append({"check": f"sex_{sex}_only",
                                  "n": len(_ss), "rho": rho, "p": p})

# ── 4. Controls included (secondary check) ──
_with_ctrl = fat_all[fat_all["CategoriaFAS"].isin(_fas_order)].copy()
if "y_score_ensemble" not in _with_ctrl.columns:
    _ens_tmp = ensemble[ensemble["classifier"] == TARGET_CLF].copy()
    _with_ctrl = _with_ctrl.merge(
        _ens_tmp[["SubjectID", "y_score_ensemble"]],
        on="SubjectID", how="left")
_with_ctrl["FAS_ordinal"] = _with_ctrl["CategoriaFAS"].map(
    {"NO HAY FATIGA": 0, "FATIGA": 1, "FATIGA EXTREMA": 2})
_valid_wc = _with_ctrl[["FAS_ordinal", "y_score_ensemble"]].dropna()
if len(_valid_wc) > 20:
    rho, p = spearmanr(_valid_wc["FAS_ordinal"], _valid_wc["y_score_ensemble"])
    print(f"  4. All (COVID+CONTROL, n={len(_valid_wc)}): "
          f"Spearman ρ = {rho:.4f}, p = {p:.4f}")
    _sens_results.append({"check": "all_incl_ctrl", "n": len(_valid_wc),
                          "rho": rho, "p": p})

# ── 5. EQ-VAS as alternative outcome ──
if "EQ-VAS" in fat_valid.columns:
    _eq = fat_valid[["FAS_ordinal", "EQ-VAS"]].dropna()
    if len(_eq) > 10:
        rho, p = spearmanr(_eq["FAS_ordinal"], _eq["EQ-VAS"])
        print(f"  5. EQ-VAS vs FAS ordinal (n={len(_eq)}): "
              f"Spearman ρ = {rho:.4f}, p = {p:.4f}")
        _sens_results.append({"check": "EQVAS_vs_FAS",
                              "n": len(_eq), "rho": rho, "p": p})

# Save
if _sens_results:
    pd.DataFrame(_sens_results).to_csv(
        TABLES_DIR / "fatigue_sensitivity_analyses.csv", index=False)
    print(f"\n  Saved → fatigue_sensitivity_analyses.csv")

print("\n✓ Sensitivity analyses complete.")
"""))

# §10h Fatigue figures
fatigue_cells.append(make_cell("code", r"""# ═══════════════════════════════════════════════════════════════════════
# §10h  FATIGUE FIGURES (publication-quality)
# ═══════════════════════════════════════════════════════════════════════

_fas_order = ["NO HAY FATIGA", "FATIGA", "FATIGA EXTREMA"]
_fas_palette = {"NO HAY FATIGA": "#2ECC71", "FATIGA": "#F39C12",
                "FATIGA EXTREMA": "#E74C3C"}

# ── Figure 1: AD-likeness by CategoriaFAS (violin + strip) ──
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# Panel A: AD-likeness score
ax = axes[0]
_pdata = fat_valid[fat_valid["CategoriaFAS"].isin(_fas_order)].copy()
_pdata["CategoriaFAS"] = pd.Categorical(_pdata["CategoriaFAS"],
                                         categories=_fas_order, ordered=True)
sns.violinplot(data=_pdata, x="CategoriaFAS", y="y_score_ensemble",
               palette=_fas_palette, ax=ax, inner=None, alpha=0.4, cut=0)
sns.stripplot(data=_pdata, x="CategoriaFAS", y="y_score_ensemble",
              color="black", alpha=0.4, size=3, jitter=0.2, ax=ax)
# Group medians
for i, grp in enumerate(_fas_order):
    med = _pdata.loc[_pdata["CategoriaFAS"] == grp, "y_score_ensemble"].median()
    ax.hlines(med, i - 0.3, i + 0.3, colors="red", linewidths=2)
ax.set_xlabel("Fatigue Category")
ax.set_ylabel("AD-Likeness Score")
ax.set_title("A) AD-Likeness by Fatigue")
ax.tick_params(axis="x", rotation=15)

# Panel B: S_sig
ax = axes[1]
if "S_sig" in _pdata.columns and _pdata["S_sig"].notna().sum() > 10:
    sns.violinplot(data=_pdata, x="CategoriaFAS", y="S_sig",
                   palette=_fas_palette, ax=ax, inner=None, alpha=0.4, cut=0)
    sns.stripplot(data=_pdata, x="CategoriaFAS", y="S_sig",
                  color="black", alpha=0.4, size=3, jitter=0.2, ax=ax)
    ax.set_xlabel("Fatigue Category")
    ax.set_ylabel("AD Signature Score (S_sig)")
    ax.set_title("B) Signature Score by Fatigue")
    ax.tick_params(axis="x", rotation=15)
else:
    ax.text(0.5, 0.5, "S_sig not available", transform=ax.transAxes,
            ha="center", va="center")

# Panel C: Recon error
ax = axes[2]
if "recon_error_ensemble" in _pdata.columns:
    sns.violinplot(data=_pdata, x="CategoriaFAS", y="recon_error_ensemble",
                   palette=_fas_palette, ax=ax, inner=None, alpha=0.4, cut=0)
    sns.stripplot(data=_pdata, x="CategoriaFAS", y="recon_error_ensemble",
                  color="black", alpha=0.4, size=3, jitter=0.2, ax=ax)
    ax.set_xlabel("Fatigue Category")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("C) Recon Error by Fatigue")
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Model-derived metrics by fatigue group (COVID only)",
             fontsize=12, y=1.02)
plt.tight_layout()
save_fig(fig, "fig_fatigue_violin_main.png", close=False)
if SHOW_FIGURES: plt.show()
else: plt.close()

# ── Figure 2: Effect-size forest plot (extreme vs none) ──
_evn_path = TABLES_DIR / "fatigue_extreme_vs_none.csv"
if _evn_path.exists():
    _evn = pd.read_csv(_evn_path)
    if len(_evn) > 0:
        fig, ax = plt.subplots(figsize=(8, max(4, len(_evn)*0.6)))
        y_pos = range(len(_evn))
        ax.errorbar(_evn["cliffs_delta"], y_pos,
                    xerr=[_evn["cliffs_delta"] - _evn["delta_CI_lo"],
                          _evn["delta_CI_hi"] - _evn["cliffs_delta"]],
                    fmt="o", color="#2C3E50", capsize=4, markersize=6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(_evn["variable"], fontsize=9)
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        # Shade negligible zone
        ax.axvspan(-0.11, 0.11, alpha=0.08, color="grey")
        ax.set_xlabel("Cliff's δ (FATIGA EXTREMA − NO HAY FATIGA)")
        ax.set_title("Effect sizes: Extreme fatigue vs No fatigue\n"
                     "(shaded = negligible |δ| < 0.11)")
        ax.invert_yaxis()
        plt.tight_layout()
        save_fig(fig, "fig_fatigue_effect_sizes.png", close=False)
        if SHOW_FIGURES: plt.show()
        else: plt.close()

# ── Figure 3: Ordinal trend scatter ──
if "FAS_ordinal" in fat_valid.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    _jitter = np.random.RandomState(SEED_GLOBAL).uniform(-0.15, 0.15, len(fat_valid))
    ax.scatter(fat_valid["FAS_ordinal"] + _jitter,
              fat_valid["y_score_ensemble"],
              c=[_fas_palette.get(c, "grey") for c in fat_valid["CategoriaFAS"]],
              s=25, alpha=0.6, edgecolors="none")

    # Group means with CI
    for i, grp in enumerate(_fas_order):
        vals = fat_valid.loc[fat_valid["CategoriaFAS"] == grp, "y_score_ensemble"].dropna()
        if len(vals) > 0:
            mu, lo, hi = bootstrap_ci(vals.values)
            ax.errorbar(i, mu, yerr=[[mu-lo], [hi-mu]], fmt="D",
                       color="black", markersize=8, capsize=5, zorder=10)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(_fas_order, fontsize=9)
    ax.set_xlabel("Fatigue Category (ordinal)")
    ax.set_ylabel("AD-Likeness Score")
    ax.set_title("AD-Likeness trend across fatigue burden\n(diamonds = group mean ± 95% CI)")
    plt.tight_layout()
    save_fig(fig, "fig_fatigue_ordinal_trend.png", close=False)
    if SHOW_FIGURES: plt.show()
    else: plt.close()

print("✓ Fatigue figures complete.")
"""))

# §10i Fatigue interpretation
fatigue_cells.append(make_cell("markdown", r"""### §10 — Fatigue Analysis: Interpretation Summary

**Findings to be filled after execution.** Below is the template for
the interpretation; the specific results will be evaluated once the
notebook runs.

**Key questions answered:**

1. **Is AD-likeness associated with fatigue burden?**
   Evaluate Kruskal-Wallis p-value and post-hoc comparisons.
   If q_FDR > 0.05, the result is null at conventional thresholds.

2. **Is the effect monotonic?**
   Evaluate Spearman ρ between FAS_ordinal and y_score_ensemble.
   Direction and CI width matter more than p-value significance.

3. **Does it survive confound adjustment?**
   Evaluate the FAS_ordinal coefficient in the multivariable model
   after controlling for Age, Sex, severity, and recovery.

4. **Is the effect stronger for extreme fatigue?**
   Evaluate the FATIGA EXTREMA vs NO HAY FATIGA contrast.
   Report Cliff's δ with CI.

5. **Is the result clinically meaningful?**
   Even if statistically significant, small effect sizes (|δ| < 0.28)
   may lack practical relevance.  Sample-size limitations
   (FATIGA EXTREMA ≈ 34, NO HAY FATIGA ≈ 25 within COVID)
   warrant conservative interpretation.

**Honest reporting policy.**
If the fatigue signal is null or negligible, this is explicitly stated.
Sample-size limitations are acknowledged.
Sensitivity analyses are checked for consistency.
"""))

# Insert all fatigue cells
for i, fc in enumerate(fatigue_cells):
    cells.insert(_fat_insert + i, fc)

# Rebuild id_map
id_map = {}
for i, c in enumerate(cells):
    id_map[c.get("id", "")] = i

# ══════════════════════════════════════════════════════════════════
# FIX 16: Update Subject Selection markdown (now §11)
# ══════════════════════════════════════════════════════════════════
idx_subj = find_cell("fc142a5f")
if idx_subj is not None:
    set_source(cells[idx_subj], r"""## §11 — Subject Selection for Interpretability

Deterministic selection of clinically extreme, complete-data subjects:
- **AD-like & in-distribution**: high score + low recon error
- **AD-like & OOD**: high score + high recon error
- **CN-like boundary**: below threshold but near boundary

Clinical annotation (MOCA, fatigue, severity) included where available.""")

# ══════════════════════════════════════════════════════════════════
# FIX 17: Update §F markdown
# ══════════════════════════════════════════════════════════════════
idx_f = find_cell("284acbdc")
if idx_f is not None:
    set_source(cells[idx_f], r"""## §F — Outputs Index & Provenance

Notebook version: `v5.0`

**Changes from v4.x:**
- Removed latent-space UMAP (§10) and K-Means clustering (§14) due to
  latent non-identifiability across independently trained fold VAEs.
- Removed cross-channel averaging in network summaries; now channel-specific.
- Simplified OOD gate to ADNI P95 reconstruction error (primary).
- Renamed "risk categories" to "AD-Likeness categories."
- Reframed §12a as score-stratified connectomic characterisation.
- Normalised network-pair decomposition by cardinality.
- Added comprehensive fatigue analysis (§10) centred on CategoriaFAS.
- All cohort counts are tensor-restricted.
""")

# ══════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✓ Notebook refactored and saved: {NB_PATH}")
print(f"  Total cells: {len(cells)}")
