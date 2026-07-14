# Audit: `site_geometry_topology_20260705` — usable fold-local persistent homology?

## Verdict: NO — the directory contains no persistent-homology output of any kind

**This directory was not rerun for this task, per instructions.** It was inspected read-only.

## What is actually in the directory

```
results/revision_bspc_2026/post_revision_exploratory_20260630/site_geometry_topology_20260705/
├── command_log.json      (2,478 bytes)
└── tda_library_check.md  (3,578 bytes)
```

Two files only. No CSVs, no figures, no per-fold barcode/persistence-diagram data, no
Mapper graphs.

## What happened when this was run (2026-07-05)

The task was a 3-step TDA plan: Step 1 (H0 Vietoris-Rips persistence per fold),
Step 2 (Mapper graphs), Step 3 (cross-fold consistency of the topology). It never
reached Step 1. `tda_library_check.md` records a Step-0 dependency check that
failed closed, by design:

- All four candidate TDA libraries — `ripser`, `giotto_tda`, `kmapper`, `gudhi` —
  were checked across every Python environment on the machine (project `.venv`,
  home `venv`, and four conda envs including `vae_ad`) and were **missing in
  all of them**.
- The task's own guardrail ("fail closed if libraries or latent caches are
  missing — do not substitute or approximate") was honored: no `pip install`
  was run, and no scipy/sklearn clustering substitute was used in place of
  actual Vietoris-Rips persistent homology.
- The input data needed to proceed (`recover035_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache/fold_{1-5}_{trainDev,test}_latent_mu.csv`,
  384-dim latent mu, on the big disk) **was confirmed present and is not the
  blocking factor** — only the missing libraries blocked progress.
- `command_log.json` explicitly lists `steps_completed: []` and
  `steps_blocked: ["step1_h0_persistence", "step2_mapper_graphs",
  "step3_cross_fold_consistency"]`.

I re-checked the same four libraries in the `vae_ad` conda environment just now
(read-only import check, no install attempted): all four are **still missing**
as of this task. Nothing has changed since 2026-07-05.

## Conclusion for the SIPAIM freeze

There is **no usable fold-local persistent-homology analysis** to include in
the SIPAIM snapshot from this directory — not a stale one, not a partial one,
none at all. The directory is legitimately empty of results; it is a
well-documented "blocked at Step 0" report, not a hidden or incomplete
analysis. Recommended handling for this freeze:

- Do **not** copy this directory into `results/sipaim_2026/` — it has no
  scientific content, only a dependency-check log.
- If the SIPAIM manuscript's methods or limitations section references
  topology/persistent-homology analysis of the latent space, it must state
  that this analysis was attempted and blocked by a missing dependency
  (`ripser`/`gudhi`/`giotto-tda`/`kmapper` unavailable in the compute
  environment), not that it was run and found negative/inconclusive — those
  are different claims and only the former is true.
- `docs/revision_bspc_2026/site_geometry_analysis/site_geometry_analysis_protocol_v4.tex`
  already reflects this correctly: its header comments list
  `h0_persistence_by_fold.pdf` and `mapper_graphs_by_fold.pdf` under "NOT yet
  generated ... TDA libraries not installed" and do not include any
  `\sitefig` macro invoking them. No manuscript correction is needed on this
  point; this audit simply confirms that status is still accurate today.
