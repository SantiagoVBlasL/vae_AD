# PREPARED — NOT EXECUTED

The clean reproducibility rerun is prepared. No VAE training, ComBat fitting,
downstream diagnostic-classifier fitting, or run-output creation was executed.
The intended run directory does not exist as of this preflight:

`results/revision_bspc_2026/recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706`

## Experiment identity

- Locked reference:
  `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`
- New run:
  `recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706`
- Branch checked:
  `exploratory/post-revision-20260630`
- Git HEAD observed during preparation:
  `dc77722464cdcdb61f7603e069af2e15103d7aea`
- The worktree is dirty. The guarded launch therefore snapshots and hashes the
  exact live sources and records Git status plus tracked/index diffs before
  execution. It does not claim that Git HEAD alone identifies executed code.

## Exact-match validation

The candidate and locked reference configs were compared programmatically.
After removing the six intended harmonization fields, their parameter maps are
identical. Channel names, selected channels, and split strategy are also
identical. The only scientific parameter additions are:

| Field | Candidate value |
|---|---|
| `input_harmonization_mode` | `foldwise_combat` |
| `input_harmonization_batch_col` | `Manufacturer` |
| `input_harmonization_covariates` | `["Age", "Sex"]` |
| `input_harmonization_excluded_covariates` | `["ResearchGroup_Mapped"]` |
| `input_harmonization_fit_scope` | `outer_train_dev_only` |
| `input_harmonization_vectorization` | `upper_offdiag_by_channel` |

The guarded Python launcher additionally asserts channels `[1,0,2]`,
latent dimension 384, beta 3.75, batch size 64, scheduler T0 80, horizon
10,000, patience 560, five outer folds, five inner folds, seed 42, and
diagnostic-classifier covariates Age/Sex. It rejects Diagnosis as either a
preserved ComBat covariate or a missing excluded covariate.

The launch workflow compares the newly written train/dev and outer-test
SubjectID order against the locked fold manifests after every fold. Any
mismatch blocks downstream diagnostic-classifier reconstruction.

## Leakage and lineage guards

- Each harmonizer is fitted on the fold-local outer train/dev VAE pool.
- Outer-test subjects are checked for zero overlap before fitting.
- The fitted transform is applied frozen to classifier train/dev and
  outer-test tensors.
- Diagnosis (`ResearchGroup_Mapped`) is excluded from ComBat.
- The downstream latent cache is written in the same trainer process from the
  already harmonized, fold-normalized tensors and frozen VAE posterior means.
- Cache lineage JSON states and hashes the fitted harmonizer and explicitly
  records `raw_input_cache_reused=false`.
- The downstream diagnostic-classification helper refuses caches without this
  lineage, tunes only within train/dev, estimates score normalization from
  inner OOF train/dev scores, and selects its threshold from those inner OOF
  scores only.

## Artifacts enforced by the guarded launch

Before training starts, the script writes:

- exact source snapshot for the trainer, ComBat helper, launch helpers,
  downstream diagnostic-classification helper, and imported project package;
- live-source and snapshot SHA256 manifests;
- resolved candidate config;
- Git HEAD, branch, porcelain status, worktree diff, index diff, and submodule
  status;
- SHA256 and file statistics for the global tensor and metadata;
- the exact trainer command and a no-write trainer dry-run log.

Each fold must write:

- train/dev and outer-test subject manifests;
- fitted full-tensor and per-channel ComBat objects plus hashes;
- leakage guard, fit audit, pre/post channel summaries, and deterministic
  pre/post sample subject tensors;
- VAE normalization parameters;
- a fold-specific VAE checkpoint;
- full VAE history with reconstruction, KL, validation losses, beta,
  learning-rate start/end trajectories, selected checkpoint epoch and terminal
  epoch;
- correctly harmonized train/dev and outer-test posterior-mean caches and
  lineage JSON.

After all five folds and split-parity checks pass, the script freezes the
fold-safe L2 logistic downstream diagnostic classifiers and writes inner-OOF
train/dev predictions, one outer-fold prediction per subject, foldwise and
pooled metrics, serialized fold classifiers, and their SHA256 manifest.

## Non-overwrite behavior

The shell launcher refuses to proceed if any of these exist:

- the new local run directory;
- the corresponding large-disk run directory;
- either new split-preview file.

The historical fold-ComBat run is never used as an output and is not modified.
The downstream helper also refuses to overwrite any classifier model or
prediction output.

## Prepared commands

Safe validation only:

```bash
scripts/revision_bspc_2026/guarded_launch_foldcombat_cleanrepro_20260706.sh --preflight-only
```

Future explicit launch:

```bash
scripts/revision_bspc_2026/guarded_launch_foldcombat_cleanrepro_20260706.sh --confirm-training
```

Read-only monitoring:

```bash
scripts/revision_bspc_2026/monitor_foldcombat_cleanrepro_20260706.sh
```

## Validation performed during preparation

- Bash syntax checks: PASS.
- Python compilation checks: PASS.
- JSON syntax check: PASS.
- exact candidate-versus-reference parameter-diff assertion: PASS.
- ComBat dependency import: PASS
  (`neurocombat_sklearn.CombatModel`).
- guarded launcher and trainer dry-run: PASS.
- monitor status: `NOT_CREATED`.
- training execution: **NO**.
- new run directory creation: **NO**.

The Matplotlib import used a temporary cache under `/tmp` during dry-run
because the user cache directory was not writable. This did not create a run
artifact and did not affect validation.

## Prepared-source fingerprints

| File | SHA256 |
|---|---|
| clean config | `753d8971eaefb89c6d98c6c143443d2d947be4dd2e1d5add9b1a37b78fd7abe6` |
| trainer | `2554f060a9ee88cd5c4e56218543b78d512bf940456c745268d736851ea95015` |
| foldwise ComBat helper | `dfb2695224997483f6ebc8569e5c3b66a79f65c85ea07135c6890d75ec3ecf3f` |
| guarded Python launcher | `d748bf2be352d25a1d160442ca6e454e134babcccae9bc6160c9468598a40afe` |
| downstream diagnostic-classification helper | `f689687a32414f3e084026e5e5b15c115d6ccb9ef1a0c3dd7e18c10272afffdf` |
| guarded shell launcher | `d102c105c4bca0ef3e886fcbaa4af279641ad37a94d3597782699340b70aa1fb` |
| monitor | `a4a8e12ed0c9ad9b830653b5d2b19a9be35af785f82f2ab6a93664d438d919f1` |

These fingerprints document the prepared state only. At actual launch, the
guard regenerates the fingerprints, snapshots those exact bytes, and aborts if
the live source changes between snapshot and trainer invocation.
