# Manuscript-safe wording

## Concise Methods sentence

The canonical OASIS3 frozen external evaluation used 180 subject-session observations (90 CN/90 AD) from selected Siemens TrioTim resting-state BOLD runs whose OASIS3 JSON sidecars reported TR = 2.2 s; connectivity was computed run-wise from selected preprocessed ROI time series with 164 time points per selected run and averaged within subject/session (2 observations with one selected run, 177 with two, 1 with four).

## Longer provenance note

The score cohort was anchored to the subject-level ensemble rows that produced the frozen `runwise164_pilot_parity` OASIS result (ROC-AUC 0.6478, PR-AUC 0.6678). Selected BOLD run identifiers were matched explicitly to `OASIS3_MR_json.csv` by subject, session label, and run token. All selected rows reported Siemens/TrioTim and TR = 2.2 s. The 164-point value refers to each selected preprocessed ROI time series/run used for run-level connectome estimation; for most observations, two such run-level connectomes were normalized and averaged, so 328 is a subject/session total across selected runs rather than a single selected run length.

## Limitations sentence

The official OASIS3 JSON metadata did not contain a volume-count column, so timepoint counts were traced from immutable preprocessing/run-selection manifests and tensor-construction artifacts rather than from the sidecar CSV itself.
