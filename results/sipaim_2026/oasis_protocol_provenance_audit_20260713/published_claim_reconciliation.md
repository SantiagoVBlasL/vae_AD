# Published claim reconciliation

| claim                                             | status              | evidence                                                                                                                                                                                                                                              |
|:--------------------------------------------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| TR = 2.5 s                                        | CONTRADICTED        | All 180 canonical observations match selected OASIS3_MR_json bold-rest run rows with RepetitionTime=2.2 s. Non-selected bold-rest test runs at 2.5 s exist for some sessions but are not the selected tensor/scoring runs.                            |
| 164 volumes per analysed scan                     | PARTIALLY_CONFIRMED | Confirmed as 164 time points per selected BOLD run/ROI time series. Not correct as a uniform subject-session total: 177 observations used two selected runs (328 total), 2 used one run (164 total), and 1 used four runs (656 total).                |
| all 164 volumes retained in the primary analysis  | CONFIRMED           | For the primary runwise164 build, scripts do not crop selected 164-point ROI time series; connectomes are computed per selected run and then averaged at connectome level.                                                                            |
| approximately comparable to seven minutes of ADNI | PARTIALLY_CONFIRMED | Each selected OASIS run is 164 x 2.2 s = 360.8 s (6.01 min), which is near but below seven minutes. The canonical observation is usually an average of two 6.01-min run-level connectomes, not one continuous seven-minute series.                    |
| Siemens TrioTim / TR = 2.2 s / 328 raw volumes    | PARTIALLY_CONFIRMED | Siemens TrioTim and TR=2.2 s are confirmed for all 180 selected observations. 328 is the modal subject/session total from two selected 164-point runs (177/180), not a uniform single-run raw acquisition; 2 observations total 164 and 1 totals 656. |

## Supporting distributions

### TR counts

|   normalized_tr_seconds |   n |
|------------------------:|----:|
|                     2.2 | 180 |

### Total selected timepoint counts

|   raw_volumes |   n |
|--------------:|----:|
|           164 |   2 |
|           328 | 177 |
|           656 |   1 |
