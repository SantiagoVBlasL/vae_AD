# Terminology reconciliation

Locked wording rules, verified against the rebuilt files in this repair
package by direct grep (not by re-derivation):

| Rule | Status |
|---|---|
| TR=2.2 s is acquisition metadata from selected OASIS3 sidecars, not a raw-volume claim | Preserved verbatim from original; TR=2.2 s confirmed for 180/180 rows by independent recount of `oasis180_scan_crosswalk.csv` |
| 164 = selected preprocessed ROI time points per run (not a subject-session total) | Preserved verbatim from original |
| 328 = modal total across two selected 164-point runs, not the length of one raw acquisition | Preserved verbatim from original; independently reproduced as 177/180 observations |
| 656 occurs for one observation with four selected runs | Preserved verbatim from original; independently reproduced as 1/180 observations |
| `OASIS3_MR_json.csv` has no native volume-count field | Preserved verbatim from original |
| The phrase "328 raw volumes" | Absent from both rebuilt files (`328 raw volumes` not found by case-insensitive search) |

## Independent recount from the canonical crosswalk (this repair run)

- n_canonical = 180 (subjects = 180)
- Siemens/TrioTim = 180/180
- TR=2.2s = 180/180
- 1 selected run: 2 observations
- 2 selected runs: 177 observations
- 4 selected runs: 1 observations
- Unresolved observations: 0

All values above match `oasis180_protocol_summary.md`, `published_claim_reconciliation.md`,
and `verification_gates.json` in the original 20260713 audit package exactly.
