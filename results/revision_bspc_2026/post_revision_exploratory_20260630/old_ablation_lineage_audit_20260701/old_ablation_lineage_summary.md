# Old Ablation Lineage Summary

**Audit date:** 2026-07-01
**Auditor:** claude-sonnet-4-6 (read-only)
**Output dir:** `results/revision_bspc_2026/post_revision_exploratory_20260630/old_ablation_lineage_audit_20260701/`

---

## 1. Overview

This document consolidates the lineage of all channel-ablation runs that preceded the planned
FAST++ 800-epoch screen (Part A of the post-revision exploratory package). The source of the
AUC≈0.78–0.79 results attributed to [5,2,1] (DistanceCorr + MI_KNN + Pearson_Full) is traced
through three distinct experiments. All existing runs are fully evaluated; the FAST++ 800-epoch
package has been planned separately but not yet launched.

---

## 2. Chronological Run Lineage

### 2.1 FAST Greedy v1 — `greedy_fast_channel_selection` (2026-05-15)

**Settings:**
- Script: `run_vae_clf_ad_inference.py`
- outer_folds=5 (later flagged as stale for within-FAST ranking purposes)
- latent_dim=128, beta=2.5, epochs=960, T0=80, cycles=12
- Metadata: `training_ready_metadata` (pre-valsplitfix)
- Stage B: logreg_l2 classifier-only readout on saved latent mu

**What happened:**
- Evaluated all 7 single channels.
- ch0 and ch1 were flagged as pre-existing outer=5 runs — marked invalid for greedy 3x3 ranking.
- ch2, ch3, ch4, ch5, ch6 ran at outer=3 (or were later re-run).
- ch1 (PearsonFull): mean AUC=0.7969, ch5 (DistanceCorr): 0.7865 at outer=5
- Greedy trace stopped at step 1 with `[1]` as anchor; no pairs evaluated in this directory.
- **Verdict:** Superseded by the 3x3 FAST run.

### 2.2 FAST Greedy 3×3 — `greedy_fast_channel_selection_3x3` (2026-05-15/16)

**Settings:**
- Script: `run_vae_clf_ad_inference.py`
- outer_folds=3, inner_folds=3
- latent_dim=128, beta=2.5, epochs=960, T0=80, cycles=12, patience=240
- Metadata: `training_ready_metadata` (pre-valsplitfix; N unconfirmed but similar pool)
- Stage B: logreg_l2 classifier-only readout with primary threshold `inner_oof_target_sens_ge_0p70_max_spec`

**Results — singles (step 1):**

| Channel | Name | Mean AUC | SE |
|:-------:|:-----|:--------:|:---:|
| ch1 | Pearson_Full_FisherZ_Signed | 0.7621 | 0.0119 |
| ch5 | DistanceCorr | 0.7439 | 0.0049 |
| ch0 | Pearson_OMST | 0.7402 | 0.0188 |
| ch2 | MI_KNN | 0.7298 | 0.0134 |
| ch3 | dFC_AbsDiffMean | 0.6537 | 0.0218 |
| ch4 | dFC_StdDev | 0.6108 | 0.0387 |
| ch6 | Granger_F_lag1 | 0.5548 | 0.0194 |

**Greedy anchor: ch1 (PearsonFull) — best single.**

**Results — pairs from ch1 (step 2):**

| Pair | Names | Mean AUC | SE | Δ vs [1] |
|:----:|:------|:--------:|:---:|:--------:|
| [1,4] | PearsonFull + dFC_StdDev | 0.7648 | 0.0096 | +0.0027 |
| [1,2] | PearsonFull + MI_KNN | 0.7610 | 0.0134 | +0.0027 (−0.0011 vs [1,4]) |
| [1,6] | PearsonFull + Granger | 0.7547 | 0.0042 | … |
| [1,0] | PearsonFull + OMST | 0.7536 | 0.0075 | … |
| [1,5] | PearsonFull + DistanceCorr | 0.7496 | 0.0116 | … |
| [1,3] | PearsonFull + dFC_AbsDiff | 0.7464 | 0.0150 | … |

**Stop rule:** max pair delta = 0.0027 < 0.01 threshold → stop. **Accepted greedy set: [1].**

**Triple lookahead from [1,4]** (`greedy_fast_channel_selection_3x3_triple_lookahead_1_4`):
Explored [1,4,X] for all remaining X. Best triple:
- [1,4,0]: mean AUC=0.7635 (+0.0027 vs [1,4])
- [1,4,5]: mean AUC=0.7595
- [1,4,2]: mean AUC=0.7561
None of these significantly exceed [1,4] by the stop criterion.

**Key finding:** Under 3x3/960-epoch/ld=128 settings with `training_ready_metadata`,
PearsonFull is the best single channel; DistanceCorr is second but well below ch1.
**[5,2,1] was never evaluated as a complete set in this 3x3 screen** — the greedy anchor was ch1, not ch5.

### 2.3 FAST meta647 Greedy Screen — `fast128_all7_finalcohort_greedy_screen_meta647_valsplitfix_20260622` (2026-06-22)

**Settings:**
- Script: `ablation_canales.py` (DIFFERENT from run_vae_clf_ad_inference.py)
- outer_folds=3
- latent_dim=128, beta=2.5, epochs=300, T0=30, beta_cycles=4, patience=30
- Loss mode: NOT SPECIFIED in config — ablation_canales.py may use a different default
- Metadata: `patched_metadata_candidate.csv` (N=647 with valsplitfix, strict intersection)
- N_effective=646 (subject 128_S_2002 excluded as tensor-only)
- Classifier: fixed logistic regression (not full L2 grid search); `metric=auc`

**Results — single channels (step 0):**

| Channel | Name | F1 AUC | F2 AUC | F3 AUC | Mean AUC | SE |
|:-------:|:-----|:------:|:------:|:------:|:--------:|:---:|
| ch5 | DistanceCorr | 0.7288 | 0.6700 | 0.8247 | 0.7412 | 0.0451 |
| ch1 | Pearson_Full | 0.6830 | 0.7528 | 0.7641 | 0.7333 | 0.0253 |
| ch2 | MI_KNN | 0.6694 | 0.7400 | 0.7262 | 0.7119 | 0.0374 |
| ch0 | OMST | 0.5458 | 0.6663 | 0.7137 | 0.6419 | 0.0500 |
| ch4 | dFC_StdDev | 0.5815 | 0.6409 | 0.6844 | 0.6356 | 0.0298 |
| ch3 | dFC_AbsDiff | 0.6239 | 0.5984 | 0.6494 | 0.6239 | 0.0147 |
| ch6 | Granger | 0.5542 | 0.5153 | 0.5550 | 0.5415 | 0.0131 |

**Critical observation:** DistanceCorr (ch5) is the best single channel at 300 epochs / ld=128,
whereas PearsonFull (ch1) was best at 960 epochs / ld=128 in the 3x3 run. Channel rankings
shifted with model capacity and training depth.

**Greedy path:**

| Step | Set | Mean AUC | SE | Δ |
|:----:|:----|:--------:|:---:|:---:|
| 0 | [5] DistanceCorr | 0.7412 | 0.0451 | — |
| 1 | [5,2] DistanceCorr+MI_KNN | 0.7815 | 0.0218 | +0.0403 |
| 2 | [5,2,1] +Pearson_Full | 0.7897 | 0.0130 | +0.0083 |
| 3 | [5,2,1,4] +dFC_StdDev | 0.7696 | 0.0061 | **−0.0201** |

**Best set: [5,2,1] at mean AUC=0.7897.**
**1-SE parsimonious: [5,2] at 0.7815 (within 1-SE threshold of 0.7767).**

**[1,0,2] was NOT evaluated in this FAST screen.** The greedy path anchored on ch5 and never
explored ch1 as the anchor or the complete [1,0,2] combination.

---

## 3. Source of the "AUC≈0.78–0.79" Result

The AUC≈0.78–0.79 result for channel sets involving DistanceCorr originates from:

1. **FAST meta647 [5,2,1]: mean AUC=0.7897** (3-fold, 300 epochs, ld=128, `ablation_canales.py`)
   — This is the primary source cited in discussions. It was labeled "exploratory screening only."

2. **FAST meta647 [5,2]: mean AUC=0.7815** (same settings, 1-SE parsimonious set)

3. **FAST 3x3 v2 [1] (outer=3): mean AUC=0.7621** (960 epochs, ld=128, `run_vae_clf_ad_inference.py`)
   — The best FAST result under the standard greedy screen with the full pipeline.

4. **FAST v1 [1] (outer=5 STALE): mean AUC=0.7969** (but outer=5 invalidates for greedy screening)

**The AUC≈0.79 seen in the FAST meta647 run is NOT comparable to the FULL [1,0,2] AUC=0.7952.**
The comparison is confounded by: different script, different epochs (300 vs 10000), different
latent_dim (128 vs 384), different loss mode (unknown vs mse_sum_batchmean_current),
different n_folds (3 vs 5), and no OOF-ECDF calibration in the FAST run.

---

## 4. Epoch / Epoch-count Reconciliation

A discrepancy between "300 epochs" and "960 epochs" cited in different documents is resolved:

| Run | Script | Epochs | T0 | Cycles |
|:----|:-------|:------:|:--:|:------:|
| FAST v1 greedy (outer=5) | run_vae_clf_ad_inference.py | 960 | 80 | 12 |
| FAST 3x3 greedy v2 | run_vae_clf_ad_inference.py | 960 | 80 | 12 |
| FAST meta647 (`ablation_canales.py`) | ablation_canales.py | **300** | 30 | 4 |

The "300 epoch" figure comes exclusively from the `ablation_canales.py` meta647 run.
All greedy runs using `run_vae_clf_ad_inference.py` used 960 epochs.
Neither is comparable to the FULL 10000-epoch runs or the planned FAST++ 800-epoch runs.

**The "300 epochs" in audit texts refers to the meta647 ablation_canales run only.**

---

## 5. Why [5,2,1] Was Considered Hypothesis-Generating Only

From `fast_meta647_ablation_results_audit_20260622/final_recommendation.md` (2026-06-22):

1. FAST AUC=0.7897 is 3-fold raw, uncalibrated, at ld=128/300ep.
   The delta vs FULL [1,0,2] OOF-ECDF AUC=0.7952 is only -0.0055 — but this comparison
   is confounded by all the parameter differences above.

2. FAST [5,2,1] does not exceed [1,0,2] by the required >0.02 margin to justify FULL 5×5
   training based on FAST alone (per the stated promotion policy at the time).

3. [1,0,2] was not evaluated in the FAST meta647 greedy screen, so FAST cannot establish
   channel superiority of [5,2,1] vs [1,0,2].

4. FAST uses `ablation_canales.py`, not the standard pipeline; results are not reproducible
   under the promoted model's codebase with the same settings.

5. The 1-SE parsimonious set [5,2] (AUC=0.7815) is even lower, making the evidence for
   DistanceCorr's necessity fragile.

**Conclusion:** FAST [5,2,1] was used as a hypothesis to motivate one controlled FULL run
(ch521 FULL 5×5), not as evidence of channel superiority.

---

## 6. ch521 FULL 5×5 Run Summary

[5,2,1] was evaluated at FULL settings (ld=384, beta=3.75, epochs=10000, 5×5 CV)
in two runs (one without `_meta647` suffix, one with). The `_meta647` version used
patched_metadata_candidate.csv and is the authoritative decision artifact.

**Stage A pooled logreg:** AUC=0.7615, PR-AUC=0.5016 — operating point very conservative
(Sens=0.175, Spec=0.980); canonical logreg hyperparameter driven to extreme L2.

**Stage B (OOF-ECDF calibrated):**
- AUC=0.7764, PR-AUC=0.5260 (via `recover035_ch521_latent384_beta3p75_stageB_oof_score_calibration_20260623`)
- Reference: AUC=0.7952, PR-AUC=0.5739 (Δ: −0.0187 AUC, −0.0479 PR-AUC)
- Bootstrap: p(cand ≤ ref)=0.938 for AUC; CI [-0.043, +0.005]

**Decision: REJECTED.** Both ADNI gates failed. See `ch521_stageb_decision_summary.md`.

---

## 7. Channel Ranking Stability Across Conditions

| Condition | Best single | 2nd | 3rd | Notes |
|:----------|:------------|:----|:----|:------|
| FAST 3x3 (960ep, ld=128, run_vae_clf) | **ch1** (0.7621) | ch5 (0.7439) | ch0 (0.7402) | Standard pipeline |
| FAST meta647 (300ep, ld=128, ablation_canales) | **ch5** (0.7412) | ch1 (0.7333) | ch2 (0.7119) | Different script + fewer epochs |
| FAST++ planned (800ep, ld=384, run_vae_clf) | TBD | TBD | TBD | FAST++ screen; expected ch1 ≈ winner |

**Takeaway:** Channel rankings are not stable across scripts, epoch counts, and latent dims.
The meta647 run's ranking (ch5 > ch1) is artifact of the short/shallow ablation_canales run.
Under the standard pipeline (run_vae_clf_ad_inference.py), ch1 consistently leads.

---

## 8. Gaps and Missing Information

- The `training_ready_metadata` used in the 3x3 FAST runs does not have a confirmed subject count
  in the available audit files (no foldwise pool table was found for the 3x3 run).
- The loss mode used by `ablation_canales.py` is not specified in the fast_meta647 config.
  This is an unresolved uncertainty; it may use a per-channel averaging form or raw MSE.
- No FULL 5×5 single-channel runs exist for ch2 (MI_KNN) or ch0 (OMST) alone with ld=384.
  The FAST++ 800-epoch planned run is the first systematic FULL-pipeline single-channel screen
  at the promoted architecture's latent_dim.

---

## 9. Integrity Notes

- All files read as read-only. No modifications were made.
- No training was launched.
- No OASIS scoring was performed.
- No manuscript files were modified.
- Where file sizes appear as "UNKNOWN" in the inventory, they were too large to enumerate
  in the find output or the file content was read directly without stat.
