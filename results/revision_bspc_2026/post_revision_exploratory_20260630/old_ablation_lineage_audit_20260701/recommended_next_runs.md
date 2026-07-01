# Recommended Next Runs

**Audit date:** 2026-07-01
**Auditor:** claude-sonnet-4-6 (read-only)
**Context:** Lineage review of old ablation evidence prior to launching FAST++ 800-epoch screen

---

## Executive Summary

The old ablation lineage does **not** justify reproducing [5,2,1] at FAST++ settings.
The [5,2,1] hypothesis was already tested at FULL 5×5 depth and **rejected**. No evidence
supports revisiting it under a new FAST++ screen unless the goal is strictly documentary.

The FAST++ 800-epoch planned screen already covers the most important comparison points.
The mandatory runs in that package are sufficient. The optional [3,4,0,1] is marginal.
Two adjustments are recommended based on the lineage review.

---

## 1. Do NOT reproduce the old [5,2,1] at FAST++ settings

**Reason:** The hypothesis was already falsified at FULL depth (AUC=0.7764, REJECTED).
Repeating it at 800 epochs / 3 folds would produce a result that is:
- Not new (outcome predictable from FULL)
- Not comparable to the FAST++ positive control [1,0,2] (different channel order, different loss mode)
- Not generative of new scientific value for the revision

**Verdict: DO NOT RUN.**

---

## 2. Do NOT add [5,2] as a parsimonious control

**Reason:** The [5,2] 1-SE result came from `ablation_canales.py` at 300 epochs/ld=128,
which is incommensurable with FAST++ runs using `run_vae_clf_ad_inference.py` at 800 epochs/ld=384.
DistanceCorr was artificially elevated as best-single under shallow training conditions and
did not transfer to the FULL 5×5 run. There is no scientific rationale to spend ~2–3 h
of GPU time reproducing an artifact of a methodologically inferior screen.

**Verdict: DO NOT RUN.**

---

## 3. Do NOT add [5,2,0] or [5,0,1] as additional arms

**Reason:** These were not evaluated in the old runs at any depth. They differ from the
planned FAST++ arms in channel order and would add complexity without a clear hypothesis.
The FAST++ package already covers [5,0,1] as the `ablation_ch5_0_1_distCorr_OMST_pearsonFull`
arm. Running additional DistanceCorr anchored sets would violate the rule against post-hoc
cherry-picking and would consume ~2–4 h GPU time each with no pre-specified hypothesis.

**Note:** `ch5_0_1` IS already in the planned FAST++ package — it is the arm that replaces
ch1 with ch5 relative to [1,0,2]. This is the correct scientific question: does DistanceCorr
bring value when the rest of [1,0,2] architecture is preserved?

**Verdict: NO additional DistanceCorr arms needed.**

---

## 4. Proceed with planned FAST++ 800-epoch screen as-is

The planned FAST++ package (`partA_channel_ablation_preflight`) covers exactly the right
scientific questions given the lineage:

| Arm | Channels | Rationale |
|:----|:--------|:---------|
| ablation_ch1_pearsonFull | [1] | Baseline single; sanity check |
| ablation_ch1_0_pearsonFull_OMST | [1,0] | Promoted pair minus MI-kNN |
| **ablation_ch1_0_2_pearsonFull_OMST_MI** | **[1,0,2]** | **Positive control = reference channels** |
| ablation_ch4_0_1_dfcStd_OMST_pearsonFull | [4,0,1] | dFC_StdDev replaces MI-kNN |
| ablation_ch3_0_1_dfcMean_OMST_pearsonFull | [3,0,1] | dFC_AbsDiffMean replaces MI-kNN |
| ablation_ch5_0_1_distCorr_OMST_pearsonFull | [5,0,1] | DistanceCorr replaces MI-kNN |
| ablation_ch3_4_0_1_dfcBoth_OMST_pearsonFull (optional) | [3,4,0,1] | 4-channel with both dFC types |

**Run order:**
1. Positive control [1,0,2] first — must pass gate before running experimental arms.
2. If positive control fails gate, diagnose and do not proceed.
3. [1], [1,0], [4,0,1], [3,0,1], [5,0,1] in any order.
4. Optional [3,4,0,1] only if disk space and time allow.

**Gate criteria (from preflight_status.md):**
- ROC-AUC ≥ 0.760 (positive control should be ~0.76–0.79)
- PR-AUC ≥ 0.500
- Fold-AUC std ≤ 0.07
- Manufacturer leakage BA ≤ 0.45
- Active units ≥ 50

---

## 5. Validate recon_loss_mode for cross-channel comparability

**Critical note from lineage review:**
- Old FAST runs (both `ablation_canales.py` and `run_vae_clf_ad_inference.py`) used unspecified
  or channel-count-dependent loss modes.
- Old FAST meta647 used `ablation_canales.py` whose loss mode is unknown.
- The 3x3 FAST runs used `mse_sum_batchmean_current` (sums over all channels without normalising).
- The FAST++ planned runs use `offdiag_channelmean_sum` — this is the correct choice for
  cross-channel comparability and is confirmed in the preflight configs.

**No action needed:** The preflight package already sets `recon_loss_mode=offdiag_channelmean_sum`
in all 7 FAST++ configs. This is a known critical difference from all prior FAST runs.
AUCs from FAST++ are NOT directly comparable to old FAST AUCs; this must be stated clearly
in any manuscript or supplementary text that presents both.

---

## 6. What to Expect from FAST++ [1,0,2] (positive control)

Based on the lineage:
- FAST 3x3 [1] alone achieved AUC=0.7621 at ld=128 (with mse_sum_batchmean_current).
- FAST++ [1,0,2] at ld=384, 800 epochs, offdiag_channelmean_sum will likely land in 0.74–0.78.
- The FULL 5x5 reference is 0.7952 (10× more epochs, OOF-ECDF calibrated).
- An 800-epoch gate of AUC≥0.760 is conservative but realistic for the positive control.

If the positive control lands below 0.74, this suggests either:
- `offdiag_channelmean_sum` suppresses AUC relative to `mse_sum_batchmean_current` at short run lengths
- Or a configuration error — inspect per-fold AUCs for VAE val split issues before concluding failure.

---

## 7. Summary Decision Table

| Option | Verdict | Reason |
|:-------|:-------:|:-------|
| Reproduce [5,2,1] at FAST++ 800ep | **NO** | Falsified at FULL; no new value |
| Add [5,2] parsimonious control | **NO** | Artifact of ablation_canales/300ep; not transferable |
| Add [5,2,0] or [5,0,1] (extra) | **NO** | [5,0,1] already in FAST++ package |
| Run FAST++ planned 6 required arms | **YES** | Correct scientific question; preflight PASS |
| Run FAST++ optional [3,4,0,1] | **OPTIONAL** | Low priority; only if time/disk allow |
| Validate offdiag_channelmean_sum | **ALREADY DONE** | Confirmed in all 7 preflight configs |
