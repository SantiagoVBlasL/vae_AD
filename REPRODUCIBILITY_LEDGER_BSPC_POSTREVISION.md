# BSPC post-revision exploratory ledger

## Frozen submission

- Frozen tag: bspc-r1-resubmission-20260629
- Submission branch: revision_bspc_2026
- Exploratory branch: exploratory/post-revision-20260630

## Final submitted model

- Run: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
- Channels: [1,0,2]
  - 1 = Pearson_Full_FisherZ_Signed
  - 0 = Pearson_OMST_GCE_Signed_Weighted
  - 2 = MI_KNN_Symmetric
- Latent dim: 384
- Beta: 3.75
- Readout: Logistic Regression L2, latent mu + Age/Sex

## Post-revision goals

1. FAST++ 800-epoch controlled channel ablation.
2. Latent site/manufacturer geometry analysis.
3. Site/cohort threshold-transfer analysis.
4. Foldwise harmonization / ComBat sensitivity.
5. Longer-term: graph-VAE prototype and ADNI multibanda inventory.

## FAST++ design

- Epochs: 800
- Scheduler T0: 80
- Beta cycles: 10
- Beta: 3.75
- Latent dim: 384
- Reconstruction loss: offdiag_channelmean_sum or equivalent
- Stage B: LogReg L2
- First run: positive control [1,0,2]
- Do not launch all candidates before confirming the control.

## Candidate channel sets

- [1]
- [1,0]
- [1,0,2]
- [4,0,1]
- [3,0,1]
- [5,0,1]
- optional: [3,4,0,1]

## Safety rules

- Do not modify the frozen submission tag.
- Do not overwrite promoted model artifacts.
- Do not commit tensors, checkpoints, joblib files, npy/npz files, TIFF/EPS/ZIP exports, or private data.
- Use python3 or the project conda environment, never bare `python` on this machine.
