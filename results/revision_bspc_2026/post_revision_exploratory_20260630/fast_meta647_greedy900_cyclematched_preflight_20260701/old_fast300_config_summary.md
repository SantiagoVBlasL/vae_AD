# Old FAST300 Config Summary

- Run id: `fast128_all7_finalcohort_greedy_screen_meta647_valsplitfix_20260622`
- Label: exploratory FAST screening, not confirmatory FULL model selection
- Script: `/home/diego/proyectos/vae_AD/scripts/ablation_canales.py`
- Tensor: `/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz`
- Metadata: `/home/diego/proyectos/vae_AD/results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv`
- Output root: `/media/diego/Datos/vae_AD_results/revision_bspc_2026/fast128_all7_finalcohort_greedy_screen_meta647_valsplitfix_20260622`
- Candidate channels: `[0, 1, 2, 3, 4, 5, 6]`
- Epochs: 300
- Latent dim: 128
- Beta: 2.5
- LR scheduler: cosine_warm, T0=30
- Beta cycles: 4
- Beta cycle length: 75.0 epochs
- Cyclical beta ratio increase: 0.4
- Outer folds/repeats: 3 x 1
- Seed: 42
- Early stopping patience: 30
- VAE validation split ratio: 0.2
- Classifier: fixed logreg ablation readout with Age/Sex metadata, class weight enabled=True
- Strict metadata intersection: True
- Strict VAE val split abort: True

Old selected greedy path from audit:

|   step |   n_channels | channel_indices   | channel_names                               |   fold1_auc |   fold2_auc |   fold3_auc |   mean_auc |   std_auc |   se_auc |   delta_vs_prev |
|-------:|-------------:|:------------------|:--------------------------------------------|------------:|------------:|------------:|-----------:|----------:|---------:|----------------:|
|      0 |            1 | [5]               | DistanceCorr                                |      0.7288 |      0.67   |      0.8247 |     0.7412 |    0.0781 |   0.0451 |        nan      |
|      1 |            2 | [5, 2]            | DistanceCorr+MI_KNN                         |      0.7497 |      0.8231 |      0.7716 |     0.7815 |    0.0377 |   0.0218 |          0.0403 |
|      2 |            3 | [5, 2, 1]         | DistanceCorr+MI_KNN+Pearson_Full            |      0.7742 |      0.7794 |      0.8156 |     0.7897 |    0.0226 |   0.013  |          0.0083 |
|      3 |            4 | [5, 2, 1, 4]      | DistanceCorr+MI_KNN+Pearson_Full+dFC_StdDev |      0.7576 |      0.7778 |      0.7734 |     0.7696 |    0.0106 |   0.0061 |         -0.0201 |
