# OASIS Mega 90/90 External Inference Model Panel

Scored the mega-OASIS 90CN/90AD tensors with ADNI-trained fold VAEs and ADNI-only Stage B OOF-ECDF readout reconstruction.

Primary convention: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.

Outputs include predictions, primary/foldwise metrics, confusion matrices, score distributions, ADNI-vs-OASIS generalization deltas, interpretation labels, and final recommendation.

Guardrails: no OASIS training, threshold fitting, calibration fitting, tensor modification, metadata modification, or model artifact modification.
