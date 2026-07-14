# Final clean fold-wise input-ComBat audit

## Verdict

PASS_MANUSCRIPT_GRADE_AUDIT

Fold-safety verdict: **FOLD_SAFE**. The clean run loads the original global tensor and applies ComBat fold-locally after splitting; this is expected and was not treated as a global pre-harmonized tensor requirement.

## Internal downstream diagnostic classifier

Primary clean ComBat ADNI OOF result, using posterior latent means + Age + Sex, class-weighted L2 logistic regression, five inner folds, inner-OOF ECDF normalization, and the prespecified inner-OOF threshold:

- ROC-AUC 0.7493; PR-AUC 0.5327
- balanced accuracy 0.6960; sensitivity 0.7320; specificity 0.6600; F1 0.5259; Brier 0.2432
- confusion matrix: TN=198, FP=102, FN=26, TP=71

Locked reference under the same convention: ROC-AUC 0.7952; PR-AUC 0.5739; balanced accuracy 0.7260.

Paired 10,000 diagnosis-stratified subject bootstrap, clean minus locked:

- delta ROC-AUC -0.0459 [-0.0807, -0.0118]
- delta PR-AUC -0.0412 [-0.1151, +0.0266]

## Manufacturer information and transport

Manufacturer decoding mean BACC: locked 0.6659 ± 0.0506; clean ComBat 0.4404 ± 0.0533; paired mean delta -0.2254. Lower values indicate less manufacturer-decodable latent information.

Philips CN FPR rows are included in `cleanrepro_manufacturer_transport.csv`; primary subgroup quantity is Philips CN false-positive rate.

| arm                         | manufacturer   |   n |   n_cn |   n_ad |   cn_false_positive_rate |   sensitivity |   specificity |   roc_auc |   pr_auc |   tn |   fp |   fn |   tp |
|:----------------------------|:---------------|----:|-------:|-------:|-------------------------:|--------------:|--------------:|----------:|---------:|-----:|-----:|-----:|-----:|
| locked                      | Philips        | 145 |     99 |     46 |                 0.454545 |      0.782609 |      0.545455 |  0.727273 | 0.570815 |   54 |   45 |   10 |   36 |
| clean_foldwise_input_combat | Philips        | 145 |     99 |     46 |                 0.323232 |      0.695652 |      0.676768 |  0.729688 | 0.575259 |   67 |   32 |   14 |   32 |

## Frozen OASIS inference

Clean OASIS used only ADNI-fitted preprocessing, serialized fold ComBat objects, VAE checkpoints, frozen downstream diagnostic classifiers, ECDF mappings and thresholds. No OASIS fitting, calibration, or threshold selection was performed.

- Locked OASIS: ROC-AUC 0.6478; PR-AUC 0.6678; balanced accuracy 0.5667; sensitivity 0.2667; specificity 0.8667; Brier 0.2581
- Clean ComBat OASIS: ROC-AUC 0.6460; PR-AUC 0.6478; balanced accuracy 0.5833; sensitivity 0.3444; specificity 0.8222; Brier 0.2575

## Distribution shift

ADNI-vs-OASIS paired locked-vs-clean latent distances: w1_mean_all_dims: 4/5 folds closer, mean delta -0.0031; pca10_w1_mean: 2/5 folds closer, mean delta +0.0246; gaussian_w2: 3/5 folds closer, mean delta -0.1453. External alignment is considered consistently improved only when all folds move closer for all distance definitions.

## Training reproducibility

Compared to the historical fold-ComBat run, selected/terminal epochs and validation objectives are summarized in `cleanrepro_training_reproducibility.csv`.

|   fold |   selected_checkpoint_epoch |   terminal_epoch |   selected_validation_objective |   historical_selected_checkpoint_epoch |   historical_terminal_epoch | selected_epoch_reproduced   | terminal_epoch_reproduced   | validation_loss_reproduced_allclose   |
|-------:|----------------------------:|-----------------:|--------------------------------:|---------------------------------------:|----------------------------:|:----------------------------|:----------------------------|:--------------------------------------|
|      1 |                        2879 |             3439 |                         33927.2 |                                   2879 |                        3439 | True                        | True                        | True                                  |
|      2 |                        3187 |             3747 |                         33440.1 |                                   3187 |                        3747 | True                        | True                        | True                                  |
|      3 |                        2542 |             3102 |                         33871.7 |                                   2542 |                        3102 | True                        | True                        | True                                  |
|      4 |                        2954 |             3514 |                         33645   |                                   2954 |                        3514 | True                        | True                        | True                                  |
|      5 |                        3360 |             3920 |                         34112.8 |                                   3360 |                        3920 | True                        | True                        | True                                  |

## Deliverables

All requested CSV deliverables and this report were written in this directory. The audit command log is `command_log.json`.
