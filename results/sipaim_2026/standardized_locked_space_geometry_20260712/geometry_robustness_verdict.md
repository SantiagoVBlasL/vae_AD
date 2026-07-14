# Geometry robustness verdict

## Protocol compliance

- Raw 384-dimensional latent coordinates were never concatenated, averaged, or compared across folds.
- Each fold used only that fold's ADNI train/dev latents to define an AD-minus-CN centroid direction.
- The direction was normalized to unit Euclidean norm, oriented so AD has the larger mean projection, centered at the 50/50 ADNI CN/AD centroid midpoint, and scaled by pooled ADNI train/dev projection SD.
- OASIS subjects were projected independently in each fold; only the resulting dimensionless scalar projections were averaged across folds per subject.
- The five folds are not treated as independent experiments; uncertainty is a subject-level bootstrap over the fixed trained ensemble.
- Bootstrap uncertainty is conditional on the saved trained models and does not include VAE retraining variability.
- No Procrustes, training, inference, target fitting, model selection, or manuscript editing was performed.

## Correct arm scope

Geometry is restricted to the shared locked VAE latent coordinate system:

- `locked_frozen_transfer`
- `external_dataset_combat_adni_reference`
- `external_dataset_combat_no_reference` as secondary

The previous ADNI-fitted Siemens-ComBat arm and every independently retrained VAE arm are excluded from this geometry audit.

## Numerical summary

- Locked projected AUC: 0.6372; AD-CN separation: 0.4540; marginal shift: -0.9348.
- Dataset-ComBat ADNI-reference projected AUC: 0.6369; AD-CN separation: 0.4939; marginal shift: -0.4728.
- Dataset-ComBat no-reference projected AUC: 0.6370; AD-CN separation: 0.4803; marginal shift: -0.6613.
- ADNI-reference deltas vs locked: projected AUC -0.0002; AD-CN separation +0.0398.

## Substantive conclusion

The correction changes the geometric framing but not the substantive interpretation. Dataset-ComBat, especially with ADNI as reference, moves the OASIS marginal centroid projection much closer to the ADNI midpoint in the locked VAE scalar geometry, but it does not provide evidence of improved clinical separation or projected ranking versus locked frozen transfer. The no-reference arm remains secondary because it is transductive and not the strict frozen-transfer condition.
