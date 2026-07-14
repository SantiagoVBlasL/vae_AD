# OASIS external Dataset-batch ComBat limitations

This analysis is transductive and post-hoc: OASIS connectivity features and Age/Sex are used to estimate an unsupervised Dataset-batch ComBat mapping. OASIS diagnosis labels are not used for harmonisation or any scoring-model operation, but the external feature distribution is observed during harmonisation.

Feature-space ComBat is applied before a VAE that was trained on unadapted ADNI connectivity matrices. The ADNI-reference variant is the methodologically cleaner feature-space variant because it targets the original ADNI training domain. The no-reference variant targets a pooled ADNI/OASIS space and is therefore exploratory for a frozen ADNI VAE.

The official locked OASIS scoring convention did not serialize fold-specific downstream diagnostic classifier estimators. To remain comparable to the published locked OASIS reference, this script reconstructs the downstream readout from ADNI train/dev latents only and applies the locked ECDF/threshold convention to OASIS. No OASIS labels enter that reconstruction. If a stricter interpretation requires already-serialized diagnostic readouts only, the current repository does not support a feature-space external-batch ComBat primary analysis without first freezing and versioning those locked readouts.

Do not interpret improved operating-point sensitivity/specificity alone as evidence of improved transport unless ROC-AUC/PR-AUC paired bootstrap intervals support it.
