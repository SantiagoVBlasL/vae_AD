# OASIS label-definition sensitivity interpretation

## Guardrails

- No model was trained.
- No VAE inference or preprocessing was run.
- Raw data were not modified.
- OASIS labels were not used for calibration, threshold selection, harmonisation, or model selection.
- CDR 0.5 was kept as its own ordinal group and excluded from the strict binary analysis.

## Direct answers

1. **Does external ROC-AUC improve when restricting OASIS to high-confidence CDR-defined CN/AD?** Yes for the locked frozen-transfer arm: ROC-AUC changed from 0.6478 on OASIS-current-180 to 0.7093 on OASIS-strict-128. PR-AUC changed from 0.6678 to 0.5565. This is a label-definition sensitivity result, not a model-selection criterion.

2. **Are CDR 0.5 subjects intermediate in model score between CDR 0 and CDR >= 1?** Yes descriptively for locked scores: mean score CDR 0 = 0.3190, CDR 0.5 = 0.4032, CDR >= 1 = 0.4908. Spearman trend across ordinal CDR groups was rho=0.2790, p=0.0001491.

3. **Does the current OASIS-180 result underestimate performance because many positives are very mild/ambiguous?** Likely yes. The current labels include 51 CDR 0.5 subjects as AD and 1 CDR 0.5 subject as CN. Removing CDR 0.5 and evaluating only CDR 0 vs CDR >= 1 increases the locked ROC-AUC, consistent with ambiguous/mild positives depressing binary discrimination.

4. **Primary external result recommendation.** Keep OASIS-current-180 as the primary external result and report OASIS-strict-128 as a planned/transparent label-definition sensitivity. Switching the primary result to strict-128 would discard the originally evaluated external cohort and can look post-hoc, even though it clarifies that stricter CDR labels yield better apparent transportability.

## Bootstrap note

OASIS-strict-128 is a subset/relabeling sensitivity rather than the same target population as OASIS-current-180. Therefore the metrics table reports separate diagnosis-stratified subject-bootstrap confidence intervals for each set, instead of a paired current-vs-strict bootstrap delta.
