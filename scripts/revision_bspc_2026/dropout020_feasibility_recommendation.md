# Dropout=0.20 Feasibility Recommendation

## Information Bottleneck (IB) Perspective
In the variational Information Bottleneck framework, the objective is to maximize $I(Z;Y) - \beta I(X;Z)$. In this unsupervised setting, the target is the reconstruction $X$, but our downstream goal is clinical generalizability (invariant to site/scanner noise).

Our previous result showed that relaxing dropout from 0.15 to 0.10 *degraded* performance (AUC: 0.7788 $\rightarrow$ 0.7435). This strongly suggests the model is currently **regularization-bound**. With a lower dropout rate, the model possessed excessive capacity, allowing nuisance high-frequency noise (e.g., scanner effects in the Pearson graphs) to bypass the bottleneck, leading to representation collapse on the out-of-site validation folds.

Given that modifying `dropout_rate_vae` impacts both spatial (`Dropout2d`) and dense (`Dropout`) regularization simultaneously across >35M parameters, tightening it to `0.20` is a mathematically meaningful, single-parameter IB-tightening test. It applies a symmetric, stricter variational bound that forces the model to encode only the most robust, low-frequency connectome motifs.

## Risks of Increasing Dropout to 0.20

1. **Over-compression:** With 20% spatial dropout across four layers and 20% dense dropout on the 16M parameter transition matrices, there is a moderate risk of over-regularization. The model might fail to reconstruct fine-grained details, leading to underfitting.
2. **Loss of Disease Signal:** The subtle, distributed connectivity changes characteristic of early AD/MCI might be aggressively dropped if they are statistically weaker than the primary default-mode network structural backbones.
3. **Scanner/Site Leakage:** This risk is effectively **reduced** by increasing dropout. However, if over-compression forces the model to cling to the most dominant variance in the dataset (which could, perversely, be a major scanner batch effect), it might still fail to generalize.
4. **Fold Instability:** High dropout rates can increase variance during training. In a 5x5 nested CV, this could lead to higher inter-fold variability in convergence, potentially making the final OOF metrics noisier or harder to interpret.

## Executive Decision & Recommendation
**Do not launch the `dropout020` training run.** 

While theoretically defensible as a single controlled IB-tightening test, it is highly likely that the locked configuration (`dropout=0.15`, AUC=0.7788) represents the optimal Pareto frontier for this architecture and dataset size. Optimizing the VAE dropout any further risks crossing into test-set burn-out and p-hacking the representation based on downstream validation metrics, which violates the premise of a *diagnosis-agnostic* generative model.

The most scientifically rigorous path forward is to hold the representation fixed and direct efforts toward **read-only audits** (calibration analysis, threshold optimization for PR-AUC) to prove the clinical robustness of the current 0.7788 AUC model to reviewers.
