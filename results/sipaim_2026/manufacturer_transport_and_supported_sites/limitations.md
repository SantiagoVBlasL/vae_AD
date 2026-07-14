# Limitations

- SiteCode is reconstructed from ADNI SubjectID and validated against numeric Site3 where present; it is an acquisition-site proxy, not a complete scanner/protocol descriptor.
- Fold-local site decoding residualizes diagnosis, Age, and Sex, but residual confounding by protocol, scanner software, motion, and timepoint regime may remain.
- Classifier/readout leave-one-site-out freezes the final VAE latent representation; it does not test a VAE trained without the target site.
- OASIS Wasserstein analysis uses existing final runwise164 fold-level latent vectors. No raw OASIS data were reopened and no OASIS inference was rerun.
- External performance versus Wasserstein shift has only five fold-level points and is descriptive only.
- No TDA, Mapper, decoder traversal, adversarial training, CovBat, CORAL, or new VAE training was performed.
