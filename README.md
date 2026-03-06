# vae_AD

Pipeline CNN β-VAE + clasificadores clásicos para AD vs CN.

## Setup
- Entorno conda recomendado: `vae_ad`
- Locks: ver `configs/vae_ad.lock.yml` y `configs/vae_ad.conda-explicit.txt`

## Ejecutar
Ejemplo:
python scripts/run_vae_clf_ad_inference.py --help

## Datos
Ver `data/README.md` (los tensores grandes no se versionan).

## Arquitectura del pipeline de interpretabilidad

Para un trazado completo de ejecución (call graph, roles de cada módulo e
interfaces de datos) del pipeline que arranca desde
`notebooks/05_run_interpretability_ad.ipynb`, ver:

➜ **[docs/interpretability_pipeline.md](docs/interpretability_pipeline.md)**

## How `03_a` and `03_b` are generated

Both COVID analysis notebooks depend on upstream scripts that must run first.
See **[docs/repo_dataflow.md](docs/repo_dataflow.md)** for the full dependency
graph including all scripts and result folders.

### Notebook `03_a` — COVID Transfer Inference

```bash
# Step 1 — Generate core inference tables (predictions, recon error, latent dist)
conda run -n vae_ad python scripts/inference_covid_from_adcn.py

# Step 2 — Run notebook through §6 (derives ADNI thresholds)
#          Then run optional helper scripts:
conda run -n vae_ad python scripts/ood_policy_adni.py
conda run -n vae_ad python scripts/analysis_covid_paper_fixes.py

# Step 3 — Execute full notebook
conda run -n vae_ad jupyter nbconvert --to notebook --execute \
  notebooks/03_a_inference_covid_from_adcn.ipynb
```

### Notebook `03_b` — COVID Classification Probing

```bash
# Step 1 — Run probing experiment (5 ADNI encoders × 5 COVID CV folds)
conda run -n vae_ad python scripts/run_covid_classification_probing.py

# Step 2 — Execute audit notebook
conda run -n vae_ad jupyter nbconvert --to notebook --execute \
  notebooks/03_b_covid_classification_probing_audit.ipynb
```
