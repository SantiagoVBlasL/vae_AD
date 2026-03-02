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
