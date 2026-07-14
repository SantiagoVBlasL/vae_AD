# Manuscript source of truth

## Overleaf es la única fuente editable

El manuscrito SIPAIM 2026 se escribe y mantiene exclusivamente en Overleaf.
Ningún archivo `.tex` ni PDF compilado dentro de este repositorio es, ni
será nunca, la fuente editable del manuscrito. Esta convención rige para
todo el proyecto, no solo para SIPAIM (aplica igualmente al manuscrito de
BSPC).

Consecuencias prácticas:

- No se edita el manuscrito a partir de una copia local.
- No se generan claims ni redacciones "leyendo" un `.tex` local como si
  fuera autoritativo.
- Cualquier discrepancia entre una copia local y Overleaf se resuelve
  siempre a favor de Overleaf.

## Qué conserva el repositorio

El repositorio conserva lo que sostiene la reproducibilidad del trabajo,
no el manuscrito en sí:

- Código (`scripts/sipaim_2026/`) necesario para reproducir los análisis.
- Configuraciones (`configs/sipaim_2026/`) de los runs citados.
- Evidencia agregada (`results/sipaim_2026/`): tablas y métricas a nivel de
  fold/grupo que sustentan los claims del paper — nunca tablas a nivel de
  sujeto individual.
- Figuras canónicas (`figures/sipaim_2026/`) y sus datos agregados de
  figura.
- Documentación de métodos, decisiones de análisis y procedimiento de
  reproducibilidad (`docs/sipaim_2026/`).

## Exports del manuscrito

Los exports del manuscrito (PDF compilado, `.tex` descargado de Overleaf)
se guardan, cuando corresponda, **solo como snapshots de submission** —
es decir, como una instantánea congelada asociada a un envío concreto
(p. ej. al momento de un submit o resubmit), nunca como un archivo de
trabajo que se edita localmente ni como insumo para futuras ediciones.
Un snapshot de submission es un artefacto de archivo histórico, no la
fuente de verdad vigente.

## Copias locales que NO son canónicas

Las copias locales de `site_geometry_analysis_protocol_v4` (tanto el
`.tex` bajo `manuscript/sipaim_2026/` como la copia anterior bajo
`docs/revision_bspc_2026/site_geometry_analysis/`) **no son canónicas**.
Son, en el mejor de los casos, borradores de trabajo o snapshots
intermedios que ya divergieron entre sí y del manuscrito vigente en
Overleaf. No deben citarse, extenderse ni tratarse como fuente en ningún
análisis, auditoría o commit futuro.
