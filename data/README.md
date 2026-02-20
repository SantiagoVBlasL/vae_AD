# data/

Este directorio contiene datos locales (NO versionados en git).

Sugerido:
- metadata (CSV) sí puede versionarse (ej: SubjectsData_*.csv, mappings).
- tensores y volúmenes grandes NO: .npz/.npy/.nii/.nii.gz

Ejemplo de paths esperados por los scripts:
- data/.../GLOBAL_TENSOR_*.npz
- data/SubjectsData_*.csv
