# -*- coding: utf-8 -*-
"""
betavae_xai.utils.run_io

I/O helpers "paper-grade" para reproducibilidad:
- _compute_file_sha256: fingerprint estable (streaming) de archivos grandes (npz, pt, joblib, etc.)
- _safe_json_dump: JSON robusto (Path, numpy scalars/arrays, pandas, torch, datetime, sets, etc.)
  con escritura atómica (evita JSON truncados si se corta el proceso).

Nota:
- Estas funciones están pensadas para configs/artefactos de runs (run_config.json, feature_columns.json, etc.)
  y para registrar fingerprints de datasets/tensores.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from datetime import datetime, date
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

try:
    import pandas as pd  # opcional
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import torch  # opcional
except Exception:  # pragma: no cover
    torch = None  # type: ignore


PathLike = Union[str, os.PathLike, Path]


def _compute_file_sha256(path: PathLike, block_size: int = 1024 * 1024) -> str:
    """
    Hash estable (streaming) del archivo. Útil como 'fingerprint' del tensor/dataset.
    - block_size default: 1MB (balance razonable entre overhead y throughput)

    Parameters
    ----------
    path : str | PathLike
        Ruta al archivo.
    block_size : int
        Tamaño del bloque de lectura.

    Returns
    -------
    str
        sha256 hex digest.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_json_dump(
    obj: Any,
    out_path: PathLike,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
) -> None:
    """
    Dump JSON robusto para objetos típicos del pipeline:
    - numpy scalars/arrays
    - pandas scalars/Series/DataFrame (si pandas está disponible)
    - torch tensors (si torch está disponible)
    - Path
    - datetime/date
    - set/tuple
    - bytes

    Además:
    - escritura atómica: escribe a un tmp file y luego reemplaza.

    Parameters
    ----------
    obj : Any
        Objeto serializable (con fallback).
    out_path : str | PathLike
        Ruta de salida.
    indent, ensure_ascii, sort_keys
        Parámetros de json.dump.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        # --- pathlib ---
        if isinstance(o, Path):
            return str(o)

        # --- datetime/date ---
        if isinstance(o, (datetime, date)):
            # datetime: idealmente isoformat() con tz si existe
            try:
                return o.isoformat()
            except Exception:
                return str(o)

        # --- numpy scalars ---
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()

        # --- numpy arrays ---
        if isinstance(o, np.ndarray):
            # Para configs normalmente son chicos (ROIs, labels). Convertimos a list.
            return o.tolist()

        # --- pandas ---
        if pd is not None:
            try:
                if isinstance(o, (pd.Timestamp,)):
                    return o.isoformat()
                if isinstance(o, (pd.Series,)):
                    return o.to_list()
                if isinstance(o, (pd.DataFrame,)):
                    return o.to_dict(orient="records")
                # pandas NA scalars
                if hasattr(pd, "isna") and pd.isna(o):
                    return None
            except Exception:
                pass

        # --- torch ---
        if torch is not None:
            try:
                if isinstance(o, torch.Tensor):
                    # ojo: puede ser gigante; para configs suele ser chico.
                    return o.detach().cpu().numpy().tolist()
            except Exception:
                pass

        # --- bytes ---
        if isinstance(o, (bytes, bytearray)):
            # representable, estable
            return o.hex()

        # --- sets/tuples ---
        if isinstance(o, (set, tuple)):
            return list(o)

        # --- mappings con keys no-string (por seguridad) ---
        if isinstance(o, Mapping):
            try:
                return {str(k): v for k, v in o.items()}
            except Exception:
                return str(o)

        # fallback final
        return str(o)

    # Escritura atómica: tmp en el mismo directorio (evita problemas cross-device)
    tmp_fd: Optional[int] = None
    tmp_path: Optional[Path] = None
    try:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=out_path.name + ".",
            suffix=".tmp",
            dir=str(out_path.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)

        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            tmp_fd = None  # ya lo cerrará el context manager
            json.dump(
                obj,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                default=_default,
            )
            f.write("\n")

        tmp_path.replace(out_path)

    finally:
        # cleanup por si algo falla antes del replace()
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


__all__ = [
    "_compute_file_sha256",
    "_safe_json_dump",
]
