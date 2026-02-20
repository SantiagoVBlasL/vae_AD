# src/betavae_xai/utils/checkpoints.py
import torch
import copy
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copia segura a CPU (evita que el dict apunte a tensores en GPU)."""
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out

def average_state_dicts(state_dicts_cpu: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SWA simple: promedio aritmético de parámetros float (en CPU).
    Para buffers/no-float, deja el valor del último dict.
    """
    if not state_dicts_cpu:
        raise ValueError("state_dicts_cpu vacío")
    keys = list(state_dicts_cpu[0].keys())
    for sd in state_dicts_cpu[1:]:
        if list(sd.keys()) != keys:
            raise ValueError("state_dicts con keys distintas (no se puede promediar)")

    avg: Dict[str, Any] = {}
    n = float(len(state_dicts_cpu))
    for k in keys:
        v0 = state_dicts_cpu[0][k]
        if torch.is_tensor(v0) and v0.dtype.is_floating_point:
            acc = torch.zeros_like(v0, dtype=torch.float32)
            for sd in state_dicts_cpu:
                acc += sd[k].to(dtype=torch.float32)
            avg[k] = (acc / n).to(dtype=v0.dtype)
        else:
            avg[k] = copy.deepcopy(state_dicts_cpu[-1][k])
    return avg

def load_checkpoint_state_dict(ckpt_path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Carga un checkpoint torch y devuelve state_dict.
    Soporta dos formatos:
      - {"state_dict": ..., ...}
      - state_dict directo
    """
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        # podría ser state_dict directo (keys de parámetros)
        return obj
    raise ValueError(f"Formato de checkpoint no soportado en {ckpt_path}")

def safe_save_checkpoint(payload: Dict[str, Any], ckpt_path: Path, fold_idx_str: str) -> None:
    """
    Guardado robusto de checkpoints (evita crashear el fold si hay I/O issues).
    """
    try:
        torch.save(payload, ckpt_path)
    except Exception as e:
        logger.warning(f"  {fold_idx_str} No se pudo guardar checkpoint {ckpt_path.name}: {e}")

def cycle_index(epoch: int, total_epochs: int, n_cycles: int) -> int:
    """Devuelve índice de ciclo (0-based) para un epoch dado."""
    if n_cycles <= 0:
        return 0
    epoch_per_cycle = float(total_epochs) / float(n_cycles)
    return int(epoch / epoch_per_cycle)

def is_cycle_end(epoch: int, total_epochs: int, n_cycles: int) -> bool:
    """
    True si este epoch es el último epoch del ciclo (o el último epoch total).
    Maneja epoch_per_cycle no entero.
    """
    if n_cycles <= 0:
        return False
    cur_c = cycle_index(epoch, total_epochs, n_cycles)
    if (epoch + 1) >= total_epochs:
        return True
    next_c = cycle_index(epoch + 1, total_epochs, n_cycles)
    return next_c != cur_c