# src/betavae_xai/utils/stats.py
import math
import numpy as np
from typing import List, Tuple

def nanmean_se(x: List[float]) -> Tuple[float, float, int]:
    """
    Calcula (mean, SE, n_eff) ignorando NaNs. 
    SE = std(ddof=1) / sqrt(n_eff).
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    
    if arr.size == 0:
        return np.nan, np.nan, 0
    
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, np.nan, int(arr.size)
    
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(arr.size)
    return mean, float(se), int(arr.size)

def format_splits(vals: List[float], precision: int = 4) -> str:
    """
    Serialización compacta de una lista de floats para CSV/logs 
    (separados por punto y coma), manejando NaNs.
    """
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("nan")
        else:
            out.append(f"{float(v):.{precision}f}")
    return ";".join(out)