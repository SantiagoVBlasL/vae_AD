"""Helper utilities for the LOSO master audit notebook (BSPC 2026 revision).

This module intentionally keeps analysis notebooks cleaner by centralizing:
- robust project/output path resolution
- artifact diagnostics/loading helpers
- plotting/export style helpers
- small statistical utilities used in LOSO audit reporting
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def find_repo_root(start: Optional[Path] = None, max_levels: int = 10) -> Path:
    """Find repository root by scanning parent directories for expected structure."""
    start_path = (start or Path.cwd()).resolve()

    candidates: List[Path] = [start_path, *list(start_path.parents)[:max_levels]]
    for c in candidates:
        if (c / "data").exists() and (c / "scripts" / "revision_bspc_2026").exists():
            return c

    fallback = Path("/home/diego/proyectos/vae_AD").resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Could not detect repository root. Expected data/ and scripts/revision_bspc_2026/."
    )


def results_root_candidates(repo_root: Path) -> List[Path]:
    """Return preferred and fallback result roots in priority order."""
    return [
        repo_root / "results" / "revision_bspc_2026",
        repo_root / "notebooks" / "revision_bspc_2026" / "results" / "revision_bspc_2026",
    ]


def resolve_artifact(relative_path: str, result_roots: Sequence[Path]) -> Optional[Path]:
    """Resolve an artifact path from a list of result root candidates."""
    for root in result_roots:
        p = root / relative_path
        if p.exists():
            return p
    return None


def build_artifact_diagnostic(
    artifact_relpaths: Dict[str, str], result_roots: Sequence[Path]
) -> pd.DataFrame:
    """Build a compact found/missing diagnostics table for notebook display."""
    rows: List[Dict[str, str]] = []
    for key, rel in artifact_relpaths.items():
        resolved = resolve_artifact(rel, result_roots)
        rows.append(
            {
                "artifact": key,
                "relative_path": rel,
                "status": "found" if resolved else "missing",
                "resolved_path": str(resolved) if resolved else "",
            }
        )
    return pd.DataFrame(rows)


def load_csv_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load CSV when present; otherwise return None."""
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def load_json_optional(path: Optional[Path]):
    """Load JSON when present; otherwise return None."""
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Create directory (including parents) and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def site_to_int(site: object) -> Optional[int]:
    """Extract numeric site id from labels like 'site_305' or raw ints."""
    if pd.isna(site):
        return None
    if isinstance(site, (int, np.integer)):
        return int(site)
    if isinstance(site, float) and np.isfinite(site):
        return int(site)

    m = re.search(r"(\d+)", str(site))
    return int(m.group(1)) if m else None


def parse_site_series(series: pd.Series) -> pd.Series:
    """Vectorized site number parsing."""
    return series.apply(site_to_int).astype("Int64")


def standardized_mean_difference(x: Iterable[float], y: Iterable[float]) -> float:
    """Compute Cohen's d (pooled-SD standardized mean difference)."""
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    xa = xa[np.isfinite(xa)]
    ya = ya[np.isfinite(ya)]
    if len(xa) < 2 or len(ya) < 2:
        return np.nan

    vx = xa.var(ddof=1)
    vy = ya.var(ddof=1)
    pooled = np.sqrt(((len(xa) - 1) * vx + (len(ya) - 1) * vy) / (len(xa) + len(ya) - 2))
    if pooled == 0:
        return 0.0
    return (xa.mean() - ya.mean()) / pooled


def bootstrap_auc_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for ROC AUC without model retraining."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.shape[0] != s.shape[0] or y.shape[0] == 0:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}

    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        sb = s[idx]
        if np.unique(yb).size < 2:
            continue
        aucs.append(float(roc_auc_score(yb, sb)))

    if not aucs:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}

    auc_arr = np.asarray(aucs, dtype=float)
    q_low = (1.0 - alpha) / 2.0
    q_high = 1.0 - q_low
    return {
        "auc": float(np.mean(auc_arr)),
        "ci_low": float(np.quantile(auc_arr, q_low)),
        "ci_high": float(np.quantile(auc_arr, q_high)),
        "n_boot": int(len(auc_arr)),
    }


def calibration_slope_intercept(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> Dict[str, float]:
    """Estimate calibration slope/intercept via logistic recalibration fit."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.shape[0] != p.shape[0] or y.shape[0] == 0:
        return {"slope": np.nan, "intercept": np.nan}

    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(logit, y)
    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Compute expected calibration error (ECE) using equal-width bins."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.shape[0] != p.shape[0] or y.shape[0] == 0:
        return np.nan

    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def setup_pub_style() -> None:
    """Set consistent, publication-grade matplotlib style."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.grid": True,
        }
    )


def export_figure(
    fig,
    out_dir: Path,
    stem: str,
    save_pdf: bool = True,
    dpi: int = 300,
) -> List[Path]:
    """Export matplotlib figure as high-resolution PNG (and optional PDF)."""
    out = []
    out_png = out_dir / f"{stem}.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    out.append(out_png)

    if save_pdf:
        out_pdf = out_dir / f"{stem}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        out.append(out_pdf)

    return out
