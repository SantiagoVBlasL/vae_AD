#!/usr/bin/env python3
"""Minimal fold-wise ComBat helpers for pre-VAE input harmonization.

This module is intentionally small and explicit. It provides a train/apply
interface for preflight validation; it does not modify tensors on disk and does
not change the main training script by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

try:
    from neurocombat_sklearn import CombatModel
except Exception:  # pragma: no cover - reported by dependency_status()
    CombatModel = None  # type: ignore[assignment]


MANUFACTURER_CODE = {"GE": 0.0, "Philips": 1.0, "SIEMENS": 2.0}


class CompatSparseOneHotEncoder(OneHotEncoder):
    """Compatibility shim for neurocombat_sklearn with newer scikit-learn."""

    def __init__(
        self,
        *,
        categories: Any = "auto",
        drop: Any = None,
        sparse: bool = True,
        dtype: Any = float,
        handle_unknown: str = "error",
        min_frequency: Any = None,
        max_categories: Any = None,
        feature_name_combiner: str = "concat",
    ) -> None:
        self.sparse = sparse
        super().__init__(
            categories=categories,
            drop=drop,
            sparse_output=sparse,
            dtype=dtype,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories,
            feature_name_combiner=feature_name_combiner,
        )


def patch_neurocombat_sklearn_ohe() -> None:
    try:
        import neurocombat_sklearn.neurocombat_sklearn as ncs  # type: ignore

        ncs.OneHotEncoder = CompatSparseOneHotEncoder
    except Exception:
        return


@dataclass
class FittedCombatChannel:
    model: Any
    variable_mask: np.ndarray
    channel_index: int
    channel_name: str
    tri_rows: np.ndarray
    tri_cols: np.ndarray
    n_rois: int


@dataclass
class FittedCombatTensor:
    channels: list[FittedCombatChannel]
    audit_rows: list[dict[str, Any]]
    channel_indices: list[int]
    channel_names: list[str]


def dependency_status() -> dict[str, Any]:
    return {
        "neurocombat_sklearn_CombatModel_available": CombatModel is not None,
        "implementation": "neurocombat_sklearn.CombatModel" if CombatModel is not None else "",
    }


def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    low = text.lower()
    if "philips" in low:
        return "Philips"
    if "siemens" in low:
        return "SIEMENS"
    if low in {"ge", "general electric"} or "general electric" in low:
        return "GE"
    return text or "UNKNOWN"


def normalize_sex(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text in {"F", "FEMALE", "0"}:
        return "F"
    if text in {"M", "MALE", "1"}:
        return "M"
    return "UNKNOWN"


def covariates_for_combat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [c for c in ["Manufacturer", "Age", "Sex"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ComBat covariate columns: {missing}")
    mfr = df["Manufacturer"].map(normalize_manufacturer)
    unknown = sorted(set(mfr.astype(str)) - set(MANUFACTURER_CODE))
    if unknown:
        raise ValueError(f"Unsupported Manufacturer values for ComBat coding: {unknown}")
    sex = df["Sex"].map(normalize_sex)
    if sex.eq("UNKNOWN").any():
        subjects = df.loc[sex.eq("UNKNOWN"), "SubjectID"].astype(str).tolist() if "SubjectID" in df.columns else []
        raise ValueError(f"Missing/unknown Sex values for ComBat covariate model: {subjects[:20]}")
    age = pd.to_numeric(df["Age"], errors="coerce")
    if age.isna().any():
        subjects = df.loc[age.isna(), "SubjectID"].astype(str).tolist() if "SubjectID" in df.columns else []
        raise ValueError(f"Missing Age values for ComBat covariate model: {subjects[:20]}")
    sites = mfr.map(MANUFACTURER_CODE).to_numpy(dtype=float).reshape(-1, 1)
    sex_cov = sex.map({"F": 0.0, "M": 1.0}).to_numpy(dtype=float).reshape(-1, 1)
    age_cov = age.to_numpy(dtype=float).reshape(-1, 1)
    return sites, sex_cov, age_cov


def upper_offdiag_features(channel_mats: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    if channel_mats.ndim != 3 or channel_mats.shape[1] != channel_mats.shape[2]:
        raise ValueError(f"Expected [N,R,R] channel matrices, got {channel_mats.shape}")
    tri = np.triu_indices(int(channel_mats.shape[1]), k=1)
    return np.asarray(channel_mats[:, tri[0], tri[1]], dtype=np.float64), tri


def reconstruct_symmetric(
    edge_features: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
    *,
    n_rois: int,
    diagonal_values: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros((edge_features.shape[0], n_rois, n_rois), dtype=np.float64)
    out[:, tri[0], tri[1]] = edge_features
    out[:, tri[1], tri[0]] = edge_features
    if diagonal_values is not None:
        idx = np.arange(n_rois)
        out[:, idx, idx] = diagonal_values
    return out


def fit_channel_combat(
    train_channel_mats: np.ndarray,
    train_meta: pd.DataFrame,
    *,
    channel_index: int,
    channel_name: str,
) -> tuple[FittedCombatChannel | None, dict[str, Any]]:
    if CombatModel is None:
        return None, {"status": "failed", "reason": "neurocombat_sklearn.CombatModel unavailable"}
    patch_neurocombat_sklearn_ohe()
    X, tri = upper_offdiag_features(train_channel_mats)
    finite = np.isfinite(X).all(axis=0)
    variable = finite & (np.nanvar(X, axis=0) > 1e-12)
    if int(variable.sum()) == 0:
        return None, {
            "status": "failed",
            "reason": "No finite non-constant upper-triangle features available",
            "n_features": int(X.shape[1]),
            "n_variable_features": 0,
        }
    sites, sex, age = covariates_for_combat(train_meta)
    model = CombatModel()
    model.fit(X[:, variable], sites, discrete_covariates=sex, continuous_covariates=age)
    wrapper = FittedCombatChannel(
        model=model,
        variable_mask=variable,
        channel_index=int(channel_index),
        channel_name=str(channel_name),
        tri_rows=tri[0],
        tri_cols=tri[1],
        n_rois=int(train_channel_mats.shape[1]),
    )
    return wrapper, {
        "status": "fit_ok",
        "reason": "",
        "n_fit": int(train_channel_mats.shape[0]),
        "n_features": int(X.shape[1]),
        "n_variable_features": int(variable.sum()),
        "n_constant_or_nonfinite_features_carried_through": int((~variable).sum()),
        "manufacturer_levels_fit": ";".join(sorted(train_meta["Manufacturer"].map(normalize_manufacturer).unique())),
        "protected_covariates": "Age+Sex",
        "batch": "Manufacturer",
        "excluded_covariates": "Diagnosis",
    }


def transform_channel_combat(
    wrapper: FittedCombatChannel,
    channel_mats: np.ndarray,
    meta: pd.DataFrame,
    *,
    preserve_diagonal: bool = True,
) -> np.ndarray:
    X, tri = upper_offdiag_features(channel_mats)
    if not (np.array_equal(tri[0], wrapper.tri_rows) and np.array_equal(tri[1], wrapper.tri_cols)):
        raise ValueError("Upper-triangle feature ordering mismatch.")
    sites, sex, age = covariates_for_combat(meta)
    out = np.asarray(X, dtype=np.float64).copy()
    out[:, wrapper.variable_mask] = wrapper.model.transform(
        out[:, wrapper.variable_mask],
        sites,
        discrete_covariates=sex,
        continuous_covariates=age,
    )
    diag = np.diagonal(channel_mats, axis1=1, axis2=2) if preserve_diagonal else None
    return reconstruct_symmetric(out, tri, n_rois=wrapper.n_rois, diagonal_values=diag)


def fit_apply_tensor_combat_channelwise(
    train_tensor: np.ndarray,
    train_meta: pd.DataFrame,
    apply_tensor: np.ndarray,
    apply_meta: pd.DataFrame,
    *,
    channel_indices: Sequence[int],
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if train_tensor.ndim != 4 or apply_tensor.ndim != 4:
        raise ValueError("Expected train/apply tensors with shape [N,C,R,R].")
    transformed = np.asarray(apply_tensor, dtype=np.float64).copy()
    audit_rows: list[dict[str, Any]] = []
    for pos, channel_index in enumerate(channel_indices):
        channel_name = channel_names[pos] if pos < len(channel_names) else f"channel_{channel_index}"
        wrapper, audit = fit_channel_combat(
            train_tensor[:, pos],
            train_meta,
            channel_index=int(channel_index),
            channel_name=str(channel_name),
        )
        audit.update({"channel_position": int(pos), "selected_channel_index": int(channel_index), "channel_name": channel_name})
        audit_rows.append(audit)
        if wrapper is None:
            continue
        transformed[:, pos] = transform_channel_combat(wrapper, apply_tensor[:, pos], apply_meta)
    return transformed, audit_rows


def fit_tensor_combat_channelwise(
    train_tensor: np.ndarray,
    train_meta: pd.DataFrame,
    *,
    channel_indices: Sequence[int],
    channel_names: Sequence[str],
) -> FittedCombatTensor:
    """Fit one frozen ComBat model per selected channel on train rows only."""
    if train_tensor.ndim != 4:
        raise ValueError(f"Expected train tensor [N,C,R,R], got {train_tensor.shape}")
    if train_tensor.shape[1] != len(channel_indices):
        raise ValueError(
            f"Selected channel count mismatch: tensor C={train_tensor.shape[1]}, "
            f"channel_indices={len(channel_indices)}"
        )
    wrappers: list[FittedCombatChannel] = []
    audit_rows: list[dict[str, Any]] = []
    for pos, channel_index in enumerate(channel_indices):
        channel_name = channel_names[pos] if pos < len(channel_names) else f"channel_{channel_index}"
        wrapper, audit = fit_channel_combat(
            train_tensor[:, pos],
            train_meta,
            channel_index=int(channel_index),
            channel_name=str(channel_name),
        )
        audit.update(
            {
                "channel_position": int(pos),
                "selected_channel_index": int(channel_index),
                "channel_name": str(channel_name),
            }
        )
        audit_rows.append(audit)
        if wrapper is None:
            raise RuntimeError(f"ComBat fit failed for channel {channel_name}: {audit}")
        wrappers.append(wrapper)
    return FittedCombatTensor(
        channels=wrappers,
        audit_rows=audit_rows,
        channel_indices=[int(x) for x in channel_indices],
        channel_names=[str(x) for x in channel_names],
    )


def transform_tensor_combat_channelwise(
    fitted: FittedCombatTensor,
    tensor: np.ndarray,
    meta: pd.DataFrame,
    *,
    preserve_diagonal: bool = True,
) -> np.ndarray:
    """Apply frozen channel-wise ComBat models without refitting."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected apply tensor [N,C,R,R], got {tensor.shape}")
    if tensor.shape[1] != len(fitted.channels):
        raise ValueError(
            f"Selected channel count mismatch: tensor C={tensor.shape[1]}, "
            f"fitted channels={len(fitted.channels)}"
        )
    transformed = np.asarray(tensor, dtype=np.float64).copy()
    for pos, wrapper in enumerate(fitted.channels):
        transformed[:, pos] = transform_channel_combat(
            wrapper,
            tensor[:, pos],
            meta,
            preserve_diagonal=preserve_diagonal,
        )
    return transformed


def manufacturer_centroid_separability_proxy(
    tensor: np.ndarray,
    meta: pd.DataFrame,
    *,
    channel_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Cheap manufacturer separability proxy from upper-triangle centroid distances."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor [N,C,R,R], got {tensor.shape}")
    mfr = meta["Manufacturer"].map(normalize_manufacturer).astype(str).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for c in range(tensor.shape[1]):
        X, _ = upper_offdiag_features(tensor[:, c])
        levels = [x for x in sorted(mfr.unique()) if x in MANUFACTURER_CODE]
        centroid_distances: list[float] = []
        for i, a in enumerate(levels):
            xa = X[mfr.eq(a).to_numpy()]
            if xa.shape[0] == 0:
                continue
            ca = np.nanmean(xa, axis=0)
            for b in levels[i + 1 :]:
                xb = X[mfr.eq(b).to_numpy()]
                if xb.shape[0] == 0:
                    continue
                cb = np.nanmean(xb, axis=0)
                centroid_distances.append(float(np.linalg.norm(ca - cb) / np.sqrt(max(1, X.shape[1]))))
        channel_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
        rows.append(
            {
                "channel_position": int(c),
                "channel_name": str(channel_name),
                "manufacturer_levels": ";".join(levels),
                "mean_pairwise_centroid_distance_per_edge": float(np.mean(centroid_distances))
                if centroid_distances
                else np.nan,
                "max_pairwise_centroid_distance_per_edge": float(np.max(centroid_distances))
                if centroid_distances
                else np.nan,
                "n_pairs": int(len(centroid_distances)),
            }
        )
    return rows


def channel_shift_summary(
    before: np.ndarray,
    after: np.ndarray,
    *,
    split_name: str,
    channel_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize off-diagonal location/scale shifts after frozen harmonization."""
    if before.shape != after.shape:
        raise ValueError(f"before/after shape mismatch: {before.shape} vs {after.shape}")
    rows: list[dict[str, Any]] = []
    for c in range(before.shape[1]):
        xb, _ = upper_offdiag_features(before[:, c])
        xa, _ = upper_offdiag_features(after[:, c])
        diff = xa - xb
        channel_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
        rows.append(
            {
                "split": str(split_name),
                "channel_position": int(c),
                "channel_name": str(channel_name),
                "pre_mean": float(np.nanmean(xb)),
                "post_mean": float(np.nanmean(xa)),
                "delta_mean": float(np.nanmean(xa) - np.nanmean(xb)),
                "pre_std": float(np.nanstd(xb)),
                "post_std": float(np.nanstd(xa)),
                "delta_std": float(np.nanstd(xa) - np.nanstd(xb)),
                "mean_abs_delta": float(np.nanmean(np.abs(diff))),
                "max_abs_delta": float(np.nanmax(np.abs(diff))),
                "finite_fraction_post": float(np.isfinite(xa).mean()),
            }
        )
    return rows
