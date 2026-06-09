"""Postprocessing utilities for anisoNET field predictions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .pinn import robust_normalize


def mask_field(field: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    """Set prediction values outside the tissue mask to zero."""

    return np.where(np.asarray(tissue_mask, dtype=bool), field, 0.0).astype(np.float32)


def smooth_inside_mask(field: np.ndarray, tissue_mask: np.ndarray, *, sigma: float) -> np.ndarray:
    """Apply normalized Gaussian smoothing inside tissue without exterior leakage."""

    arr = np.asarray(field, dtype=np.float32)
    mask_float = np.asarray(tissue_mask, dtype=np.float32)
    weighted = gaussian_filter(arr * mask_float, sigma=sigma)
    support = gaussian_filter(mask_float, sigma=sigma)
    smoothed = weighted / (support + 1e-8)
    return np.where(mask_float > 0, smoothed, 0.0).astype(np.float32)


def normalize_masked_field(field: np.ndarray, tissue_mask: np.ndarray, *, percentile: float = 99.5) -> np.ndarray:
    """Robustly normalize a field and preserve zero exterior background."""

    normalized = robust_normalize(field, percentile=percentile)
    return mask_field(normalized, tissue_mask)
