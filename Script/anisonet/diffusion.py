"""Diffusion-field construction utilities for anisoNET."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray, rgb2hed
from skimage.transform import resize

from .preprocessing import clip_and_normalize, spot_grid_indices


BARRIER_MODULES: Dict[str, Tuple[str, ...]] = {
    "CNS_Myelin": ("Mbp", "Plp1", "Mobp"),
    "Stromal_Fibrosis": ("Col1a1", "Col1a2", "Col3a1", "Fn1", "Acta2", "Vim"),
    "Liver_Periportal": ("Cyp2f2",),
    "Kidney_Tubular": ("Lcn2", "Slc34a1", "Aqp1", "Umod"),
}


def _to_dense_vector(matrix_like: object) -> np.ndarray:
    if hasattr(matrix_like, "toarray"):
        return np.asarray(matrix_like.toarray()).reshape(-1)
    return np.asarray(matrix_like).reshape(-1)


def compute_barrier_score(
    adata: object,
    genes: Iterable[str],
    *,
    percentile: float = 99.0,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Compute a normalized barrier score from a set of marker genes."""

    active_genes = tuple(g for g in genes if g in adata.var_names)
    if not active_genes:
        raise ValueError("None of the requested barrier genes are present in adata.var_names")
    raw_sum = _to_dense_vector(adata[:, list(active_genes)].X.sum(axis=1))
    return clip_and_normalize(raw_sum, percentile=percentile), active_genes


def infer_barrier_module(adata: object, modules: Dict[str, Tuple[str, ...]] | None = None) -> Tuple[str, Tuple[str, ...]]:
    """Infer the dominant barrier module from total marker abundance."""

    modules = modules or BARRIER_MODULES
    scores = {}
    for name, genes in modules.items():
        active = [g for g in genes if g in adata.var_names]
        if active:
            scores[name] = float(_to_dense_vector(adata[:, active].X.sum(axis=1)).sum())
        else:
            scores[name] = 0.0
    best_name = max(scores, key=scores.get)
    if scores[best_name] <= 0:
        raise ValueError("No supported barrier module was detected in this sample")
    return best_name, tuple(g for g in modules[best_name] if g in adata.var_names)


def build_barrier_grid(
    coords_norm: np.ndarray,
    barrier_score: np.ndarray,
    *,
    grid_size: int = 200,
    sigma: float = 1.5,
) -> np.ndarray:
    """Rasterize spot-level barrier scores onto a smoothed grid."""

    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid[grid_y, grid_x] = np.asarray(barrier_score, dtype=np.float32)
    smoothed = gaussian_filter(grid, sigma=sigma)
    return smoothed / (smoothed.max() + 1e-8)


def build_optical_diffusion_grid(
    hires_image: np.ndarray,
    *,
    grid_size: int = 200,
    d_min: float = 0.0002,
    d_free: float = 0.02,
    optical_exponent: float = 2.0,
    flip_y: bool = True,
) -> np.ndarray:
    """Build a histology-derived scalar diffusion coefficient grid."""

    structural_resistance = build_histology_resistance_grid(
        hires_image,
        grid_size=grid_size,
        mode="brightness",
        flip_y=flip_y,
    )
    porosity_like = 1.0 - structural_resistance
    return d_min + (d_free - d_min) * (porosity_like ** optical_exponent)


def build_histology_resistance_grid(
    hires_image: np.ndarray,
    *,
    grid_size: int = 200,
    mode: str = "brightness",
    flip_y: bool = True,
) -> np.ndarray:
    """Build a normalized H&E structural resistance proxy.

    ``brightness`` preserves the historical optical prior. ``hematoxylin`` uses
    color deconvolution as a nuclei-rich tissue-density proxy.
    """

    if mode == "brightness":
        gray = rgb2gray(hires_image) if hires_image.ndim == 3 else hires_image.astype(float)
        gray_grid = resize(gray, (grid_size, grid_size), anti_aliasing=True, preserve_range=True)
        gray_norm = _robust_minmax(gray_grid)
        resistance = 1.0 - gray_norm
    elif mode == "hematoxylin":
        if hires_image.ndim != 3:
            raise ValueError("hematoxylin mode requires an RGB H&E image")
        rgb = np.asarray(hires_image, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        hed = rgb2hed(np.clip(rgb, 0.0, 1.0))
        hematoxylin = hed[:, :, 0]
        hematoxylin_grid = resize(
            hematoxylin,
            (grid_size, grid_size),
            anti_aliasing=True,
            preserve_range=True,
        )
        resistance = _robust_minmax(hematoxylin_grid, lower=1.0, upper=99.0)
    else:
        raise ValueError("mode must be either 'brightness' or 'hematoxylin'")
    if flip_y:
        resistance = np.flipud(resistance)
    return resistance.astype(np.float32)


def diffusion_from_histology_resistance(
    histology_resistance_grid: np.ndarray,
    *,
    d_min: float = 0.0002,
    d_free: float = 0.02,
    optical_exponent: float = 2.0,
) -> np.ndarray:
    """Convert a normalized structural resistance grid to diffusion."""

    resistance = np.clip(np.asarray(histology_resistance_grid, dtype=np.float32), 0.0, 1.0)
    porosity_like = 1.0 - resistance
    return (d_min + (d_free - d_min) * (porosity_like ** optical_exponent)).astype(np.float32)


def _robust_minmax(values: np.ndarray, *, lower: float = 0.0, upper: float = 100.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(arr, lower))
    hi = float(np.percentile(arr, upper))
    return (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-8)


def build_diffusion_grid(
    optical_diffusion_grid: np.ndarray,
    barrier_grid: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    alpha: float = 4.0,
    background_diffusion: float = 0.005,
) -> np.ndarray:
    """Fuse optical and transcriptomic priors into a scalar diffusion grid."""

    if optical_diffusion_grid.shape != barrier_grid.shape or optical_diffusion_grid.shape != tissue_mask.shape:
        raise ValueError("optical_diffusion_grid, barrier_grid, and tissue_mask must have matching shapes")
    fused = optical_diffusion_grid * np.exp(-alpha * barrier_grid)
    return np.where(tissue_mask, fused, background_diffusion).astype(np.float32)


def build_source_grid(
    coords_norm: np.ndarray,
    source_score: np.ndarray,
    *,
    grid_size: int = 200,
    sigma: float = 1.5,
) -> np.ndarray:
    """Rasterize normalized source expression onto a smoothed source grid."""

    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid[grid_y, grid_x] = np.asarray(source_score, dtype=np.float32)
    return gaussian_filter(grid, sigma=sigma).astype(np.float32)


def resistance_from_diffusion(diffusion_grid: np.ndarray, *, mode: str = "inverse") -> np.ndarray:
    """Convert a diffusion coefficient grid to a resistance representation."""

    diffusion = np.asarray(diffusion_grid, dtype=np.float32)
    if np.any(diffusion <= 0):
        raise ValueError("diffusion_grid must be strictly positive")
    if mode == "inverse":
        return 1.0 / diffusion
    if mode == "neg_log10":
        return -np.log10(diffusion)
    raise ValueError("mode must be either 'inverse' or 'neg_log10'")
