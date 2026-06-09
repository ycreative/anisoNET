"""Preprocessing helpers for anisoNET spatial transcriptomics workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.transform import resize


@dataclass(frozen=True)
class SpatialSample:
    """Container for a loaded Visium-like spatial sample."""

    adata: object
    sample_dir: Path
    library_id: Optional[str]
    hires_image: Optional[np.ndarray]
    spatial_scale: Optional[float]


def load_visium_sample(sample_dir: str | Path, *, normalize: bool = True) -> SpatialSample:
    """Load a 10x Visium sample with Scanpy.

    Scanpy is imported lazily so that lightweight utilities can be tested
    without requiring a full spatial-omics environment.
    """

    import scanpy as sc

    sample_path = Path(sample_dir)
    adata = sc.read_visium(str(sample_path))
    adata.var_names_make_unique()
    if normalize:
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)

    library_id = None
    hires_image = None
    spatial_scale = None
    if "spatial" in adata.uns and adata.uns["spatial"]:
        library_id = next(iter(adata.uns["spatial"].keys()))
        spatial_info = adata.uns["spatial"][library_id]
        hires_image = spatial_info.get("images", {}).get("hires")
        spatial_scale = spatial_info.get("scalefactors", {}).get("tissue_hires_scalef")

    return SpatialSample(
        adata=adata,
        sample_dir=sample_path,
        library_id=library_id,
        hires_image=hires_image,
        spatial_scale=spatial_scale,
    )


def get_hires_image(sample: SpatialSample) -> np.ndarray:
    """Return the high-resolution histology image for a loaded sample."""

    if sample.hires_image is None:
        raise ValueError(f"No hires image found in sample: {sample.sample_dir}")
    return np.asarray(sample.hires_image)


def normalize_coordinates(
    adata: object,
    *,
    hires_image: Optional[np.ndarray] = None,
    spatial_scale: Optional[float] = None,
    flip_y: bool = True,
) -> np.ndarray:
    """Normalize spatial coordinates into [0, 1] x [0, 1].

    If a hires image and scale factor are available, coordinates are normalized
    in image space. Otherwise, min-max normalization is used as a fallback.
    """

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if hires_image is not None and spatial_scale is not None:
        height, width = hires_image.shape[:2]
        coords_norm = (coords * float(spatial_scale)) / np.array([width, height], dtype=np.float32)
    else:
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)

    coords_norm = np.clip(coords_norm, 0.0, 1.0)
    if flip_y:
        coords_norm = coords_norm.copy()
        coords_norm[:, 1] = 1.0 - coords_norm[:, 1]
    return coords_norm


def spot_grid_indices(coords_norm: np.ndarray, grid_size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Map normalized coordinates to integer grid indices."""

    if coords_norm.ndim != 2 or coords_norm.shape[1] != 2:
        raise ValueError("coords_norm must have shape (n_spots, 2)")
    scale = grid_size - 1
    grid_x = np.clip((coords_norm[:, 0] * scale).astype(int), 0, scale)
    grid_y = np.clip((coords_norm[:, 1] * scale).astype(int), 0, scale)
    return grid_x, grid_y


def build_spot_mask(
    coords_norm: np.ndarray,
    *,
    grid_size: int = 200,
    dilation_iterations: int = 1,
) -> np.ndarray:
    """Build a boolean tissue-support mask from spot coordinates."""

    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid_size)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[grid_y, grid_x] = True
    if dilation_iterations > 0:
        mask = binary_dilation(mask, iterations=dilation_iterations)
    return mask


def build_tissue_mask_from_hires(
    hires_image: np.ndarray,
    *,
    grid_size: int = 200,
    flip_y: bool = True,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Build a coarse tissue mask from a histology image using Otsu thresholding."""

    gray = rgb2gray(hires_image) if hires_image.ndim == 3 else hires_image.astype(float)
    threshold = threshold_otsu(gray)
    tissue = binary_fill_holes(gray < threshold)

    if keep_largest_component:
        labels = label(tissue)
        counts = np.bincount(labels.ravel())
        if len(counts) > 1:
            counts[0] = 0
            tissue = labels == counts.argmax()

    tissue_grid = resize(
        tissue.astype(float),
        (grid_size, grid_size),
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    ) > 0.5
    if flip_y:
        tissue_grid = np.flipud(tissue_grid)
    return tissue_grid


def keep_largest_mask_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a boolean mask."""

    labels = label(np.asarray(mask, dtype=bool))
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.asarray(mask, dtype=bool)
    counts[0] = 0
    return labels == counts.argmax()


def clip_and_normalize(values: np.ndarray, *, percentile: float = 99.0) -> np.ndarray:
    """Clip extreme values and normalize to [0, 1]."""

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    vmax = np.percentile(arr, percentile)
    clipped = np.clip(arr, 0, vmax)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
