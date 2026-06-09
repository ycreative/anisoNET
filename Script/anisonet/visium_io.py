"""Lightweight Visium readers used by anisoNET workflows.

These helpers avoid importing Scanpy for preprocessing steps that only need
coordinates, histology, and a small set of gene-expression vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


@dataclass(frozen=True)
class VisiumLiteSample:
    sample_dir: Path
    matrix: sparse.csc_matrix
    genes: tuple[str, ...]
    barcodes: tuple[str, ...]
    spatial_pixels: np.ndarray
    hires_image: np.ndarray
    spatial_scale: float
    library_size: np.ndarray


def _decode_array(values: np.ndarray) -> tuple[str, ...]:
    return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values)


def read_10x_h5_matrix(path: str | Path) -> tuple[sparse.csc_matrix, tuple[str, ...], tuple[str, ...]]:
    """Read a 10x Genomics HDF5 matrix as genes x barcodes."""

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        data = group["data"][()]
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        shape = tuple(group["shape"][()])
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        genes = _decode_array(group["features"]["name"][()])
        barcodes = _decode_array(group["barcodes"][()])
    return matrix, genes, barcodes


def read_spatial_positions(path: str | Path, barcodes: Iterable[str]) -> np.ndarray:
    """Read Space Ranger tissue positions and return pixel coordinates."""

    columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    frame = pd.read_csv(path, header=None, names=columns)
    frame = frame.set_index("barcode")
    missing = [barcode for barcode in barcodes if barcode not in frame.index]
    if missing:
        raise ValueError(f"{len(missing)} matrix barcodes are missing from tissue_positions_list.csv")
    ordered = frame.loc[list(barcodes)]
    return ordered[["pxl_col", "pxl_row"]].to_numpy(dtype=np.float32)


def load_visium_lite(sample_dir: str | Path) -> VisiumLiteSample:
    """Load a standardized Visium directory without Scanpy."""

    sample_path = Path(sample_dir)
    matrix, genes, barcodes = read_10x_h5_matrix(sample_path / "filtered_feature_bc_matrix.h5")
    spatial_dir = sample_path / "spatial"
    spatial_pixels = read_spatial_positions(spatial_dir / "tissue_positions_list.csv", barcodes)

    with (spatial_dir / "scalefactors_json.json").open("r", encoding="utf-8") as handle:
        scalefactors = json.load(handle)
    spatial_scale = float(scalefactors["tissue_hires_scalef"])

    with Image.open(spatial_dir / "tissue_hires_image.png") as image:
        hires_image = np.asarray(image.convert("RGB"))

    library_size = np.asarray(matrix.sum(axis=0)).reshape(-1).astype(np.float32)
    return VisiumLiteSample(
        sample_dir=sample_path,
        matrix=matrix,
        genes=genes,
        barcodes=barcodes,
        spatial_pixels=spatial_pixels,
        hires_image=hires_image,
        spatial_scale=spatial_scale,
        library_size=library_size,
    )


def normalized_gene_vector(sample: VisiumLiteSample, gene: str, *, target_sum: float = 1e4) -> np.ndarray:
    """Return log1p-normalized expression for one gene."""

    try:
        gene_idx = sample.genes.index(gene)
    except ValueError as exc:
        raise ValueError(f"Gene '{gene}' was not found in {sample.sample_dir}") from exc
    raw = np.asarray(sample.matrix[gene_idx, :].toarray()).reshape(-1).astype(np.float32)
    normalized = raw / (sample.library_size + 1e-8) * target_sum
    return np.log1p(normalized)


def normalized_gene_sum(sample: VisiumLiteSample, genes: Iterable[str], *, target_sum: float = 1e4) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return log1p-normalized summed expression for a marker set."""

    active = tuple(g for g in genes if g in sample.genes)
    if not active:
        raise ValueError(f"None of the requested genes were found in {sample.sample_dir}")
    indices = [sample.genes.index(g) for g in active]
    raw = np.asarray(sample.matrix[indices, :].sum(axis=0)).reshape(-1).astype(np.float32)
    normalized = raw / (sample.library_size + 1e-8) * target_sum
    return np.log1p(normalized), active


def normalize_pixel_coordinates(sample: VisiumLiteSample, *, flip_y: bool = True) -> np.ndarray:
    """Normalize full-resolution spatial pixels to hires-image coordinates."""

    height, width = sample.hires_image.shape[:2]
    coords = (sample.spatial_pixels * sample.spatial_scale) / np.array([width, height], dtype=np.float32)
    coords = np.clip(coords, 0.0, 1.0)
    if flip_y:
        coords = coords.copy()
        coords[:, 1] = 1.0 - coords[:, 1]
    return coords.astype(np.float32)
