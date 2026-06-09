"""Quantitative evaluation metrics for anisoNET fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .preprocessing import spot_grid_indices


@dataclass(frozen=True)
class FieldMetrics:
    values: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)


def sample_grid_at_spots(grid: np.ndarray, coords_norm: np.ndarray) -> np.ndarray:
    """Sample a 2D grid at normalized spot coordinates."""

    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid.shape[0])
    return np.asarray(grid[grid_y, grid_x], dtype=np.float32)


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation with NaN protection."""

    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def mse(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    return float(np.mean((x_arr - y_arr) ** 2))


def high_low_mask(values: np.ndarray, *, high_quantile: float = 0.9, low_quantile: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return masks for high and low quantiles."""

    arr = np.asarray(values, dtype=np.float32)
    high = arr >= np.quantile(arr, high_quantile)
    low = arr <= np.quantile(arr, low_quantile)
    return high, low


def evaluate_field(
    *,
    coords_norm: np.ndarray,
    source_spot_values: np.ndarray,
    prediction_grid: np.ndarray,
    barrier_grid: np.ndarray,
    tissue_mask: np.ndarray,
) -> FieldMetrics:
    """Evaluate a predicted field against source and barrier structure."""

    source = np.asarray(source_spot_values, dtype=np.float32).reshape(-1)
    pred_spots = sample_grid_at_spots(prediction_grid, coords_norm)
    barrier_spots = sample_grid_at_spots(barrier_grid, coords_norm)

    tissue_values = prediction_grid[tissue_mask]
    background_values = prediction_grid[~tissue_mask]
    roughness = field_roughness_metrics(prediction_grid, tissue_mask)

    high_barrier, low_barrier = high_low_mask(barrier_spots)
    high_barrier_pred = pred_spots[high_barrier]
    low_barrier_pred = pred_spots[low_barrier]

    background_mean = float(np.mean(background_values)) if background_values.size else float("nan")
    tissue_mean = float(np.mean(tissue_values)) if tissue_values.size else float("nan")
    leakage_ratio = background_mean / (tissue_mean + 1e-8)

    values = {
        "spot_pearson_source": pearsonr_safe(pred_spots, source),
        "spot_mse_source": mse(pred_spots, source),
        "spot_pearson_barrier": pearsonr_safe(pred_spots, barrier_spots),
        "prediction_min": float(np.min(prediction_grid)),
        "prediction_max": float(np.max(prediction_grid)),
        "prediction_mean_tissue": tissue_mean,
        "prediction_mean_background": background_mean,
        "background_to_tissue_ratio": float(leakage_ratio),
        "prediction_mean_high_barrier_spots": float(np.mean(high_barrier_pred)) if high_barrier_pred.size else float("nan"),
        "prediction_mean_low_barrier_spots": float(np.mean(low_barrier_pred)) if low_barrier_pred.size else float("nan"),
        "high_to_low_barrier_prediction_ratio": float(
            np.mean(high_barrier_pred) / (np.mean(low_barrier_pred) + 1e-8)
        )
        if high_barrier_pred.size and low_barrier_pred.size
        else float("nan"),
        **roughness,
    }
    return FieldMetrics(values)


def field_roughness_metrics(field: np.ndarray, tissue_mask: np.ndarray) -> Dict[str, float]:
    """Compute simple high-frequency roughness metrics inside tissue."""

    arr = np.asarray(field, dtype=np.float32)
    mask = np.asarray(tissue_mask, dtype=bool)
    gy, gx = np.gradient(arr)
    grad_mag = np.sqrt(gx**2 + gy**2)
    laplacian = (
        np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
        - 4 * arr
    )
    grad_tissue = grad_mag[mask]
    lap_tissue = laplacian[mask]
    if grad_tissue.size == 0:
        return {
            "roughness_grad_mean": float("nan"),
            "roughness_grad_p95": float("nan"),
            "roughness_laplacian_energy": float("nan"),
        }
    return {
        "roughness_grad_mean": float(np.mean(grad_tissue)),
        "roughness_grad_p95": float(np.percentile(grad_tissue, 95)),
        "roughness_laplacian_energy": float(np.mean(lap_tissue**2)),
    }
