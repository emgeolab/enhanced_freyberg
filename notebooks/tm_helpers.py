"""Small numerical helpers for the TM 3-layer synthetic-truth notebook."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def exponential_covariance(xy, correlation_range, std):
    """Return an exponential covariance matrix for coordinates in xy."""
    xy = np.asarray(xy, dtype=float)
    std = np.asarray(std, dtype=float)
    distance = cdist(xy, xy)
    correlation = np.exp(-distance / float(correlation_range))
    covariance = correlation * np.outer(std, std)
    return covariance


def exponential_time_covariance(n, correlation_range, std):
    """Return an exponential covariance matrix on an equally spaced time index."""
    index = np.arange(int(n), dtype=float)[:, None]
    distance = np.abs(index - index.T)
    std = np.asarray(std, dtype=float)
    correlation = np.exp(-distance / float(correlation_range))
    covariance = correlation * np.outer(std, std)
    return covariance


def stable_cholesky(covariance, initial_jitter=1.0e-12, max_tries=8):
    """Compute a Cholesky factor, adding only a minimal diagonal jitter if required."""
    covariance = np.asarray(covariance, dtype=float)
    jitter = float(initial_jitter)
    eye = np.eye(covariance.shape[0])
    for _ in range(int(max_tries)):
        try:
            return np.linalg.cholesky(covariance + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Covariance matrix is not positive definite after jitter attempts.")


def draw_bounded_log_values(rng, mean_log10, chol, lower_log10, upper_log10):
    """Draw one log10-Gaussian vector and enforce parameter bounds by clipping."""
    mean_log10 = np.asarray(mean_log10, dtype=float)
    lower_log10 = np.asarray(lower_log10, dtype=float)
    upper_log10 = np.asarray(upper_log10, dtype=float)
    standard_normal = rng.standard_normal(mean_log10.size)
    draw = mean_log10 + chol @ standard_normal
    return np.clip(draw, lower_log10, upper_log10)


def active_cell_xy(idomain_layer, delr, delc):
    """Return active-cell row/column indices and x/y cell-centre coordinates."""
    row, col = np.where(np.asarray(idomain_layer) > 0)
    x = (col + 0.5) * float(delr)
    y = (row + 0.5) * float(delc)
    return row, col, np.column_stack([x, y])


def scatter_to_full_grid(active_values, row, col, shape, inactive_fill):
    """Place active-cell values in a full 2-D grid."""
    full = np.full(shape, float(inactive_fill), dtype=float)
    full[row, col] = np.asarray(active_values, dtype=float)
    return full


def read_mf6_observation_csv(path):
    """Read an MF6 observation CSV and normalize column names to lower case."""
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def reset_directory(path):
    """Remove an existing directory and recreate it empty."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
