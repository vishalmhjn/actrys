"""One-shot bias correction of the prior OD matrix and search bounds."""

from __future__ import annotations

import numpy as np


def bias_factor(
    method: str,
    simulated_counts: np.ndarray,
    observed_counts: np.ndarray,
    *,
    observed_for_weighted: np.ndarray | None = None,
    incidence: np.ndarray | None = None,
    fill_nan: bool = False,
) -> float | np.ndarray:
    """Estimated over/under-estimation of the prior demand from simulated vs observed counts.

    * ``naive``: one global factor ``sum(simulated) / sum(observed)``.
    * ``weighted``: one factor per OD variable, the incidence-weighted mean of
      the per-detector ratios ``simulated / observed``. With ``fill_nan``,
      variables that cross no detector get the naive factor.
    """
    naive = np.sum(simulated_counts) / np.sum(observed_counts)
    if method == "naive":
        return naive
    if method == "weighted":
        if incidence is None:
            raise ValueError("The weighted bias correction needs the incidence matrix")
        obs = observed_counts if observed_for_weighted is None else observed_for_weighted
        ratios = simulated_counts / (obs + 1e-8)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = (incidence @ ratios).flatten() / incidence.sum(axis=1)
        if fill_nan:
            factor = np.nan_to_num(factor, nan=naive)
        return factor.flatten()
    raise ValueError(f"Unknown bias correction method {method!r}; use 'naive' or 'weighted'")


def search_bounds(x: np.ndarray, lower: float, upper: float | np.ndarray) -> np.ndarray:
    """Box bounds ``[lower * x_i, upper_i * x_i]`` around a starting point.

    ``upper`` is a scalar or one factor per variable.
    """
    x = np.asarray(x)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), x.shape)
    return np.column_stack([lower * x, upper * x]).astype(float)
