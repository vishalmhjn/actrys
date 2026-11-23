"""Goodness-of-fit (GoF) measures."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.preprocessing import StandardScaler

_EPS = 1e-8


def _rmsn(data: np.ndarray, sim: np.ndarray) -> np.ndarray:
    sum_diff = ((data - sim) ** 2).sum()
    return np.round(np.sqrt(len(data) * sum_diff) / data.sum(), 6)


def _rmse(data: np.ndarray, sim: np.ndarray) -> np.ndarray:
    return np.sqrt(((data - sim) ** 2).mean())


def _smape(data: np.ndarray, sim: np.ndarray) -> np.ndarray:
    diff = np.abs(data - sim) / (0.5 * (data + sim + _EPS))
    return np.round(diff.sum() / len(data), 6)


def _rmsp(data: np.ndarray, sim: np.ndarray) -> np.ndarray:
    diff = ((data - sim) / (0.5 * (data + sim + _EPS))) ** 2
    return np.round(np.sqrt(diff.sum() / len(diff)), 6)


def _wmape(data: np.ndarray, sim: np.ndarray) -> np.ndarray:
    return np.array(np.sum(np.abs(data - sim)) / np.sum(data))


def _composite(data: np.ndarray, sim: np.ndarray) -> np.ndarray:
    return _rmsn(data, sim) + _smape(data, sim)


ESTIMATORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "rmsn": _rmsn,
    "rmse": _rmse,
    "smape": _smape,
    "rmsp": _rmsp,
    "wmape": _wmape,
    "composite": _composite,
}


def gof(data: np.ndarray, simulated: np.ndarray, estimator: str = "rmsn") -> float:
    """Goodness of fit between observed ``data`` and ``simulated`` values (lower is better)."""
    data = np.asarray(data)
    simulated = np.asarray(simulated)
    if data.shape != simulated.shape:
        raise ValueError(f"Shape mismatch: {data.shape} vs {simulated.shape}")
    try:
        fn = ESTIMATORS[estimator]
    except KeyError:
        raise ValueError(
            f"Unknown estimator {estimator!r}; choose from {sorted(ESTIMATORS)}"
        ) from None
    return fn(data, simulated).tolist()


def median_gof(data: np.ndarray, simulated: np.ndarray, estimator: str = "rmsn") -> float:
    """Median/mean-normalised variants of :func:`gof` (``rmsn``, ``smape``, ``rmsp``, ``composite``)."""
    data = np.asarray(data)
    simulated = np.asarray(simulated)
    if data.shape != simulated.shape:
        raise ValueError(f"Shape mismatch: {data.shape} vs {simulated.shape}")
    if estimator == "rmsn":
        sum_diff = ((data - simulated) ** 2).sum()
        val = np.round(np.sqrt(sum_diff / len(data)) / np.mean(data), 6)
    elif estimator == "smape":
        diff = np.abs(data - simulated) / (0.5 * (data + simulated + _EPS))
        val = np.round(np.median(diff), 6)
    elif estimator in ("rmsp", "composite"):
        val = ESTIMATORS[estimator](data, simulated)
    else:
        raise ValueError(f"median_gof does not support estimator {estimator!r}")
    return val.tolist()


def _standardize(data: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(data.reshape(-1, 1))
    return (
        scaler.transform(data.reshape(-1, 1)).flatten(),
        scaler.transform(simulated.reshape(-1, 1)).flatten(),
    )


def _weighted_geh(data: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    return (data + 1e-11) * (data - simulated) ** 2 / (0.5 * np.abs(data + simulated + 1e-11))


def squared_deviation(
    data: np.ndarray,
    simulated: np.ndarray,
    which_metric: str = "standardized_weighted_geh",
) -> np.ndarray:
    """Element-wise deviation used to build the W-SPSA gradient."""
    data = np.asarray(data)
    simulated = np.asarray(simulated)
    if data.shape != simulated.shape:
        raise ValueError(f"Shape mismatch: {data.shape} vs {simulated.shape}")

    if which_metric == "standardized_weighted_geh":
        return _weighted_geh(*_standardize(data, simulated))
    if which_metric == "scaled_weighted_geh":
        peak = np.max(data)
        return _weighted_geh(data / peak, simulated / peak)
    if which_metric == "unweighted_geh":
        return _weighted_geh(data, simulated)
    if which_metric == "standardized_rmse":
        d, s = _standardize(data, simulated)
        return np.sqrt((d - s) ** 2 / len(d))
    raise ValueError(f"Unknown deviation metric {which_metric!r}")
