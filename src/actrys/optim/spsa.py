"""Simultaneous perturbation stochastic approximation (SPSA) and variants.

Originally based on the ``noisyopt`` package by Andreas Mayer (MIT licence,
https://github.com/andim/noisyopt) and adapted for OD-matrix estimation:

* perturbations are relative (``x ± c_k (x + 1) Δ``) and rounded to integers,
* :func:`minimize_wspsa` implements W-SPSA (Lu et al., 2015), where a weight
  matrix restricts each gradient component to the measurements it influences,
* :func:`minimize_pcspsa` runs SPSA in a principal-component space.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from scipy.optimize import OptimizeResult
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from actrys.config import SPSAGains
from actrys.noise import RNG

POSITIVE = "positive"
NEGATIVE = "negative"


class Objective(Protocol):
    """Interface the SPSA family needs from a calibration objective."""

    def deviations(self, x: np.ndarray, perturbation: str) -> np.ndarray:
        """Weighted element-wise deviations at ``x`` (used for the gradient).

        ``perturbation`` is ``"positive"`` or ``"negative"`` so that simulators
        can keep the two concurrent evaluations apart (e.g. separate files).
        """

    def loss(self, x: np.ndarray) -> float:
        """Scalar loss at ``x`` without side effects on the calibration history."""

    def evaluate(self, x: np.ndarray) -> float:
        """Scalar loss at an accepted iterate; recorded in the calibration history."""


def gain_sequence(gains: SPSAGains, k: int, niter: int) -> tuple[float, float]:
    """Return ``(a_k, c_k)`` for (0-based) iteration ``k``."""
    big_a = 0.01 * niter
    ak = gains.a / (k + 1.0 + big_a) ** gains.alpha
    ck = gains.c / (k + 1.0) ** gains.gamma
    return ak, ck


def _projector(bounds: np.ndarray | None) -> Callable[[np.ndarray], np.ndarray]:
    if bounds is None:
        return lambda x: x
    bounds = np.asarray(bounds)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lambda x: np.clip(x, lo, hi)


def _perturbations(x: np.ndarray, ck: float, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step = np.multiply(x + 1, ck * delta)
    return np.round(x + step), np.round(x - step)


class _PairEvaluator:
    """Evaluates the +/- perturbations, optionally on two threads.

    Threads (not processes) are used on purpose: for SUMO the work happens in
    external processes, and the objective may hold large shared arrays.
    """

    def __init__(self, objective: Objective, parallel: bool):
        self.objective = objective
        self.parallel = parallel

    def __call__(self, xplus: np.ndarray, xminus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.parallel:
            return (
                self.objective.deviations(xplus, POSITIVE),
                self.objective.deviations(xminus, NEGATIVE),
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_plus = pool.submit(self.objective.deviations, xplus, POSITIVE)
            fut_minus = pool.submit(self.objective.deviations, xminus, NEGATIVE)
            return fut_plus.result(), fut_minus.result()


def _check_niter(niter: int) -> None:
    if niter < 1:
        raise ValueError(f"niter must be >= 1, got {niter}")


def minimize_spsa(
    objective: Objective,
    x0: np.ndarray,
    *,
    gains: SPSAGains,
    niter: int = 100,
    bounds: np.ndarray | None = None,
    reps: int = 1,
    rng: RNG,
    parallel: bool = False,
    callback: Callable[[np.ndarray], None] | None = None,
) -> OptimizeResult:
    """Standard SPSA with relative perturbations and a relative (``x``-scaled) step."""
    _check_niter(niter)
    project = _projector(bounds)
    evaluate_pair = _PairEvaluator(objective, parallel)
    x = np.array(x0)
    n = len(x)
    fun = np.nan
    for k in range(niter):
        ak, ck = gain_sequence(gains, k, niter)
        grads = []
        for _ in range(reps):
            delta = rng.choice([-1, 1], size=n)
            xplus, xminus = _perturbations(x, ck, delta)
            pos, neg = evaluate_pair(xplus, xminus)
            grads.append(np.sum(pos - neg) / (2 * ck * delta))
        grad = np.mean(grads, axis=0)
        x = project(x - ak * grad * x)
        fun = objective.evaluate(x)
        if callback is not None:
            callback(x)
    return OptimizeResult(
        fun=fun,
        x=x,
        nit=niter,
        nfev=2 * niter * reps,
        success=True,
        message="terminated after reaching max number of iterations",
    )


def minimize_wspsa(
    objective: Objective,
    x0: np.ndarray,
    weights: np.ndarray | sp.spmatrix,
    *,
    gains: SPSAGains,
    niter: int = 100,
    bounds: np.ndarray | None = None,
    reps: int = 1,
    momentum: float = 0.7,
    rng: RNG,
    parallel: bool = False,
    callback: Callable[[np.ndarray], None] | None = None,
) -> OptimizeResult:
    """Weighted SPSA (Lu et al., 2015) with heavy-ball momentum.

    ``weights`` has one row per decision variable and one column per element
    of the deviation vector returned by ``objective.deviations``; entry
    ``(i, j)`` is non-zero when variable ``i`` influences measurement ``j``.

    Every one of the ``reps`` replications draws its own perturbation
    direction; their gradient estimates are averaged.
    """
    _check_niter(niter)
    project = _projector(bounds)
    evaluate_pair = _PairEvaluator(objective, parallel)
    x = np.array(x0)
    n = len(x)
    if weights.shape[0] != n:
        raise ValueError(f"weights has {weights.shape[0]} rows but x0 has {n} elements")
    prev_change: np.ndarray | float = 0
    fun = np.nan
    ak = ck = np.nan
    for k in range(niter):
        ak, ck = gain_sequence(gains, k, niter)
        grads = []
        for _ in range(reps):
            delta = rng.choice([-1, 1], size=n)
            xplus, xminus = _perturbations(x, ck, delta)
            pos, neg = evaluate_pair(xplus, xminus)
            grads.append(weights @ (pos - neg) / (2 * ck * (x + 1) * delta))
        grad = np.mean(grads, axis=0)
        x = project(x - ak * grad - momentum * prev_change)
        prev_change = ak * grad
        fun = objective.evaluate(x)
        if callback is not None:
            callback(x)
    return OptimizeResult(
        fun=fun,
        x=x,
        nit=niter,
        nfev=2 * niter * reps,
        spsa_a=ak,
        spsa_c=ck,
        success=True,
        message="terminated after reaching max number of iterations",
    )


def pca_components_for_variance(
    x0: np.ndarray, num_features: int, explained_variance: float = 0.95
) -> int:
    """Number of principal components explaining at most ``explained_variance`` (at least 1)."""
    data = StandardScaler().fit_transform(np.asarray(x0).reshape(-1, num_features))
    cumulative = PCA().fit(data).explained_variance_ratio_.cumsum()
    return max(int(np.sum(cumulative <= explained_variance)), 1)


def minimize_pcspsa(
    objective: Objective,
    x0: np.ndarray,
    num_features: int,
    n_components: int,
    *,
    gains: SPSAGains,
    niter: int = 100,
    reps: int = 1,
    rng: RNG,
    random_state: int | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
) -> OptimizeResult:
    """SPSA in the space of principal components of the (time x feature) OD matrix.

    ``x0`` is reshaped to ``(-1, num_features)`` (one row per time interval),
    standardised and projected onto ``n_components`` components; SPSA then
    perturbs the component scores, using the scalar loss for the gradient.
    """
    _check_niter(niter)
    scaler = StandardScaler()
    data = scaler.fit_transform(np.array(x0).reshape(-1, num_features))
    pca = PCA(n_components=n_components, random_state=random_state)
    z = pca.fit_transform(data)
    n_comp = z.shape[1]
    z = z.flatten()

    def to_x(scores: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(pca.inverse_transform(scores.reshape(-1, n_comp))).flatten()

    x = to_x(z)
    fun = np.nan
    for k in range(niter):
        ak, ck = gain_sequence(gains, k, niter)
        grads = []
        for _ in range(reps):
            delta = rng.choice([-1, 1], size=len(z))
            zplus = z + np.multiply(z, ck * delta)
            zminus = z - np.multiply(z, ck * delta)
            grads.append(
                (objective.loss(to_x(zplus)) - objective.loss(to_x(zminus))) / (zplus - zminus)
            )
        grad = np.mean(grads, axis=0)
        z = z - ak * grad * z
        x = to_x(z)
        fun = objective.evaluate(x)
        if callback is not None:
            callback(x)
    return OptimizeResult(
        fun=fun,
        x=x,
        nit=niter,
        nfev=2 * niter * reps,
        success=True,
        message="terminated after reaching max number of iterations",
    )
