"""Random number handling and synthetic noise."""

from __future__ import annotations

import numpy as np

#: Anything exposing the numpy random API used here (``standard_normal``, ``choice``, ``integers``/``randint``).
RNG = np.random.Generator | np.random.RandomState


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a random generator; ``None`` gives a non-reproducible (OS-seeded) one."""
    return np.random.default_rng(seed)


def add_noise(
    x: np.ndarray, perc_var: float, mu: float = 1.0, rng: RNG | None = None
) -> np.ndarray:
    """Return ``trunc(mu * x) + trunc(N(0, 1) * perc_var * x)`` as integers.

    This is the perturbation used to create a biased/noisy prior from a true OD
    matrix (``mu`` is the bias factor, ``perc_var`` the relative std. deviation)
    and to add measurement noise to synthetic counts. One normal draw is taken
    per element, in order, even if ``perc_var`` is zero.
    """
    rng = np.random.mtrand._rand if rng is None else rng
    x = np.asarray(x).reshape(-1)
    draws = rng.standard_normal(len(x))
    return (np.trunc(mu * x) + np.trunc(draws * perc_var * x)).astype(np.int64)


def clip_negative(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, 0, x)
