import numpy as np
import pytest
import scipy.sparse as sp

from actrys.config import SPSAGains
from actrys.optim.spsa import (
    gain_sequence,
    minimize_pcspsa,
    minimize_spsa,
    minimize_wspsa,
    pca_components_for_variance,
)
from reference import minimize_wspsa_loop


class Quadratic:
    """Separable least squares towards ``target``; deviations are per-element squared errors."""

    def __init__(self, target):
        self.target = np.asarray(target, float)
        self.evaluated = []

    def deviations(self, x, perturbation="positive"):
        return (x - self.target) ** 2

    def loss(self, x):
        return float(np.sum((x - self.target) ** 2))

    def evaluate(self, x):
        self.evaluated.append(np.array(x))
        return self.loss(x)


def test_gain_sequence():
    ak, ck = gain_sequence(SPSAGains(a=1, c=2, alpha=1, gamma=1), k=0, niter=100)
    assert ak == pytest.approx(1 / 2) and ck == pytest.approx(2.0)


def test_wspsa_matches_reference_loop():
    target = np.array([50.0, 80, 20, 140, 60])
    x0 = np.array([30.0, 100, 10, 100, 90])
    bounds = np.column_stack([0.1 * x0, 3 * x0])
    w = (np.random.default_rng(1).uniform(size=(5, 5)) > 0.4).astype(np.int8) | np.eye(
        5, dtype=np.int8
    )
    obj = Quadratic(target)

    np.random.seed(11)
    expected = minimize_wspsa_loop(
        lambda x, weighted: obj.deviations(x) if weighted else obj.loss(x),
        x0,
        w,
        bounds,
        25,
        0.5,
        0.1,
        0.602,
        0.101,
        2,
        0.3,
    )
    np.random.seed(11)
    res = minimize_wspsa(
        Quadratic(target),
        x0,
        sp.csr_matrix(w),
        gains=SPSAGains(0.5, 0.1),
        niter=25,
        bounds=bounds,
        reps=2,
        momentum=0.3,
        rng=np.random.mtrand._rand,
    )
    np.testing.assert_allclose(res.x, expected, rtol=1e-12)


@pytest.mark.parametrize("parallel", [False, True])
def test_wspsa_converges(parallel):
    target = np.array([50.0, 80, 20, 140])
    obj = Quadratic(target)
    x0 = np.full(4, 60.0)
    res = minimize_wspsa(
        obj,
        x0,
        np.eye(4),
        gains=SPSAGains(0.2, 0.05),
        niter=60,
        bounds=np.column_stack([np.zeros(4), np.full(4, 500)]),
        momentum=0.0,
        rng=np.random.default_rng(0),
        parallel=parallel,
    )
    assert obj.loss(res.x) < 0.2 * obj.loss(x0)
    assert len(obj.evaluated) == 60 and res.fun == obj.loss(res.x)


def test_spsa_improves():
    target = np.array([50.0, 80, 20, 140])
    obj = Quadratic(target)
    x0 = np.full(4, 60.0)
    res = minimize_spsa(
        obj,
        x0,
        gains=SPSAGains(1e-5, 0.05),
        niter=40,
        bounds=np.column_stack([np.zeros(4), np.full(4, 500)]),
        rng=np.random.default_rng(0),
    )
    assert obj.loss(res.x) < obj.loss(x0)


def test_pcspsa_runs():
    rng = np.random.default_rng(0)
    target = rng.uniform(10, 100, 3 * 8)
    x0 = target * rng.uniform(0.5, 1.5, target.size)
    n_comp = pca_components_for_variance(x0, 8)
    assert n_comp >= 1
    res = minimize_pcspsa(
        Quadratic(target),
        x0,
        8,
        n_comp,
        gains=SPSAGains(1e-3, 0.01),
        niter=5,
        rng=rng,
        random_state=0,
    )
    assert res.x.shape == x0.shape


def test_niter_validated():
    with pytest.raises(ValueError):
        minimize_spsa(
            Quadratic([1.0]),
            np.array([1.0]),
            gains=SPSAGains(),
            niter=0,
            rng=np.random.default_rng(),
        )


def test_weight_rows_validated():
    with pytest.raises(ValueError, match="rows"):
        minimize_wspsa(
            Quadratic([1.0, 2]),
            np.array([1.0, 2]),
            np.eye(3),
            gains=SPSAGains(),
            niter=1,
            rng=np.random.default_rng(),
        )
