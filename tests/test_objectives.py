import numpy as np
import pytest

from actrys.calibration.bias import bias_factor, search_bounds
from actrys.calibration.objectives import AssignmentObjective, stack_deviations
from actrys.calibration.state import BestSolution
from actrys.config import ObjectiveWeights
from actrys.metrics import squared_deviation
from reference import assignment_counts_loop


def test_assignment_counts_match_reference_loop():
    rng = np.random.default_rng(0)
    a = (rng.uniform(size=(40, 12)) * (rng.uniform(size=(40, 12)) > 0.7)).astype("float32")
    x = rng.uniform(-5, 300, 40)
    obj = AssignmentObjective(a, np.ones(12), x, x, ObjectiveWeights(1, 1, 0), "wmape")
    np.testing.assert_array_equal(obj.counts(x), assignment_counts_loop(x, a))


def test_assignment_objective_tracks_best():
    rng = np.random.default_rng(1)
    a = rng.uniform(size=(6, 4)).astype("float32")
    true = rng.integers(10, 100, 6)
    obs = AssignmentObjective(a, np.ones(4), true, true, ObjectiveWeights(1, 0, 0), "wmape").counts(
        true
    )
    best = BestSolution()
    obj = AssignmentObjective(a, obs, true, true, ObjectiveWeights(0.5, 0.5, 0), "wmape", best=best)
    assert obj.evaluate(true * 2) > obj.evaluate(true)
    assert best.od.tolist() == true.tolist() and best.gof == pytest.approx(0)
    assert obj.deviations(true * 2).shape == (4 + 6,)


def test_stack_deviations_order_and_zero_weights():
    c, o, s = (
        (np.array([1.0, 2, 3]), np.array([1.5, 2, 2])),
        (np.array([4.0, 5]), np.array([4.0, 7])),
        None,
    )
    out = stack_deviations(ObjectiveWeights(2, 1, 0), c, o, s)
    np.testing.assert_allclose(out, np.hstack([2 * squared_deviation(*c), squared_deviation(*o)]))
    with pytest.raises(ValueError):
        stack_deviations(ObjectiveWeights(0, 0, 1), c, o, None)


def test_bias_factor_and_bounds():
    sim, obs = np.array([20.0, 40]), np.array([10.0, 20])
    assert bias_factor("naive", sim, obs) == 2
    inc = np.array([[1.0, 0], [0, 1], [0, 0]])
    f = bias_factor(
        "weighted", np.array([20.0, 10]), np.array([10.0, 20]), incidence=inc, fill_nan=True
    )
    np.testing.assert_allclose(f, [2, 0.5, 1.0])
    with pytest.raises(ValueError):
        bias_factor("magic", sim, obs)
    b = search_bounds(np.array([10, 20]), 0.1, 2.0)
    np.testing.assert_allclose(b, [[1, 20], [2, 40]])
    # per-variable upper factors
    np.testing.assert_allclose(
        search_bounds(np.array([1, 0]), 0.1, np.array([5.0, 7.0])),
        [[0.1, 5.0], [0, 0]],
    )
