import numpy as np
import pytest

from actrys.metrics import gof, median_gof, squared_deviation

DATA = np.array([10.0, 20.0, 30.0, 40.0])
SIM = np.array([12.0, 18.0, 33.0, 35.0])


def test_perfect_fit_is_zero():
    for est in ("rmsn", "rmse", "smape", "rmsp", "wmape", "composite"):
        assert gof(DATA, DATA, est) == pytest.approx(0.0, abs=1e-6)


def test_known_values():
    diff = DATA - SIM
    assert gof(DATA, SIM, "wmape") == pytest.approx(np.abs(diff).sum() / DATA.sum())
    assert gof(DATA, SIM, "rmse") == pytest.approx(np.sqrt((diff**2).mean()))
    assert gof(DATA, SIM, "rmsn") == pytest.approx(
        np.sqrt(len(DATA) * (diff**2).sum()) / DATA.sum(), abs=1e-6
    )
    assert gof(DATA, SIM, "composite") == pytest.approx(
        gof(DATA, SIM, "rmsn") + gof(DATA, SIM, "smape")
    )


def test_returns_python_float():
    assert type(gof(DATA, SIM, "wmape")) is float


def test_errors():
    with pytest.raises(ValueError, match="Shape mismatch"):
        gof(DATA, SIM[:2])
    with pytest.raises(ValueError, match="Unknown estimator"):
        gof(DATA, SIM, "nope")
    with pytest.raises(ValueError):
        median_gof(DATA, SIM, "wmape")
    with pytest.raises(ValueError, match="Unknown deviation"):
        squared_deviation(DATA, SIM, "nope")


def test_median_gof():
    assert median_gof(DATA, DATA, "smape") == pytest.approx(0.0, abs=1e-6)
    assert median_gof(DATA, SIM, "rmsp") == gof(DATA, SIM, "rmsp")


@pytest.mark.parametrize(
    "metric",
    ["standardized_weighted_geh", "scaled_weighted_geh", "unweighted_geh", "standardized_rmse"],
)
def test_squared_deviation_shape_and_zero(metric):
    dev = squared_deviation(DATA, SIM, metric)
    assert dev.shape == DATA.shape
    assert np.allclose(squared_deviation(DATA, DATA, metric), 0)
