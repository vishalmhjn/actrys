import json

import numpy as np
import pytest

from actrys.calibration.synthetic_pipeline import SyntheticCalibration
from actrys.config import (
    AutotuneConfig,
    BiasCorrectionConfig,
    SPSAGains,
    SyntheticCalibrationConfig,
    SyntheticScenarioConfig,
)
from actrys.synthetic.scenario import create_assignment, generate_scenario, link_speeds

SMALL = SyntheticScenarioConfig(num_zones=6, num_detectors=40, start_hour=7, end_hour=10)


def test_scenario_shapes(rng):
    sc = generate_scenario(SMALL, rng=rng)
    n_pairs, t = 36, 3
    assert sc.od.shape == (n_pairs, t)
    assert sc.assignment.shape == (n_pairs * t, 40 * t)
    assert sc.counts.shape == sc.speeds.shape == (40 * t, 1)
    counts, speeds = sc.simulate(sc.od_vector)
    np.testing.assert_allclose(counts.reshape(-1, 1), sc.counts)
    np.testing.assert_allclose(speeds.reshape(-1, 1), sc.speeds)


def test_assignment_pattern_is_fixed_over_time(rng):
    od = np.ones((10, 3))
    a = create_assignment(od, 5, temporal_influence=8, network_correlation=0.5, rng=rng)
    for det in range(5):
        rows0 = np.flatnonzero(a[:, det])
        assert rows0.max() < 10  # first interval only sees first-interval demand
        for k in (1, 2):
            rows = set(np.flatnonzero(a[:, k * 5 + det]))
            assert rows == {r + lag * 10 for r in rows0 for lag in range(k + 1)}


def test_link_speeds_branches():
    q = np.array([0.0, 7000.0, 1e9])
    free, congested = link_speeds(q, 1, 1), link_speeds(q, 1, 0)
    assert free[0] == pytest.approx(120) and congested[0] == pytest.approx(0)
    assert free[1] == pytest.approx(60) == congested[1]  # capacity
    assert free[2] == 0 and congested[2] == 0  # beyond capacity -> 0


def _cfg(tmp_path, **kw):
    base = dict(
        output_dir=tmp_path,
        run_name="t",
        scenario=SMALL,
        iterations=15,
        bagging_runs=2,
        od_noise_percent=20,
        od_bias=0.8,
        plot=False,
        seed=0,
    )
    base.update(kw)
    return SyntheticCalibrationConfig(**base)


@pytest.mark.parametrize("algorithm", ["wspsa", "spsa", "pcspsa"])
def test_pipeline_runs_and_writes_outputs(tmp_path, algorithm):
    cfg = _cfg(tmp_path, algorithm=algorithm)
    results = SyntheticCalibration(cfg).run()
    out = tmp_path / "t"
    assert (out / "mean_0.7_2_od.csv").exists() and (out / "0_od.csv").exists()
    stored = json.loads(next(out.glob("results_*.json")).read_text())
    assert stored["f_val"] == results["f_val"] and len(results["f_val"]) == 2 * 15


def test_wspsa_reduces_weighted_loss(tmp_path):
    cfg = _cfg(tmp_path, iterations=40, bagging_runs=1, gains=SPSAGains(1, 0.3, 0.7, 0.01))
    results = SyntheticCalibration(cfg).run()
    f = results["f_val"]
    assert np.mean(f[-5:]) < f[0]


def test_reproducible_with_seed(tmp_path):
    r1 = SyntheticCalibration(_cfg(tmp_path / "a")).run()
    r2 = SyntheticCalibration(_cfg(tmp_path / "b")).run()
    assert r1["f_val"] == r2["f_val"]


def test_bias_correction_and_autotune(tmp_path):
    cfg = _cfg(
        tmp_path,
        bias_correction=BiasCorrectionConfig(enabled=True, method="naive"),
        autotune=AutotuneConfig(enabled=True, n_init=2, n_iter=1, iterations=3),
        plot=True,
    )
    results = SyntheticCalibration(cfg).run()
    assert results["spsa_a"] != cfg.gains.a
    assert list((tmp_path / "t").glob("*.png"))


@pytest.mark.parametrize("method", ["naive", "weighted"])
def test_only_bias_correction(tmp_path, method):
    cfg = _cfg(
        tmp_path, bias_correction=BiasCorrectionConfig(enabled=True, method=method, only=True)
    )
    results = SyntheticCalibration(cfg).run()
    assert len(results["od_val"]) == 2 and (tmp_path / "t" / "mean_od.csv").exists()


def test_weighted_bias_correction_runs(tmp_path):
    cfg = _cfg(tmp_path, bias_correction=BiasCorrectionConfig(enabled=True, method="weighted"))
    cal = SyntheticCalibration(cfg)
    p = cal.build_problem()
    bounds = cal.search_bounds(p)
    assert bounds.shape == (len(p.prior), 2) and np.isfinite(bounds).all()
    assert np.all(bounds[:, 0] <= bounds[:, 1])
    cal.run()


def test_bagging_result_is_best_iterate(tmp_path):
    cfg = _cfg(tmp_path, bagging_runs=1, gains=SPSAGains(1, 0.3, 0.7, 0.01))
    cal = SyntheticCalibration(cfg)
    results = cal.run()
    od = np.loadtxt(tmp_path / "t" / "0_od.csv", delimiter=",", skiprows=1)[:, 1]
    p = SyntheticCalibration(cfg).build_problem()  # same seed -> same problem
    obj = SyntheticCalibration(cfg).objective(p, p.prior)
    assert obj.loss(od) == pytest.approx(min(results["f_val"]))
