"""End-to-end SUMO calibration on a tiny grid scenario (needs SUMO)."""

import dataclasses
import json

import numpy as np
import pytest

from actrys.calibration.sumo_pipeline import SumoCalibration
from actrys.config import (
    AutotuneConfig,
    BiasCorrectionConfig,
    DemandCalibrationConfig,
    ScenarioConfig,
    SimulationConfig,
    SumoCalibrationConfig,
    SupplyCalibrationConfig,
)
from conftest import requires_sumo
from sumo_scenario import build

pytestmark = [requires_sumo, pytest.mark.sumo]


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    return build(tmp_path_factory.mktemp("grid"))


def _cfg(scenario, run_name, **demand):
    d = dict(bagging_runs=1, in_loop_iterations=2, autotune=AutotuneConfig(enabled=False))
    d.update(demand)
    return SumoCalibrationConfig(
        scenario=ScenarioConfig(directory=scenario, run_name=run_name),
        demand=DemandCalibrationConfig(**d),
        simulation=SimulationConfig(routing_threads=1),
        n_sequential=1,
        plot=True,
        seed=0,
    )


def test_demand_calibration(scenario):
    results = SumoCalibration(_cfg(scenario, "demand")).run()
    run = scenario / "demand"
    # prior + 2 W-SPSA iterations
    assert len(results["f_val"]) == 3
    assert (run / "mean_0_1_od.csv").exists() and (run / "best_out.csv").exists()
    assert list(run.glob("img_results_*.png"))
    stored = json.loads(next(run.glob("results_*.json")).read_text())
    assert stored["supply_params"]["rerouting_prob"] == 0.9


def test_out_of_loop_autotune_and_supply(scenario):
    cfg = _cfg(
        scenario,
        "full",
        sim_out_loop=True,
        sim_in_loop=True,
        out_of_loop_iterations=2,
        autotune=AutotuneConfig(enabled=True, n_init=2, n_iter=1, iterations=2),
    )
    cfg = dataclasses.replace(cfg, supply=SupplyCalibrationConfig(enabled=True, n_init=2, n_iter=1))
    results = SumoCalibration(cfg).run()
    assert len(results["f_val_supply"]) == 1
    assert len(results["tuned_spsa_gains"]) == 1
    assert np.isfinite(results["best_supply_count_rmsn"])
    # the calibrated supply parameters are those of the best trial
    bo, supply = results["supply_bo_params"], results["supply_params"]
    assert supply["rerouting_prob"] == bo["rerouting_probability"]
    assert supply["rerouting_period"] == 1 + int(bo["rerouting_period"])


def test_supply_only(scenario):
    cfg = _cfg(scenario, "supply", enabled=False)
    cfg = dataclasses.replace(cfg, supply=SupplyCalibrationConfig(enabled=True, n_init=2, n_iter=0))
    results = SumoCalibration(cfg).run()
    assert len(results["f_val_supply"]) == 1


def test_only_bias_correction(scenario):
    cfg = _cfg(scenario, "bias", bias_correction=BiasCorrectionConfig(only=True))
    results = SumoCalibration(cfg).run()
    assert len(results["f_val"]) == 2
    assert (scenario / "bias" / "mean_od.csv").exists()
