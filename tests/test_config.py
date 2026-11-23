from pathlib import Path

import pytest

from actrys.config import (
    ObjectiveWeights,
    SumoCalibrationConfig,
    SyntheticCalibrationConfig,
    TimeWindow,
    load_config,
    replace_path,
)
from conftest import REPO


@pytest.mark.parametrize(
    "name,cls",
    [
        ("munich.toml", SumoCalibrationConfig),
        ("munich_real_data.toml", SumoCalibrationConfig),
        ("synthetic.toml", SyntheticCalibrationConfig),
    ],
)
def test_example_configs_load(name, cls):
    cfg = load_config(cls, REPO / "configs" / name)
    assert isinstance(cfg, cls)


def test_paths_resolve_relative_to_config_file():
    cfg = load_config(SumoCalibrationConfig, REPO / "configs" / "munich.toml")
    assert cfg.scenario.directory == (REPO / "munich").resolve()


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[demand]\nbaging_runs = 3\n")
    with pytest.raises(ValueError, match="baging_runs"):
        load_config(SumoCalibrationConfig, p)


def test_literal_validation(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('estimator = "mape"\n')
    with pytest.raises(ValueError, match="estimator"):
        load_config(SumoCalibrationConfig, p)


def test_time_window():
    w = TimeWindow(
        start_hour=7, end_hour=9, warm_up_hours=2, cool_down_hours=0, interval_seconds=3600
    )
    assert w.demand_hours.tolist() == [5, 6, 7, 8]
    assert w.evaluation_hours.tolist() == [7, 8]
    assert (w.simulation_begin, w.simulation_end) == (18000, 32400)
    w15 = TimeWindow(7, 8, 1, 0, 900)
    assert len(w15.demand_hours) == 8 and len(w15.evaluation_hours) == 4
    assert w15.simulation_end == 8 * 3600


def test_weights_cannot_all_be_zero():
    with pytest.raises(ValueError):
        ObjectiveWeights(0, 0, 0)


def test_replace_path():
    cfg = SumoCalibrationConfig()
    new = replace_path(cfg, "demand.gains.a", 0.5)
    assert new.demand.gains.a == 0.5 and cfg.demand.gains.a == 0.1
    assert replace_path(cfg, "scenario.directory", Path("x")).scenario.directory == Path("x")
