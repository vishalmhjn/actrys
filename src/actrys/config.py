"""Typed configuration for actrys runs.

Configurations are plain dataclasses that can be loaded from TOML files
(see ``configs/`` in the repository). Relative paths in a config file are
resolved against the directory containing that file, so runs do not depend on
the current working directory.
"""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np

Estimator = Literal["rmsn", "rmse", "smape", "rmsp", "wmape", "composite"]
BiasCorrectionMethod = Literal["naive", "weighted"]
Algorithm = Literal["spsa", "wspsa", "pcspsa"]
T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeWindow:
    """Analysis period of a dynamic (SUMO) scenario.

    All hours are hours of the day. Demand is simulated for
    ``[start_hour - warm_up_hours, end_hour + cool_down_hours)`` (clipped to
    the day) while measurements are compared only for the evaluation hours.
    """

    start_hour: int = 7
    end_hour: int = 9
    warm_up_hours: int = 2
    cool_down_hours: int = 0
    interval_seconds: int = 3600

    @property
    def interval_hours(self) -> float:
        return self.interval_seconds / 3600

    @property
    def demand_hours(self) -> np.ndarray:
        """Start hour of every simulated demand interval (``TOD`` in the paper code)."""
        return np.arange(
            max(self.start_hour - self.warm_up_hours, 0),
            min(24, self.end_hour + self.cool_down_hours),
            self.interval_hours,
        )

    @property
    def evaluation_hours(self) -> np.ndarray:
        """Start hour of every interval for which measurements are compared."""
        tod = self.demand_hours
        return np.arange(
            tod[0] + self.warm_up_hours,
            self.interval_hours + tod[-1] - self.cool_down_hours,
            self.interval_hours,
        )

    @property
    def simulation_begin(self) -> float:
        return self.demand_hours[0] * 3600

    @property
    def simulation_end(self) -> float:
        return (self.demand_hours[-1] + self.interval_hours) * 3600


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights of the multi-objective calibration function."""

    counts: float = 0.9
    od: float = 0.1
    speed: float = 0.0

    def __post_init__(self) -> None:
        if self.counts == 0 and self.od == 0 and self.speed == 0:
            raise ValueError("At least one objective weight must be non-zero")


@dataclass(frozen=True)
class SPSAGains:
    """SPSA gain sequence parameters ``a_k = a / (k + 1 + A)^alpha``, ``c_k = c / (k + 1)^gamma``."""

    a: float = 0.1
    c: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101


@dataclass(frozen=True)
class AutotuneConfig:
    """Bayesian optimization of the SPSA ``a`` and ``c`` gains (searched in log10 space)."""

    enabled: bool = True
    n_init: int = 30
    n_iter: int = 30
    log_a_bounds: tuple[float, float] = (-7.0, -1.0)
    log_c_bounds: tuple[float, float] = (-2.0, 1.0)
    random_state: int = 1
    #: SPSA iterations per trial; ``None`` uses the calibration's own iteration count
    iterations: int | None = 50


@dataclass(frozen=True)
class BiasCorrectionConfig:
    """One-shot heuristic correcting a systematic bias in the prior OD matrix."""

    enabled: bool = True
    method: BiasCorrectionMethod = "naive"
    only: bool = False
    lower_bound: float = 0.1


# --------------------------------------------------------------------------- #
# SUMO (dynamic) calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioConfig:
    """Files that make up a SUMO scenario (paths relative to ``directory``)."""

    directory: Path = Path("munich")
    run_name: str = "run"
    network: str = "network.net.xml"
    zones: str = "tazes.taz.xml"
    detectors: str = "additional.add.xml"
    demand_dir: str = "demand"
    od_prefix: str = "OD"
    od_list: str = "od_list.txt"

    def path(self, name: str) -> Path:
        return self.directory / name


@dataclass(frozen=True)
class DataConfig:
    """Where the "observed" measurements come from."""

    synthetic_counts: bool = True
    #: noise (in %) and multiplicative bias applied to the true OD to create the prior
    od_noise_percent: float = 10
    od_bias: float = 1.0
    #: noise (in %) added to synthetic counts
    count_noise_percent: float = 0
    #: real-world data, only used when ``synthetic_counts`` is false
    real_counts: Path | None = None
    real_speeds: Path | None = None
    matchable_detectors: Path | None = None


@dataclass(frozen=True)
class SupplyParams:
    """Mesoscopic supply / route-choice parameters passed to ``sumo``."""

    rerouting_prob: float = 0.9
    tls_tt_penalty: float = 6.0
    meso_minor_penalty: int = 10
    rerouting_period: int = 30
    rerouting_adaptation: int = 5
    rerouting_adaptation_steps: int = 157
    meso_tls_flow_penalty: float = 0.9
    priority_factor: float = 0.01

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DemandCalibrationConfig:
    enabled: bool = True
    algorithm: Algorithm = "wspsa"
    bias_correction: BiasCorrectionConfig = field(default_factory=BiasCorrectionConfig)
    autotune: AutotuneConfig = field(default_factory=AutotuneConfig)
    #: gains of the W-SPSA running with the simulator in the loop
    gains: SPSAGains = field(default_factory=lambda: SPSAGains(a=0.1, c=0.1))
    #: gains of the W-SPSA running on the analytical approximation (simulator out of the loop)
    out_of_loop_gains: SPSAGains = field(
        default_factory=lambda: SPSAGains(a=1.0, c=1.0, alpha=0.7, gamma=0.01)
    )
    spsa_reps: int = 1
    momentum: float = 0.3
    out_of_loop_momentum: float = 0.7
    sim_in_loop: bool = True
    sim_out_loop: bool = False
    outer_iterations: int = 1
    in_loop_iterations: int = 5
    out_of_loop_iterations: int = 1
    bagging_runs: int = 3
    #: start bagging runs > 0 from the best solution of the previous run
    warm_start: bool = False
    #: exploration noise (in %) added to the starting point of every bagging run
    exploration_noise_percent: float = 10
    #: a and c are divided by ``learning_decay_factor ** iterations`` after every SPSA call
    learning_decay_factor: float = 1.05
    #: incidence ratios below this threshold are dropped from the W-SPSA weight matrix
    wspsa_threshold: float = 0.01


@dataclass(frozen=True)
class SupplyCalibrationConfig:
    enabled: bool = False
    n_init: int = 50
    n_iter: int = 50
    random_state: int = 1
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "rerouting_probability": (0.1, 0.5),
            "rerouting_period": (30, 90),
            "rerouting_adaptation": (5, 120),
            "rerouting_adaptation_steps": (120, 240),
            "meso_minor_penalty": (0, 60),
            "tls_tt_penalty": (0, 1),
            "meso_tls_flow_penalty": (0, 1),
            "priority_factor": (0.1, 1),
        }
    )


@dataclass(frozen=True)
class SimulationConfig:
    routing_threads: int = 16
    #: evaluate the +/- SPSA perturbations concurrently
    parallel_perturbations: bool = True
    #: "meso" averages lane detectors of an edge, "micro" sums them
    sim_type: Literal["meso", "micro"] = "meso"


@dataclass(frozen=True)
class SumoCalibrationConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    time: TimeWindow = field(default_factory=TimeWindow)
    data: DataConfig = field(default_factory=DataConfig)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    estimator: Estimator = "wmape"
    demand: DemandCalibrationConfig = field(default_factory=DemandCalibrationConfig)
    supply: SupplyCalibrationConfig = field(default_factory=SupplyCalibrationConfig)
    initial_supply: SupplyParams = field(default_factory=SupplyParams)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    #: number of sequential demand -> supply calibration rounds
    n_sequential: int = 3
    #: simulate the scenario demand first (needed for synthetic counts)
    simulate_base_scenario: bool = True
    plot: bool = True
    seed: int | None = None


# --------------------------------------------------------------------------- #
# Analytical (synthetic, static assignment) calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticScenarioConfig:
    """Parameters of the randomly generated static-assignment scenario."""

    num_zones: int = 50
    num_detectors: int = 500
    interval_hours: float = 1.0
    start_hour: int = 7
    end_hour: int = 10
    lower_limit_od: float = 0
    upper_limit_od: float = 100
    od_distribution: Literal["uniform", "beta"] = "beta"
    network_correlation: float = 0.2

    @property
    def num_od_pairs(self) -> int:
        return self.num_zones**2


@dataclass(frozen=True)
class SyntheticCalibrationConfig:
    output_dir: Path = Path("synthetic_sims")
    run_name: str = "run"
    scenario: SyntheticScenarioConfig = field(default_factory=SyntheticScenarioConfig)
    weights: ObjectiveWeights = field(default_factory=lambda: ObjectiveWeights(0.5, 0.5, 0.0))
    estimator: Estimator = "wmape"
    algorithm: Algorithm = "wspsa"
    od_noise_percent: float = 50
    od_bias: float = 0.6
    count_noise_percent: float = 0
    gains: SPSAGains = field(default_factory=lambda: SPSAGains(a=3, c=0.01, alpha=0.7, gamma=0.01))
    spsa_reps: int = 1
    iterations: int = 30
    momentum: float = 0.7
    bagging_runs: int = 10
    bias_correction: BiasCorrectionConfig = field(
        default_factory=lambda: BiasCorrectionConfig(enabled=False, method="weighted")
    )
    autotune: AutotuneConfig = field(
        default_factory=lambda: AutotuneConfig(
            enabled=False,
            n_init=50,
            n_iter=100,
            log_a_bounds=(-3.0, 4.0),
            log_c_bounds=(-2.0, 1.0),
            random_state=2,
            iterations=None,
        )
    )
    plot: bool = True
    seed: int | None = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _build(cls: type, data: dict[str, Any], base_dir: Path, where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown key(s) in [{where}]: {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, base_dir, f"{where}.{name}" if where else name)
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, base_dir: Path, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(args[0], value, base_dir, where)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ValueError(f"[{where}] must be a table")
        return _build(tp, value, base_dir, where)
    if tp is Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    if origin is tuple:
        return tuple(value)
    if origin is dict:
        _, val_tp = typing.get_args(tp)
        return {k: _coerce(val_tp, v, base_dir, f"{where}.{k}") for k, v in value.items()}
    if origin is Literal:
        if value not in typing.get_args(tp):
            raise ValueError(f"[{where}] must be one of {typing.get_args(tp)}, got {value!r}")
        return value
    if tp is float and isinstance(value, int):
        return float(value)
    return value


def load_config(cls: type[T], path: str | Path) -> T:
    """Load a TOML file into the configuration dataclass ``cls``."""
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return _build(cls, data, path.parent.resolve(), "")


def replace_path(cfg: Any, dotted: str, value: Any) -> Any:
    """Return a copy of a (nested) frozen config with ``dotted`` set to ``value``."""
    head, _, rest = dotted.partition(".")
    if not rest:
        return dataclasses.replace(cfg, **{head: value})
    return dataclasses.replace(cfg, **{head: replace_path(getattr(cfg, head), rest, value)})
