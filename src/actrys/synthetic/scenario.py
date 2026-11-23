"""Analytical (static assignment) scenario generator for prototyping.

A random assignment matrix maps time-dependent OD flows to detector counts;
speeds follow from a quadratic fundamental diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import invgauss
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

from actrys.config import SyntheticScenarioConfig
from actrys.noise import RNG

#: Bi-modal (morning/evening peak) daily demand profile, one value per hour.
DAILY_PROFILE = np.array(
    [1, 1, 0.5, 0.5, 1, 1, 2, 2.5, 3, 3.5, 3, 2.5, 2, 2, 1.5, 1.5, 2, 2.5, 3, 2.5, 2.5, 2, 2, 1]
)


def fit_temporal_kernel(hours: np.ndarray, profile: np.ndarray) -> GaussianProcessRegressor:
    """Fit a GP (Matern + rational quadratic kernel) to a temporal demand profile."""
    kernel = Matern() + RationalQuadratic()
    return GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
        hours.reshape(-1, 1), profile.reshape(-1, 1)
    )


def create_od_matrix(
    num_zones: int,
    start_hour: float,
    end_hour: float,
    interval: float = 1,
    lower_limit_od: float = 0,
    upper_limit_od: float = 100,
    od_distribution: str = "uniform",
    temporal_noise: float = 0.1,
    *,
    rng: RNG,
) -> np.ndarray:
    """Synthetic demand of shape ``(num_zones**2, n_intervals)``.

    A static OD vector is drawn from ``od_distribution`` and scaled over time
    by the daily profile, with multiplicative Gaussian noise per interval.
    """
    n_pairs = num_zones**2
    n_intervals = int((end_hour - start_hour) / interval)
    tod = np.arange(start_hour, end_hour, interval).reshape(-1, 1)
    model = fit_temporal_kernel(np.arange(24), DAILY_PROFILE)
    shape = model.predict(tod, return_std=True)[0]

    if od_distribution == "uniform":
        od_v = rng.uniform(lower_limit_od, upper_limit_od, n_pairs)
    elif od_distribution == "beta":
        # inverse-Gaussian (heavy tailed) flows, parameters adapted to the Munich example
        od_v = (invgauss.rvs(3, size=n_pairs, random_state=rng) * 9).astype("int")
    else:
        raise ValueError(f"Unknown OD distribution {od_distribution!r}; use 'uniform' or 'beta'")

    od = np.zeros((n_pairs, n_intervals), dtype="float32")
    for i in range(n_intervals):
        noise = rng.normal(0, temporal_noise, n_pairs)
        col = shape[i] * od_v * (1 + noise)
        od[:, i] = col
        od[:, i][od[:, i] < 0] = 0
        od[:, i] = np.round(od[:, i], 0)
    return od


def create_assignment(
    od: np.ndarray,
    num_detectors: int,
    temporal_influence: int = 8,
    min_sensor_correlation: float = 0.2,
    max_sensor_correlation: float = 0.4,
    network_correlation: float = 0.2,
    go_random: bool = False,
    *,
    rng: RNG,
) -> np.ndarray:
    """Random assignment matrix of shape ``(n_pairs * T, num_detectors * T)``.

    Each detector observes a random subset of (OD pair, departure interval)
    combinations from the current and up to ``temporal_influence`` past
    intervals with a random assignment ratio. ``network_correlation`` caps the
    fraction of candidates a detector may observe. Unless ``go_random`` is set,
    the OD pairs observed by a detector are fixed after the first interval
    (reasonable for uncongested networks).
    """
    n_pairs, n_intervals = od.shape
    assignment = np.zeros((n_pairs * n_intervals, num_detectors * n_intervals), dtype="float32")

    def draw(candidates: np.ndarray, upper: int) -> tuple[np.ndarray, np.ndarray]:
        n_corr = (
            int(rng.integers(1, upper)) if hasattr(rng, "integers") else int(rng.randint(1, upper))
        )
        idx = rng.permutation(candidates)[:n_corr]
        return idx, rng.uniform(min_sensor_correlation, max_sensor_correlation, n_corr)

    for k in range(n_intervals):
        n_past = min(k, temporal_influence)
        candidates = np.arange((k - n_past) * n_pairs, min(k + 1, n_intervals) * n_pairs)
        upper = int(len(candidates) * network_correlation) or 2

        for det in range(num_detectors):
            col = k * num_detectors + det
            if k == 0 or go_random:
                idx, w = draw(candidates, upper)
                assignment[idx, col] = w
            else:
                # replicate the first-interval pattern for all influencing intervals
                base_rows = np.flatnonzero(assignment[:, det])
                base_w = assignment[base_rows, det]
                for lag in range(k - n_past, k + 1):
                    assignment[base_rows + lag * n_pairs, col] = base_w
    return assignment


def link_speeds(
    q: np.ndarray,
    interval: float,
    traffic_regime: int,
    max_flow: float = 7000,
    max_speed: float = 120,
) -> np.ndarray:
    """Speed from flow via the quadratic fundamental diagram ``Q = k1 v^2 + k2 v``.

    ``traffic_regime`` selects the branch: 1 = free flow, 0 = congested.
    Flows above capacity give speed 0.
    """
    k1 = (-1 * max_flow * 4) / (max_speed**2)
    k2 = -1 * k1 * max_speed
    flow = np.asarray(q) / interval
    with np.errstate(invalid="ignore"):
        root = np.sqrt(k2**2 + 4 * flow * k1)
    v = (-k2 + root) / (2 * k1) if traffic_regime == 0 else (-k2 - root) / (2 * k1)
    v = np.nan_to_num(v, nan=0)
    return np.where(v < 0, 0, v)


@dataclass(frozen=True)
class SyntheticScenario:
    """A generated analytical scenario."""

    od: np.ndarray  #: (n_pairs, T) true demand
    assignment: np.ndarray  #: (n_pairs*T, n_detectors*T)
    counts: np.ndarray  #: (n_detectors*T, 1) true counts, interval-major
    speeds: np.ndarray  #: (n_detectors*T, 1) true speeds
    traffic_states: np.ndarray  #: (n_detectors,) 1 = free flow, 0 = congested
    interval: float
    num_detectors: int

    @property
    def od_vector(self) -> np.ndarray:
        """True demand flattened interval-major, i.e. the decision-variable layout."""
        return self.od.T.flatten()

    def simulate(self, od_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return synthetic_simulation(
            od_vector, self.assignment, self.traffic_states, self.interval, self.num_detectors
        )


def _speeds_from_counts(
    counts: np.ndarray, t_state: np.ndarray, interval: float, num_detectors: int
) -> np.ndarray:
    per_detector = counts.reshape(-1, num_detectors).T
    speed = np.zeros(per_detector.shape, dtype="float32")
    free = np.where(t_state == 1)
    congested = np.where(t_state == 0)
    # NB: regime index 1 selects the "-" root (high speed), 0 the "+" root
    speed[free, :] = link_speeds(per_detector[free, :], interval, 1)
    speed[congested, :] = link_speeds(per_detector[congested, :], interval, 0)
    return speed.T.reshape(-1, 1)


def synthetic_simulation(
    od_vector: np.ndarray,
    assignment: np.ndarray,
    t_state: np.ndarray,
    interval: float,
    num_detectors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts (``A^T x``) and speeds for an OD vector."""
    counts = assignment.T @ od_vector
    speed = _speeds_from_counts(counts, t_state, interval, num_detectors).flatten()
    return counts, speed


def generate_scenario(cfg: SyntheticScenarioConfig, *, rng: RNG) -> SyntheticScenario:
    od = create_od_matrix(
        cfg.num_zones,
        cfg.start_hour,
        cfg.end_hour,
        cfg.interval_hours,
        cfg.lower_limit_od,
        cfg.upper_limit_od,
        cfg.od_distribution,
        rng=rng,
    )
    # assume demand influences counts for up to 2 hours
    temporal_influence = 2 * int(1 / cfg.interval_hours)
    assignment = create_assignment(
        od,
        cfg.num_detectors,
        temporal_influence,
        network_correlation=cfg.network_correlation,
        rng=rng,
    )
    counts = assignment.T @ od.T.reshape(-1, 1)
    t_state = rng.binomial(1, 0.5, size=cfg.num_detectors)
    speeds = _speeds_from_counts(counts, t_state, cfg.interval_hours, cfg.num_detectors)
    return SyntheticScenario(
        od=od,
        assignment=assignment,
        counts=counts,
        speeds=speeds,
        traffic_states=t_state,
        interval=cfg.interval_hours,
        num_detectors=cfg.num_detectors,
    )
