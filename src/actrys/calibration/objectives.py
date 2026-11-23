"""Calibration objectives that do not need SUMO.

All objectives implement :class:`actrys.optim.spsa.Objective`:

* ``deviations``: weighted element-wise deviations ``[counts | od | speeds]``
  (blocks with zero weight are omitted) used for the (W-)SPSA gradient,
* ``loss``: the scalar weighted goodness of fit,
* ``evaluate``: ``loss`` at an accepted iterate, which also updates the
  calibration history / best solution.

Decision variables are clipped at zero and truncated to integers first.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

from actrys.calibration.state import BestSolution, CalibrationHistory
from actrys.config import ObjectiveWeights
from actrys.metrics import gof, squared_deviation
from actrys.synthetic.scenario import SyntheticScenario

log = logging.getLogger(__name__)


def to_demand(x: np.ndarray) -> np.ndarray:
    """Feasible integer demand from an optimizer iterate."""
    return np.where(x < 0, 0, x).astype(int)


def stack_deviations(
    weights: ObjectiveWeights,
    counts: tuple[np.ndarray, np.ndarray] | None,
    od: tuple[np.ndarray, np.ndarray] | None,
    speeds: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """``[w_c * dev(counts) | w_od * dev(od) | w_s * dev(speeds)]`` for the non-zero weights."""
    blocks = []
    for w, pair in ((weights.counts, counts), (weights.od, od), (weights.speed, speeds)):
        if w != 0:
            if pair is None:
                raise ValueError("Missing data for a term with non-zero weight")
            blocks.append(w * squared_deviation(*pair))
    if not blocks:
        raise ValueError("All objective weights are zero")
    return blocks[0] if len(blocks) == 1 else np.hstack(blocks)


class AssignmentObjective:
    """Counts approximated by a (sparse) assignment matrix: ``c_j = sum_i trunc(x_i * A_ij)``.

    Used for calibration with the simulator out of the loop and for tuning the
    SPSA gains. Speeds are not modelled, so only the count and OD terms count.
    """

    def __init__(
        self,
        assignment: np.ndarray,
        observed_counts: np.ndarray,
        od_true: np.ndarray,
        od_prior: np.ndarray,
        weights: ObjectiveWeights,
        estimator: str,
        best: BestSolution | None = None,
    ):
        coo = sp.coo_matrix(assignment)
        self._rows, self._cols, self._vals = coo.row, coo.col, coo.data
        self._n_cols = assignment.shape[1]
        self.observed_counts = np.asarray(observed_counts).flatten()
        self.od_true = od_true
        self.od_prior = od_prior
        self.weights = ObjectiveWeights(weights.counts, weights.od, 0.0)
        self.estimator = estimator
        self.best = best if best is not None else BestSolution()

    def counts(self, x: np.ndarray) -> np.ndarray:
        demand = to_demand(x)
        contrib = (demand[self._rows] * self._vals).astype(np.int64)
        return np.bincount(self._cols, weights=contrib, minlength=self._n_cols).reshape(-1, 1)

    def deviations(self, x: np.ndarray, perturbation: str = "positive") -> np.ndarray:
        demand = to_demand(x)
        sim = self.counts(demand).flatten()
        return stack_deviations(
            self.weights, (self.observed_counts, sim), (self.od_prior, demand), None
        )

    def _loss(self, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        demand = to_demand(x)
        sim = self.counts(demand)
        count_gof = np.round(gof(self.observed_counts, sim.flatten(), self.estimator), 4)
        total = self.weights.counts * count_gof + self.weights.od * gof(
            self.od_true, demand, self.estimator
        )
        return total, demand, sim

    def loss(self, x: np.ndarray) -> float:
        return self._loss(x)[0]

    def evaluate(self, x: np.ndarray) -> float:
        total, demand, sim = self._loss(x)
        self.best.offer(total, demand, counts=sim.flatten())
        return total


class SyntheticObjective:
    """Objective for the analytical scenario (counts and speeds from :class:`SyntheticScenario`)."""

    def __init__(
        self,
        scenario: SyntheticScenario,
        observed_counts: np.ndarray,
        observed_speeds: np.ndarray,
        od_true: np.ndarray,
        od_prior: np.ndarray,
        weights: ObjectiveWeights,
        estimator: str,
        history: CalibrationHistory | None = None,
    ):
        self.scenario = scenario
        self.observed_counts = np.asarray(observed_counts).flatten()
        self.observed_speeds = np.asarray(observed_speeds).flatten()
        self.od_true = od_true
        self.od_prior = od_prior
        self.weights = weights
        self.estimator = estimator
        self.history = history if history is not None else CalibrationHistory()
        #: best accepted iterate (with its counts and speeds)
        self.best = BestSolution()

    def simulate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        demand = to_demand(x)
        counts, speeds = self.scenario.simulate(demand)
        return demand, counts.reshape(-1, 1), speeds.reshape(-1, 1)

    def deviations(self, x: np.ndarray, perturbation: str = "positive") -> np.ndarray:
        demand, counts, speeds = self.simulate(x)
        return stack_deviations(
            self.weights,
            (self.observed_counts, counts.flatten()),
            (self.od_prior, demand),
            (self.observed_speeds, speeds.flatten()),
        )

    def _loss(self, demand: np.ndarray, counts: np.ndarray, speeds: np.ndarray) -> float:
        w, est = self.weights, self.estimator
        return (
            w.counts * gof(self.observed_counts, counts.flatten(), est)
            + w.od * gof(self.od_true, demand, est)
            + w.speed * gof(self.observed_speeds, speeds.flatten(), est)
        )

    def loss(self, x: np.ndarray) -> float:
        return self._loss(*self.simulate(x))

    def evaluate(self, x: np.ndarray) -> float:
        demand, counts, speeds = self.simulate(x)
        total = self._loss(demand, counts, speeds)
        count_gof = np.round(gof(self.observed_counts, counts.flatten(), self.estimator), 4)
        self.history.record(total, count_gof, gof(self.od_true, demand, self.estimator))
        self.best.offer(total, demand, counts=counts, speeds=speeds)
        log.debug(
            "Weighted %s %.6f, count %s %.6f", self.estimator, total, self.estimator, count_gof
        )
        return total
