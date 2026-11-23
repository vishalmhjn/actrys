"""Mutable calibration state shared by objectives and pipelines."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BestSolution:
    """Best evaluated iterate since the last :meth:`reset`."""

    gof: float = math.inf
    od: np.ndarray | None = None
    counts: np.ndarray | None = None
    speeds: np.ndarray | None = None

    def reset(self) -> None:
        self.gof = math.inf
        self.od = None
        self.counts = None
        self.speeds = None

    def offer(
        self,
        gof: float,
        od: np.ndarray,
        counts: np.ndarray | None = None,
        speeds: np.ndarray | None = None,
    ) -> bool:
        """Keep the candidate if it improves on the best; returns whether it did."""
        if not gof < self.gof:
            return False
        self.gof = gof
        self.od = copy.deepcopy(od)
        if counts is not None:
            self.counts = counts
        if speeds is not None:
            self.speeds = speeds
        return True


@dataclass
class CalibrationHistory:
    """Loss trajectory of the evaluated iterates (keys as in ``results_*.json``)."""

    f_val: list = field(default_factory=list)
    count_val: list = field(default_factory=list)
    speed_val: list = field(default_factory=list)
    od_val: list = field(default_factory=list)
    f_val_supply: list = field(default_factory=list)
    best_supply_count_rmsn: float = 100

    def record(self, f: float, count: float, od: float, speed: float | None = None) -> None:
        self.f_val.append(f)
        self.count_val.append(count)
        self.od_val.append(od)
        if speed is not None:
            self.speed_val.append(speed)
