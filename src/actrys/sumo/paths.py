"""File layout of a calibration run folder (``<scenario>/<run_name>/``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from actrys.config import ScenarioConfig, TimeWindow
from actrys.sumo.simulator import EvaluationOutputs


@dataclass(frozen=True)
class VariantPaths:
    """Files of one simulation "slot": the main evaluation run or a +/- perturbation.

    The perturbation slots get their own detector file whose outputs carry a
    ``_positive`` / ``_negative`` suffix, so both can run concurrently.
    """

    trips: Path
    routes: Path
    additional: Path
    detector_output: Path
    edge_data: Path

    @classmethod
    def in_folder(cls, folder: Path, interval_seconds: int, suffix: str = "") -> VariantPaths:
        s = f"_{suffix}" if suffix else ""
        return cls(
            trips=folder / f"trips.trips{s}.xml",
            routes=folder / f"routes.rou{s}.xml",
            additional=folder / f"additional.add{s}.xml",
            detector_output=folder / f"out{s}.xml",
            edge_data=folder / f"edge_data_{interval_seconds}{s}",
        )


@dataclass(frozen=True)
class RunPaths:
    folder: Path
    main: VariantPaths
    positive: VariantPaths
    negative: VariantPaths
    evaluation: EvaluationOutputs
    observed_counts: Path
    observed_speeds: Path
    od_prefix: str = "OD"

    @classmethod
    def create(cls, scenario: ScenarioConfig, window: TimeWindow) -> RunPaths:
        folder = scenario.directory / scenario.run_name
        folder.mkdir(parents=True, exist_ok=True)
        iv = window.interval_seconds
        main = VariantPaths.in_folder(folder, iv)
        return cls(
            folder=folder,
            main=main,
            positive=VariantPaths.in_folder(folder, iv, "positive"),
            negative=VariantPaths.in_folder(folder, iv, "negative"),
            evaluation=EvaluationOutputs(
                summary=folder / "summary",
                tripinfo=folder / "trips_output",
                vehroutes=main.routes,
            ),
            observed_counts=folder / "real_counts_complete.csv",
            observed_speeds=folder / f"real_edge_data_{iv}.csv",
            od_prefix=scenario.od_prefix,
        )

    def variant(self, perturbation: str | None) -> VariantPaths:
        return {None: self.main, "positive": self.positive, "negative": self.negative}[perturbation]

    def od_base(self, tag: str, perturbation: str | None = None) -> Path:
        """Base name of written demand files, e.g. ``OD_n.txt`` / ``OD_n_positive.txt``."""
        s = f"_{perturbation}" if perturbation else ""
        return self.folder / f"{self.od_prefix}_{tag}{s}.txt"

    def file(self, name: str) -> Path:
        return self.folder / name
