"""Reading and writing time-dependent OD matrices in SUMO's O-format.

A scenario's demand is a set of files ``<prefix>_<from>_<to>.txt`` (one per
demand interval, e.g. ``OD_7.0_8.0.txt``) of the form::

    $OR;D2
    * From-Time  To-Time
    07.00 8.00
    * Factor
    1.00
    <origin> <destination> <flow>
    ...

The calibration works on a flat *OD vector*: the (factor-scaled) flows of all
files concatenated interval by interval, in file row order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_LINES = 5


def period_path(base: Path, hour: float, interval_hours: float) -> Path:
    """``<dir>/OD.txt`` -> ``<dir>/OD_7.0_8.0.txt`` for the interval starting at ``hour``."""
    base = Path(base)
    return base.with_name(f"{base.stem}_{float(hour)}_{float(hour + interval_hours)}{base.suffix}")


def period_paths(base: Path, hours: Iterable[float], interval_hours: float) -> list[Path]:
    return [period_path(base, h, interval_hours) for h in hours]


@dataclass(frozen=True)
class ODPeriodFile:
    """One parsed O-format file."""

    header: list[str]  #: the first lines, written back verbatim
    factor: float
    origins: list[str]
    destinations: list[str]
    flows: np.ndarray

    @classmethod
    def read(cls, path: Path) -> ODPeriodFile:
        lines = Path(path).read_text().splitlines(keepends=True)
        if len(lines) < HEADER_LINES or not lines[0].startswith("$"):
            raise ValueError(f"{path} is not a SUMO O-format file")
        header = lines[:HEADER_LINES]
        # non-comment lines after the format marker: time range, then factor
        values = [ln.split() for ln in header[1:] if ln.strip() and not ln.lstrip().startswith("*")]
        if len(values) != 2:
            raise ValueError(
                f"{path}: expected time range and factor in the first {HEADER_LINES} lines"
            )
        factor = float(values[1][0])
        origins, destinations, flows = [], [], []
        for ln in lines[HEADER_LINES:]:
            parts = ln.split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}: malformed OD line {ln!r}")
            origins.append(parts[0])
            destinations.append(parts[1])
            flows.append(float(parts[2]))
        return cls(header, factor, origins, destinations, np.asarray(flows, dtype=float))

    def write(self, path: Path, scaled_flows: np.ndarray) -> None:
        """Write ``scaled_flows`` (already multiplied by ``factor``) with this file's header and OD pairs."""
        if len(scaled_flows) != len(self.origins):
            raise ValueError(f"Expected {len(self.origins)} flows, got {len(scaled_flows)}")
        rows = [
            f"{o} {d} {int(float(f / self.factor))}"
            for o, d, f in zip(self.origins, self.destinations, scaled_flows, strict=True)
        ]
        Path(path).write_text("".join(self.header) + "\n".join(rows))


class ODDemand:
    """Time-dependent OD demand of a scenario, mapped to/from the flat OD vector."""

    def __init__(self, periods: Sequence[ODPeriodFile]):
        if not periods:
            raise ValueError("No demand intervals")
        n = len(periods[0].origins)
        if any(len(p.origins) != n for p in periods):
            raise ValueError("All demand intervals must contain the same OD pairs")
        self.periods = list(periods)

    @classmethod
    def read(cls, base: Path, hours: Iterable[float], interval_hours: float) -> ODDemand:
        return cls([ODPeriodFile.read(p) for p in period_paths(base, hours, interval_hours)])

    @property
    def n_pairs(self) -> int:
        return len(self.periods[0].origins)

    @property
    def n_intervals(self) -> int:
        return len(self.periods)

    def __len__(self) -> int:
        return self.n_pairs * self.n_intervals

    def vector(self) -> np.ndarray:
        """Factor-scaled flows of all intervals, concatenated."""
        return np.concatenate([p.flows * p.factor for p in self.periods])

    def write(
        self, base: Path, od_vector: np.ndarray, hours: Iterable[float], interval_hours: float
    ) -> list[Path]:
        """Write ``od_vector`` as O-format files ``<base>_<from>_<to>.txt``; returns the paths."""
        od_vector = np.asarray(od_vector)
        if len(od_vector) != len(self):
            raise ValueError(f"OD vector has {len(od_vector)} elements, expected {len(self)}")
        paths = period_paths(base, hours, interval_hours)
        if len(paths) != self.n_intervals:
            raise ValueError(f"{len(paths)} intervals requested, demand has {self.n_intervals}")
        n = self.n_pairs
        for k, (period, path) in enumerate(zip(self.periods, paths, strict=True)):
            period.write(path, od_vector[k * n : (k + 1) * n])
        return paths


def read_od_pairs(path: Path) -> list[str]:
    """Unique OD pairs (``"<o>___<d>"``) of an ``od_list.txt`` (``o d t`` columns), in file order."""
    pairs: dict[str, None] = {}
    with Path(path).open() as fh:
        header = fh.readline().split()
        try:
            io, idd = header.index("o"), header.index("d")
        except ValueError:
            raise ValueError(f"{path}: expected a header with 'o' and 'd' columns") from None
        for line in fh:
            parts = line.split()
            if parts:
                pairs.setdefault(f"{parts[io]}___{parts[idd]}", None)
    return list(pairs)
