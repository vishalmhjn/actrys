"""Parsing SUMO outputs and aligning simulated with observed measurements.

SUMO XML files are parsed directly (no ``xml2csv`` round trip). All
measurement tables use the columns ``det_id`` (edge id), ``hour`` (interval
start in seconds) and ``count`` / ``speed``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from actrys.noise import RNG, add_noise

DETECTOR_TAGS = ("e1Detector", "inductionLoop")


# --------------------------------------------------------------------------- #
# XML readers
# --------------------------------------------------------------------------- #
def _iter_elements(path: Path, tag: str) -> Iterator[ET.Element]:
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == tag:
            yield elem
            elem.clear()


def _numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df:
            df[col] = pd.to_numeric(df[col])
    return df


def lane_to_edge(lane_id: str) -> str:
    """``"-1032#2_0"`` -> ``"-1032#2"`` (strip the lane index)."""
    return lane_id.rsplit("_", 1)[0]


@dataclass(frozen=True)
class DetectorLayout:
    """Induction-loop detectors of a scenario, in the order of the additional file."""

    ids: tuple[str, ...]
    edges: tuple[str, ...]  #: edge of every detector (parallel to ``ids``)

    @classmethod
    def read(cls, path: Path, tags: Iterable[str] = DETECTOR_TAGS) -> DetectorLayout:
        tags = set(tags)
        ids, edges = [], []
        for elem in ET.parse(path).getroot().iter():
            if elem.tag in tags:
                det_id = elem.get("id")
                lane = elem.get("lane")
                if det_id is None or lane is None:
                    raise ValueError(f"{path}: detector without id/lane attribute")
                ids.append(det_id)
                edges.append(lane_to_edge(lane))
        if not ids:
            raise ValueError(f"{path}: no detectors with tags {sorted(tags)}")
        return cls(tuple(ids), tuple(edges))

    @property
    def edge_of(self) -> dict[str, str]:
        return dict(zip(self.ids, self.edges, strict=True))

    @property
    def unique_edges(self) -> list[str]:
        """Detected edges in order of first appearance in the additional file."""
        return list(dict.fromkeys(self.edges))


def read_detector_output(path: Path) -> pd.DataFrame:
    """Induction-loop output (``<interval begin= end= id= entered= .../>``)."""
    rows = [dict(e.attrib) for e in _iter_elements(path, "interval")]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{path}: no detector intervals")
    return _numeric(df, ["begin", "end", "entered", "nVehEntered", "nVehContrib", "flow", "speed"])


def read_edge_data(path: Path) -> pd.DataFrame:
    """Edge-based mean data: one row per (interval, edge) with ``begin`` and ``edge_id``/``speed``."""
    rows = []
    for _, interval in ET.iterparse(path, events=("end",)):
        if interval.tag != "interval":
            continue
        begin = float(interval.get("begin"))
        for edge in interval.iter("edge"):
            row = dict(edge.attrib)
            row["edge_id"] = row.pop("id")
            row["begin"] = begin
            rows.append(row)
        interval.clear()
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{path}: no edge data")
    if "speed" not in df:
        df["speed"] = np.nan
    return _numeric(df, ["speed", "sampledSeconds", "entered", "left", "density"])


def read_vehroutes(path: Path) -> pd.DataFrame:
    """Vehicle routes; for rerouted vehicles the last (= final) route is used."""
    rows = []
    for veh in _iter_elements(path, "vehicle"):
        routes = list(veh.iter("route"))
        rows.append(
            {
                "vehicle_id": veh.get("id"),
                "depart": veh.get("depart"),
                "from_taz": veh.get("fromTaz"),
                "to_taz": veh.get("toTaz"),
                "route_edges": routes[-1].get("edges") if routes else None,
            }
        )
    df = pd.DataFrame(rows, columns=["vehicle_id", "depart", "from_taz", "to_taz", "route_edges"])
    df = df.drop_duplicates(subset=["vehicle_id"], keep="last")
    df["depart"] = pd.to_numeric(df["depart"], errors="coerce")
    return df


def read_tripinfo(path: Path) -> pd.DataFrame:
    rows = [
        {"vehicle_id": e.get("id"), "arrival": e.get("arrival")}
        for e in _iter_elements(path, "tripinfo")
    ]
    df = pd.DataFrame(rows, columns=["vehicle_id", "arrival"])
    df["arrival"] = pd.to_numeric(df["arrival"], errors="coerce")
    return df


def write_detector_csv(df: pd.DataFrame, path: Path) -> None:
    """Write detector output like SUMO's ``xml2csv`` would (``interval_*`` columns, ``;``-separated)."""
    df.add_prefix("interval_").to_csv(path, sep=";", index=False)


def write_detector_variant(source: Path, target: Path, suffix: str) -> None:
    """Copy an additional file, appending ``_<suffix>`` to every output ``file`` attribute.

    ``out.xml`` -> ``out_<suffix>.xml``, ``edge_data_300`` -> ``edge_data_300_<suffix>``.
    Used so the +/- SPSA perturbations can be simulated concurrently.
    """
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(source)
    for elem in tree.getroot().iter():
        name = elem.get("file")
        if name is not None:
            p = Path(name)
            elem.set("file", str(p.with_name(f"{p.stem}_{suffix}{p.suffix}")))
    tree.write(target, encoding="UTF-8", xml_declaration=True)


# --------------------------------------------------------------------------- #
# Measurement processing
# --------------------------------------------------------------------------- #
def _window(df: pd.DataFrame, hour: float, interval_hours: float) -> pd.DataFrame:
    return df[(df["begin"] >= hour * 3600) & (df["begin"] < (hour + interval_hours) * 3600)]


def _aggregate_lanes(values: pd.Series, by: pd.Index | list, sim_type: str) -> pd.Series:
    grouped = values.groupby(by)
    if sim_type == "meso":
        return grouped.mean()
    if sim_type == "micro":
        return grouped.sum()
    raise ValueError(f"Unknown sim_type {sim_type!r}; use 'meso' or 'micro'")


def edge_counts(
    detector_output: pd.DataFrame,
    edge_of: dict[str, str],
    hours: Iterable[float],
    interval_hours: float,
    sim_type: str = "meso",
) -> pd.DataFrame:
    """Simulated counts per (edge, evaluation interval).

    Detector counts are summed over the sub-intervals of an evaluation
    interval, then lane detectors of an edge are averaged (meso: every lane
    detector sees the whole edge segment) or summed (micro). Sorted by edge.
    """
    known = detector_output[detector_output["id"].isin(edge_of.keys())]
    frames = []
    for hour in hours:
        per_det = _window(known, hour, interval_hours).groupby("id")["entered"].sum()
        per_edge = _aggregate_lanes(per_det, per_det.index.map(edge_of), sim_type)
        frames.append(
            pd.DataFrame(
                {"det_id": per_edge.index, "count": per_edge.to_numpy(), "hour": hour * 3600}
            )
        )
    return pd.concat(frames, ignore_index=True)


def synthetic_counts(
    detector_output: pd.DataFrame,
    edge_of: dict[str, str],
    hours: Iterable[float],
    interval_hours: float,
    count_noise_percent: float = 0,
    sim_type: str = "meso",
    *,
    rng: RNG,
) -> pd.DataFrame:
    """ "Observed" counts generated from a simulation run (optionally with noise; truncated to int)."""
    df = edge_counts(detector_output, edge_of, hours, interval_hours, sim_type)
    df["count"] = add_noise(df["count"].to_numpy(), count_noise_percent / 100, mu=1, rng=rng)
    return df


def synthetic_speeds(
    edge_data: pd.DataFrame,
    detected_edges: Iterable[str],
    hours: Iterable[float],
    interval_hours: float,
) -> pd.DataFrame:
    """ "Observed" mean speeds of the detected edges generated from a simulation run."""
    edges = set(detected_edges)
    known = edge_data[edge_data["edge_id"].astype(str).isin(edges)]
    frames = []
    for hour in hours:
        per_edge = _window(known, hour, interval_hours).groupby("edge_id")["speed"].mean()
        frames.append(
            pd.DataFrame(
                {"det_id": per_edge.index, "speed": per_edge.to_numpy(), "hour": hour * 3600}
            )
        )
    return pd.concat(frames, ignore_index=True)


def _match(
    observed: pd.DataFrame, simulated: pd.DataFrame, value: str, hour: float
) -> pd.DataFrame:
    obs = observed[observed["hour"] == hour * 3600].copy()
    obs["det_id"] = obs["det_id"].astype(str)
    sim = simulated.copy()
    sim["det_id"] = sim["det_id"].astype(str)
    # inner merge preserves the order of the observations
    return pd.merge(obs, sim[["det_id", value]], on="det_id", suffixes=("_obs", "_sim"))


def match_counts(
    detector_output: pd.DataFrame,
    observed: pd.DataFrame,
    edge_of: dict[str, str],
    hours: Iterable[float],
    interval_hours: float,
    sim_type: str = "meso",
) -> tuple[np.ndarray, np.ndarray]:
    """Observed and simulated counts for every observation that has a simulated counterpart.

    The order is: evaluation interval, then the row order of ``observed``.
    """
    hours = list(hours)
    sim_all = edge_counts(detector_output, edge_of, hours, interval_hours, sim_type)
    frames = [
        _match(observed, sim_all[sim_all["hour"] == hour * 3600], "count", hour) for hour in hours
    ]
    both = pd.concat(frames)
    return both["count_obs"].to_numpy(), both["count_sim"].to_numpy()


def match_speeds(
    edge_data: pd.DataFrame,
    observed: pd.DataFrame,
    hours: Iterable[float],
    interval_hours: float,
    fill_value: float = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed and simulated edge speeds; missing values are filled with ``fill_value``."""
    frames = []
    for hour in hours:
        per_edge = _window(edge_data, hour, interval_hours).groupby("edge_id")["speed"].mean()
        sim = pd.DataFrame({"det_id": per_edge.index, "speed": per_edge.to_numpy()})
        frames.append(_match(observed, sim, "speed", hour))
    both = pd.concat(frames)
    return (
        both["speed_obs"].fillna(fill_value).to_numpy(),
        both["speed_sim"].fillna(fill_value).to_numpy(),
    )
