"""Link incidence (W-SPSA weight) and assignment matrices from simulated routes.

Rows are (departure interval, OD pair) and columns (arrival interval,
detector), both interval-major::

    row = od_index + k * n_od          col = det_index + l * n_detectors

* ``assignment[row, col]``: number of detector passages by trips of the OD
  pair departing in ``k`` and arriving in ``l``, divided by the number of
  trips of that OD pair departing in ``k`` (an assignment-matrix estimate).
* ``weight[row, col]``: share of the origin's departures in ``k`` that belong
  to the OD pair, wherever that OD pair crosses the detector.

Columns of the warm-up intervals are dropped from the returned matrices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncidenceMatrices:
    weight: np.ndarray
    assignment: np.ndarray


def _interval_index(times: np.ndarray, start: float, interval: float, n: int) -> np.ndarray:
    """Index ``i`` of the interval ``[start + i*interval, start + (i+1)*interval)``; -1 outside."""
    edges = start + interval * np.arange(n + 1)
    idx = np.searchsorted(edges, times, side="right") - 1
    idx[(idx >= n) | np.isnan(times)] = -1
    return idx


def detector_incidence(
    trips: pd.DataFrame,
    od_pairs: Sequence[str],
    detectors: Sequence[str],
    *,
    t_start: float,
    t_end: float,
    t_warmup: float,
    interval: float,
    threshold: float | None = None,
    binary: bool = False,
) -> IncidenceMatrices:
    """Build the weight and assignment matrices.

    Parameters
    ----------
    trips:
        One row per vehicle with ``from_taz``, ``to_taz``, ``depart``,
        ``arrival`` (seconds) and ``route_edges`` (space-separated edge ids).
    od_pairs:
        Decision-variable order, as ``"<origin>___<destination>"``.
    detectors:
        Column order: the (edge) ids of the detectors.
    t_start, t_end, t_warmup, interval:
        Analysis period in seconds; intervals start at ``t_start - t_warmup``.
    threshold, binary:
        If ``threshold`` is given, weights ``<= threshold`` are set to 0 and,
        with ``binary``, weights ``> threshold`` to 1.
    """
    od_pairs = list(od_pairs)
    detectors = list(dict.fromkeys(str(d) for d in detectors))
    t0 = t_start - t_warmup
    n_int = len(range(int(t0), int(t_end), int(interval)))
    n_od, n_det = len(od_pairs), len(detectors)

    weight = np.zeros((n_od * n_int, n_det * n_int), dtype="float32")
    assignment = np.zeros_like(weight)

    df = pd.DataFrame(
        {
            "od_pair": trips["from_taz"].astype(str) + "___" + trips["to_taz"].astype(str),
            "k": _interval_index(trips["depart"].to_numpy(float), t0, interval, n_int),
            "l": _interval_index(trips["arrival"].to_numpy(float), t0, interval, n_int),
            "route_edges": trips["route_edges"].astype(str),
        }
    )
    df["origin"] = df["od_pair"].str.split("___").str[0]
    df = df[df["k"] >= 0]

    # trips per (departure interval, origin) and per (departure interval, OD pair)
    origin_trips = df.groupby(["k", "origin"]).size()
    od_trips = df.groupby(["k", "od_pair"]).size()

    od_index = pd.Series(np.arange(n_od), index=pd.Index(od_pairs))
    det_index = pd.Series(np.arange(n_det), index=pd.Index(detectors))

    passages = df[df["od_pair"].isin(od_index.index) & (df["l"] >= 0)]
    passages = passages.assign(edge=passages["route_edges"].str.split(" ")).explode("edge")
    passages = passages[passages["edge"].isin(det_index.index)]
    if not passages.empty:
        hits = (
            passages.groupby(["k", "od_pair", "origin", "l", "edge"], sort=False)
            .size()
            .reset_index(name="n")
            .merge(od_trips.rename("n_od").reset_index(), on=["k", "od_pair"])
            .merge(origin_trips.rename("n_origin").reset_index(), on=["k", "origin"])
        )
        n_od_trips = hits["n_od"].to_numpy()
        rows = hits["od_pair"].map(od_index).to_numpy() + hits["k"].to_numpy() * n_od
        cols = hits["edge"].map(det_index).to_numpy() + hits["l"].to_numpy() * n_det
        assignment[rows, cols] = hits["n"].to_numpy(np.float32) / n_od_trips.astype(np.float32)
        weight[rows, cols] = (n_od_trips / hits["n_origin"].to_numpy()).astype(np.float32)

    if threshold is not None:
        if binary:
            weight = np.where(weight > threshold, 1, 0).astype("float32")
        else:
            weight[weight <= threshold] = 0

    first_col = int(t_warmup / interval) * n_det
    return IncidenceMatrices(weight[:, first_col:], assignment[:, first_col:])


def trips_table(vehroutes: pd.DataFrame, tripinfo: pd.DataFrame) -> pd.DataFrame:
    """Join routes and trip info (vehicles that completed their trip)."""
    return pd.merge(vehroutes, tripinfo, on="vehicle_id")
