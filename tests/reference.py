"""Straightforward loop implementations used as reference specifications.

The vectorised implementations in ``actrys`` must reproduce them exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detector_incidence_loop(
    df_merge,
    od_pairs,
    detectors,
    t_start,
    t_end,
    t_warmup,
    interval,
    threshold=False,
    threshold_value=0.1,
    binary_rounding=False,
):
    """Reference for :func:`actrys.sumo.incidence.detector_incidence`."""
    df_merge = df_merge.copy()
    intervals = range(int(t_start - t_warmup), int(t_end), int(interval))
    num_detectors = len(pd.unique(pd.Series(detectors, dtype=object)))
    n_od, n_int = len(od_pairs), len(intervals)
    weight = np.zeros((n_od * n_int, num_detectors * n_int), dtype="float32")
    assign = np.zeros_like(weight)
    df_merge["origin"] = df_merge["od_pair"].apply(lambda x: x.split("___")[0])
    unique_dets = pd.unique(pd.Series(detectors, dtype=object))
    for k, depr in enumerate(intervals):
        df_temp = df_merge[
            (df_merge.vehicle_depart >= depr) & (df_merge.vehicle_depart < depr + interval)
        ]
        for i, od in enumerate(od_pairs):
            origin = od.split("___")[0]
            origin_trips = len(df_temp[df_temp.origin == origin])
            if origin_trips != 0:
                temp_od = df_temp[df_temp.od_pair == od]
                if len(temp_od) != 0:
                    for l_, arrv in enumerate(intervals):
                        temp = temp_od[
                            (temp_od.tripinfo_arrival >= arrv)
                            & (temp_od.tripinfo_arrival < arrv + interval)
                        ]
                        if len(temp) != 0:
                            edges = []
                            for route in temp.route_edges.astype("str"):
                                edges.extend(route.split(" "))
                            for q, detector in enumerate(unique_dets):
                                for e in edges:
                                    if str(detector) == e:
                                        assign[i + k * n_od, q + l_ * num_detectors] += 1
                                        weight[i + k * n_od, q + l_ * num_detectors] = (
                                            len(temp_od) / origin_trips
                                        )
                    assign[i + k * n_od, :] /= len(temp_od)
    if threshold:
        if binary_rounding:
            weight[weight > threshold_value] = 1
            weight[weight <= threshold_value] = 0
        else:
            weight[weight <= threshold_value] = 0
    first = int(t_warmup / interval) * num_detectors
    return weight[:, first:], assign[:, first:]


def assignment_counts_loop(x, assignment):
    """Reference for :meth:`actrys.calibration.objectives.AssignmentObjective.counts`."""
    x = np.where(x < 0, 0, x).astype(int)
    counts = np.zeros((assignment.shape[1], 1))
    for i in range(assignment.shape[1]):
        counts[i] = np.sum(np.multiply(x, assignment[:, i]).astype(int))
    return counts


def add_noise_loop(x, perc_var, mu=1):
    return [int(mu * i) + int(np.random.randn() * perc_var * i) for i in x]


def prepare_weight_matrix_dense(W, weight_counts, weight_od, weight_speed):
    binary = np.where(W > 0, 1, 0)
    blocks = []
    if weight_counts != 0:
        blocks.append(binary)
    if weight_od != 0:
        blocks.append(np.eye(W.shape[0]))
    if weight_speed != 0:
        blocks.append(binary)
    return np.hstack(blocks).astype("int8")


def minimize_wspsa_loop(func, x0, w_matrix, bounds, niter, a, c, alpha, gamma, reps, momentum):
    """Reference for :func:`actrys.optim.spsa.minimize_wspsa` (bounded, sequential evaluation)."""
    big_a = 0.01 * niter
    bounds = np.asarray(bounds)
    x = np.array(x0)
    n = len(x0)
    prev_change = 0
    for k in range(niter):
        ak = a / (k + 1.0 + big_a) ** alpha
        ck = c / (k + 1.0) ** gamma
        save_grad = []
        for _ in range(reps):
            delta = np.random.choice([-1, 1], size=n)
            xplus = np.round(x + np.multiply(x + 1, ck * delta))
            xminus = np.round(x - np.multiply(x + 1, ck * delta))
            pos, neg = func(xplus, True), func(xminus, True)
            save_grad.append(w_matrix @ (pos - neg) / (2 * ck * (x + 1) * delta))
        grad = np.mean(save_grad, axis=0)
        x = np.clip(x - ak * grad - momentum * prev_change, bounds[:, 0], bounds[:, 1])
        prev_change = ak * grad
        func(x, False)
    return x
