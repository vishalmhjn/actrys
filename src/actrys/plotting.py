"""Diagnostic plots: 45-degree (true vs estimated) plots and loss curves."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from actrys.config import ObjectiveWeights  # noqa: E402

if TYPE_CHECKING:
    from actrys.calibration.state import CalibrationHistory


def _diagonal(ax, upper: float) -> None:
    ax.plot([0, upper], [0, upper])
    ax.set_xlim([-2, upper + 5])
    ax.set_ylim([-2, upper + 5])


def plot_45_degree(
    fig,
    ax,
    true_od,
    estimated_od,
    init_od,
    weights: ObjectiveWeights,
    true_counts,
    simulated_counts,
    initial_counts,
    true_speeds,
    simulated_speeds,
    initial_speeds,
    spsa_a: float,
    spsa_c: float,
    noise: float,
    bias: float,
    only_out_of_simulator: bool = False,
):
    """3x2 grid: OD, counts and speeds, initial (left) and calibrated (right) vs. true."""
    true_od, estimated_od, init_od = (
        np.asarray(v).flatten() for v in (true_od, estimated_od, init_od)
    )
    true_counts, simulated_counts, initial_counts = (
        np.asarray(v).flatten() for v in (true_counts, simulated_counts, initial_counts)
    )
    true_speeds, simulated_speeds, initial_speeds = (
        np.asarray(v).flatten() for v in (true_speeds, simulated_speeds, initial_speeds)
    )

    ax[0, 0].scatter(
        true_od,
        init_od,
        c="r",
        alpha=0.6,
        s=2,
        label=f"Initial (Bias: {int(100 * (bias - 1))}%, Noise: ±{noise}%) vs true",
    )
    ax[0, 1].scatter(true_od, estimated_od, c="c", alpha=0.6, s=2, label="Estimated vs True")
    sns.regplot(
        x=true_od,
        y=estimated_od,
        ax=ax[0, 1],
        scatter=False,
        robust=False,
        ci=68,
        line_kws={"alpha": 0.4},
    )
    od_max = np.maximum(np.max(true_od), np.max(estimated_od))
    for a in (ax[0, 0], ax[0, 1]):
        _diagonal(a, od_max)
        a.legend()

    count_max = np.max(np.maximum.reduce([initial_counts, simulated_counts, true_counts]))
    ax[1, 0].scatter(true_counts, initial_counts, c="b", alpha=0.6, s=2, label="Initial vs True")
    ax[1, 1].scatter(
        true_counts, simulated_counts, c="b", alpha=0.5, s=2, label="Simulated vs True"
    )
    for a in (ax[1, 0], ax[1, 1]):
        _diagonal(a, count_max)

    speed_max = np.maximum(np.max(true_speeds), np.max(simulated_speeds))
    ax[2, 0].scatter(true_speeds, initial_speeds, c="b", alpha=0.5, s=2, label="Initial vs True")
    if not only_out_of_simulator:
        sns.regplot(
            x=true_speeds,
            y=simulated_speeds,
            ax=ax[2, 1],
            robust=False,
            scatter=False,
            line_kws={"alpha": 0.4},
        )
        ax[2, 1].scatter(
            true_speeds, simulated_speeds, c="b", alpha=0.6, s=2, label="Simulated vs True"
        )
    for a in (ax[2, 0], ax[2, 1]):
        _diagonal(a, speed_max)

    for row, (name, weight) in enumerate(
        (("ODs", weights.od), ("Counts", weights.counts), ("Speed", weights.speed))
    ):
        ax[row, 0].set_title(name)
        ax[row, 0].set_xlabel("True")
        ax[row, 0].set_ylabel("Initial")
        ax[row, 1].set_title(f"Weight = {weight}")
        ax[row, 1].set_xlabel("True")
        ax[row, 1].set_ylabel("Estimated" if row == 0 else "Simulated")
    fig.suptitle(f"a {spsa_a:0.5f} , c {spsa_c:0.5f}")
    return fig, ax


def save_45_degree(path: Path, *args, **kwargs) -> None:
    fig, ax = plt.subplots(3, 2, figsize=(7, 9))
    plot_45_degree(fig, ax, *args, **kwargs)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_loss_curve(
    path: Path, history: CalibrationHistory, a: float, c: float, estimator: str
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].plot(history.f_val, label="Weighted loss")
    ax[1].plot(history.count_val, label="Count loss")
    ax[1].plot(history.speed_val, label="Speed loss")
    ax[1].plot(history.od_val, label="OD loss")
    ax[0].set_ylim([0, max(history.f_val, default=1)])
    ax[1].set_ylim([0, max([*history.speed_val, *history.count_val, *history.od_val], default=1)])
    for a_ in ax:
        a_.set_ylabel(estimator, fontsize=16)
        a_.set_title(f"a {a:0.5f} , c {c:0.5f}", fontsize=16)
    fig.suptitle("Result of OD Estimation", fontsize=16)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_loss_curve_synthetic(
    path: Path, history: CalibrationHistory, a: float, c: float, estimator: str = "wmape"
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history.f_val, label="Weighted loss")
    ax.set_ylabel(estimator, fontsize=16)
    ax.set_ylim([0, max(history.f_val, default=1)])
    ax.set_title(f"a {a:0.5f} , c {c:0.5f}", fontsize=16)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_sumo_plots(
    calibration,
    results,
    x_true,
    init_seq,
    true_counts,
    true_speeds,
    initial,
    final,
    od_estimate,
    bagging: int,
) -> None:
    """Result plots of a SUMO calibration run ."""
    cfg = calibration.cfg
    d, w, data = cfg.demand, cfg.weights, cfg.data
    out, ts, est = calibration.paths.folder, calibration.timestamp, cfg.estimator
    tag = (
        f"n_{data.od_noise_percent}_b_{int(100 * data.od_bias)}_{d.outer_iterations}_{w.od}_{est}_"
        f"{w.counts}_{w.speed}_{d.algorithm}_{d.wspsa_threshold}.png"
    )
    a, c = d.gains.a, d.gains.c
    best = calibration.best
    common = (w, true_counts)
    if best.counts is not None and best.speeds is not None:
        save_45_degree(
            out / f"img_results_{bagging}_{tag}",
            x_true,
            od_estimate,
            init_seq,
            *common,
            best.counts,
            initial.simulated_counts,
            true_speeds,
            best.speeds,
            initial.simulated_speeds,
            a,
            c,
            data.od_noise_percent,
            data.od_bias,
        )
    save_loss_curve(
        out
        / (
            f"img_loss_{ts}_n_{data.od_noise_percent}_b_{int(100 * data.od_bias)}__{d.outer_iterations}_"
            f"{est}_{w.od}_{w.counts}_{w.speed}_{d.algorithm}_{d.wspsa_threshold}.png"
        ),
        calibration.history,
        a,
        c,
        est,
    )
    only_out = (not d.sim_in_loop) and d.sim_out_loop
    save_45_degree(
        out / f"img_results_{ts}_{tag}",
        x_true,
        od_estimate,
        init_seq,
        *common,
        final.simulated_counts,
        initial.simulated_counts,
        true_speeds,
        final.simulated_speeds,
        initial.simulated_speeds,
        a,
        c,
        data.od_noise_percent,
        data.od_bias,
        only_out,
    )
