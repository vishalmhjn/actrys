"""Demand calibration on the analytical (static assignment) scenario.

Useful for prototyping and for studying the optimizers: no simulator or
input files are required.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from actrys.calibration.bias import bias_factor, search_bounds
from actrys.calibration.objectives import SyntheticObjective, to_demand
from actrys.calibration.state import CalibrationHistory
from actrys.calibration.tuning import bayes_maximize
from actrys.config import SPSAGains, SyntheticCalibrationConfig
from actrys.metrics import gof
from actrys.noise import RNG, add_noise, clip_negative, make_rng
from actrys.optim.spsa import (
    minimize_pcspsa,
    minimize_spsa,
    minimize_wspsa,
    pca_components_for_variance,
)
from actrys.optim.weights import wspsa_weight_matrix
from actrys.synthetic.scenario import SyntheticScenario, generate_scenario

log = logging.getLogger(__name__)


@dataclass
class SyntheticProblem:
    """True scenario, observations and the perturbed prior."""

    scenario: SyntheticScenario
    od_true: np.ndarray
    measured_counts: np.ndarray  #: (m, 1)
    true_speeds: np.ndarray  #: (m, 1)
    prior: np.ndarray
    prior_counts: np.ndarray  #: (m, 1) counts of the prior
    prior_speeds: np.ndarray  #: (m, 1)


class SyntheticCalibration:
    def __init__(
        self,
        cfg: SyntheticCalibrationConfig,
        rng: RNG | None = None,
        scenario: SyntheticScenario | None = None,
    ):
        self.cfg = cfg
        self.rng = rng if rng is not None else make_rng(cfg.seed)
        self._scenario = scenario
        self.out = Path(cfg.output_dir) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.history = CalibrationHistory()
        self.timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    # ------------------------------------------------------------- problem
    def build_problem(self) -> SyntheticProblem:
        cfg = self.cfg
        scenario = (
            self._scenario
            if self._scenario is not None
            else generate_scenario(cfg.scenario, rng=self.rng)
        )
        true_counts = scenario.counts.astype(np.int64).reshape(-1, 1)
        true_speeds = scenario.speeds.astype(np.int64).reshape(-1, 1)
        measured = add_noise(true_counts, cfg.count_noise_percent / 100, 1, self.rng).reshape(-1, 1)
        od_true = scenario.od_vector
        prior = clip_negative(add_noise(od_true, cfg.od_noise_percent / 100, cfg.od_bias, self.rng))
        counts, speeds = scenario.simulate(prior)
        return SyntheticProblem(
            scenario,
            od_true,
            measured,
            true_speeds,
            prior,
            counts.reshape(-1, 1),
            speeds.reshape(-1, 1),
        )

    def _bias_factor(self, p: SyntheticProblem) -> float | np.ndarray | None:
        bc = self.cfg.bias_correction
        if self.cfg.weights.counts == 0 or not bc.enabled:
            return None
        return bias_factor(
            bc.method,
            p.prior_counts,
            p.measured_counts,
            incidence=p.scenario.assignment,
            fill_nan=True,
        )

    def bias_correction(self, p: SyntheticProblem) -> np.ndarray:
        factor = self._bias_factor(p)
        return p.prior.copy() if factor is None else p.prior / factor

    def search_bounds(self, p: SyntheticProblem) -> np.ndarray:
        bc, factor = self.cfg.bias_correction, self._bias_factor(p)
        if factor is None:
            return search_bounds(p.prior, bc.lower_bound, 3)
        return search_bounds(p.prior, bc.lower_bound, (2 / factor) - bc.lower_bound)

    def objective(self, p: SyntheticProblem, prior: np.ndarray) -> SyntheticObjective:
        return SyntheticObjective(
            p.scenario,
            p.measured_counts,
            p.true_speeds,
            p.od_true,
            prior,
            self.cfg.weights,
            self.cfg.estimator,
            self.history,
        )

    def calibrate(
        self,
        objective: SyntheticObjective,
        x0: np.ndarray,
        gains: SPSAGains,
        bounds: np.ndarray,
        niter: int,
    ):
        cfg = self.cfg
        if cfg.algorithm == "wspsa":
            weights = wspsa_weight_matrix(objective.scenario.assignment, cfg.weights)
            return minimize_wspsa(
                objective,
                x0,
                weights,
                gains=gains,
                niter=niter,
                bounds=bounds,
                reps=cfg.spsa_reps,
                momentum=cfg.momentum,
                rng=self.rng,
            )
        if cfg.algorithm == "spsa":
            return minimize_spsa(
                objective,
                x0,
                gains=gains,
                niter=niter,
                bounds=bounds,
                reps=cfg.spsa_reps,
                rng=self.rng,
            )
        if cfg.algorithm == "pcspsa":
            n_feat = cfg.scenario.num_od_pairs
            n_comp = pca_components_for_variance(x0, n_feat)
            return minimize_pcspsa(
                objective,
                x0,
                n_feat,
                n_comp,
                gains=gains,
                niter=niter,
                reps=cfg.spsa_reps,
                rng=self.rng,
                random_state=int(self.rng.integers(2**31))
                if hasattr(self.rng, "integers")
                else None,
            )
        raise ValueError(f"Unknown algorithm {cfg.algorithm!r}")

    def tune_gains(self, p: SyntheticProblem, x0: np.ndarray, bounds: np.ndarray) -> SPSAGains:
        cfg, at, w, est = self.cfg, self.cfg.autotune, self.cfg.weights, self.cfg.estimator
        niter = at.iterations if at.iterations is not None else cfg.iterations

        def score(log_spsa_a: float, log_spsa_c: float) -> float:
            gains = SPSAGains(
                np.power(10, log_spsa_a), np.power(10, log_spsa_c), cfg.gains.alpha, cfg.gains.gamma
            )
            res = self.calibrate(self.objective(p, x0.copy()), x0, gains, bounds, niter)
            demand = to_demand(res["x"])
            counts, _ = p.scenario.simulate(demand)
            return -(
                w.counts * gof(p.measured_counts.flatten(), counts.reshape(-1, 1).flatten(), est)
                + w.od * gof(p.od_true, demand, est)
            )

        best = bayes_maximize(
            score,
            {"log_spsa_a": at.log_a_bounds, "log_spsa_c": at.log_c_bounds},
            at.n_init,
            at.n_iter,
            at.random_state,
            self.out / "logs_spsa_tune.json",
        )
        return SPSAGains(
            np.power(10, best["params"]["log_spsa_a"]),
            np.power(10, best["params"]["log_spsa_c"]),
            cfg.gains.alpha,
            cfg.gains.gamma,
        )

    # ----------------------------------------------------------------- run
    def run(self) -> dict:
        cfg, est = self.cfg, self.cfg.estimator
        p = self.build_problem()
        corrected = self.bias_correction(p)
        if cfg.bias_correction.only:
            return self._only_bias_correction(p, corrected)
        bounds = self.search_bounds(p)

        log.info("Bias-corrected OD %s: %s", est, gof(p.od_true, corrected, est))
        gains = cfg.gains
        if cfg.weights.counts != 0 and cfg.autotune.enabled:
            gains = self.tune_gains(p, corrected.copy(), bounds)
            log.info("Tuned SPSA gains: a=%g, c=%g", gains.a, gains.c)

        od_bag = []
        for bagging in range(cfg.bagging_runs):
            log.info("Bagging run %d", bagging)
            objective = self.objective(p, corrected.copy())
            self.calibrate(objective, corrected.copy(), gains, bounds, cfg.iterations)
            best = objective.best
            od_bag.append(best.od)
            if cfg.plot:
                self._plot_run(
                    p,
                    best.od,
                    best.counts,
                    best.speeds,
                    gains,
                    self._name(f"results_{bagging}a_{gains.a}", f"_{cfg.run_name}.png"),
                )
            pd.DataFrame({"real": p.od_true, "simulated": best.od.flatten()}).to_csv(
                self.out / f"{bagging}_od.csv", index=None
            )

        od_mean = np.mean(np.array(od_bag), axis=0).astype(int)
        counts, speeds = p.scenario.simulate(od_mean)
        counts, speeds = counts.reshape(-1, 1), speeds.reshape(-1, 1)
        log.info("Final bagged OD %s: %s", est, gof(p.od_true, od_mean, est))
        return self._save(p, od_mean, counts, speeds, gains)

    # -------------------------------------------------------------- output
    def _name(self, head: str, tail: str) -> str:
        cfg = self.cfg
        return (
            f"{head}n_{int(cfg.od_noise_percent)}_b_{int(100 * cfg.od_bias)}_{cfg.scenario.num_zones}_"
            f"{cfg.scenario.num_detectors}_{cfg.iterations}_{cfg.weights.od}_{cfg.weights.counts}_"
            f"{cfg.weights.speed}_{cfg.algorithm}{tail}"
        )

    def _plot_run(self, p, od, counts, speeds, gains, filename):
        from actrys import plotting

        plotting.save_45_degree(
            self.out / filename,
            p.od_true,
            od,
            p.prior,
            self.cfg.weights,
            p.measured_counts,
            counts,
            p.prior_counts,
            p.true_speeds,
            speeds,
            p.prior_speeds,
            gains.a,
            gains.c,
            self.cfg.od_noise_percent,
            self.cfg.od_bias,
        )

    def _save(self, p, od_mean, counts, speeds, gains) -> dict:
        cfg, beta = self.cfg, self.cfg.momentum
        results = {
            "f_val": self.history.f_val,
            "count_val": self.history.count_val,
            "od_val": self.history.od_val,
            "spsa_a": gains.a,
            "spsa_c": gains.c,
            "n_iterations": cfg.iterations,
            "noise_param": cfg.od_noise_percent,
            "bias_param": cfg.od_bias,
            "spsa_reps": cfg.spsa_reps,
            "weight_counts": cfg.weights.counts,
            "weight_od": cfg.weights.od,
            "weight_speed": cfg.weights.speed,
            "beta_momentum": beta,
            "final_od_gof": gof(p.od_true, od_mean, cfg.estimator),
        }
        pd.DataFrame(
            {
                "real": p.measured_counts.flatten(),
                "initial": p.prior_counts.flatten(),
                "simulated": counts.flatten(),
            }
        ).to_csv(self.out / f"{beta}_counts.csv")
        pd.DataFrame(
            {
                "real": p.true_speeds.flatten(),
                "initial": p.prior_speeds.flatten(),
                "simulated": speeds.flatten(),
            }
        ).to_csv(self.out / f"{beta}_speeds.csv")
        pd.DataFrame(
            {
                "real": p.od_true.flatten(),
                "initial": p.prior.flatten(),
                "simulated": od_mean.flatten(),
            }
        ).to_csv(self.out / f"mean_{beta}_{cfg.bagging_runs}_od.csv", index=None)
        _write_json(results, self.out / f"results_{beta}{self.timestamp}.json")
        if cfg.plot:
            from actrys import plotting

            plotting.save_loss_curve_synthetic(
                self.out / self._name(f"loss_a_{gains.a}", f"{cfg.run_name}.png"),
                self.history,
                gains.a,
                gains.c,
            )
            self._plot_run(
                p,
                od_mean,
                counts,
                speeds,
                gains,
                self._name(f"results_{beta}_a_{gains.a}", f"{cfg.run_name}.png"),
            )
        return results

    def _only_bias_correction(self, p: SyntheticProblem, corrected: np.ndarray) -> dict:
        cfg, est = self.cfg, self.cfg.estimator
        counts, speeds = p.scenario.simulate(corrected)
        counts, speeds = counts.reshape(-1, 1), speeds.reshape(-1, 1)
        for c, o in ((p.prior_counts, p.prior), (counts, corrected)):
            self.history.count_val.append(gof(p.measured_counts, c, est))
            self.history.od_val.append(gof(p.od_true, o, est))
        pd.DataFrame(
            {
                "real": p.measured_counts.flatten(),
                "initial": p.prior_counts.flatten(),
                "simulated": counts.flatten(),
            }
        ).to_csv(self.out / "counts.csv")
        pd.DataFrame(
            {
                "real": p.true_speeds.flatten(),
                "initial": p.prior_speeds.flatten(),
                "simulated": speeds.flatten(),
            }
        ).to_csv(self.out / "speeds.csv")
        pd.DataFrame(
            {
                "real": p.od_true.flatten(),
                "initial": p.prior.flatten(),
                "simulated": corrected.flatten(),
            }
        ).to_csv(self.out / "mean_od.csv", index=None)
        results = {
            "f_val": self.history.f_val,
            "count_val": self.history.count_val,
            "od_val": self.history.od_val,
            "noise_param": cfg.od_noise_percent,
            "bias_param": cfg.od_bias,
        }
        _write_json(results, self.out / f"results_{self.timestamp}.json")
        return results


def _write_json(data: dict, path: Path) -> None:
    with Path(path).open("w") as fh:
        json.dump(data, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def run_synthetic_calibration(cfg: SyntheticCalibrationConfig, rng: RNG | None = None) -> dict:
    return SyntheticCalibration(cfg, rng).run()
