"""Sequential demand/supply calibration of a SUMO scenario.

Stages (repeated ``n_sequential`` times):

1. **Demand** (OD) calibration
   a. simulate the prior, build the W-SPSA incidence/assignment matrices,
   b. one-shot bias correction of the prior and search bounds,
   c. (optional) Bayesian tuning of the SPSA gains on the assignment-matrix
      approximation,
   d. ``bagging_runs`` W-SPSA runs (simulator in and/or out of the loop),
      averaged into the demand estimate.
2. **Supply** calibration of route-choice / meso parameters with Bayesian
   optimization.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from actrys.calibration.bias import bias_factor, search_bounds
from actrys.calibration.objectives import AssignmentObjective, stack_deviations, to_demand
from actrys.calibration.state import BestSolution, CalibrationHistory
from actrys.calibration.tuning import bayes_maximize
from actrys.config import SPSAGains, SumoCalibrationConfig, SupplyParams
from actrys.metrics import gof
from actrys.noise import RNG, add_noise, clip_negative, make_rng
from actrys.optim.spsa import NEGATIVE, POSITIVE, minimize_spsa, minimize_wspsa
from actrys.optim.weights import wspsa_weight_matrix
from actrys.sumo.incidence import IncidenceMatrices, detector_incidence, trips_table
from actrys.sumo.od import ODDemand, period_paths, read_od_pairs
from actrys.sumo.outputs import (
    DetectorLayout,
    match_counts,
    match_speeds,
    read_detector_output,
    read_edge_data,
    read_tripinfo,
    read_vehroutes,
    synthetic_counts,
    synthetic_speeds,
    write_detector_csv,
    write_detector_variant,
)
from actrys.sumo.paths import RunPaths
from actrys.sumo.simulator import SumoSimulator

log = logging.getLogger(__name__)


def supply_from_bo(base: SupplyParams, p: dict[str, float]) -> SupplyParams:
    """Supply parameters for a Bayesian-optimization candidate ``p``.

    Used for both the trials and the final result, so the calibrated
    parameters are exactly those of the best evaluated trial.
    """
    return dataclasses.replace(
        base,
        rerouting_prob=float(p["rerouting_probability"]),
        tls_tt_penalty=float(p["tls_tt_penalty"]),
        meso_minor_penalty=int(p["meso_minor_penalty"]),
        rerouting_period=1 + int(p["rerouting_period"]),
        rerouting_adaptation=1 + int(p["rerouting_adaptation"]),
        rerouting_adaptation_steps=int(p["rerouting_adaptation_steps"]),
        meso_tls_flow_penalty=float(p["meso_tls_flow_penalty"]),
        priority_factor=float(p["priority_factor"]),
    )


@dataclass
class Measurements:
    """Observed vs. simulated measurements of one simulation run."""

    observed_counts: np.ndarray
    simulated_counts: np.ndarray
    observed_speeds: np.ndarray
    simulated_speeds: np.ndarray
    detector_output: pd.DataFrame


class SumoEvaluator:
    """Simulates an OD vector and returns the matched measurements."""

    def __init__(
        self,
        cfg: SumoCalibrationConfig,
        paths: RunPaths,
        simulator: SumoSimulator,
        template: ODDemand,
        layout: DetectorLayout,
    ):
        self.cfg = cfg
        self.paths = paths
        self.simulator = simulator
        self.template = template
        self.layout = layout
        self.edge_of = layout.edge_of
        self.observed_counts: pd.DataFrame | None = None
        self.observed_speeds: pd.DataFrame | None = None

    def load_observations(self) -> None:
        self.observed_counts = pd.read_csv(self.paths.observed_counts)
        self.observed_speeds = pd.read_csv(self.paths.observed_speeds)

    def match(self, perturbation: str | None = None) -> Measurements:
        variant = self.paths.variant(perturbation)
        window = self.cfg.time
        det_out = read_detector_output(variant.detector_output)
        obs_c, sim_c = match_counts(
            det_out,
            self.observed_counts,
            self.edge_of,
            window.evaluation_hours,
            window.interval_hours,
            self.cfg.simulation.sim_type,
        )
        obs_s, sim_s = match_speeds(
            read_edge_data(variant.edge_data),
            self.observed_speeds,
            window.evaluation_hours,
            window.interval_hours,
        )
        return Measurements(obs_c, sim_c, obs_s, sim_s, det_out)

    def run(
        self,
        od_base: Path,
        od_vector: np.ndarray,
        supply: SupplyParams,
        perturbation: str | None = None,
    ) -> Measurements:
        """Write the demand, route it, simulate and match the outputs.

        Runs without ``perturbation`` are evaluation runs that also write the
        trip info / vehicle routes needed for the incidence matrix.
        """
        window = self.cfg.time
        variant = self.paths.variant(perturbation)
        od_files = self.template.write(
            od_base, od_vector, window.demand_hours, window.interval_hours
        )
        self.simulator.generate_trips(od_files, variant.trips, variant.routes)
        outputs = self.paths.evaluation if perturbation is None else None
        self.simulator.simulate(variant.trips, variant.additional, supply, outputs)
        return self.match(perturbation)


class SumoObjective:
    """Objective with SUMO in the loop (see :class:`actrys.optim.spsa.Objective`)."""

    def __init__(
        self,
        pipeline: SumoCalibration,
        od_true: np.ndarray,
        od_prior: np.ndarray,
        supply: SupplyParams,
        demand_tag: str = "n",
    ):
        self.p = pipeline
        self.od_true = od_true
        self.od_prior = od_prior
        self.supply = supply
        self.demand_tag = demand_tag

    def _run(self, x: np.ndarray, perturbation: str | None) -> tuple[np.ndarray, Measurements]:
        demand = to_demand(x)
        od_base = self.p.paths.od_base(self.demand_tag, perturbation)
        return demand, self.p.evaluator.run(od_base, demand, self.supply, perturbation)

    def deviations(self, x: np.ndarray, perturbation: str = POSITIVE) -> np.ndarray:
        demand, m = self._run(x, perturbation)
        return stack_deviations(
            self.p.cfg.weights,
            (m.observed_counts.flatten(), m.simulated_counts.flatten()),
            (self.od_prior, demand),
            (m.observed_speeds.flatten(), m.simulated_speeds.flatten()),
        )

    def _scores(self, demand: np.ndarray, m: Measurements) -> tuple[float, float, float, float]:
        est, w = self.p.cfg.estimator, self.p.cfg.weights
        count_gof = round(gof(m.observed_counts.flatten(), m.simulated_counts.flatten(), est), 4)
        speed_gof = round(gof(m.observed_speeds.flatten(), m.simulated_speeds.flatten(), est), 4)
        od_gof = gof(self.od_true, demand, est)
        return (
            w.counts * count_gof + w.speed * speed_gof + w.od * od_gof,
            count_gof,
            speed_gof,
            od_gof,
        )

    def loss(self, x: np.ndarray) -> float:
        demand, m = self._run(x, POSITIVE)
        return self._scores(demand, m)[0]

    def evaluate(self, x: np.ndarray) -> float:
        demand, m = self._run(x, None)
        total, count_gof, speed_gof, od_gof = self._scores(demand, m)
        p, est = self.p, self.p.cfg.estimator
        history = p.history

        # snapshots of the solutions once the OD error has stabilised
        if len(history.od_val) > 1 and np.std(history.od_val) < 0.10:
            k = p.interim_counter
            pd.DataFrame({"real": self.od_true.flatten(), "simulated": demand.flatten()}).to_csv(
                p.paths.file(f"interim_{k}_od.csv"), index=None
            )
            pd.DataFrame(
                {"real": m.observed_counts.flatten(), "simulated": m.simulated_counts.flatten()}
            ).to_csv(p.paths.file(f"interim_{k}_counts.csv"), index=None)
            p.interim_counter += 1

        history.record(total, count_gof, od_gof, speed_gof)
        log.info(
            "Weighted %s = %s | counts %s | speeds %s | OD %s",
            est,
            total,
            count_gof,
            speed_gof,
            od_gof,
        )

        if p.best.offer(total, demand, counts=m.simulated_counts, speeds=m.simulated_speeds):
            p.template.write(
                p.paths.od_base("best"), demand, p.cfg.time.demand_hours, p.cfg.time.interval_hours
            )
            p.best_supply = self.supply
            p.save_best_outputs(m)
        return total


class SumoCalibration:
    def __init__(self, cfg: SumoCalibrationConfig, rng: RNG | None = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else make_rng(cfg.seed)
        sc = cfg.scenario
        self.paths = RunPaths.create(sc, cfg.time)
        self.simulator = SumoSimulator(
            sc.path(sc.network), sc.path(sc.zones), cfg.time, cfg.simulation.routing_threads
        )
        self.demand_base = sc.path(sc.demand_dir) / f"{sc.od_prefix}.txt"
        self.template = ODDemand.read(
            self.demand_base, cfg.time.demand_hours, cfg.time.interval_hours
        )
        self.layout = DetectorLayout.read(sc.path(sc.detectors))
        self.od_pairs = read_od_pairs(sc.path(sc.demand_dir) / sc.od_list)
        self.evaluator = SumoEvaluator(cfg, self.paths, self.simulator, self.template, self.layout)
        self.history = CalibrationHistory()
        self.best = BestSolution()
        self.best_supply: SupplyParams | None = None
        self.interim_counter = 0
        self.timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.tuned_gains: list[dict] = []
        self.supply_bo_params: dict | None = None

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        """Simulate the base scenario and create/copy the observations."""
        cfg, paths, window = self.cfg, self.paths, self.cfg.time
        source = cfg.scenario.path(cfg.scenario.detectors)
        shutil.copyfile(source, paths.main.additional)
        for suffix, variant in ((POSITIVE, paths.positive), (NEGATIVE, paths.negative)):
            write_detector_variant(source, variant.additional, suffix)

        if cfg.simulate_base_scenario:
            od_files = period_paths(self.demand_base, window.demand_hours, window.interval_hours)
            self.simulator.generate_trips(od_files, paths.main.trips, paths.main.routes)
            self.simulator.simulate(
                paths.main.trips, paths.main.additional, cfg.initial_supply, paths.evaluation
            )

        if cfg.data.synthetic_counts:
            synthetic_counts(
                read_detector_output(paths.main.detector_output),
                self.layout.edge_of,
                window.evaluation_hours,
                window.interval_hours,
                cfg.data.count_noise_percent,
                cfg.simulation.sim_type,
                rng=self.rng,
            ).to_csv(paths.observed_counts, index=None)
            synthetic_speeds(
                read_edge_data(paths.main.edge_data),
                self.layout.unique_edges,
                window.evaluation_hours,
                window.interval_hours,
            ).to_csv(paths.observed_speeds, index=None)
        else:
            if cfg.data.real_counts is None or cfg.data.real_speeds is None:
                raise ValueError(
                    "data.real_counts and data.real_speeds are required without synthetic counts"
                )
            pd.read_csv(cfg.data.real_counts).to_csv(paths.observed_counts, index=None)
            pd.read_csv(cfg.data.real_speeds).to_csv(paths.observed_speeds, index=None)
        self.evaluator.load_observations()

    # ------------------------------------------------------------- utilities
    def incidence_detectors(self) -> list[str]:
        """Column order of the incidence matrices: the detector order of the count vector."""
        observed = self.evaluator.observed_counts
        if not self.cfg.data.synthetic_counts:
            if self.cfg.data.matchable_detectors is None:
                raise ValueError("data.matchable_detectors is required without synthetic counts")
            matchable = pd.read_csv(self.cfg.data.matchable_detectors)
            observed = observed[observed["det_id"].isin(matchable["det_id"])]
        return [str(d) for d in observed["det_id"].unique()]

    def incidence(self) -> IncidenceMatrices:
        """W-SPSA weights and assignment matrix from the latest evaluation run."""
        ev, w = self.paths.evaluation, self.cfg.time
        trips = trips_table(read_vehroutes(ev.vehroutes), read_tripinfo(ev.tripinfo))
        return detector_incidence(
            trips,
            self.od_pairs,
            self.incidence_detectors(),
            t_start=w.start_hour * 3600,
            t_end=w.end_hour * 3600,
            t_warmup=w.warm_up_hours * 3600,
            interval=w.interval_seconds,
            threshold=self.cfg.demand.wspsa_threshold,
            binary=True,
        )

    def _check_shapes(self, weight: np.ndarray, n_od: int, n_counts: int, n_speeds: int) -> None:
        if self.cfg.demand.algorithm != "wspsa":
            return
        if weight.shape[0] != n_od:
            raise ValueError(f"Weight matrix has {weight.shape[0]} rows, OD vector {n_od} elements")
        if weight.shape[1] != n_counts:
            raise ValueError(
                f"Weight matrix has {weight.shape[1]} columns, but there are {n_counts} counts"
            )
        if weight.shape[1] != n_speeds:
            raise ValueError(
                f"Weight matrix has {weight.shape[1]} columns, but there are {n_speeds} speeds"
            )

    def save_best_outputs(self, m: Measurements) -> None:
        for iv in (300, 900, 3600):
            src = self.paths.file(f"edge_data_{iv}")
            if src.exists():
                shutil.copyfile(src, self.paths.file(f"best_edge_data_{iv}"))
        write_detector_csv(m.detector_output, self.paths.file("best_out.csv"))

    def simulate(self, tag: str, od: np.ndarray, supply: SupplyParams) -> Measurements:
        return self.evaluator.run(self.paths.od_base(tag), od, supply)

    # ---------------------------------------------------------- optimizers
    def _calibrate_in_loop(self, x0, od_true, od_prior, weight, supply, gains: SPSAGains, bounds):
        d = self.cfg.demand
        objective = SumoObjective(self, od_true, od_prior, supply)
        common = dict(
            gains=gains,
            niter=d.in_loop_iterations,
            bounds=bounds,
            reps=d.spsa_reps,
            rng=self.rng,
            parallel=self.cfg.simulation.parallel_perturbations,
        )
        if d.algorithm == "wspsa":
            w = wspsa_weight_matrix(weight, self.cfg.weights)
            return minimize_wspsa(objective, x0, w, momentum=d.momentum, **common)
        if d.algorithm == "spsa":
            return minimize_spsa(objective, x0, **common)
        raise ValueError(f"Algorithm {d.algorithm!r} is not supported with SUMO")

    def _calibrate_out_of_loop(
        self,
        x0,
        od_true,
        od_prior,
        assignment,
        weight,
        observed_counts,
        gains: SPSAGains,
        niter: int,
        bounds,
    ):
        d = self.cfg.demand
        if d.algorithm != "wspsa":
            raise ValueError(
                "Calibration with the simulator out of the loop requires algorithm = 'wspsa'"
            )
        weights = dataclasses.replace(self.cfg.weights, speed=0.0)
        objective = AssignmentObjective(
            assignment,
            observed_counts,
            od_true,
            od_prior,
            weights,
            self.cfg.estimator,
            best=self.best,
        )
        return minimize_wspsa(
            objective,
            x0,
            wspsa_weight_matrix(weight, weights),
            gains=gains,
            niter=niter,
            bounds=bounds,
            reps=d.spsa_reps,
            momentum=d.out_of_loop_momentum,
            rng=self.rng,
        )

    def _tune_gains(
        self, x0, od_true, od_prior, assignment, weight, observed_counts, eval_counts, bounds
    ):
        d, w, est = self.cfg.demand, self.cfg.weights, self.cfg.estimator
        at = d.autotune
        niter = at.iterations if at.iterations is not None else d.out_of_loop_iterations

        def score(log_spsa_a: float, log_spsa_c: float) -> float:
            gains = dataclasses.replace(
                self.cfg.demand.out_of_loop_gains, a=10**log_spsa_a, c=10**log_spsa_c
            )
            res = self._calibrate_out_of_loop(
                x0, od_true, od_prior, assignment, weight, observed_counts, gains, niter, bounds
            )
            demand = to_demand(res["x"])
            sim = AssignmentObjective(assignment, eval_counts, od_true, od_prior, w, est).counts(
                demand
            )
            return -(
                w.counts * gof(eval_counts.flatten(), sim.flatten(), est)
                + w.od * gof(od_true, demand, est)
            )

        best = bayes_maximize(
            score,
            {"log_spsa_a": at.log_a_bounds, "log_spsa_c": at.log_c_bounds},
            at.n_init,
            at.n_iter,
            at.random_state,
            self.paths.file("logs_spsa_tune.json"),
        )
        return np.power(10, best["params"]["log_spsa_a"]), np.power(
            10, best["params"]["log_spsa_c"]
        )

    # --------------------------------------------------------------- stages
    def run(self) -> dict:
        cfg, d, w, est = self.cfg, self.cfg.demand, self.cfg.weights, self.cfg.estimator
        self.prepare()

        base = self.evaluator.match()
        true_counts, true_speeds = base.observed_counts, base.observed_speeds
        log.info("Initial count %s: %s", est, gof(true_counts, base.simulated_counts, est))
        log.info("Initial speed %s: %s", est, gof(true_speeds, base.simulated_speeds, est))

        supply = cfg.initial_supply
        x_base_true = None
        init_seq = None
        initial_meas: Measurements | None = None
        od_estimate = self.template.vector()
        final: Measurements | None = None
        bagging = 0

        for seq in range(cfg.n_sequential):
            if d.enabled:
                if seq == 0:
                    x_od = self.template.vector()
                    x_base_true = x_od.copy()
                    init = clip_negative(
                        add_noise(x_od, cfg.data.od_noise_percent / 100, cfg.data.od_bias, self.rng)
                    )
                    init_seq = init.copy()
                else:
                    x_od = self.best.od
                    init = to_demand(x_od)

                initial_meas = self.simulate("init", init, supply)
                count_init = initial_meas.observed_counts
                mats = self.incidence()
                self._check_shapes(mats.weight, len(x_od), len(true_counts), len(true_speeds))

                corrected, bounds_fn = self._bias_correction(
                    init, initial_meas, true_counts, mats.weight
                )
                rmsn_c = gof(count_init, initial_meas.simulated_counts, est)
                rmsn_s = gof(initial_meas.observed_speeds, initial_meas.simulated_speeds, est)
                rmsn_od = gof(x_base_true, init, est)

                if d.bias_correction.only:
                    return self._only_bias_correction(
                        corrected,
                        init_seq,
                        x_base_true,
                        initial_meas,
                        supply,
                        rmsn_c,
                        rmsn_s,
                        rmsn_od,
                    )

                bounds = bounds_fn()
                log.info(
                    "Prior: count %s %s, speed %s %s, OD %s %s",
                    est,
                    rmsn_c,
                    est,
                    rmsn_s,
                    est,
                    rmsn_od,
                )
                self.history.record(
                    w.od * rmsn_od + w.counts * rmsn_c + w.speed * rmsn_s, rmsn_c, rmsn_od, rmsn_s
                )
                log.info("Bias-corrected OD %s: %s", est, gof(x_base_true, corrected, est))

                weight_out = mats.weight.copy()
                weight_in = mats.weight.copy()
                assignment = mats.assignment

                in_loop_gains = d.gains
                if w.counts != 0 and d.autotune.enabled:
                    a, c = self._tune_gains(
                        corrected.copy(),
                        x_od,
                        corrected.copy(),
                        assignment,
                        weight_out,
                        true_counts,
                        count_init,
                        bounds,
                    )
                    self.tuned_gains.append(
                        {"sequence": seq, "spsa_a": float(a), "spsa_c": float(c)}
                    )
                    in_loop_gains = dataclasses.replace(d.gains, a=float(a), c=float(c))
                    log.info("Tuned SPSA gains: a=%g, c=%g", a, c)

                od_bag = []
                for bagging in range(d.bagging_runs):
                    log.info("Bagging run %d", bagging)
                    start = corrected if not (d.warm_start and bagging > 0) else self.best.od
                    x_prior = np.array(start, copy=True)
                    x = clip_negative(
                        add_noise(start, d.exploration_noise_percent / 100, 1, self.rng)
                    )
                    gains = in_loop_gains
                    out_gains = d.out_of_loop_gains
                    for _ in range(d.outer_iterations):
                        if d.sim_out_loop:
                            self.best.reset()
                            res = self._calibrate_out_of_loop(
                                x,
                                x_base_true,
                                x_prior,
                                assignment,
                                weight_out,
                                true_counts,
                                out_gains,
                                d.out_of_loop_iterations,
                                bounds,
                            )
                            decay = d.learning_decay_factor**d.out_of_loop_iterations
                            out_gains = dataclasses.replace(
                                out_gains, a=out_gains.a / decay, c=out_gains.c / decay
                            )
                            x = res["x"]
                            self.simulate("n", x, supply)
                            mats = self.incidence()
                            weight_out, assignment = mats.weight, mats.assignment
                            weight_in = weight_out.copy()
                            self._check_shapes(
                                weight_out, len(x_od), len(true_counts), len(true_speeds)
                            )
                        if d.sim_in_loop:
                            self.best.reset()
                            res = self._calibrate_in_loop(
                                x, x_base_true, x_prior, weight_in, supply, gains, bounds
                            )
                            decay = d.learning_decay_factor**d.in_loop_iterations
                            gains = dataclasses.replace(gains, a=gains.a / decay, c=gains.c / decay)
                            x = res["x"]
                            last = weight_in.copy()
                            mats = self.incidence()
                            assignment = mats.assignment
                            weight_in = 0.1 * last + 0.9 * mats.weight
                            self._check_shapes(
                                weight_in, len(x_od), len(true_counts), len(true_speeds)
                            )
                    if self.best.od is None:
                        raise RuntimeError(
                            "Demand calibration produced no solution (no SPSA stage enabled?)"
                        )
                    od_bag.append(self.best.od)
                    pd.DataFrame({"real": x_base_true, "simulated": self.best.od.flatten()}).to_csv(
                        self.paths.file(f"{bagging}_od.csv"), index=None
                    )

                od_estimate = np.mean(np.array(od_bag), axis=0).astype(int)
                self.best.od = od_estimate.copy()
                log.info(
                    "Final bagged OD %s: %s", est, np.round(gof(x_base_true, od_estimate, est), 4)
                )
                final = self.simulate("mean", od_estimate, supply)
                self._save_comparison(
                    f"mean_{seq}_{d.bagging_runs}",
                    true_counts,
                    true_speeds,
                    x_base_true,
                    init_seq,
                    initial_meas,
                    final,
                    od_estimate,
                )
            else:
                self.best.od = od_estimate

            if cfg.supply.enabled:
                supply = self._calibrate_supply(supply, true_counts)
            final = self.simulate("mean", od_estimate, supply)

        results = self._results(supply)
        if cfg.plot and d.enabled:
            from actrys import plotting

            plotting.save_sumo_plots(
                self,
                results,
                x_base_true,
                init_seq,
                true_counts,
                true_speeds,
                initial_meas,
                final,
                od_estimate,
                bagging,
            )
        return results

    def _bias_correction(self, init, initial: Measurements, true_counts, weight):
        cfg, bc = self.cfg, self.cfg.demand.bias_correction
        if cfg.weights.counts == 0:
            return init, lambda: search_bounds(init, bc.lower_bound, 3)
        factor = bias_factor(
            bc.method,
            initial.simulated_counts,
            initial.observed_counts,
            observed_for_weighted=true_counts,
            incidence=weight,
            fill_nan=True,
        )
        upper = (2 / factor) - bc.lower_bound
        corrected = init / factor if bc.enabled else init
        return corrected, lambda: search_bounds(init, bc.lower_bound, upper)

    def _only_bias_correction(
        self, corrected, init_seq, x_true, initial, supply, rmsn_c, rmsn_s, rmsn_od
    ):
        w, est = self.cfg.weights, self.cfg.estimator
        self.history.record(
            w.od * rmsn_od + w.counts * rmsn_c + w.speed * rmsn_s, rmsn_c, rmsn_od, rmsn_s
        )
        after = self.simulate("init", corrected, supply)
        c = gof(after.observed_counts, after.simulated_counts, est)
        s = gof(after.observed_speeds, after.simulated_speeds, est)
        o = gof(x_true, corrected, est)
        self.history.record(w.od * o + w.counts * c + w.speed * s, c, o, s)
        pd.DataFrame(
            {
                "real": after.observed_counts.flatten(),
                "initial": initial.simulated_counts.flatten(),
                "simulated": after.simulated_counts.flatten(),
            }
        ).to_csv(self.paths.file("counts.csv"))
        pd.DataFrame(
            {
                "real": after.observed_speeds.flatten(),
                "initial": initial.simulated_speeds.flatten(),
                "simulated": after.simulated_speeds.flatten(),
            }
        ).to_csv(self.paths.file("speeds.csv"))
        pd.DataFrame(
            {
                "real": x_true.flatten(),
                "initial": init_seq.flatten(),
                "simulated": corrected.flatten(),
            }
        ).to_csv(self.paths.file("mean_od.csv"), index=None)
        results = self._history_dict()
        results.update(noise_param=self.cfg.data.od_noise_percent, bias_param=self.cfg.data.od_bias)
        _write_json(results, self.paths.file(f"results_{self.timestamp}.json"))
        return results

    def _calibrate_supply(self, supply: SupplyParams, true_counts: np.ndarray) -> SupplyParams:
        sc, est = self.cfg.supply, self.cfg.estimator
        od = self.best.od

        def score(**params: float) -> float:
            m = self.simulate("best", od, supply_from_bo(supply, params))
            return -gof(m.observed_counts, m.simulated_counts, est)

        best = bayes_maximize(
            score,
            dict(sc.bounds),
            sc.n_init,
            sc.n_iter,
            sc.random_state,
            self.paths.file("logs_supply.json"),
        )
        p = best["params"]
        updated = supply_from_bo(supply, p)
        target = abs(best["target"])
        self.history.best_supply_count_rmsn = min(self.history.best_supply_count_rmsn, target)
        self.history.f_val_supply.append(target)
        self.supply_bo_params = dict(p)
        return updated

    # -------------------------------------------------------------- outputs
    def _save_comparison(
        self, prefix, true_counts, true_speeds, x_true, init_seq, initial, final, od
    ):
        frames = {
            "counts": {
                "real": true_counts.flatten(),
                "initial": initial.simulated_counts.flatten(),
                "simulated": final.simulated_counts.flatten(),
            },
            "speeds": {
                "real": true_speeds.flatten(),
                "initial": initial.simulated_speeds.flatten(),
                "simulated": final.simulated_speeds.flatten(),
            },
            "od": {
                "real": x_true.flatten(),
                "initial": init_seq.flatten(),
                "simulated": od.flatten(),
            },
        }
        for kind, data in frames.items():
            pd.DataFrame(data).to_csv(self.paths.file(f"{prefix}_{kind}.csv"), index=None)

    def _history_dict(self) -> dict:
        h = self.history
        out = {
            "f_val": h.f_val,
            "f_val_supply": h.f_val_supply,
            "count_val": h.count_val,
            "speed_val": h.speed_val,
            "od_val": h.od_val,
            "best_supply_count_rmsn": h.best_supply_count_rmsn,
        }
        if self.best_supply is not None:
            out.update(self.best_supply.as_dict())
        return out

    def _results(self, supply: SupplyParams) -> dict:
        cfg, d = self.cfg, self.cfg.demand
        results = self._history_dict()
        results.update(
            spsa_a_init=d.gains.a,
            spsa_c_init=d.gains.c,
            n_iterations=d.outer_iterations,
            sim_in_iterations=d.in_loop_iterations,
            noise_param=cfg.data.od_noise_percent,
            bias_param=cfg.data.od_bias,
            spsa_reps=d.spsa_reps,
            weight_counts=cfg.weights.counts,
            weight_od=cfg.weights.od,
            weight_speed=cfg.weights.speed,
            wspsa_thrshold=d.wspsa_threshold,
            supply_params=supply.as_dict(),
            tuned_spsa_gains=self.tuned_gains,
        )
        if self.supply_bo_params is not None:
            results["supply_bo_params"] = self.supply_bo_params
        _write_json(results, self.paths.file(f"results_{self.timestamp}.json"))
        return results


def _write_json(data: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with Path(path).open("w") as fh:
        json.dump(data, fh, default=default)


def run_sumo_calibration(cfg: SumoCalibrationConfig, rng: RNG | None = None) -> dict:
    return SumoCalibration(cfg, rng).run()
