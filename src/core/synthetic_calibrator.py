import sys
import os
from datetime import datetime

import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

import utilities

from optimization_handler.gof import gof_eval, squared_deviation
from optimization_handler.optimizer import SolutionFinder
from synthetic_experiment import (
    synthetic_scenario_orchestrator,
    synthetic_simulation,
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.util import load_logs


temp_folder_name = sys.argv[1]
run_scenario = sys.argv[2]
noise_param = int(float(sys.argv[3]))
bias_param = float(sys.argv[4])
spsa_a = float(sys.argv[5])
spsa_c = float(sys.argv[6])
spsa_reps = int(sys.argv[7])
n_iterations = int(sys.argv[8])
which_algo = sys.argv[9]
file_idenfier = sys.argv[10]
bagging_run = int(sys.argv[11])
beta_momentum_param = float(sys.argv[12])
count_noise = int(sys.argv[13])
weight_counts = float(sys.argv[14])
weight_od = float(sys.argv[15])
weight_speed = float(sys.argv[16])
spsa_autotune = eval(sys.argv[17])
correction_heuristic = eval(sys.argv[18])
only_bias_correction = eval(sys.argv[19])
bias_correction_method = sys.argv[20]


best_od = []
estimator = "wmape"
stochastic_solution_counter = 0

timestr = str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

pre_string = "../../synthetic_sims" + "/" + temp_folder_name

try:
    os.mkdir(pre_string)
except FileExistsError:
    pass


res_dict = dict()
res_dict["f_val"] = []
res_dict["count_val"] = []
res_dict["od_val"] = []
estimated = 0
simulated_counts = 0
simulated_speeds = 0

best_rmsn = 10000000


debug = False


def save_params(d, params):
    for param in params:
        d[str(param)] = eval(param)
    return d


def add_noise(x, perc_var, mu=0):
    """Add noise to a synthetic data"""
    noisy_od = []
    for i in x:
        noisy_od.append(int(mu * i) + int(np.random.randn() * perc_var * i))
    return noisy_od


def objective_function(
    X,
    X_true,
    X_prior,
    W,
    count_init,
    speed_init,
    traffic_state,
    interval,
    num_detectors,
    weighted=False,
    eval_rmsn=False,
    which_perturb="positive",
):
    """This is the objective function which estimates the rmsn between the
    True and Simulaed detector counts
    """

    X = np.array([i for i in np.where(X < 0, 0, X)])
    X = X.astype(int)
    global simulated_counts, simulated_speeds
    simulated_counts, simulated_speeds = synthetic_simulation(
        X, W, traffic_state, interval, num_detectors
    )

    simulated_counts = simulated_counts.reshape(-1, 1)
    simulated_speeds = simulated_speeds.reshape(-1, 1)

    global weight_counts, weight_od, weight_speed

    if weighted == False:
        count_rmsn = np.round(
            gof_eval(
                count_init.flatten(), simulated_counts.flatten(), estimator=estimator
            ),
            4,
        )
        rmsn = (
            weight_counts
            * gof_eval(
                count_init.flatten(), simulated_counts.flatten(), estimator=estimator
            )
            + weight_od * gof_eval(X_true, X, estimator=estimator)
            + weight_speed
            * gof_eval(
                speed_init.flatten(), simulated_speeds.flatten(), estimator=estimator
            )
        )
        od_rmsn = gof_eval(X_true, X, estimator=estimator)

    else:
        if weight_counts != 0:
            unw_sd_counts = squared_deviation(
                count_init.flatten(), simulated_counts.flatten()
            )
            sd_counts = weight_counts * unw_sd_counts
        else:
            sd_counts = np.array([])
        if weight_od != 0:
            unw_sd_od = squared_deviation(X_prior, X)
            sd_od = weight_od * unw_sd_od
        else:
            sd_od = np.array([])
        if weight_speed != 0:
            unw_sd_speeds = squared_deviation(
                speed_init.flatten(), simulated_speeds.flatten()
            )
            sd_speed = weight_speed * unw_sd_speeds
        else:
            sd_speed = np.array([])

        if sd_counts.size:
            if sd_od.size:
                if sd_speed.size:
                    rmsn = np.hstack((sd_counts, sd_od, sd_speed))
                else:
                    rmsn = np.hstack((sd_counts, sd_od))
            else:
                if sd_speed.size:
                    rmsn = np.hstack((sd_counts, sd_speed))
                else:
                    rmsn = sd_counts
        else:
            if sd_od.size:
                if sd_speed.size:
                    rmsn = np.hstack((sd_od, sd_speed))
                else:
                    rmsn = sd_od
            else:
                if sd_speed.size:
                    rmsn = sd_speed
                else:
                    raise ("All weights cannot be zero")  # type: ignore

    if eval_rmsn == True:
        global res_dict
        if len(res_dict["od_val"]) > 1:
            if np.std(res_dict["od_val"]) < 0.10:
                global stochastic_solution_counter
                stochastic_solution_counter += 1
        res_dict["f_val"].append(rmsn)
        res_dict["count_val"].append(count_rmsn)
        res_dict["od_val"].append(od_rmsn)
        global estimated, best_rmsn, best_od
        estimated = X
        if rmsn < best_rmsn:
            best_od = X.copy()
        print(
            f"Weighted {estimator} {rmsn:.6f}, Count {estimator} {count_rmsn:.6f}",
            end="\r",
        )

    return rmsn


def calibration_handler(
    obj_func,
    x0,
    x_true,
    x_prior,
    W,
    orig_count,
    orig_speed,
    t_state,
    interval,
    num_detectors,
    num_od_pairs,
    beta_momentum,
    network_correlation,
    spsa_a,
    spsa_c,
    bounds,
):
    """Wrapper for calibration
    TODO when the random pertubation lands in a negative range
    As expected, SUMO gives an errors and does not generates new trips
    So optimizer uses the same routes as last time for goodness evaluation
    Need to think a better way to handle this, so that negative values are totally
    avoided"""

    sf = SolutionFinder(obj_func, bounds=bounds, x0=x0)
    global which_algo
    print("===========  Optimization using " + which_algo + " ===============")
    if which_algo == "spsa":
        result = sf.spsa(
            path_args=(
                x_true,
                x_prior,
                W,
                orig_count,
                orig_speed,
                t_state,
                interval,
                num_detectors,
            ),
            paired=False,
            a=spsa_a,
            c=spsa_c,
            reps=spsa_reps,
            bounds=sf.bounds,
            niter=n_iterations,
            disp=True,
        )

    elif which_algo == "pcspsa":
        pca = PCA()
        x0 = x0.reshape(-1, num_od_pairs)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(x0)
        pca.fit(X_train)
        number_compenents = len(
            pca.explained_variance_ratio_.cumsum()[
                pca.explained_variance_ratio_.cumsum() <= 0.95
            ]
        )
        if number_compenents == 0:
            number_compenents = 1
        pca = PCA(n_components=number_compenents)
        pca.fit(X_train)
        result = sf.pcspsa(
            num_features=num_od_pairs,
            pca_model=pca,
            var_scaler=scaler,
            path_args=(
                x_true,
                x_prior,
                W,
                orig_count,
                orig_speed,
                t_state,
                interval,
                num_detectors,
            ),
            paired=False,
            a=spsa_a,
            c=spsa_c,
            reps=spsa_reps,
            bounds=None,
            niter=n_iterations,
            disp=True,
        )

    elif which_algo == "wspsa":
        global weight_counts, weight_od, weight_speed
        if weight_counts != 0:
            if weight_od != 0:
                if weight_speed != 0:
                    weight_wspsa = np.zeros(
                        (W.shape[0], W.shape[1] + W.shape[0] + W.shape[1]), dtype="int8"
                    )
                    weight_wspsa[:, : W.shape[1]] = np.where(W > 0, 1, 0)
                    weight_wspsa[:, W.shape[1] : W.shape[1] + W.shape[0]] = np.eye(
                        W.shape[0]
                    )
                    weight_wspsa[:, W.shape[1] + W.shape[0] :] = np.where(W > 0, 1, 0)
                else:
                    weight_wspsa = np.zeros(
                        (W.shape[0], W.shape[1] + W.shape[0]), dtype="int8"
                    )
                    weight_wspsa[:, : W.shape[1]] = np.where(W > 0, 1, 0)
                    weight_wspsa[:, W.shape[1] :] = np.eye(W.shape[0])
            else:
                if weight_speed != 0:
                    weight_wspsa = np.zeros(
                        (W.shape[0], W.shape[1] + W.shape[1]), dtype="int8"
                    )
                    weight_wspsa[:, : W.shape[1]] = np.where(W > 0, 1, 0)
                    weight_wspsa[:, W.shape[1] :] = np.where(W > 0, 1, 0)
                else:
                    weight_wspsa = np.where(W > 0, 1, 0).astype("int8")
        else:
            if weight_od != 0:
                if weight_speed != 0:
                    weight_wspsa = np.zeros(
                        (W.shape[0], W.shape[0] + W.shape[1]), dtype="int8"
                    )
                    weight_wspsa[:, : W.shape[0]] = np.eye(W.shape[0])
                    weight_wspsa[:, W.shape[0] :] = np.where(W > 0, 1, 0)
                else:
                    weight_wspsa = np.eye(W.shape[0], dtype="int8")
            else:
                if weight_speed != 0:
                    weight_wspsa = np.where(W > 0, 1, 0).astype("int8")
                else:
                    raise ("All the weights cannot be zero")

        result = sf.w_spsa(
            weights_wspsa=weight_wspsa,
            path_args=(
                x_true,
                x_prior,
                W,
                orig_count,
                orig_speed,
                t_state,
                interval,
                num_detectors,
            ),
            paired=False,
            a=spsa_a,
            c=spsa_c,
            gamma=0.01,
            alpha=0.7,
            param_momentum=beta_momentum,
            network_correlation=network_correlation,
            reps=spsa_reps,
            bounds=sf.bounds,
            niter=n_iterations,
            disp=True,
        )
        return result


def spsa_tune_function(log_spsa_a, log_spsa_c):
    spsa_a = np.power(10, log_spsa_a)
    spsa_c = np.power(10, log_spsa_c)

    res = calibration_handler(
        objective_function,
        init_iter,
        X_OD,
        X_prior,
        W,
        measured_counts,
        real_speed,
        t_state,
        interval,
        num_detectors,
        num_od_pairs,
        beta_momentum_param,
        network_correlation,
        spsa_a,
        spsa_c,
        bounds=BOUNDS,
    )

    X = res["x"]
    X = np.array([int(i) for i in np.where(X < 0, 0, X)])
    simulated_counts, simulated_speeds = synthetic_simulation(
        X, W, t_state, interval, num_detectors
    )

    simulated_counts = simulated_counts.reshape(-1, 1)
    simulated_speeds = simulated_speeds.reshape(-1, 1)

    global weight_counts, weight_od, weight_speed
    rmsn = weight_counts * gof_eval(
        measured_counts.flatten(), simulated_counts.flatten(), estimator=estimator
    ) + weight_od * gof_eval(X_OD, X, estimator=estimator)

    return -rmsn


if __name__ == "__main__":
    execute_scenario = eval(run_scenario)
    num_od = 50  # 50
    num_od_pairs = num_od**2
    num_detectors = 500  # 1000 #600
    interval = 1
    TOD_S = 7
    TOD_E = 10
    plot = True
    network_correlation = 0.2
    if execute_scenario:
        OD, W, sim, speed, t_state = synthetic_scenario_orchestrator(
            num_od,
            TOD_S,
            TOD_E,
            num_detectors,
            interval,
            lower_limit_od=0,
            upper_limit_od=100,
            od_distribution="beta",
            network_correlation=network_correlation,
        )

    real_count = np.array([int(i) for i in sim]).reshape(-1, 1)
    real_speed = np.array([int(i) for i in speed]).reshape(-1, 1)

    measured_counts = np.array(
        add_noise(real_count, int(count_noise) / 100, mu=1)
    ).reshape(-1, 1)

    shape = OD.shape

    X_OD = OD.T.flatten()

    initial_solution = np.array(add_noise(X_OD, int(noise_param) / 100, mu=bias_param))

    init = np.array(np.where(initial_solution < 0, 0, initial_solution))

    initial_counts, initial_speed = synthetic_simulation(
        init, W, t_state, interval, num_detectors
    )
    initial_counts = initial_counts.reshape(-1, 1)
    initial_speed = initial_speed.reshape(-1, 1)

    if weight_counts != 0:
        # Automatic optimization bounds determination
        if correction_heuristic == True:
            if bias_correction_method == "naive":
                estimated_bias_factor = np.sum(initial_counts) / np.sum(measured_counts)
            elif bias_correction_method == "weighted":
                estimated_bias_counts = initial_counts / (measured_counts + 1e-8)
                estimated_bias_factor = (W @ estimated_bias_counts).flatten() / W.sum(
                    axis=1
                )
                print(estimated_bias_factor)
                estimated_bias_factor = estimated_bias_factor.flatten()
            else:
                raise ("Please enter a valid bias correction method")
            domain_lower_bound = 0.1
            domain_upper_bound = (2 / estimated_bias_factor) - domain_lower_bound

            print(estimated_bias_factor)
            corrected_od = init / estimated_bias_factor

            X_domain = init.copy()
            if bias_correction_method == "naive":
                print("here")
                BOUNDS = np.array(
                    [[domain_lower_bound * i, domain_upper_bound * i] for i in X_domain]
                )
            else:
                BOUNDS = np.array(
                    [
                        [domain_lower_bound * i, domain_upper_bound[i] * i]
                        for i in X_domain
                    ]
                )

        else:
            corrected_od = init.copy()
            X_domain = init.copy()
            domain_lower_bound = 0.1
            domain_upper_bound = 3
            BOUNDS = np.array(
                [[domain_lower_bound * i, domain_upper_bound * i] for i in X_domain]
            )
    else:
        corrected_od = init.copy()
        X_domain = init.copy()
        domain_lower_bound = 0.1
        domain_upper_bound = 3
        BOUNDS = np.array(
            [[domain_lower_bound * i, domain_upper_bound * i] for i in X_domain]
        )

    if only_bias_correction:
        sim_init_counts, sim_init_speeds = synthetic_simulation(
            corrected_od, W, t_state, interval, num_detectors
        )
        sim_init_counts = sim_init_counts.reshape(-1, 1)
        sim_init_speeds = sim_init_speeds.reshape(-1, 1)

        rmsn_c = gof_eval(measured_counts, initial_counts, estimator=estimator)
        rmsn_s = gof_eval(real_speed, initial_speed, estimator=estimator)
        rmsn_od = gof_eval(X_OD, init, estimator=estimator)
        res_dict["count_val"].append(rmsn_c)
        res_dict["od_val"].append(rmsn_od)

        rmsn_c = gof_eval(measured_counts, sim_init_counts, estimator=estimator)
        rmsn_s = gof_eval(real_speed, sim_init_speeds, estimator=estimator)
        rmsn_od = gof_eval(X_OD, corrected_od, estimator=estimator)
        res_dict["count_val"].append(rmsn_c)
        res_dict["od_val"].append(rmsn_od)

        save_counts = pd.DataFrame(
            {
                "real": measured_counts.flatten(),
                "initial": initial_counts.flatten(),
                "simulated": sim_init_counts.flatten(),
            }
        )
        save_counts.to_csv(pre_string + "/" + "counts.csv")

        save_speeds = pd.DataFrame(
            {
                "real": real_speed.flatten(),
                "initial": initial_speed.flatten(),
                "simulated": sim_init_speeds.flatten(),
            }
        )
        save_speeds.to_csv(pre_string + "/" + "speeds.csv")

        save_od = pd.DataFrame(
            {
                "real": X_OD.flatten(),
                "initial": init.flatten(),
                "simulated": corrected_od.flatten(),
            }
        )
        save_od.to_csv(pre_string + "/mean_" + "od.csv", index=None)

        res_dict = save_params(res_dict, ["noise_param", "bias_param"])

        with open(pre_string + "/results_" + timestr + ".json", "w") as fp:
            json.dump(res_dict, fp)
    else:
        rmsn_od_bias_corrected = gof_eval(X_OD, corrected_od, estimator=estimator)

        print("B-N Corrected OD " + estimator + ": " + str(rmsn_od_bias_corrected))

        if debug:
            plt.imshow(
                initial_counts.reshape(-1, num_detectors).T,
                cmap="hot",
                interpolation="nearest",
            )
            plt.xlabel("Time")
            plt.ylabel("Counts")
            plt.tight_layout()
            plt.show()

        # SPSA parameter tuner
        init_iter = corrected_od.copy()
        X_prior = init_iter.copy()

        if weight_counts != 0:
            if spsa_autotune == True:
                pbounds = {"log_spsa_a": (-3, 4), "log_spsa_c": (-2, 1)}
                bounds_transformer = SequentialDomainReductionTransformer()
                optimizer = BayesianOptimization(
                    f=spsa_tune_function,
                    pbounds=pbounds,
                    random_state=2,
                    allow_duplicate_points=True,
                )
                try:
                    ### previously seen points are used to explore the new points
                    load_logs(optimizer, logs=[pre_string + "/logs_spsa_tune.json"])
                except FileNotFoundError:
                    logger = JSONLogger(path=pre_string + "/logs_spsa_tune.json")

                logger = JSONLogger(path=pre_string + "/logs_spsa_tune.json")

                optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
                optimizer.maximize(
                    init_points=50,
                    n_iter=100,
                )

                best_params = optimizer.max["params"]
                spsa_a = np.power(10, best_params["log_spsa_a"])
                spsa_c = np.power(10, best_params["log_spsa_c"])
            else:
                print("Using manual values of SPSA c and a")
                print(spsa_a, spsa_c)

        SPA = False

        for beta_momentum in [beta_momentum_param]:
            od_bag_list = []

            for bagging in range(0, bagging_run):
                print(f"Run: {bagging}")

                if SPA == False:
                    init_iter = corrected_od.copy()
                else:
                    if bagging == 0:
                        init_iter = corrected_od.copy()
                    else:
                        init_iter = best_od.copy()

                best_rmsn = 1000000000
                best_od = []

                res = calibration_handler(
                    objective_function,
                    init_iter,
                    X_OD,
                    X_prior,
                    W,
                    measured_counts,
                    real_speed,
                    t_state,
                    interval,
                    num_detectors,
                    num_od_pairs,
                    beta_momentum,
                    network_correlation,
                    spsa_a,
                    spsa_c,
                    bounds=BOUNDS,
                )
                fig, ax = plt.subplots(3, 2, figsize=(7, 9))

                fig, ax = utilities.plot_45_degree_plots(
                    fig,
                    ax,
                    X_OD,
                    best_od,
                    init,
                    weight_counts,
                    weight_od,
                    weight_speed,
                    measured_counts,
                    simulated_counts,
                    initial_counts,
                    real_speed,
                    simulated_speeds,
                    initial_speed,
                    spsa_a,
                    spsa_c,
                    noise_param,
                    bias_param,
                )

                plt.tight_layout()
                plt.savefig(
                    pre_string
                    + "/results_"
                    + str(bagging)
                    + "a_"
                    + str(spsa_a)
                    + "n_"
                    + str(noise_param)
                    + "_"
                    + "b_"
                    + str(int(100 * bias_param))
                    + "_"
                    + str(num_od)
                    + "_"
                    + str(num_detectors)
                    + "_"
                    + str(n_iterations)
                    + "_"
                    + str(weight_od)
                    + "_"
                    + str(weight_counts)
                    + "_"
                    + str(weight_speed)
                    + "_"
                    + str(which_algo)
                    + "_"
                    + file_idenfier
                    + ".png",
                    dpi=300,
                )
                plt.close()

                od_bag_list.append(best_od)
                save_od = pd.DataFrame({"real": X_OD, "simulated": best_od.flatten()})
                save_od.to_csv(
                    pre_string + "/" + str(bagging) + "_" + "od.csv",
                    index=None,
                )

            od_mean = np.mean(np.array(od_bag_list), axis=0).astype(int)
            sim_final_counts, sim_final_speeds = synthetic_simulation(
                od_mean, W, t_state, interval, num_detectors
            )

            sim_final_counts = sim_final_counts.reshape(-1, 1)
            sim_final_speeds = sim_final_speeds.reshape(-1, 1)
            if plot:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax = utilities.plot_loss_curve_synthetic(ax, res_dict, spsa_a, spsa_c)
                plt.savefig(
                    pre_string
                    + "/loss_"
                    + "a_"
                    + str(spsa_a)
                    + "n_"
                    + str(noise_param)
                    + "_"
                    + "b_"
                    + str(int(100 * bias_param))
                    + "_"
                    + str(num_od)
                    + "_"
                    + str(num_detectors)
                    + "_"
                    + str(n_iterations)
                    + "_"
                    + str(weight_od)
                    + "_"
                    + str(weight_counts)
                    + "_"
                    + str(weight_speed)
                    + "_"
                    + str(which_algo)
                    + file_idenfier
                    + "_"
                    + ".png",
                    dpi=300,
                )
                plt.legend()
                plt.close()

                fig, ax = plt.subplots(3, 2, figsize=(7, 9))

                fig, ax = utilities.plot_45_degree_plots(
                    fig,
                    ax,
                    X_OD,
                    od_mean,
                    init,
                    weight_counts,
                    weight_od,
                    weight_speed,
                    measured_counts,
                    sim_final_counts,
                    initial_counts,
                    real_speed,
                    sim_final_speeds,
                    initial_speed,
                    spsa_a,
                    spsa_c,
                    noise_param,
                    bias_param,
                )

                plt.tight_layout()
                plt.savefig(
                    pre_string
                    + "/results_"
                    + str(beta_momentum)
                    + "_"
                    + "a_"
                    + str(spsa_a)
                    + "n_"
                    + str(noise_param)
                    + "_"
                    + "b_"
                    + str(int(100 * bias_param))
                    + "_"
                    + str(num_od)
                    + "_"
                    + str(num_detectors)
                    + "_"
                    + str(n_iterations)
                    + "_"
                    + str(weight_od)
                    + "_"
                    + str(weight_counts)
                    + "_"
                    + str(weight_speed)
                    + "_"
                    + str(which_algo)
                    + "_"
                    + file_idenfier
                    + ".png",
                    dpi=300,
                )

                if debug:
                    plt.imshow(
                        simulated_counts.reshape(-1, num_detectors).T,
                        cmap="hot",
                        interpolation="nearest",
                    )
                    plt.xlabel("Time")
                    plt.ylabel("Counts")
                    plt.tight_layout()
                    plt.show()

                save_counts = pd.DataFrame(
                    {
                        "real": measured_counts.flatten(),
                        "initial": initial_counts.flatten(),
                        "simulated": sim_final_counts.flatten(),
                    }
                )
                save_counts.to_csv(
                    pre_string + "/" + str(beta_momentum) + "_" + "counts.csv"
                )

                save_speeds = pd.DataFrame(
                    {
                        "real": real_speed.flatten(),
                        "initial": initial_speed.flatten(),
                        "simulated": sim_final_speeds.flatten(),
                    }
                )
                save_speeds.to_csv(
                    pre_string + "/" + str(beta_momentum) + "_" + "speeds.csv"
                )

                save_od = pd.DataFrame(
                    {
                        "real": X_OD.flatten(),
                        "initial": init.flatten(),
                        "simulated": od_mean.flatten(),
                    }
                )

                save_od.to_csv(
                    pre_string
                    + "/mean_"
                    + str(beta_momentum)
                    + "_"
                    + str(bagging_run)
                    + "_"
                    + "od.csv",
                    index=None,
                )

                res_dict = save_params(
                    res_dict,
                    [
                        "spsa_a",
                        "spsa_c",
                        "n_iterations",
                        "noise_param",
                        "bias_param",
                        "spsa_reps",
                        "weight_counts",
                        "weight_od",
                        "weight_speed",
                        "beta_momentum",
                    ],
                )

                with open(
                    pre_string + "/results_" + str(beta_momentum) + timestr + ".json",
                    "w",
                ) as fp:
                    json.dump(res_dict, fp)
