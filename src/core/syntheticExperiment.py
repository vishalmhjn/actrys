from numpy.lib.function_base import vectorize
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sys import getsizeof
from scipy.stats import invgauss


from aprioriDemand import temporal_kernel


def create_dummy_OD(
    num_taz,
    TOD_START,
    TOD_END,
    interval=1,
    lower_limit_od=0,
    upper_limit_od=100,
    od_distribution="normal",
    upper_limit_temporal_noise=0.1,
):
    """Creates a synthetic demand maatrix based on the predefined.

    Parameters
    ----------
    num_taz : int, required
        number of Traffic Analysis Zones

    TOD_START : int, required
        Start time of the Analysis period

    TOD_END : int, required
        End time of the Analysis period

    interval: float, required
        Size of the time interval e.g., For 15 minutes, use 0.25

    lower_limit_od: int, required
        Lower limit of the number of trips between two TAZs

    upper_limit_od: int, required
        Upper limit of the number of trips between two TAZs

    upper_limit_temporal_noise: float, required
        Upper limit of the temporal noise (uncertainty) in the trips between two TAZs

    od_distribution: str, required
        Distribution of the demand, e.g., normal, beta, gamma

    Raises
    ------
    """

    x = np.array(list(range(0, 24))).reshape(-1, 1)

    # A bi-peak temporal trend during the day
    y = np.array(
        [
            1,
            1,
            0.5,
            0.5,
            1,
            1,
            2,
            2.5,
            3,
            3.5,
            3,
            2.5,
            2,
            2,
            1.5,
            1.5,
            2,
            2.5,
            3,
            2.5,
            2.5,
            2,
            2,
            1,
        ]
    ).reshape(-1, 1)

    num_intervals = int((TOD_END - TOD_START) / interval)
    # print(num_intervals)

    TOD = np.arange(TOD_START, TOD_END, interval).reshape(-1, 1)

    _, temporal_model = temporal_kernel(x, y, interval)

    demand_shape = temporal_model.predict(TOD, return_std=True)[0]

    OD_vector = np.zeros((num_taz**2, num_intervals), dtype="float32")

    print(
        "===== Using " + od_distribution + " distribution for creating OD vector ======"
    )

    if od_distribution == "uniform":
        od_v = np.random.uniform(lower_limit_od, upper_limit_od, num_taz**2)
    elif od_distribution == "beta":
        # specifically adapting mu and scale to Munich Example
        mu = 3
        scale = 9
        od_v = (invgauss.rvs(mu, size=num_taz**2) * scale).astype("int")
        # plt.hist(od_v, bins=1000, log=False, cumulative=False, density=False,fill=False);
        # plt.show()
    else:
        raise ("Specify the distribution of the OD vectors such as normal, beta, ...")

    for i in range(0, num_intervals):
        temporal_noise = np.random.normal(
            0, upper_limit_temporal_noise, num_taz * num_taz
        )

        # print(temporal_noise)
        # add temporal noise
        OD_vector[:, i] = demand_shape[i] * od_v * (1 + temporal_noise)

        OD_vector[:, i][OD_vector[:, i] < 0] = 0
        OD_vector[:, i] = np.round(OD_vector[:, i], 0)
    return OD_vector


def create_assignment(
    OD,
    num_detectors,
    temporal_influence=8,
    min_sensor_correlation=0.2,
    max_sensor_correlation=0.4,
    network_correlation=0.2,
    go_random=False,
):
    """Creates an assignment matrix and return the coefficients.

    Randomness in OD-Detector is OFF by default.

    Parameters
    ----------
    OD : array, required
        OD Matrix of dimensions (od-pairs,time_intervals)

    num_detectors : int, required
        number of detectors in the network for count data

    temporal_influence : int, required
        Total intervals (current+past) which have an effect on the network counts

    min_sensor_correlation: float, required
        Minimum correlation between 0 and 1 between OD flow and sensor counts

    max_sensor_correlation: float, required
        Maximum correlation between 0 and 1 between OD flow and sensor counts

    network_correlation: float, required
        Maximum number of ODs responsible for the counts on a detector
        High network correlation will require the learning rate to be lower

    go_random: bool, optional
        go_random is a variable which if False changes the OD-detector dependence
        over time both spatially and temporally, so the problem complexity increases, as the same detectors
        do not record the same ODs over time. It is False because this is not reasonable
        for traffic networks in uncongested networks. For congested or stochastic route choice behavior
        use True.

    Raises
    ------
    """

    # print(len(OD))
    intervals = OD.shape[1]

    num_od_pairs = len(OD)

    # if np.sqrt(num_od_pairs) > 20:
    # 	network_correlation = 0.01
    print(f"Number of OD pairs: {num_od_pairs}")
    print(f"Network correlation: {network_correlation}")

    # Change the dtype to float 32 to use only 4 bytes
    X = np.zeros((len(OD) * intervals, num_detectors * intervals), dtype="float32")

    for k in range(0, intervals):
        # print(k)
        num_correlated_intervals = min(k, temporal_influence)
        max_interval = min(k + 1, intervals)

        sample_list_od = list(
            range(
                (k - num_correlated_intervals) * num_od_pairs,
                max_interval * num_od_pairs,
            )
        )

        # upper limit for spatially and temporally  correlated ODs
        upper_limit = int(len(sample_list_od) * network_correlation)

        if upper_limit == 0:
            upper_limit = 2

        for i in range(0, num_detectors):
            if k == 0:
                """The detectors are selected randomly for first interval and then the same
                dependence of OD-detectors is used for the further intervals."""

                random.shuffle(sample_list_od)
                # select number of correlated ODs
                num_correlated_od = random.choice(list(range(1, upper_limit)))
                # select indices of correlated ODs
                indices_correlated_od = sample_list_od[:num_correlated_od]
                weights_correlated_od = np.random.uniform(
                    min_sensor_correlation, max_sensor_correlation, num_correlated_od
                )
                X[
                    list(indices_correlated_od), k * num_detectors + i
                ] = weights_correlated_od

            else:
                if go_random:
                    random.shuffle(sample_list_od)
                    num_correlated_od = random.choice(list(range(1, upper_limit)))
                    indices_correlated_od = sample_list_od[:num_correlated_od]
                    weights_correlated_od = np.random.uniform(
                        min_sensor_correlation,
                        max_sensor_correlation,
                        num_correlated_od,
                    )
                    X[
                        list(indices_correlated_od), k * num_detectors + i
                    ] = weights_correlated_od

                else:
                    indices_correlated_od = np.where(X[:, i] != 0)[0]
                    num_correlated_od = len(indices_correlated_od)

                    # set_indices_correlated_od = []
                    for l in range(k - num_correlated_intervals, k + 1):
                        # set_indices_correlated_od.append(indices_correlated_od + l*num_od_pairs)
                        X[
                            list(indices_correlated_od + l * num_od_pairs),
                            k * num_detectors + i,
                        ] = X[list(indices_correlated_od), i]

                    # # copy the weights for the same detectors from the first interval
                    # X[list(set_indices_correlated_od),k*num_detectors+i]  = X[list(indices_correlated_od),i]
    return X


def link_speeds(q, interval, traffic_regime, max_flow=7000, max_speed=120):
    """
    Approximates synthetic link speeds based on the flow.
    The parameters of the fundamental diagram are randomly sampled from the
    prespecfied distribution. The speed is approximated by the quadratic relationship of the
    form Q = k1 * v**2 + k2 * v2 . Based on the shape of the fundamental diagram, it can be said that
    k1<0, whereas k2>0. To resolve the ambiguity we need a data driven procedure, which can inform us about
    whaat values are feasible for a speed at a specific time.
    Randomness in OD-Detector is OFF by default.

    Parameters
    ----------
    q : array, required
        Count matrix from the synthetic simulation

    interval: float, required
        Interval to calculate the hourly flow

    traffic_regime: int, reguired
        1: Free-flow, 0: Congestion
    """

    k1 = (-1 * max_flow * 4) / (max_speed**2)
    k2 = -1 * k1 * max_speed

    flow = q / interval
    v = [
        (-k2 + np.sqrt(k2**2 + 4 * flow * k1)) / (2 * k1),
        (-k2 - np.sqrt(k2**2 + 4 * flow * k1)) / (2 * k1),
    ][traffic_regime]
    # print(v)
    v = np.nan_to_num(v, 0)
    v = np.where(v < 0, 0, v)
    return v


def synthetic_simulation(OD_long, W, t_state, interval, num_detectors):
    """OD is input OD
    W is assignment matrix
    """
    counts = W.T @ OD_long
    reshape_counts = counts.reshape(-1, num_detectors).T
    speed = np.zeros(
        (reshape_counts.shape[0], reshape_counts.shape[1]), dtype="float32"
    )
    free = np.where(t_state == 1)
    congestion = np.where(t_state == 0)
    vectorize_speeds = np.vectorize(link_speeds)
    speed[free, :] = vectorize_speeds(reshape_counts[free, :], interval, 1)
    speed[congestion, :] = vectorize_speeds(reshape_counts[congestion, :], interval, 0)
    speed = speed.T.reshape(-1, 1).flatten()
    return counts, speed


def synthetic_scenario_orchestrator(
    num_od,
    TOD_START,
    TOD_END,
    num_detectors,
    time_interval,
    lower_limit_od,
    upper_limit_od,
    od_distribution,
    network_correlation=0.2,
):
    """one function to call them all"""

    OD = create_dummy_OD(
        num_od,
        TOD_START,
        TOD_END,
        time_interval,
        lower_limit_od,
        upper_limit_od,
        od_distribution,
    )

    # assuming influence of 2 hours, so for 15 minute intervals, it is 8
    temporal_influence = 2 * int(1 / time_interval)

    W = create_assignment(
        OD, num_detectors, temporal_influence, network_correlation=network_correlation
    )
    OD_long = OD.T.reshape(-1, 1)
    counts = W.T @ OD_long

    traffic_states = np.random.binomial(1, 0.5, size=num_detectors)

    free = np.where(traffic_states == 1)
    congestion = np.where(traffic_states == 0)

    reshape_counts = counts.reshape(-1, num_detectors).T
    vectorize_speeds = np.vectorize(link_speeds)
    speed = np.zeros(
        (reshape_counts.shape[0], reshape_counts.shape[1]), dtype="float32"
    )

    speed[free, :] = vectorize_speeds(reshape_counts[free, :], time_interval, 1)
    speed[congestion, :] = vectorize_speeds(
        reshape_counts[congestion, :], time_interval, 0
    )

    speed = speed.T.reshape(-1, 1)
    return OD, W, counts, speed, traffic_states


if __name__ == "__main__":
    interval = 0.25
    num_detectors = 50
    num_od = 10
    TOD_S = 0
    TOD_E = 24

    plot = True

    OD, W, counts, speed, _ = synthetic_scenario_orchestrator(
        num_od,
        TOD_S,
        TOD_E,
        num_detectors,
        interval,
        lower_limit_od=0,
        upper_limit_od=100,
        od_distribution="beta",
    )
    counts = counts.reshape(-1, num_detectors).T

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.imshow(OD, cmap="hot", interpolation="nearest")
        plt.xlabel("Intervals")
        plt.ylabel("OD pair")
        plt.tight_layout()
        plt.savefig(
            "../../images/od_matrix"
            + str(int(np.sqrt(OD.shape[0])))
            + "_"
            + str(interval)
            + ".png",
            dpi=300,
        )

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.imshow(W, cmap="hot", interpolation="nearest")
        plt.xlabel("Count Detector-Intervals")
        plt.ylabel("OD pair - Intervals")
        plt.tight_layout()
        plt.savefig(
            "../../images/assignment_matrix"
            + str(int(np.sqrt(OD.shape[0])))
            + "_"
            + str(interval)
            + ".png",
            dpi=300,
        )

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plt.imshow(counts, cmap="hot", interpolation="nearest")
        plt.xlabel("Interval")
        plt.ylabel("Count Detector")
        plt.tight_layout()
        plt.savefig(
            "../../images/count_matrix"
            + str(int(np.sqrt(OD.shape[0])))
            + "_"
            + str(interval)
            + ".png",
            dpi=300,
        )
