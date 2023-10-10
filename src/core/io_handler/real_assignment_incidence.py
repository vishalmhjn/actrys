#### To generate the weight matrix for W-SPSA on the fly

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pickle
import subprocess
import os

from sim_handler.scenario_generator import config
from io_handler.output_processor import additonal_identifier

period = int(3600 * config["DEMAND_INTERVAL"])
trip_impact = 3600


def xml_to_csv(input_file):
    """Convert an XML file to CSV using SUMO's xml2csv tool."""
    SUMO_HOME = os.getenv("SUMO_HOME")
    subprocess.run(SUMO_HOME + "/tools/xml/xml2csv.py " + input_file, shell=True)


def get_od_pairs_df(path_od_sample):
    """Get OD pairs dataframe from a sample OD matrix file."""
    dfod = pd.read_csv(path_od_sample, sep=" ")
    dfod["od_pair"] = dfod.o.astype("str") + "___" + dfod.d.astype("str")
    return dfod


def get_routes_df(path_routes):
    """Get routes dataframe from a SUMO route output file."""
    dfr = pd.read_csv(path_routes[:-3] + "csv", sep=";")
    dfr.drop_duplicates(subset=["vehicle_id"], keep="last", inplace=True)
    dfr["od_pair"] = (
        dfr.vehicle_fromTaz.astype("str") + "___" + dfr.vehicle_toTaz.astype("str")
    )
    return dfr


def get_trip_summary_df(path_trip_summary):
    """Get trip summary dataframe from a SUMO trip summary output file."""
    return pd.read_csv(path_trip_summary + ".csv", sep=";")


def generate_detector_incidence(
    path_od_sample,
    path_routes,
    path_trip_summary,
    path_additional,
    path_true_count,
    path_match_detectors,
    path_output_weight,
    path_output_assignment,
    scenario,
    threshold=False,
    threshold_value=0.1,
    is_synthetic=False,
    binary_rounding=False,
    time_interval=period,
    t_start=config["TOD_START"] * 3600,
    t_end=config["TOD_END"] * 3600,
    t_impact=trip_impact,
    t_warmup=config["WARM_UP_PERIOD"] * 3600,
    do_plot=True,
    do_save=True,
):
    """Extract link incidence matrix from the simulated routes in SUMO

    Parameters
    ----------
    path_od_sample : str, required
        path to the sample OD matrix file in SUMO format

    path_routes : str, required
        path to the SUMO route output file

    path_trip_summary : str, required
        path to the SUMO trip summary output file

    path_additional: str, required
        path to the SUMO edge detector file i.e., additional.add.xml

    path_true_count: str, required
        path to the true counts

    path_match_detectors: str, required
        path to the detector file to match the counts from real-world data

    path_output: str, required
        path to the output file where they should be saved

    scenario: str, required
        name of the scenario

    threshold: bool, optional
        if incidence ratio to be ignored below a threshold value

    threshold_value: float, optional
        value of the threshold value below which link incidence ratio is ignored

    is_synthetic: bool, optional
        if this is a scenario using synthetic counts. set FALSE if using real count data

    binary_rounding: bool, optional
        if the link incidence ratios to be rounded to 0 or 1 based on the threshold_value

    time_interval: int, optional
        interval of analysis of weight matrices, this is same as calibration interval

    t_start: int, optional
        start time of the simulation

    t_end: int, optional
        end time of the simulation

    t_impact: int, optional
        time upto which future impact of the current demand is possible

    t_warmup: int, optional
        warm-up period for the simulation

    do_plot: bool, optional
        if the link incidence matrices are to be plotted

    do_save: bool, optional
        Set True if want to save the link incidence matrix

    Returns
    ------
    numpy.ndarray
        weight matrix

    numpy.ndarray
        link incidence matrix
    """
    SUMO_HOME = os.getenv("SUMO_HOME")

    xml_to_csv(path_trip_summary)
    xml_to_csv(path_routes)
    xml_to_csv(path_additional)

    dfod = get_od_pairs_df(path_od_sample)

    dfr = get_routes_df(path_routes)
    dft = get_trip_summary_df(path_trip_summary)

    df_merge = pd.merge(
        left=dfr, right=dft, left_on="vehicle_id", right_on="tripinfo_id"
    )

    df_merge = df_merge[
        [
            "vehicle_id",
            "vehicle_depart",
            "tripinfo_arrival",
            "route_edges",
            "vehicle_fromTaz",
            "vehicle_toTaz",
            "od_pair",
        ]
    ]

    if is_synthetic:
        edge_col = "edge_id"
        dtd = pd.read_csv(path_additional[:-3] + "csv", sep=";")
        dtd.dropna(subset=[additonal_identifier], inplace=True)
        dtd[edge_col] = dtd[additonal_identifier].apply(lambda x: x.split("_")[1])
    else:
        edge_col = "det_id"
        dtd = pd.read_csv(path_true_count)
        # dtd_real_vs_sim = pd.read_csv(Paths['det_counts'])
        # dtd = dtd[dtd[edge_col].isin(dtd_real_vs_sim[edge_col])]

        dtd_macthing = pd.read_csv(path_match_detectors)
        dtd = dtd[dtd[edge_col].isin(dtd_macthing["det_id"])]

    od_pairs = dfod.od_pair.unique()
    num_detectors = len(dtd[edge_col].unique())

    intervals = range(t_start - t_warmup, t_end, time_interval)  # time_interval

    num_od_pairs = len(od_pairs)
    num_intervals = len(intervals)

    incidence_weight_array = np.zeros(
        (num_od_pairs * num_intervals, num_detectors * num_intervals), dtype="float32"
    )
    incidence_assignment_array = np.zeros(
        (num_od_pairs * num_intervals, num_detectors * num_intervals), dtype="float32"
    )

    df_merge["origin"] = df_merge["od_pair"].apply(lambda x: x.split("___")[0])

    for k, depr in tqdm(enumerate(intervals)):
        df_temp = df_merge[
            (df_merge.vehicle_depart >= depr)
            & (df_merge.vehicle_depart < (depr + time_interval))
        ]
        for i, od in enumerate(od_pairs):
            origin = od.split("___")[0]
            origin_trips = len(df_temp[df_temp.origin == origin])
            if origin_trips != 0:
                temp_od = df_temp[df_temp.od_pair == od]
                if len(temp_od) != 0:
                    for l, arrv in enumerate(intervals):
                        temp = temp_od[
                            (temp_od.tripinfo_arrival >= arrv)
                            & (temp_od.tripinfo_arrival < (arrv + time_interval))
                        ]
                        if len(temp) != 0:
                            edges = []
                            for route in temp.route_edges.astype("str"):  # .unique():
                                edges.extend(route.split(" "))
                            for q, detector in enumerate(dtd[edge_col].unique()):
                                new_edge = str(detector)
                                for e in edges:
                                    if new_edge == e:
                                        incidence_assignment_array[
                                            i + k * len(od_pairs), q + l * num_detectors
                                        ] += 1
                                        incidence_weight_array[
                                            i + k * len(od_pairs), q + l * num_detectors
                                        ] = (
                                            len(temp_od) / origin_trips
                                        )  # 1
                    incidence_assignment_array[i + k * len(od_pairs), :] /= len(
                        temp_od
                    )  # 1

    if threshold:
        # set a threshold
        if binary_rounding:
            incidence_weight_array[incidence_weight_array > threshold_value] = 1
            incidence_weight_array[incidence_weight_array <= threshold_value] = 0
        else:
            incidence_weight_array[incidence_weight_array <= threshold_value] = 0

    if do_plot == True:
        path_plot = (
            "../../resources/weight_incidence_"
            + scenario
            + "_"
            + str(threshold_value)
            + ".png"
        )
        fig, ax = plt.subplots(1, 1, figsize=(60, 60))
        plt.imshow(incidence_weight_array, cmap="hot", interpolation="nearest")
        plt.savefig(path_plot, dpi=300)
        plt.close("all")

        path_plot = (
            "../../resources/assignment_incidence_"
            + scenario
            + "_"
            + str(threshold_value)
            + ".png"
        )
        fig, ax = plt.subplots(1, 1, figsize=(60, 60))
        plt.imshow(incidence_assignment_array, cmap="hot", interpolation="nearest")
        plt.savefig(path_plot, dpi=300)
        plt.close("all")

    if do_save == True:
        S = scipy.sparse.csr_matrix(
            incidence_weight_array[
                :, int(t_warmup / (3600 * config["DEMAND_INTERVAL"])) * num_detectors :
            ]
        )
        with open(path_output_weight, "wb") as file:
            pickle.dump(S, file)
        S = scipy.sparse.csr_matrix(
            incidence_assignment_array[
                :, int(t_warmup / (3600 * config["DEMAND_INTERVAL"])) * num_detectors :
            ]
        )
        with open(path_output_assignment, "wb") as file:
            pickle.dump(S, file)

    return (
        incidence_weight_array[
            :, int(t_warmup / (3600 * config["DEMAND_INTERVAL"])) * num_detectors :
        ],
        incidence_assignment_array[
            :, int(t_warmup / (3600 * config["DEMAND_INTERVAL"])) * num_detectors :
        ],
    )


if __name__ == "__main__":
    print("Done")
