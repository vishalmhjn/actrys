import pandas as pd
import numpy as np
from sim_handler.scenario_generator import config


def get_path_od_txt():
    """Get the path to the OD file."""
    return (
        "../../"
        + config["SCENARIO"]
        + "/"
        + config["DEMAND_SOURCE"]
        + "/"
        + config["OD_FILE_IDENTIFIER"]
        + ".txt"
    )


def get_path_od_txt_new():
    """Get the path to the new OD file."""
    return (
        "../../" + config["SCENARIO"] + "/temp/" + config["OD_FILE_IDENTIFIER"] + ".txt"
    )


def od_txt_to_dataframe(path):
    """Convert the SUMO OD matrices to a dataframe to get the demand vector or decision variables."""
    od_array = []
    for hour in config["TOD"]:
        path_od_demand = (
            path[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + config["DEMAND_INTERVAL"]))
            + ".txt"
        )

        df = pd.read_csv(path_od_demand, sep=" ").reset_index()
        demand_factor = float(df.iloc[3, 0])

        od_array.extend([i * demand_factor for i in df.iloc[4:, 2].to_list()])
    return od_array


def od_dataframe_to_txt(path_od_save, list_demand_flows, path_base_file):
    """Convert an OD dataframe back to a text file."""
    for interval, hour in enumerate(config["TOD"]):
        path_save = (
            path_od_save[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + config["DEMAND_INTERVAL"]))
            + ".txt"
        )
        path_base = (
            path_base_file[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + config["DEMAND_INTERVAL"]))
            + ".txt"
        )
        text_file = open(path_save, "w")
        base_file = open(path_base, "r")

        df = pd.read_csv(path_base, sep=" ").reset_index()
        demand_factor = float(df.iloc[3, 0])

        for counter, line in enumerate(base_file):
            if counter < 5:
                text_file.write(line)
            else:
                origin = line.split(" ")[0]
                destination = line.split(" ")[1]

                idx = interval * (len(df) - 5 + 1) + (counter - 5)

                flow = str(int(float(list_demand_flows[idx] / demand_factor)))
                if counter == len(df):
                    text_file.write(origin + " " + destination + " " + flow)
                else:
                    text_file.write(origin + " " + destination + " " + flow + "\n")

        text_file.close()
        base_file.close()
    assert len(list_demand_flows) == len(config["TOD"]) * (len(df) - 4)


if __name__ == "__main__":
    path_od_txt = get_path_od_txt()
    path_od_txt_new = get_path_od_txt_new()

    demand_flows = od_txt_to_dataframe(path_od_txt)
    print(len(demand_flows))

    od_dataframe_to_txt(path_od_txt_new, demand_flows, path_od_txt)
