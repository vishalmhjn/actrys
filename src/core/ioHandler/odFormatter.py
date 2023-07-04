import pandas as pd
import numpy as np
from simHandler.scenarioGenerator import *

PATH_OD_TXT = (
    "../../" + SCENARIO + "/" + DEMAND_SOURCE + "/" + OD_FILE_IDENTIFIER + ".txt"
)
PATH_OD_TXT_NEW = "../../" + SCENARIO + "/temp/" + OD_FILE_IDENTIFIER + ".txt"


def OD_txt_to_dataframe(path=PATH_OD_TXT):
    """To convert the SUMO od matrices to dataframe to get the
    demand vector or decision variables"""

    od_array = []
    # print(od_array)
    for hour in TOD:
        path_od_demand = (
            path[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + DEMAND_INTERVAL))
            + ".txt"
        )

        df = pd.read_csv(path_od_demand, sep=" ").reset_index()

        demand_factor = float(df.iloc[3, 0])

        od_array.extend([i * demand_factor for i in df.iloc[4:, 2].to_list()])
    # print(np.array(df.iloc[4:,2]*demand_factor))
    return od_array


def OD_dataframe_to_txt(path_od_save, list_demand_flows, path_base_file=PATH_OD_TXT):
    for interval, hour in enumerate(TOD):
        # print(interval)
        path_save = (
            path_od_save[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + DEMAND_INTERVAL))
            + ".txt"
        )
        path_base = (
            path_base_file[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + DEMAND_INTERVAL))
            + ".txt"
        )
        text_file = open(path_save, "w")
        base_file = open(path_base, "r")

        counter = 0

        df = pd.read_csv(path_base, sep=" ").reset_index()

        demand_factor = float(df.iloc[3, 0])

        for counter, line in enumerate(base_file):
            # print(counter)
            if counter < 5:
                text_file.write(line)
                # print(line)
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
    # print(len(df))

    assert len(list_demand_flows) == len(TOD) * (len(df) - 4)


if __name__ == "__main__":
    l = OD_txt_to_dataframe(PATH_OD_TXT)
    print(len(l))
    OD_dataframe_to_txt(PATH_OD_TXT_NEW, l)
