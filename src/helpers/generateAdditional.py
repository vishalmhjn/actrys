# coding=utf-8
import pandas as pd
import helpers.readClass as readClass
import helpers.sumoClass as sumoClass
from tqdm import tqdm
import sys


def create_sumo_detector_file(scenario):
    """
    Create a SUMO detector file based on the specified scenario.

    Parameters:
    - scenario (str): The scenario name ("munich" or "paris").
    """
    DETECTOR_POS = -20

    if scenario == "munich":
        NETWORK_CSV = "../scenario_munich/network.net.csv"
        DETECTOR_MAPPING = "../sharedstreets/detector_sumo_mapping_munich.csv"
        FILE = "../data/BAST/day_counts_190605.csv"
        TIME_COL = "Stunde"
        PATH_NETWORK_CSV = NETWORK_CSV
        PATH_EDGE_CSV = "../scenario_munich/network.edg.csv"
        SUMO_COLUMN = "sumo_id"
        DETECTOR_COLUMN = "munich_id"
        DETECTOR_LATITUDE = "detector_lat"
        DETECTOR_LONGITUDE = "detector_lon"
        OUTPUT = "../scenario_munich/addition_shared_streets_bast.add.xml"
    else:
        raise ValueError("Invalid scenario name.")

    detector_locations = pd.read_csv(DETECTOR_MAPPING)

    if scenario == "paris":
        # Only add detectors where the sensors recorded values during the filter time
        temp = detector_locations[
            detector_locations[DETECTOR_COLUMN].isin(list(df.iu_ac.unique()))
        ]
    else:
        temp = detector_locations

    sumo_obj = sumoClass.Sumo_Network(PATH_NETWORK_CSV, PATH_EDGE_CSV)
    df_edge_length, df_edge_lanes = sumo_obj.write_edges()

    a = """<additional>"""
    b = ""
    check_list_sumo_edge = []
    check_list_id = []

    if scenario == "munich":
        temp[DETECTOR_COLUMN] = temp.apply(
            lambda x: str(x[DETECTOR_COLUMN]) + "_" + str(int(x.direction)), axis=1
        )
        temp.drop_duplicates(subset=[DETECTOR_COLUMN], inplace=True)

    for i, j in tqdm(
        enumerate(
            zip(
                temp[SUMO_COLUMN],
                temp[DETECTOR_COLUMN],
                temp[DETECTOR_LATITUDE],
                temp[DETECTOR_LONGITUDE],
            )
        )
    ):
        try:
            len_lane = float(
                df_edge_length[df_edge_length.edge_id == j[0]]["lane_length"]
            )
            if j[0] in check_list_sumo_edge:
                pass
            elif j[1] in check_list_id:
                pass
            elif len_lane < DETECTOR_POS:
                pass
            else:
                temp_len = len(df_edge_lanes[df_edge_lanes.edge_id == j[0]])
                if temp_len == 1:
                    n_lanes = int(
                        df_edge_lanes[df_edge_lanes.edge_id == j[0]]["edge_numLanes"]
                    )
                else:
                    n_lanes = list(
                        (
                            df_edge_lanes[df_edge_lanes.edge_id == j[0]][
                                "edge_numLanes"
                            ]
                        ).astype(int)
                    )[0]
                for k in range(0, n_lanes):
                    b = (
                        b
                        + """<e1Detector file="out.xml" freq="900.0" id="""
                        + '"'
                        + "e1Detector_"
                        + j[0]
                        + "_"
                        + str(k)
                        + '"'
                        + " lane="
                        + '"'
                        + j[0]
                        + "_"
                        + str(k)
                        + '"'
                        + """ pos="""
                        + '"'
                        + str(DETECTOR_POS)
                        + '"'
                        + """ name="""
                        + '"'
                        + str(j[1])
                        + '"'
                        + """><param key="C" value="""
                        + '"'
                        + str(j[2])
                        + ", "
                        + str(j[3])
                        + '"'
                        + """/></e1Detector>"""
                    )
                check_list_sumo_edge.append(j[0])
                check_list_id.append(j[1])
        except Exception as e:
            print(e)
            pass

    c = """</additional>"""
    file_text = a + b + c

    with open(OUTPUT, "w") as f:
        f.write(file_text)
        f.close()


if __name__ == "__main__":
    # Example usage for creating SUMO detector file
    scenario = sys.argv[1]
    create_sumo_detector_file(scenario)
