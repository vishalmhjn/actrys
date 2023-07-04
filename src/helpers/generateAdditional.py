# coding=utf-8
import pandas as pd
import helpers.readClass as readClass
import helpers.sumoClass as sumoClass
from tqdm import tqdm
import sys

# To create sumo detector file : addition.add.xml
# TO DO: Right now it only considers one to one mapping,
# but one edge could be (as is the case) might be connected to more
# than one detector. This is not needed since same traffic should be on a
# detector but check for the anamolies

if __name__ == "__main__":
    scenario = sys.argv[1]

    # DETECTOR_MAPPING = "../data/best_detector_locations.csv"

    DETECTOR_POS = -20

    if scenario == "munich":
        NETWORK_CSV = "../scenario_munich/network.net.csv"
        DETECTOR_MAPPING = "../sharedstreets/detector_sumo_mapping_munich.csv"
        FILE = "../data/BAST/day_counts_190605.csv"
        # FILTER = "2020-10-07" # 19:00:00"
        TIME_COL = "Stunde"
        PATH_NETWORK_CSV = NETWORK_CSV
        PATH_EDGE_CSV = "../scenario_munich/network.edg.csv"
        SUMO_COLUMN = "sumo_id"
        DETECTOR_COLUMN = "munich_id"
        DETECTOR_LATITUDE = "detector_lat"
        DETECTOR_LONGITUDE = "detector_lon"

        paris = pd.read_csv(FILE)
        # df = paris.read_file(FILTER, time_col=TIME_COL)
        # df.dropna(inplace=True)
        # df.reset_index(inplace=True)
        # filtered output for only keeping sensors where data is available
        OUTPUT = "../scenario_munich/addition_shared_streets_bast.add.xml"
        # Renaming the columns of the current time data as they are
        # different from the archived data

    else:
        raise ValueError

    detector_locations = pd.read_csv(DETECTOR_MAPPING)

    if scenario == "paris":
        # we only add detectors where the sensors recorded values during the filter time
        # because there is no fun in having a detector with no real-time data, but in case where
        # those detectors have some other useful information like density, it might make sense to keep
        # them
        temp = detector_locations[
            detector_locations[DETECTOR_COLUMN].isin(list(df.iu_ac.unique()))
        ]
        # temp = detector_locations

    else:
        temp = detector_locations

    # we need edge file, to know the number of lanes in addition.add.xml
    # get the length and num lanes of the edges
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
        # print(df_edge_length[df_edge_length.edge_id==j[0]])
        try:
            # try except added to cater the cases when some of the detector links are not in the matched
            # sumo network. but there are many links for one detector.
            len_lane = float(
                df_edge_length[df_edge_length.edge_id == j[0]]["lane_length"]
            )
            if j[0] in check_list_sumo_edge:
                # only one detector on one sumo edge
                pass
            elif j[1] in check_list_id:
                # only one detector with one name
                pass
            ## insufficient length of the lane for detector
            elif len_lane < DETECTOR_POS:
                pass
            else:
                # print(len_lane)
                # pass
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
