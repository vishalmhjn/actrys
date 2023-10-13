import pandas as pd
from sim_handler.scenario_generator import config


def get_path_od_txt():
    """
    Get the path to the OD file.
    """
    return f"../../{config['SCENARIO']}/{config['DEMAND_SOURCE']}/{config['OD_FILE_IDENTIFIER']}.txt"


def od_txt_to_dataframe(path):
    """
    Convert the SUMO OD matrices to a dataframe to get the demand vector or decision variables.

    Parameters:
    - path (str): Path to the SUMO OD matrices file.

    Returns:
    - list: List of demand values.
    """

    od_array = []

    for hour in config["TOD"]:
        path_od_demand = (
            f"{path[:-4]}_{float(hour)}_{float(hour + config['DEMAND_INTERVAL'])}.txt"
        )

        df = pd.read_csv(path_od_demand, sep=" ").reset_index()
        demand_factor = float(df.iloc[3, 0])

        od_array.extend([i * demand_factor for i in df.iloc[4:, 2].to_list()])

    return od_array


def od_dataframe_to_txt(path_od_save, list_demand_flows, path_base_file):
    """
    Convert an OD dataframe back to a text file.

    Parameters:
    - path_od_save (str): Path to save the generated OD files.
    - list_demand_flows (list): List of demand flows.
    - path_base_file (str): Path to the base OD file.

    Raises:
    - AssertionError: Raised if the length of `list_demand_flows` does not match the expected length.

    """
    for interval, hour in enumerate(config["TOD"]):
        path_save = f"{path_od_save[:-4]}_{float(hour)}_{float(hour + config['DEMAND_INTERVAL'])}.txt"
        path_base = f"{path_base_file[:-4]}_{float(hour)}_{float(hour + config['DEMAND_INTERVAL'])}.txt"
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
                    text_file.write(f"{origin} {destination} {flow}")
                else:
                    text_file.write(f"{origin} {destination} {flow}\n")

        text_file.close()
        base_file.close()
    assert len(list_demand_flows) == len(config["TOD"]) * (len(df) - 4)


if __name__ == "__main__":
    path_od_txt = get_path_od_txt()

    demand_flows = od_txt_to_dataframe(path_od_txt)
    print(len(demand_flows))

    od_dataframe_to_txt(path_od_txt, demand_flows, path_od_txt)
