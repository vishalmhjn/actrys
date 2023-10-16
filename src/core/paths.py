import os
from sim_handler.scenario_generator import file_manager
from sim_handler.scenario_generator import config
from sim_handler.simulator import PATH_ADDITIONAL

# folder paths
temp_folder_name = os.environ.get("temp_folder_name")
run_scenario = os.environ.get("run_scenario")

SCENARIO = os.environ.get("SCENARIO")
DEMAND_SOURCE = os.environ.get("DEMAND_SOURCE")
PATH_DEMAND = os.environ.get("PATH_DEMAND")
PATH_REAL_COUNT = os.environ.get("PATH_REAL_COUNT")
PATH_REAL_SPEED = os.environ.get("PATH_REAL_SPEED")
DEMAND_DURATION = int(os.environ.get("DEMAND_INTERVAL"))

FILE_MATCH_DETECTORS = os.environ.get("FILE_MATCH_DETECTORS")
FILE_REAL_COUNTS = os.environ.get("FILE_REAL_COUNTS")
FILE_REAL_SPEEDS = os.environ.get("FILE_REAL_SPEEDS")

pre_string = "../../" + SCENARIO + "/" + temp_folder_name
path_temp_demand = pre_string + "/" + config["OD_FILE_IDENTIFIER"] + "_n.txt"
path_temp_additional = pre_string + "/" + PATH_ADDITIONAL.split("/")[-1]
path_simulation_counts = pre_string + "/out.xml"
path_simulation_speeds = pre_string + "/edge_data_" + str(DEMAND_DURATION)
path_trips, path_routes = file_manager(temp_folder_name)

path_trip_summary = pre_string + "/trips_output"
path_od_format_file = PATH_DEMAND + "/od_list.txt"
