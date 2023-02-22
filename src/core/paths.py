from scenario_generator import file_manager
import os
from scenario_generator import OD_FILE_IDENTIFIER
from simulator import PATH_ADDITIONAL

# folder paths
temp_folder_name = os.environ.get("temp_folder_name")
run_scenario = os.environ.get("run_scenario")

SCENARIO = os.environ.get("SCENARIO")
DEMAND_SOURCE = os.environ.get("DEMAND_SOURCE")
PATH_DEMAND = os.environ.get("PATH_DEMAND")
PATH_REAL_COUNT = os.environ.get("PATH_REAL_COUNT")
PATH_REAL_SPEED = os.environ.get("PATH_REAL_SPEED")
DEMAND_DURATION = int(os.environ.get("DEMAND_INTERVAL"))

pre_string = "../../"+SCENARIO+"/"+temp_folder_name
path_temp_demand = pre_string +"/"+OD_FILE_IDENTIFIER+"_n.txt"
path_temp_additional = pre_string+"/"+PATH_ADDITIONAL.split("/")[-1]#additional.add.xml"
path_simulation_counts = pre_string+"/out.xml" 
path_simulation_speeds = pre_string+"/edge_data_"+str(DEMAND_DURATION)
path_trips, path_routes = file_manager(temp_folder_name)

path_trip_summary = pre_string+"/trips_output"
path_od_format_file = PATH_DEMAND+"/od_list.txt"