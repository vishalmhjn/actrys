import numpy as np
import subprocess
import os
#### NOTES
# TAZ file creation: Manual
# OD fle creation: Manual

SUMO_PATH = os.getenv("SUMO_HOME")
# print(SUMO_PATH)

# Scenario variables
SCENARIO = os.environ.get("SCENARIO") #"sioux_falls" #"scenario_munich"
OD_FILE_IDENTIFIER = os.environ.get("OD_FILE_IDENTIFIER") #"SF" #"MR"
DEMAND_SOURCE = os.environ.get("DEMAND_SOURCE")#"syn_demand"
temp_scenario_name = os.environ.get("temp_scenario_name") #"temp"

# Static Files
PATH_ZONE = os.environ.get("PATH_ZONE") #"../../"+SCENARIO+"/taZes.taz.xml" #taZes5.taz.xml" 
PATH_DEMAND = os.environ.get("PATH_DEMAND") #"../../"+SCENARIO+"/"+DEMAND_SOURCE+"/"+OD_FILE_IDENTIFIER+".txt"
PATH_NETWORK = os.environ.get("PATH_NETWORK") #"../../"+SCENARIO+"/network.net.xml"

DEMAND_INTERVAL = int(os.environ.get("DEMAND_INTERVAL"))/3600

TOD_START= int(float(os.environ.get("TOD_START")))
TOD_END= int(float(os.environ.get("TOD_END")))

# Warm period will depend on how the previous ODs effect the current counts
WARM_UP_PERIOD = 2
COOL_DOWN_PERIOD = 0

TOD = np.arange(max(TOD_START-WARM_UP_PERIOD, 0), min(24, TOD_END+COOL_DOWN_PERIOD), DEMAND_INTERVAL)

def create_trips(cmd_string):
    '''This function uses the demand defined in OD Matrix and 
    creates trips'''
    subprocess.run(cmd_string, shell=True)

def call_duarouter(cmd_string):
    '''This function generates routes from Trips by 
    simply calling DuaRouter'''
    subprocess.run(cmd_string, shell=True)

def call_sumo_fast(cmd_string):
    '''This function generates routes from Trips by 
    using rerouting modules'''
    subprocess.run(cmd_string, shell=True)

def create_scenario(path_trips, 
                    path_routes, 
                    path_demand = PATH_DEMAND):
    '''This is a two step process, it first calls
    od2trips to creates trips. The created trips are
    usede to generate routes using duaroutes'''
    path_od_demand = ""

    for hour in TOD:
        # print(TOD)
        path_od_demand += path_demand[:-4] +\
                            "_" + str(float(hour)) + \
                            "_" + str(float(hour+DEMAND_INTERVAL)) +\
                            ".txt" + ","
    
    path_od_demand = path_od_demand[:-1]

    create_trips(cmd_string="od2trips"+\
                        " -n "+	PATH_ZONE +\
                        " -d "+ path_od_demand +\
                        " -o " + path_trips)

    call_duarouter(cmd_string="duarouter"+\
                                " -n "+ PATH_NETWORK +\
                                " -r "+ path_trips +\
                                " --ignore-errors --no-warnings --no-step-log"+\
                                " -o "+ path_routes)

def trip_validator(path_trips, path_routes):
    '''This is a trip validator for deleting trips
    which cannot be pefromed due to network connectivity'''

    # path compatible with sumo1.9.0
    subprocess.run("python "+\
                     SUMO_PATH + "/tools/purgatory/route2trips.py "+path_routes+\
                    " > "+ path_trips, shell=True)



def file_manager(temp_folder):
    try: 
        os.mkdir("../../"+SCENARIO+"/"+temp_folder)
    except FileExistsError:
        pass
        # Dynamic Files
    PATH_TRIPS = "../../"+SCENARIO+"/"+ temp_folder+"/trips.trips.xml"
    PATH_ROUTES = "../../"+SCENARIO+"/"+ temp_folder+"/routes.rou.xml"
    return PATH_TRIPS, PATH_ROUTES

if __name__ == "__main__":

    PATH_TRIPS, PATH_ROUTES = file_manager(temp_scenario_name)
    create_scenario(path_trips=PATH_TRIPS, path_routes=PATH_ROUTES)
    trip_validator(PATH_TRIPS, PATH_ROUTES)