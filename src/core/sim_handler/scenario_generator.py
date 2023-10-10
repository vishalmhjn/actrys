import numpy as np
import subprocess
import os

# Configuration
config = {
    "SUMO_PATH": os.getenv("SUMO_HOME"),
    "SCENARIO": os.environ.get("SCENARIO"),
    "OD_FILE_IDENTIFIER": os.environ.get("OD_FILE_IDENTIFIER"),
    "DEMAND_SOURCE": os.environ.get("DEMAND_SOURCE"),
    "temp_scenario_name": os.environ.get("temp_scenario_name"),
    "PATH_ZONE": os.environ.get("PATH_ZONE"),
    "PATH_DEMAND": os.environ.get("PATH_DEMAND"),
    "PATH_NETWORK": os.environ.get("PATH_NETWORK"),
    "DEMAND_INTERVAL": int(os.environ.get("DEMAND_INTERVAL")) / 3600,
    "TOD_START": int(float(os.environ.get("TOD_START"))),
    "TOD_END": int(float(os.environ.get("TOD_END"))),
    "WARM_UP_PERIOD": 2,
    "COOL_DOWN_PERIOD": 0,
}

TOD = np.arange(
    max(config["TOD_START"] - config["WARM_UP_PERIOD"], 0),
    min(24, config["TOD_END"] + config["COOL_DOWN_PERIOD"]),
    config["DEMAND_INTERVAL"],
)

config["TOD"] = TOD


def run_command(cmd_string):
    subprocess.run(cmd_string, shell=True)


def create_trips(cmd_string):
    run_command(cmd_string)


def call_duarouter(cmd_string):
    run_command(cmd_string)


def create_scenario(path_trips, path_routes, path_demand=config["PATH_DEMAND"]):
    path_od_demand = ""

    for hour in TOD:
        path_od_demand += (
            path_demand[:-4]
            + "_"
            + str(float(hour))
            + "_"
            + str(float(hour + config["DEMAND_INTERVAL"]))
            + ".txt"
            + ","
        )

    path_od_demand = path_od_demand[:-1]

    create_trips(
        cmd_string="od2trips"
        + f" -n {config['PATH_ZONE']} -d {path_od_demand} -o {path_trips}"
    )

    call_duarouter(
        cmd_string="duarouter"
        + f" -n {config['PATH_NETWORK']} -r {path_trips} --ignore-errors --no-warnings --no-step-log -o {path_routes}"
    )


def trip_validator(path_trips, path_routes):
    run_command(
        f"python {config['SUMO_PATH']}/tools/purgatory/route2trips.py {path_routes} > {path_trips}"
    )


def file_manager(temp_folder):
    try:
        os.mkdir(f"../../{config['SCENARIO']}/{temp_folder}")
    except FileExistsError:
        pass

    PATH_TRIPS = f"../../{config['SCENARIO']}/{temp_folder}/trips.trips.xml"
    PATH_ROUTES = f"../../{config['SCENARIO']}/{temp_folder}/routes.rou.xml"
    return PATH_TRIPS, PATH_ROUTES


if __name__ == "__main__":
    PATH_TRIPS, PATH_ROUTES = file_manager(config["temp_scenario_name"])
    create_scenario(path_trips=PATH_TRIPS, path_routes=PATH_ROUTES)
    trip_validator(PATH_TRIPS, PATH_ROUTES)
