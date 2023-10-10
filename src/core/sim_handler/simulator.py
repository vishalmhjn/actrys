import subprocess
import os
from shutil import copyfile
from sim_handler.scenario_generator import config, TOD

# Constants for command-line options
ROUTING_HOW_REROUTE = "reroute"
ROUTING_HOW_DUR = "duarouter"

PATH_ADDITIONAL = os.environ.get("PATH_ADDITIONAL")


def call_sumo(cmd_string):
    # print(cmd_string)
    subprocess.run(cmd_string + " --mesosim --no-warnings", shell=True)  # --verbose


def copy_additonal(PATH_ORIG_ADDITIONAL, path_temp):
    copyfile(PATH_ORIG_ADDITIONAL, path_temp)


def run_simulation(
    path_trips,
    path_routes,
    path_temp_additional,
    evaluation_run=True,
    routing_how=ROUTING_HOW_REROUTE,
    routing_threads=16,
    rerouting_prob=0.5,
    rerouting_period=50,
    rerouting_adaptation=16,
    rerouting_adaptation_steps=180,
    tls_tt_penalty=0,
    meso_minor_penalty=0,
    meso_tls_flow_penalty=0,
    priority_factor=0,
):
    cmd_string = (
        f'sumo -n {config["PATH_NETWORK"]} -r {path_trips} --additional-files {path_temp_additional} '
        f"-b {TOD[0] * 3600} -e {(1 + TOD[-1]) * 3600} --no-step-log"
    )

    if routing_how == ROUTING_HOW_REROUTE:
        cmd_string += (
            f" --device.rerouting.probability {rerouting_prob} --device.rerouting.threads {routing_threads} "
            f"--device.rerouting.period {int(rerouting_period)} --device.rerouting.synchronize "
            f"--device.rerouting.adaptation-interval {int(rerouting_adaptation)} "
            f"--device.rerouting.adaptation-steps {int(rerouting_adaptation_steps)} "
            f"--meso-tls-flow-penalty {meso_tls_flow_penalty} --meso-junction-control true "
            f"--meso-tls-penalty {tls_tt_penalty} --meso-minor-penalty {int(meso_minor_penalty)} "
            f"--weights.priority-factor {priority_factor}"
        )

        if evaluation_run:
            cmd_string += (
                f' --summary ../../{config["SCENARIO"]}/{config["temp_scenario_name"]}/summary '
                f'--tripinfo-output ../../{config["SCENARIO"]}/{config["temp_scenario_name"]}/trips_output '
                f'--vehroute-output ../../{config["SCENARIO"]}/{config["temp_scenario_name"]}/routes.rou.xml'
            )

    else:
        cmd_string = (
            f'sumo -n {config["PATH_NETWORK"]} -r {path_routes} --additional-files {path_temp_additional} '
            f"-b {TOD[0] * 3600} -e {(1 + TOD[-1]) * 3600}"
        )

    subprocess.run(cmd_string + " --mesosim --no-warnings", shell=True, check=True)


if __name__ == "__main__":
    print(TOD)
    path_temp_additional = (
        f'../../{config["SCENARIO"]}/{config["temp_scenario_name"]}/additional.add.xml'
    )

    # Define PATH_TRIPS and PATH_ROUTES here
    PATH_TRIPS = "your_trips_path"
    PATH_ROUTES = "your_routes_path"

    print(PATH_TRIPS)
    print(PATH_ROUTES)
    run_simulation(
        PATH_TRIPS,
        PATH_ROUTES,
        path_temp_additional,
        routing_how=ROUTING_HOW_REROUTE,
        routing_threads=16,
    )
