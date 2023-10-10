import subprocess
import numpy as np
from shutil import copyfile
import os

from sim_handler.scenario_generator import *

PATH_ADDITIONAL = os.environ.get("PATH_ADDITIONAL")


def copy_additonal(PATH_ORIG_ADDITIONAL, path_temp):
    copyfile(PATH_ORIG_ADDITIONAL, path_temp)


def call_sumo(cmd_string):
    # print(cmd_string)
    subprocess.run(cmd_string + " --mesosim --no-warnings", shell=True)  # --verbose


def run_simulation(
    path_trips,
    path_routes,
    path_temp_additional,
    evaluation_run=True,
    routing_how="default",
    **kwargs
):
    """Copied from https://sumo.dlr.de/docs/Demand/Automatic_Routing.html
    --device.rerouting.probability <FLOAT>	-1	The probability for a vehicle to have a routing device (-1 is equivalent to 0 here)
    --device.rerouting.explicit <STRING>		Assign a device to named vehicles
    --device.rerouting.deterministic	false	The devices are set deterministic using a fraction of 1000 (with the defined probability)
    --device.rerouting.period <STRING>	0	The period with which the vehicle shall be rerouted
    --device.rerouting.pre-period <STRING>	60	The rerouting period before insertion/depart
    --device.rerouting.adaptation-interval <INT>	1	The interval for updating the edge weights.
    --device.rerouting.adaptation-weight <FLOAT>	0.0 (disabled)	The weight of prior edge weights for exponential averaging from [0, 1].
    --device.rerouting.adaptation-steps <INT>	180	The number of adaptation steps for averaging (enable for values > 0).
    --device.rerouting.with-taz	false	Use traffic assignment zones (TAZ/districts) as routing end points
    --device.rerouting.init-with-loaded-weights	false	Use option --weight-files for initializing the edge weights at simulation start
    --weights.priority-factor: Consider edge priorities in addition to travel times, weighted by factor
    """

    if routing_how == "reroute":
        # rerouting needs trips file or even route files as input

        routing_threads = kwargs.get("routing_threads", "7")
        rerouting_prob = str(float(kwargs.get("rerouting_prob", "0.5")))
        rerouting_period = str(int(kwargs.get("rerouting_period", "50")))
        rerouting_adaptation = str(int(kwargs.get("rerouting_adaptation", "16")))
        rerouting_adaptation_steps = str(
            int(kwargs.get("rerouting_adaptation_steps", "180"))
        )
        tls_tt_penalty = str(float(kwargs.get("tls_tt_penalty", "0")))
        meso_minor_penalty = str(int(kwargs.get("meso_minor_penalty", "0")))
        meso_tls_flow_penalty = str(float(kwargs.get("meso_tls_flow_penalty", "0")))
        priority_factor = str(float(kwargs.get("priority_factor", "0")))

        sim_string = (
            "sumo"
            + " -n "
            + PATH_NETWORK
            + " -r "
            + path_trips
            + " --additional-files "
            + path_temp_additional
            + " -b "
            + str(TOD[0] * 3600)
            + " -e "
            + str((1 + TOD[-1]) * 3600)
            + " --device.rerouting.probability "
            + rerouting_prob
            + " --device.rerouting.threads "
            + routing_threads
            + " --device.rerouting.period "
            + rerouting_period
            + " --device.rerouting.synchronize --no-step-log"
            + " --device.rerouting.adaptation-interval "
            + rerouting_adaptation
            + " --device.rerouting.adaptation-steps "
            + rerouting_adaptation_steps
            + " --meso-tls-flow-penalty "
            + meso_tls_flow_penalty
            + " --meso-junction-control true"
            + " --meso-tls-penalty "
            + tls_tt_penalty
            + " --meso-minor-penalty "
            + meso_minor_penalty
            + " --weights.priority-factor "
            + priority_factor
        )
        # " --device.fcd.probability 0.25" +\
        # " --fcd-output " + str(path_trips[:-15])+"fcd_dump" +\
        # " --device.fcd.period 10"
        #  " --device.rerouting.adaptation-weight 0.3"+\

        if evaluation_run == True:
            sim_string += (
                " --summary "
                + "../../"
                + SCENARIO
                + "/"
                + temp_scenario_name
                + "/summary"
                + " --tripinfo-output "
                + "../../"
                + SCENARIO
                + "/"
                + temp_scenario_name
                + "/trips_output"
                + " --vehroute-output "
                + "../../"
                + SCENARIO
                + "/"
                + temp_scenario_name
                + "/routes.rou.xml"
            )
        # print(sim_string)
        call_sumo(cmd_string=sim_string)

    else:
        # duarouter or due based assignment will need rourtes files as input which
        # are again the output of the duarouter or duaiterate based assignment
        call_sumo(
            cmd_string="sumo"
            + " -n "
            + PATH_NETWORK
            + " -r "
            + path_routes
            + " --additional-files "
            + path_temp_additional
            + " -b "
            + str(TOD[0] * 3600)
            + " -e "
            + str((1 + TOD[-1]) * 3600)
        )


if __name__ == "__main__":
    print(TOD)
    path_temp_additional = (
        "../../" + SCENARIO + "/" + temp_scenario_name + "/additional.add.xml"
    )

    PATH_TRIPS, PATH_ROUTES = file_manager(temp_scenario_name)
    # copy_additonal(PATH_ORIG_ADDITIONAL = PATH_ADDITIONAL, path_temp=path_temp_additional)

    print(PATH_TRIPS)
    print(PATH_ROUTES)
    run_simulation(
        PATH_TRIPS, PATH_ROUTES, path_temp_additional, routing_how="reroute", threads=16
    )
