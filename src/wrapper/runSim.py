"""Wrapper file
"""
import subprocess
import sys

SCENARIO = sys.argv[1]

BIAS = 0.6
NOISE = 20

SPSA_A = 0.00001
SPSA_C = 0.01
SPSA_A_OUT = 1
SPSA_C_OUT = 1
SPSA_REPS = 1
NUM_ITERATIONS = 1
SIM_OUT_ITERATIONS = 1
SIM_IN_ITERATIONS = 5
WSPSA_THRESHOLD = 0.01

ESTIMATOR = "wmape"

BIAS_CORRECTION_METHOD = "naive"

DEMAND_INTERVAL = 3600

CALIBRATE_DEMAND = True
CALIBRATE_SUPPLY = True
SET_SPA = False

BAGGING_RUNS = 3
MOMENTUM_BETA = 0.3
AUTO_TUNE_SPSA = True
BAGGING = BAGGING_RUNS
CORRECTION_HEURISTIC = True
ONLY_BIAS_CORRECTION = False
BIAS_CORRECTION_TYPE = BIAS_CORRECTION_METHOD

for i, (WEIGHT_PROFILES, COUNT_NOISE) in enumerate(
    zip(
        [(10, 0, 0)],
        [
            0,
        ],
    )
):
    subprocess.run(
        f"sh wrapper.sh \
                    free_{DEMAND_INTERVAL}_{BIAS}_{NOISE}_{WEIGHT_PROFILES[0]}_{WEIGHT_PROFILES[1]}_{WEIGHT_PROFILES[2]}_{COUNT_NOISE}_sequential_{BIAS_CORRECTION_TYPE}_2a \
                    True True \
                    {NOISE} {BIAS} \
                    {SPSA_A} {SPSA_C} \
                    {SPSA_A_OUT} {SPSA_C_OUT} \
                    {SPSA_REPS} \
                    True False \
                    {NUM_ITERATIONS} {SIM_IN_ITERATIONS} {SIM_OUT_ITERATIONS} \
                    wspsa {WSPSA_THRESHOLD} \
                    {CALIBRATE_SUPPLY} {CALIBRATE_DEMAND} \
                    {SET_SPA} \
                    {ESTIMATOR} \
                    {SCENARIO} \
                    {WEIGHT_PROFILES[0]} \
                    {WEIGHT_PROFILES[1]} \
                    {WEIGHT_PROFILES[2]} \
                    {BAGGING} \
                    {COUNT_NOISE} \
                    {CORRECTION_HEURISTIC} \
                    {AUTO_TUNE_SPSA} \
                    {MOMENTUM_BETA} \
                    {DEMAND_INTERVAL} \
                    {ONLY_BIAS_CORRECTION} \
                    {BIAS_CORRECTION_TYPE}",
        shell=True,
        check=True,
    )
