"""Wrapper file
"""
import subprocess

SPSA_A = 0.01
SPSA_C = 0.01
SPSA_REPS = 1
NUM_ITERATIONS = 100
ONLY_BIAS_CORRECTION = False
CORRECTION_HEURISTIC_METHOD = "weighted"

BIAS = 0.6
NOISE = 60
MOMENTUM = 0.7

WEIGHT_PROFILES = (1, 0, 0)
COUNT_NOISE = 0
AUTO_TUNE_SPSA = False
BAGGING = 10
CORRECTION_HEURISTIC = True

SCENARIO = "test"

subprocess.run(
    f"python ../core/syntheticCalibrator.py \
                {CORRECTION_HEURISTIC_METHOD}4_50_{BIAS}_{NOISE}_{WEIGHT_PROFILES[0]}_{WEIGHT_PROFILES[1]}_{WEIGHT_PROFILES[2]}_{COUNT_NOISE}_spsareps_{SPSA_REPS} True \
                {NOISE} {BIAS} \
                {SPSA_A} {SPSA_C} \
                {SPSA_REPS} {NUM_ITERATIONS} \
                wspsa {SCENARIO} \
                {BAGGING} \
                {MOMENTUM} \
                {COUNT_NOISE} \
                {WEIGHT_PROFILES[0]} \
                {WEIGHT_PROFILES[1]} \
                {WEIGHT_PROFILES[2]} \
                {AUTO_TUNE_SPSA} \
                {CORRECTION_HEURISTIC} \
                {ONLY_BIAS_CORRECTION} \
                {CORRECTION_HEURISTIC_METHOD}",
    shell=True,
    check=True,
)
