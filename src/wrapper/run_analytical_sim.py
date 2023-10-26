"""Wrapper file
"""
import subprocess

SPSA_A = 3
SPSA_C = 0.01
SPSA_REPS = 1
NUM_ITERATIONS = 30
ONLY_BIAS_CORRECTION = False
CORRECTION_HEURISTIC_METHOD = "weighted"

BIAS = 0.6
NOISE = 50
MOMENTUM = 0.7

WEIGHT_PROFILES = (0.5, 0.5, 0)
COUNT_NOISE = 0
AUTO_TUNE_SPSA = False
BAGGING = 10
CORRECTION_HEURISTIC = False

SCENARIO = "test"

subprocess.run(
    f"python ../core/synthetic_calibrator.py \
                ta{CORRECTION_HEURISTIC_METHOD}_{CORRECTION_HEURISTIC}_{BIAS}_{NOISE}_{WEIGHT_PROFILES[0]}_{WEIGHT_PROFILES[1]}_{WEIGHT_PROFILES[2]}_{COUNT_NOISE}_spsareps_{SPSA_REPS} True \
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
