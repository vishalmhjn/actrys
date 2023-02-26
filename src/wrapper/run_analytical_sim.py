import subprocess
from tqdm import tqdm
import numpy as np
spsa_a = 0.000001
spsa_c = .01
spsa_reps = 1
num_iterations = 100
only_bias_correction=False
correction_heuristic_method = "weighted"

bias = 0.8
noise = 20
momentum = 0.7

weight_profiles = (1,0,0)
count_noise = 0
auto_tune_spsa = False
only_bias_correction=False
bagging = 10
spsa_reps = 1
correction_heuristic = True

scenario = "test"

subprocess.run(f"python ../core/synthetic_calibrator.py \
                {correction_heuristic_method}_50_{bias}_{noise}_{weight_profiles[0]}_{weight_profiles[1]}_{weight_profiles[2]}_{count_noise}_spsareps_{spsa_reps} True \
                {noise} {bias} \
                {spsa_a} {spsa_c} \
                {spsa_reps} {num_iterations} \
                wspsa {scenario} \
                {bagging} \
                {momentum} \
                {count_noise} \
                {weight_profiles[0]} \
                {weight_profiles[1]} \
                {weight_profiles[2]} \
                {auto_tune_spsa} \
                {correction_heuristic} \
                {only_bias_correction} \
                {correction_heuristic_method}",
                shell=True)