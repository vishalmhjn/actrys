import subprocess
from tqdm import tqdm

spsa_a = 0.00001
spsa_c = 0.5
bias = 0.6
noise = 20
spsa_reps = 1
num_iterations = 1
wspsa_threshold = .01
spsa_a_out = 1
spsa_c_out = 1
estimator = "wmape"
sim_in_iterations = 10
sim_out_iterations = 1
demand_interval = 3600
bagging_runs = 10

momentum_beta=0.7
auto_tune_spsa = True
bagging = bagging_runs
correction_heuristic = True
sim_in_iterations = 5
only_bias_correction=False

for i, (weight_profiles, count_noise) in enumerate(zip([
                                                        (10,0,0),\
                                                        ],
                                                        [
                                                        0,
                                                        ])):
        subprocess.run(f"sh munichmr.sh \
                        free_{demand_interval}_{bias}_{noise}_{weight_profiles[0]}_{weight_profiles[1]}_{weight_profiles[2]}_{count_noise} \
                        True True \
			            {noise} {bias} \
                        {spsa_a} {spsa_c} \
                        {spsa_a_out} {spsa_c_out} \
                        {spsa_reps} \
                        True False \
                        {num_iterations} {sim_in_iterations} {sim_out_iterations} \
                        wspsa {wspsa_threshold} \
                        False \
                        {estimator} \
                        munichmr/ \
                        {weight_profiles[0]} \
                        {weight_profiles[1]} \
                        {weight_profiles[2]} \
                        {bagging} \
                        {count_noise} \
                        {correction_heuristic} \
                        {auto_tune_spsa} \
                        {momentum_beta} \
                        {demand_interval} \
                        {only_bias_correction}",
			            shell=True)