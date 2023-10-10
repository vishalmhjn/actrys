import os

# Data Parameters
noise_param = int(os.environ.get("noise_param"))
bias_param = float(os.environ.get("bias_param"))

# SPSA Parameters
spsa_a = float(os.environ.get("spsa_a"))
spsa_c = float(os.environ.get("spsa_c"))

spsa_c_init = spsa_c
spsa_a_init = spsa_a

spsa_a_out_sim = float(os.environ.get("spsa_a_out_sim"))
spsa_c_out_sim = float(os.environ.get("spsa_c_out_sim"))

spsa_reps = int(os.environ.get("spsa_reps"))

n_iterations = int(os.environ.get("n_iterations"))
sim_in_iterations = int(os.environ.get("sim_in_iterations"))
sim_out_iterations = int(os.environ.get("sim_out_iterations"))

calibrate_supply = os.environ.get("calibrate_supply")
calibrate_demand = os.environ.get("calibrate_demand")
set_spa = os.environ.get("set_spa")

which_algo = os.environ.get("which_algo")

weight_counts = float(os.environ.get("weight_counts"))
weight_od = float(os.environ.get("weight_od"))
weight_speed = float(os.environ.get("weight_speed"))
bagging_run = int(os.environ.get("bagging_run"))

count_noise_param = int(os.environ.get("count_noise_param"))

# use correction heuristic
correction_heuristic = eval(os.environ.get("heuristic"))

spsa_autotune = eval(os.environ.get("auto_tune_spsa"))
momentum_beta = float(os.environ.get("momentum_beta"))


only_bias_correction = eval(os.environ.get("only_bias_correction"))
bias_correction_method = os.environ.get("bias_correction_method")

estimator = os.environ.get("estimator")
print("Using " + estimator + " for selecting the best fit")

if which_algo == "wspsa":
    wspsa_thrshold = float(os.environ.get("wspsa_threshold"))

# for decaying the a and c for simulation out-of-loop
learning_decay_factor = 1.05

# added noise in the true estimates to get the prior beliefs
noise_prior = 10

# exploration noise for bagging runs
exploration_noise = 0

# number of sequential calibration runs
n_sequential = 5

# number of bayesian initial exploration for spsa tuning
n_init_spsa_tune = 30

# number of bayesian iterations for spsa tuning
n_iterate_spsa_tune = 30

# number of bayesian initial exploration for supply
n_init_supply = 20

# number of bayesian iterations for supply
n_iterate_supply = 10
