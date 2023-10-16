import subprocess
import sys


def run_calibration_scenario(
    scenario,
    bias=0.7,
    noise=0,
    synthetic_counts=True,
    spsa_a=0.1,
    spsa_c=0.1,
    spsa_a_out=1,
    spsa_c_out=1,
    spsa_reps=1,
    num_iterations=1,
    sim_out_iterations=1,
    sim_in_iterations=5,
    wspsa_threshold=0.01,
    estimator="wmape",
    bias_correction_method="naive",
    demand_interval=3600,
    calibrate_demand=False,
    calibrate_supply=True,
    set_spa=False,
    bagging_runs=3,
    momentum_beta=0.3,
    auto_tune_spsa=True,
    correction_heuristic=True,
    only_bias_correction=False,
    weight_profiles=[(0.9, 0.1, 0)],
    count_noise=[0],
):
    for i, (weight_profile, count_noise_val) in enumerate(
        zip(weight_profiles, count_noise)
    ):
        subprocess.run(
            f"sh wrapper.sh \
            free_{demand_interval}_{bias}_{noise}_{weight_profile[0]}_{weight_profile[1]}_{weight_profile[2]}_{count_noise_val}_sequential_{bias_correction_method}_2a \
            True {synthetic_counts} \
            {noise} {bias} \
            {spsa_a} {spsa_c} \
            {spsa_a_out} {spsa_c_out} \
            {spsa_reps} \
            True False \
            {num_iterations} {sim_in_iterations} {sim_out_iterations} \
            wspsa {wspsa_threshold} \
            {calibrate_supply} {calibrate_demand} \
            {set_spa} \
            {estimator} \
            {scenario} \
            {weight_profile[0]} \
            {weight_profile[1]} \
            {weight_profile[2]} \
            {bagging_runs} \
            {count_noise_val} \
            {correction_heuristic} \
            {auto_tune_spsa} \
            {momentum_beta} \
            {demand_interval} \
            {only_bias_correction} \
            {bias_correction_method}",
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    SCENARIO = sys.argv[1]

    # Define your default parameters here
    DEFAULT_PARAMS = {
        "scenario": SCENARIO,
        "bias": 1,
        "noise": 10,
        "synthetic_counts": True,
        "calibrate_demand": True,
        "calibrate_supply": False
        # Add other default parameters here
    }

    # You can override default parameters as needed
    run_calibration_scenario(**DEFAULT_PARAMS)
