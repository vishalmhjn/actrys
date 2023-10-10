#source ~/.bash_profile

export temp_folder_name=$1

export run_scenario=$2
export synthetic_counts=$3

export noise_param=$4
export bias_param=$5

export spsa_a=$6
export spsa_c=$7
export spsa_a_out_sim=$8
export spsa_c_out_sim=$9
export spsa_reps=${10}

export sim_in_loop=${11}
export sim_out_loop=${12}
export n_iterations=${13}
export sim_in_iterations=${14}
export sim_out_iterations=${15}

export which_algo=${16}
export wspsa_thrshold=${17}
export calibrate_supply=${18}
export calibrate_demand=${19}
export set_spa=${20}
export estimator=${21}

export SCENARIO=${22}
export weight_counts=${23}
export weight_od=${24}
export weight_speed=${25}
export bagging_run=${26}
export count_noise_param=${27}
export heuristic=${28}
export auto_tune_spsa=${29}
export momentum_beta=${30}
export interval=${31}
export only_bias_correction=${32}
export bias_correction_method=${33}

export OD_FILE_IDENTIFIER=OD
export DEMAND_SOURCE=demand
export DEMAND_INTERVAL=$interval
export temp_scenario_name=$temp_folder_name

export PATH_ZONE=../../$SCENARIO/tazes.taz.xml
export PATH_DEMAND=../../$SCENARIO/$DEMAND_SOURCE/$OD_FILE_IDENTIFIER.txt
export PATH_NETWORK=../../$SCENARIO/network.net.xml

export PATH_ADDITIONAL=../../$SCENARIO/additional.add.xml

export PATH_SUMO_TOOLS=$SUMO_HOME/tools/

export PATH_OUTPUT_COUNT=../../$SCENARIO/$temp_scenario_name/out.xml 
export PATH_REAL_COUNT=../../$SCENARIO/$temp_scenario_name/real_counts_complete.csv

export PATH_OUTPUT_SPEED=../../$SCENARIO/$temp_scenario_name/edge_data_3600
export PATH_REAL_SPEED=../../$SCENARIO/$temp_scenario_name/real_edge_data_3600.csv

export FILE_MATCH_DETECTORS=../../ua_aqt/matchable_detectors.csv
export FILE_REAL_COUNTS=../../ua_aqt/real_counts.csv
export FILE_REAL_SPEEDS=../../ua_aqt/dummy_edge_data_3600.csv

export TOD_START=7
export TOD_END=9
python ../core/main.py

        


