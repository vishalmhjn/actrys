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
export estimator=${19}

export SCENARIO=${20}
export weight_counts=${21}
export weight_od=${22}
export weight_speed=${23}
export bagging_run=${24}
export count_noise_param=${25}
export heuristic=${26}
export auto_tune_spsa=${27}
export momentum_beta=${28}
export interval=${29}
export only_bias_correction=${30}

export OD_FILE_IDENTIFIER=OD
export DEMAND_SOURCE=demand/v2
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

export TOD_START=7
export TOD_END=10
python ../core/main.py

        


