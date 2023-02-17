import numpy as np
import pandas as pd
import subprocess
import os
from simulator import PATH_ADDITIONAL
from scenario_generator import *
from gof import gof_eval

def add_noise(x, perc_var, mu=1):
    '''Add noise to a synthetic data
    '''
    noisy_od = []
    for i in x:
        noisy_od.append( int(mu*i) + int(np.random.randn()*perc_var*i))
    return noisy_od

PATH_SUMO_TOOLS = os.environ.get("PATH_SUMO_TOOLS") #SUMO_PATH+"/tools/"

PATH_OUTPUT_COUNT = os.environ.get("PATH_OUTPUT_COUNT") #"../../"+SCENARIO+"/"+temp_scenario_name+"/out.xml" 
PATH_REAL_COUNT = os.environ.get("PATH_REAL_COUNT") #"../../"+SCENARIO+"/"+temp_scenario_name+"/real_counts.csv"

PATH_OUTPUT_SPEED = os.environ.get("PATH_OUTPUT_SPEED") #"../../"+SCENARIO+"/"+temp_scenario_name+"/out.xml" 
PATH_REAL_SPEED = os.environ.get("PATH_REAL_SPEED") #"../../"+SCENARIO+"/"+temp_scenario_name+"/real_counts.csv"

# SIMULATION_COUNT_INTERVAL = 900/3600 # to be defined when generating additional.add.xml in freq attribute
REAL_COUNT_INTERVAL = DEMAND_INTERVAL

from scenario_generator import TOD

def xml_2_csv(cmd_string):
    # print(cmd_string)
    subprocess.run(cmd_string, shell=True)

def create_synthetic_counts(path_additional, 
                            path_output_counts,
                            path_real_counts,
                            sim_type="meso",
                            count_noise=0):
    '''this function is basically for creating synthetic counts
    from the simulateed scenario'''
    xml_2_csv(cmd_string="python " +\
                    PATH_SUMO_TOOLS +\
                    "xml/xml2csv.py " +\
                    path_output_counts)
    xml_2_csv(cmd_string="python " +\
                    PATH_SUMO_TOOLS +\
                    "xml/xml2csv.py " +\
                    path_additional)

    df_additional = pd.read_csv(path_additional[:-3]+"csv", sep=";")

    temp = df_additional[['e1Detector_id']]
    df_additional = df_additional[~temp.isnull().any(axis=1)]

    
    df_simulated_counts = pd.read_csv(path_output_counts[:-3]+"csv", sep=";")

    list_count = []
    list_detector = []
    list_hour = []

    # for RMSN calculation, we donot take into account the warm-up and cool-down periods
    # remeber counts are generated at 15 min frequency or 900 seconds in out.xml
    for hour in np.arange(TOD[0]+WARM_UP_PERIOD, 
                          REAL_COUNT_INTERVAL + TOD[-1] - COOL_DOWN_PERIOD, 
                          REAL_COUNT_INTERVAL):	
        temp_simulated = df_simulated_counts[(df_simulated_counts['interval_begin'] >= hour*3600) &\
                                    (df_simulated_counts['interval_begin'] < (hour+REAL_COUNT_INTERVAL)*3600)]
        final = pd.merge(df_additional, 
                temp_simulated, 
                left_on="e1Detector_id", 
                right_on="interval_id")
        
        # get the unique identifier for a detector on a edge irrespective of the lane
        # Add the counts for each lane from the four intervals
        final = final.groupby(['e1Detector_id']).sum().reset_index()
        final['edge_detector_id'] = final['e1Detector_id'].apply(lambda x: x.split("_")[1])
        if sim_type=="meso":
            final = final.groupby(['edge_detector_id']).mean().reset_index()
        elif sim_type=="micro":
            final = final.groupby(['edge_detector_id']).sum().reset_index()
        else:
            raise("Enter a valid simulation type")


        list_count.extend(final['interval_entered'])
        list_detector.extend(final['edge_detector_id'])
        list_hour.extend(len(final)*[hour*3600])
    list_count = list(add_noise(np.array(list_count), int(count_noise)/100, mu=1))
    result = pd.DataFrame({"det_id":list_detector, "count": list_count, "hour": list_hour})
    result.to_csv(path_real_counts, index=None)
            
    

def get_true_simulated(	path_output_counts,
                        path_real_counts,
                        # path_additional,
                        flow_col= "count",#'q', 
                        detector_col= "det_id", #'iu_ac',
                        time_col= "hour", #"interval_begin",
                        sim_type = "meso",
                        ):
    '''this function is basically for processing outputs
    and getting results for RMSN evaluation
    For the real counts, not all detectors may have data
    Possibilities of error:
    Check if the number of detectors in true counts and additionals are same sd_counts.ipynb
    '''

    xml_2_csv(cmd_string="python " +\
                    PATH_SUMO_TOOLS +\
                    "xml/xml2csv.py " +\
                    path_output_counts)

    ## deprecated
    # xml_2_csv(cmd_string="python " +\
    # 					PATH_SUMO_TOOLS +\
    # 					"xml/xml2csv.py " +\
    # 					path_additional)

    # Read additional
    # df_additional = pd.read_csv(path_additional[:-3]+"csv", sep=";")
    # temp = df_additional[['e1Detector_id']]
    # df_additional = df_additional[~temp.isnull().any(axis=1)]
    
    # print(len(df_additional))

    # Read simulated out.xml
    df_simulated_counts = pd.read_csv(path_output_counts[:-3]+"csv", sep=";")
    #TODO Check for nans in out.csv and drop them
    #TODO Add checks for maintaining consitency for one-one comparison of counts

    # Read counts from real data
    df_real = pd.read_csv(path_real_counts)

    df_sim = []

    output_vector = np.empty((0,2), int) 
    for hour in np.arange(TOD[0]+WARM_UP_PERIOD, 
                          REAL_COUNT_INTERVAL + TOD[-1] - COOL_DOWN_PERIOD, 
                          REAL_COUNT_INTERVAL):	
        list_det = []
        list_count = []
        
        temp_simulated = df_simulated_counts[(df_simulated_counts['interval_begin'] >= hour*3600) &\
                                    (df_simulated_counts['interval_begin'] < (hour+REAL_COUNT_INTERVAL)*3600)]
        # to supress the SttingCopyWarning
        temp_simulated = temp_simulated.copy()

        ### add the counts for all the detector observations
        ###  within the calibration interval
        temp_simulated = temp_simulated.groupby(['interval_id']).sum().reset_index()


        # temp_real = df_real[(df_real[time_col] >= hour*3600) &\
        # 					(df_real[time_col] < (hour+REAL_COUNT_INTERVAL)*3600)]
        # get the edge ids
        detector_ids = temp_simulated['interval_id'].apply(lambda x: x.split("_")[1])
        temp_simulated.loc[:,'det_id']= detector_ids
        list_det.extend(list(temp_simulated.det_id))
        list_count.extend(list(temp_simulated['interval_entered']))
        
        # edge wise data
        temp_real = df_real[df_real[time_col]==hour*3600].copy()
        
        df_temp = pd.DataFrame({'det_id': list_det, 'count':list_count})
        df_temp['hour'] = hour*3600

        # In mesoscopic simulations, the lane based outputs are not available in SUMO
        # The lane output is same as the edge output. Therefore aggregating the lane based output
        # will overestimate the link flows by a factor of NUMBER_OF_LANES. Therefore, we use the mean option
        # which is equal to each of the output. However, if the simulation is microscopic, each output is needed.
        if sim_type=="meso":
            df_temp = df_temp.groupby(['det_id','hour']).mean().reset_index()
        elif sim_type=="micro":
            df_temp = df_temp.groupby(['det_id','hour']).sum().reset_index()
        else:
            raise("Invalid Simulation type. Is it meso or micro?")
        
        temp_real.loc[:, detector_col] = temp_real[detector_col].astype('str')
        df_temp.loc[:,'det_id'] = df_temp['det_id'].astype('str')
        temp_merge = pd.merge(temp_real, df_temp, left_on=detector_col, right_on='det_id')

        # print(f"RMSN for {hour} hour is {gof_eval(np.array(temp_merge[flow_col+'_x']), np.array(temp_merge.count_y))}")

        df_sim.append(temp_merge)



        # final = pd.merge(df_additional, 
        # 				temp_simulated, 
        # 				left_on="e1Detector_id", 
        # 				right_on="interval_id")		
        # get the unique identifier for a detector on a edge irrespective of the lane
        # final['edge_detector_id'] = final['e1Detector_id'].apply(lambda x: x.split("_")[1])

        # final = final.groupby(['e1Detector_name','interval_begin']).sum().reset_index()
        # The Paris Data is hourly, so we will not compare 15 mins or 900 s counts but the hourly  counts
        # final = final.groupby(['edge_detector_id']).sum().reset_index()

        # print(len(final))
        # final.to_csv("../../scenario/merge_raw.csv", index=None)
        # print(final)

        # compared = pd.merge(final,
        # 					temp_real,
        # 					left_on="edge_detector_id", 
        # 					right_on=detector_col)


        # # uncomment if you want to save the deetectors for creation of 
        # weight matrix for WSPSA
        # compared[detector_col].to_csv("../../scenario_munich/detectors_with_counts.csv", index=None)
        
        # print(len(compared))
        # temp_vector = np.zeros([len(compared), 2])
        # temp_vector[:,0] = compared[flow_col]
        # temp_vector[:,1] = compared['interval_entered']
        
        # output_vector = np.vstack([output_vector, temp_vector])
        # print(output_vector[:,0])
    df_both = pd.concat(df_sim)

    counts_true = np.array(df_both[flow_col+'_x'])
    counts_simulated = np.array(df_both["count_y"])
    return counts_true, counts_simulated #output_vector[:,0], output_vector[:,1] # check this again, should it be left + arrived?

def create_synthetic_speeds(path_additional, 
                            path_output_speeds,
                            path_real_speeds):
    '''this function is basically for creating synthetic counts
    from the simulateed scenario'''
    xml_2_csv(cmd_string="python " +\
                    PATH_SUMO_TOOLS +\
                    "xml/xml2csv.py -x $SUMO_HOME/data/xsd/meandata_file.xsd " +\
                    path_output_speeds)
    xml_2_csv(cmd_string="python " +\
                    PATH_SUMO_TOOLS +\
                    "xml/xml2csv.py " +\
                    path_additional)

    df_additional = pd.read_csv(path_additional[:-3]+"csv", sep=";")

    temp = df_additional[['e1Detector_id']]
    df_additional = df_additional[~temp.isnull().any(axis=1)]
    df_additional['edge_detector_id'] = df_additional['e1Detector_id'].apply(lambda x: x.split("_")[1])
    df_additional.drop_duplicates(subset="edge_detector_id", inplace=True)
    df_additional.reset_index(inplace=True, drop=True)


    
    df_simulated_speeds = pd.read_csv(path_output_speeds+".csv", sep=";")

    list_speed = []
    list_edge = []
    list_hour = []

    df_additional.edge_detector_id = df_additional.edge_detector_id.astype(str)

    # for RMSN calculation, we donot take into account the warm-up and cool-down periods
    # remeber counts are generated at 15 min frequency or 900 seconds in out.xml
    for hour in np.arange(TOD[0]+WARM_UP_PERIOD, 
                          REAL_COUNT_INTERVAL + TOD[-1] - COOL_DOWN_PERIOD, 
                          REAL_COUNT_INTERVAL):
        temp_simulated = df_simulated_speeds[(df_simulated_speeds['interval_begin'] >= hour*3600) &\
                                    (df_simulated_speeds['interval_begin'] < (hour+REAL_COUNT_INTERVAL)*3600)].copy()
        temp_simulated.edge_id = temp_simulated.edge_id.astype(str)

        final = pd.merge(df_additional, 
                temp_simulated, 
                left_on="edge_detector_id", 
                right_on="edge_id")
        
        # get the unique identifier for a detector on a edge irrespective of the lane
        # Add the counts for each lane from the four intervals
        final = final.groupby(['edge_detector_id']).mean().reset_index()
        # # final['edge_detector_id'] = final['e1Detector_id'].apply(lambda x: x.split("_")[1])
        # if sim_type=="meso":
        # 	final = final.groupby(['edge_detector_id']).mean().reset_index()
        # elif sim_type=="micro":
        # 	final = final.groupby(['edge_detector_id']).sum().reset_index()
        # else:
        # 	raise("Enter a valid simulation type")

        # print(final.columns)
        list_speed.extend(final['edge_speed'])
        list_edge.extend(final['edge_detector_id'])
        list_hour.extend(len(final)*[hour*3600])

    result = pd.DataFrame({"det_id":list_edge, "speed": list_speed, "hour": list_hour})
    result.to_csv(path_real_speeds, index=None)

def get_true_simulated_speeds(path_output_speeds,
                            path_real_speeds,
                            speed_col= "speed",#'q', 
                            detector_col= "det_id", #'iu_ac',
                            time_col= "hour", #"interval_begin",
                            ):
    '''this function is basically for processing outputs
    and getting results for RMSN evaluation
    For the real counts, not all detectors may have data
    Possibilities of error:
    Check if the number of detectors in true counts and additionals are same sd_counts.ipynb
    '''

    xml_2_csv(cmd_string="python " +\
                    PATH_SUMO_TOOLS +\
                    "xml/xml2csv.py -x $SUMO_HOME/data/xsd/meandata_file.xsd " +\
                    path_output_speeds)

    # Read simulated out.xml
    df_simulated_speeds = pd.read_csv(path_output_speeds+".csv", sep=";")
    #TODO Check for nans in out.csv and drop them
    #TODO Add checks for maintaining consitency for one-one comparison of counts

    # Read counts from real data
    df_real = pd.read_csv(path_real_speeds)

    df_sim = []


    for hour in np.arange(TOD[0]+WARM_UP_PERIOD, 
                        REAL_COUNT_INTERVAL + TOD[-1] - COOL_DOWN_PERIOD, 
                        REAL_COUNT_INTERVAL):
        list_det = []
        list_speed = []
        
        temp_simulated = df_simulated_speeds[(df_simulated_speeds['interval_begin'] >= hour*3600) &\
                                    (df_simulated_speeds['interval_begin'] < (hour+REAL_COUNT_INTERVAL)*3600)]

        temp_simulated = temp_simulated.copy()

        ### add the counts for all the detector observations
        ###  within the calibration interval
        temp_simulated = temp_simulated.groupby(['edge_id']).mean().reset_index()

        list_det.extend(list(temp_simulated.edge_id))
        list_speed.extend(list(temp_simulated['edge_speed']))
        
        # edge wise data
        temp_real = df_real[df_real[time_col]==hour*3600].copy()
        
        df_temp = pd.DataFrame({'det_id': list_det, 'speed':list_speed})
        df_temp['hour'] = hour*3600
        
        temp_real.loc[:, detector_col] = temp_real[detector_col].astype('str')
        df_temp.loc[:, 'det_id'] = df_temp['det_id'].astype('str')
        temp_merge = pd.merge(temp_real, df_temp, left_on=detector_col, right_on='det_id')

        # print(f"RMSN for {hour} hour is {gof_eval(np.array(temp_merge[flow_col+'_x']), np.array(temp_merge.count_y))}")

        df_sim.append(temp_merge)
    df_both = pd.concat(df_sim)
    df_both.loc[:, speed_col+'_x'] = df_both[speed_col+'_x'].fillna(20) # filling free speed when there are no vehicles
    df_both.loc[:, "speed_y"] = df_both["speed_y"].fillna(20) # filling free speed when there are no vehicles
    # df_both.dropna(inplace=True)
    speeds_true = np.array(df_both[speed_col+'_x'])
    speeds_simulated = np.array(df_both["speed_y"])
    # print(speeds_true-speeds_simulated)
    return speeds_true, speeds_simulated

if __name__ == "__main__":


    path_temp_additional = "../../"+SCENARIO+"/"+temp_scenario_name+"/additional.add.xml"
    # # create_synthetic_counts(path_additional=path_temp_additional,
    # # 						path_output_counts=PATH_OUTPUT_COUNT,
    # # 						path_real_counts=PATH_REAL_COUNT)

    # true, simulated = get_true_simulated(path_output_counts=PATH_OUTPUT_COUNT, 
    # 									path_real_counts=PATH_REAL_COUNT)

    # print("True counts")
    # print(true)

    # print("Simulated counts")
    # print(simulated)

    # print(f"RMSN for full interval is {gof_eval(np.array(true), np.array(simulated))}")

    create_synthetic_speeds(path_additional=path_temp_additional,
                        path_output_speeds=PATH_OUTPUT_SPEED,
                        path_real_speeds=PATH_REAL_SPEED)
    get_true_simulated_speeds(PATH_OUTPUT_SPEED,
                            PATH_REAL_SPEED,
                            speed_col= "speed",#'q', 
                            detector_col= "det_id", #'iu_ac',
                            time_col= "hour"
                            )
    

    

