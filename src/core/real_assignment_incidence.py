#### To generate the weight matrix for W-SPSA on the fly

import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pickle
import subprocess
import sys
import os
from dotenv import load_dotenv

## To load environment variables consistent  across scenarios
load_dotenv()

from scenario_generator import TOD_START, TOD_END, \
                                WARM_UP_PERIOD, COOL_DOWN_PERIOD,\
                                DEMAND_INTERVAL

period = int(3600*DEMAND_INTERVAL)
period_fraction = DEMAND_INTERVAL
trip_impact = 3600

def generate_detector_incidence(path_od_sample, 
                                path_routes,
                                path_trip_summary,
                                path_additional,
                                path_true_count,
                                path_match_detectors,
                                path_output_weight,
                                path_output_assignment,
                                scenario,
                                threshold=False,
                                threshold_value=0.1,
                                is_synthetic=False,
                                binary_rounding=False,
                                time_interval = period,
                                t_start = TOD_START*3600,
                                t_end = TOD_END*3600,
                                t_impact = trip_impact,
                                t_warmup = WARM_UP_PERIOD*3600,
                                do_plot=True,
                                do_save=True):
    print(time_interval)
    # print("Threshold: "+ str(threshold)+", "+ "Synthetic: "+ str(is_synthetic) + \
    # 	  "Cut-off: "+ str(threshold_value)+ ", Binary rounding: "+ str(binary_rounding))

    SUMO_HOME = os.getenv("SUMO_HOME")

    subprocess.run(SUMO_HOME+"/tools/xml/xml2csv.py " + path_trip_summary, shell=True)
    subprocess.run(SUMO_HOME+"/tools/xml/xml2csv.py " + path_routes, shell=True)
    subprocess.run(SUMO_HOME+"/tools/xml/xml2csv.py " + path_additional, shell=True)


    # using sumo OD matrrix file so that the order of ODs is same as the counts
    dfod = pd.read_csv(path_od_sample, sep=' ')
    dfod['od_pair'] = dfod.o.astype('str') + "___"+ dfod.d.astype("str")

    dfr = pd.read_csv(path_routes[:-3]+"csv", sep=';')
    dfr.drop_duplicates(subset=['vehicle_id'], keep='last', inplace=True)
    dft = pd.read_csv(path_trip_summary+".csv", sep=';')
    dfr['od_pair'] = dfr.vehicle_fromTaz.astype('str') + "___"+ dfr.vehicle_toTaz.astype("str")


    # len(dfr.od_pair.unique())
    df_merge = pd.merge(left=dfr, right=dft, left_on="vehicle_id",right_on="tripinfo_id")

    df_merge = df_merge[['vehicle_id', 'vehicle_depart', 'tripinfo_arrival', 'route_edges', 
                        'vehicle_fromTaz', 'vehicle_toTaz', 'od_pair']]


    if is_synthetic:
        edge_col = 'edge_id'
        dtd = pd.read_csv(path_additional[:-3]+"csv", sep=";")
        dtd.dropna(subset=["e1Detector_id"], inplace=True)
        dtd[edge_col]= dtd['e1Detector_id'].apply(lambda x: x.split("_")[1])
    else:
        edge_col = "det_id"
        dtd = pd.read_csv(path_true_count)
        # dtd_real_vs_sim = pd.read_csv(Paths['det_counts'])
        # dtd = dtd[dtd[edge_col].isin(dtd_real_vs_sim[edge_col])]
    
        if scenario=="composite_scenario_munich":
            dtd_macthing = pd.read_csv(path_match_detectors)
            dtd = dtd[dtd[edge_col].isin(dtd_macthing["det_id"])]

    od_pairs = dfod.od_pair.unique()
    num_detectors = len(dtd[edge_col].unique())
    # print(num_detectors)
    # sys.exit(0)
    intervals = range(t_start-t_warmup, t_end, time_interval) #time_interval
    print(intervals)

    num_od_pairs = len(od_pairs)
    num_intervals = len(intervals)

    incidence_weight_array = np.zeros((num_od_pairs*num_intervals, num_detectors*num_intervals), dtype="float32")
    incidence_assignment_array = np.zeros((num_od_pairs*num_intervals, num_detectors*num_intervals), dtype="float32")


    df_merge['origin'] = df_merge['od_pair'].apply(lambda x: x.split("___")[0])

    for k, depr in tqdm(enumerate(intervals)):
        df_temp = df_merge[(df_merge.vehicle_depart>=depr) & (df_merge.vehicle_depart<(depr+time_interval))]
        # for l, arrv in enumerate(intervals):
            # temp = df_temp[(df_merge.tripinfo_arrival>arrv) & (df_merge.tripinfo_arrival<arrv+t_impact)]
            # print(f"Arrival:{arrv} to {arrv+t_impact}, Departure:{depr} to {depr+time_interval}, trips: {len(temp)}")
        for i, od in enumerate(od_pairs):
            origin = od.split("___")[0]
            origin_trips = len(df_temp[df_temp.origin==origin])
            if origin_trips!=0:
                temp_od  = df_temp[df_temp.od_pair==od]
                if len(temp_od)!=0:
                    for l, arrv in enumerate(intervals):
                        temp = temp_od[(temp_od.tripinfo_arrival>=arrv) & (temp_od.tripinfo_arrival<(arrv+time_interval))]
                        if len(temp)!=0:
                            edges = []
                            for route in temp.route_edges.astype("str"):#.unique():
                                edges.extend(route.split(" "))
                            for q, detector in enumerate(dtd[edge_col].unique()):
                                new_edge = str(detector)
                                for e in edges:
                                    if new_edge==e:
                                        #  a naive approach will to assign the weights equal to 1
                                        # this is a binary approximation, a continous value between 0 and 1 can also be used 
                                        # for exact correlation specification
                                        # assign a weight equal to the proportion of trips from an OD passing a specfifc detector
                                        # alternative is to assign a weight equal to the contribution of an OD to the number of 
                                        # vehicles on a specific counter
                                        # from the origin
                                        incidence_assignment_array[i+k*len(od_pairs), q+l*num_detectors] += 1
                                        incidence_weight_array[i+k*len(od_pairs), q+l*num_detectors] = len(temp_od)/origin_trips #1 
                    incidence_assignment_array[i+k*len(od_pairs), :] /= len(temp_od) #1 
    
    if threshold:
        # set a threshold
        if binary_rounding:
            incidence_weight_array[incidence_weight_array>threshold_value]=1
            incidence_weight_array[incidence_weight_array<=threshold_value]=0
        else:
            incidence_weight_array[incidence_weight_array<=threshold_value]=0

        
    # incidence_array = incidence_array.astype("int8")
    
    if do_plot == True:
        path_plot="../../images/weight_incidence_"+scenario+"_"+str(threshold_value)+".png"
        fig, ax = plt.subplots(1,1, figsize=(60,60))
        plt.imshow(incidence_weight_array, cmap='hot', interpolation='nearest')
        plt.savefig(path_plot, dpi=300)
        plt.close("all")


        path_plot="../../images/assignment_incidence_"+scenario+"_"+str(threshold_value)+".png"
        fig, ax = plt.subplots(1,1, figsize=(60,60))
        plt.imshow(incidence_assignment_array, cmap='hot', interpolation='nearest')
        plt.savefig(path_plot, dpi=300)
        plt.close("all")

    if do_save == True:
        S = scipy.sparse.csr_matrix(incidence_weight_array[:,int(t_warmup/(3600*period_fraction))*num_detectors:])
        with open(path_output_weight,'wb') as file:
            pickle.dump(S, file)
        S = scipy.sparse.csr_matrix(incidence_assignment_array[:,int(t_warmup/(3600*period_fraction))*num_detectors:])
        with open(path_output_assignment,'wb') as file:
            pickle.dump(S, file)
    
    

    print("Weight Matrix Succeess")
    return incidence_weight_array[:,int(t_warmup/(3600*period_fraction))*num_detectors:],\
           incidence_assignment_array[:,int(t_warmup/(3600*period_fraction))*num_detectors:]

def prepare_weight_matrix(W, weight_counts, weight_od, weight_speed):

    if weight_counts!=0:
        if weight_od!=0:
            if weight_speed!=0:
                ###### Warning: Change the dtype of the weight array to float32 (4 bytes) 
                # if you want the correlation to be float values between 0 and 1. Now it is 
                # configured to int8 which takes 1 byte of space
                weight_wspsa = np.zeros((W.shape[0], W.shape[1]+W.shape[0]+W.shape[1]), dtype='int8')
                weight_wspsa[:, :W.shape[1]] = np.where(W>0, 1, 0) 
                weight_wspsa[:, W.shape[1]:W.shape[1]+W.shape[0]] = np.eye(W.shape[0])
                weight_wspsa[:, W.shape[1]+W.shape[0]:] = np.where(W>0, 1, 0)
            else:
                weight_wspsa = np.zeros((W.shape[0], W.shape[1]+W.shape[0]), dtype='int8')
                weight_wspsa[:, :W.shape[1]] = np.where(W>0, 1, 0) 
                weight_wspsa[:, W.shape[1]:] = np.eye(W.shape[0])
        else:
            if weight_speed!=0:
                weight_wspsa = np.zeros((W.shape[0], W.shape[1]+W.shape[1]), dtype='int8')
                weight_wspsa[:, :W.shape[1]] = np.where(W>0, 1, 0) 
                weight_wspsa[:, W.shape[1]:] = np.where(W>0, 1, 0)
            else:
                weight_wspsa = np.where(W>0, 1, 0).astype('int8')
    else:
        if weight_od!=0:
            if weight_speed!=0:
                weight_wspsa = np.zeros((W.shape[0], W.shape[0]+W.shape[1]), dtype='int8')
                weight_wspsa[:, :W.shape[0]] = np.eye(W.shape[0])
                weight_wspsa[:, W.shape[0]:] = np.where(W>0, 1, 0)
            else:
                weight_wspsa = np.eye(W.shape[0], dtype='int8')
        else:
            if weight_speed!=0:
                weight_wspsa = np.where(W>0, 1, 0).astype('int8')
            else:
                raise("All the weights cannot be zero")
    return weight_wspsa


if __name__ == "__main__":
    scenario = sys.argv[1]
    set_threshold = eval(sys.argv[2])
    threshold_val = float(sys.argv[3])
    is_synthetic = eval(sys.argv[4])
    binary_rounding = eval(sys.argv[5])

    if scenario == 'san_francisco':
        Paths = {'od_file' : "../../san_francisco/demand/od_list.txt",
                'trips_file' : "../../san_francisco/temp_wspsa/trips",
                'routes_file' : "../../san_francisco/temp_wspsa/routes.rou.xml",
                'detector_file' : "../../san_francisco/additional.add.xml",
                'true_counts': "../../san_francisco/true_counts/true_counts.csv",
                # 'det_counts' : "../../scenario_munich/detectors_with_counts.csv",
                'save_output' : "../../san_francisco/temp_wspsa/weight_matrix_real_counts.pickle"
                }
    elif scenario == 'munich':

            Paths = {'od_file' : "../../scenario_munich/true_demand/od_list.txt",
                'trips_file' : "../../scenario_munich/temp_wspsa/trips",
                'routes_file' : "../../scenario_munich/temp_wspsa/routes.rou.csv",
                'detector_file' : "../../scenario_munich/additional.add.csv",
                'true_counts': "../../scenario_munich/true_counts/true_counts.csv",
                #### Some detectors in Munich scenario (and composite dont match with real counts vs additional...
                #### Why? did the network change or something else??),
                'det_counts' : "../../scenario_munich/detectors_with_counts.csv",
                'save_output' : "../../scenario_munich/temp_wspsa/weight_matrix_real_counts.pickle"
                }

    elif scenario == 'msm_scenario_munich':

            Paths = {'od_file' : "../../msm_scenario_munich/demand/od_list.txt",
                'trips_file' : "../../msm_scenario_munich/wspsa/trips",
                'routes_file' : "../../msm_scenario_munich/wspsa/routes.rou.xml",
                'detector_file' : "../../msm_scenario_munich/wspsa/additional.add.csv",
                'true_counts': "../../msm_scenario_munich/true_counts/real_counts.csv",
                'det_counts' : "../../msm_scenario_munich/detectors_with_counts.csv",
                'save_output' : "../../msm_scenario_munich/wspsa/weight_matrix_thresh_"+str(set_threshold)+"_synthetic_"+str(is_synthetic)+\
                                "_"+str(threshold_val)+"_"+"binary_"+str(binary_rounding)+".pickle"
                }

    elif scenario == 'composite_scenario_munich':

            Paths = {'od_file' : "../../composite_scenario_munich/demand/od_list.txt",
                'trips_file' : "../../composite_scenario_munich/temp_munich_wspsa/trips_output",
                'routes_file' : "../../composite_scenario_munich/temp_munich_wspsa/routes.rou.xml",
                'detector_file' : "../../composite_scenario_munich/temp_munich_wspsa/additionalcomplete.add.xml",
                'true_counts': "../../composite_scenario_munich/true_counts/real_counts_complete.csv",
                #### Some detectors in Munich scenario (and composite dont match with real counts vs additional...
                #### Why? did the network change or something else??)
                'det_counts' : "../../composite_scenario_munich/matchable_detectors.csv",
                'save_weights' : "../../composite_scenario_munich/wspsa/weight_matrix_thresh_"+str(set_threshold)+"_synthetic_"+str(is_synthetic)+\
                                "_"+str(threshold_val)+"_"+"binary_"+str(binary_rounding)+".pickle",
                'save_assignment' : "../../composite_scenario_munich/wspsa/assignment_matrix.pickle"
                }

    elif scenario == 'synthetic_msm_scenario_munich':

            Paths = {'od_file' : "../../msm_scenario_munich/demand/od_list.txt",
                'trips_file' : "../../msm_scenario_munich/wspsa/trips",
                'routes_file' : "../../msm_scenario_munich/wspsa/routes.rou.xml",
                'detector_file' : "../../msm_scenario_munich/wspsa/additional.add.csv",
                'true_counts': "../../msm_scenario_munich/wspsa/synthetic_counts.csv",
                'det_counts' : "../../msm_scenario_munich/detectors_with_counts.csv",
                'save_output' : "../../msm_scenario_munich/wspsa/weight_matrix_thresh_"+str(set_threshold)+"_synthetic_"+str(is_synthetic)+\
                                "_"+str(threshold_val)+"_"+"binary_"+str(binary_rounding)+".pickle"
                }
    elif scenario == 'demo_grid':

        Paths = {'od_file' : "../../demos/demand/od_list.txt",
            'trips_file' : "../../demos/temp_outsim_wspsa/trips_output",
            'routes_file' : "../../demos/temp_outsim_wspsa/routes.rou.xml",
            'detector_file' : "../../demos/temp_outsim_wspsa/additional.add.xml",
            'true_counts': "../../demos/temp_outsim_wspsa/real_counts_complete.csv",
            #### Some detectors in Munich scenario (and composite dont match with real counts vs additional...
            #### Why? did the network change or something else??)
            'det_counts' : "",
            'save_weights' : "../../demos/wspsa/weight_matrix_thresh_"+str(set_threshold)+"_synthetic_"+str(is_synthetic)+\
                            "_"+str(threshold_val)+"_"+"binary_"+str(binary_rounding)+".pickle",
            'save_assignment' : "../../demos/wspsa/assignment_matrix.pickle"
                }
    else:
        raise("Enter a valid scenario")

    W, A = generate_detector_incidence(Paths['od_file'],
                                Paths['routes_file'],
                                Paths['trips_file'],
                                Paths['detector_file'],
                                Paths['true_counts'],
                                Paths['det_counts'],
                                Paths['save_weights'],
                                Paths['save_assignment'],
                                scenario,
                                set_threshold,
                                threshold_val,
                                is_synthetic, 
                                binary_rounding)
    print(W.shape)
    print(A.shape)
    print("Done")