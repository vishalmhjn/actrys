import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
import copy
import pandas as pd
import utilities
from scenario_generator import create_scenario, trip_validator
from simulator import run_simulation, PATH_ADDITIONAL, copy_additonal
from gof import gof_eval, squared_deviation
from output_processor import get_true_simulated, create_synthetic_counts
from output_processor import *
from od_formatter import OD_dataframe_to_txt, OD_txt_to_dataframe
from optimizer import SolutionFinder
from scenario_generator import OD_FILE_IDENTIFIER
from scenario_generator import file_manager
from wspsa_weight_incidence import prepare_weight_matrix
from real_assignment_incidence import generate_detector_incidence

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from datetime import datetime
timestr = str(datetime.now().strftime('%m_%d_%Y_%H_%M_%S') )

from params import *
from paths import *



res_dict = dict()
res_dict["f_val"] = []
res_dict["f_val_supply"] = []
res_dict["count_val"] = []
res_dict["speed_val"] = []
res_dict["od_val"] = []
res_dict["best_supply_count_rmsn"] = 100

best_rmsn = 10000000000
best_od = []
best_simulated_speeds = 0
best_simulated_counts = 0
estimated = 0
simulated_counts = 0
simulated_speeds = 0
stochastic_solution_counter = 0

def save_params(d, params):
    for param in params:
        d[str(param)] = eval(param)
    return d

def add_noise(x, perc_var, mu=1):
    '''Add noise to a synthetic data
    '''
    noisy_od = []
    for i in x:
        noisy_od.append( int(mu*i) + int(np.random.randn()*perc_var*i))
    return noisy_od

def od_to_counts(path_iter_demand,
                                od_vector,
                                path_iter_trips, 
                                path_iter_routes,
                                path_temp_additional,
                                path_iter_simulation_counts,
                                path_iter_simulation_speeds,
                                evaluation_run=True,
                                rerouting_prob=0.9,
                                tls_tt_pen=6, 
                                meso_minor_pen=10, 
                                rerouting_period=30,
                                rerouting_adaptation=5,
                                rerouting_adaptation_steps=157,
                                meso_tls_flow_penalty=0.9,
                                priority_factor = 0.01):


    OD_dataframe_to_txt(path_iter_demand,
                        od_vector,
                        path_base_file=PATH_DEMAND)
    #### this step is within the iteration of objective function to reduce the 
    #### stochasticity due to the generation of the trips
    create_scenario(path_iter_trips, path_iter_routes, path_iter_demand)
    trip_validator(path_iter_trips, path_iter_routes)
    run_simulation(path_iter_trips, 
                path_iter_routes, 
                path_temp_additional, 
                evaluation_run,
                routing_how="reroute",
                rerouting_prob=rerouting_prob,
                tls_tt_penalty=tls_tt_pen, 
                meso_minor_penalty=meso_minor_pen, 
                rerouting_period=rerouting_period,
                rerouting_adaptation=rerouting_adaptation,
                rerouting_adaptation_steps=rerouting_adaptation_steps,
                meso_tls_flow_penalty=meso_tls_flow_penalty,
                priority_factor = priority_factor)
    iter_true, iter_simulated = get_true_simulated(path_output_counts=path_iter_simulation_counts, 
                                    path_real_counts=PATH_REAL_COUNT,
                                    # path_additional=path_temp_additional,
                                    flow_col="count", 
                                    detector_col="det_id", 
                                    time_col="hour",
                                    sim_type = "meso")
    iter_true_speeds, iter_simulated_speeds = get_true_simulated_speeds(path_output_speeds = path_iter_simulation_speeds,
                                                                        path_real_speeds = PATH_REAL_SPEED,
                                                                        speed_col= "speed",#'q', 
                                                                        detector_col= "det_id", #'iu_ac',
                                                                        time_col= "hour"
                                                                        )
    return iter_true, iter_simulated, iter_true_speeds, iter_simulated_speeds

def wspsa_update_wrapper(true_count_file,
                        det_counts,
                        od_file_template = "../../"+SCENARIO+"/demand/od_list.txt",
                        routes_file = path_routes,
                        trips_output=path_trip_summary,
                        detector_file=path_temp_additional,
                        save_weight_output="../../"+SCENARIO+"/wspsa/weight_iter.pickle",
                        save_assignment_output="../../"+SCENARIO+"/wspsa/weight_iter.pickle",
                        wscenario=SCENARIO,
                        set_threshold=True,
                        threshold_val=0,
                        is_synthetic=True, 
                        binary_rounding=True):
    W, A = generate_detector_incidence(od_file_template,
                                    routes_file,
                                    trips_output,
                                    detector_file,
                                    true_count_file,
                                    det_counts,
                                    save_weight_output,
                                    save_assignment_output,
                                    wscenario,
                                    set_threshold,
                                    threshold_val,
                                    is_synthetic, 
                                    binary_rounding,
                                    do_save=False,
                                    do_plot=False)
    return W, A

def objective_function(X, 
                        path_demand, 
                        path_trips, 
                        X_true,
                        X_prior,
                        true_counts,
                        true_speeds,
                        rerouting_prob,
                        tls_tt_pen, 
                        meso_minor_pen, 
                        rerouting_period,
                        rerouting_adaptation,
                        rerouting_adaptation_steps,
                        meso_tls_flow_penalty,
                        priority_factor,
                        weighted=False,
                        eval_rmsn=False,
                        which_perturb='positive'):
    '''This is the objective function which estimates the rmsn between the 
    True and Simulaed detector counts
    TODO add a high cost when the demand value is negative'''

    # check = bool(np.all(X_OD >= 0))
    # print(X_OD)

    X = np.where(X<0, 0, X)

    X = X.astype(int)

    num_od = len(X)

    if eval_rmsn==True:
        global simulated_counts, simulated_speeds
        count_init, simulated_counts,\
            speed_init, simulated_speeds = od_to_counts(path_demand,
                                                        X,
                                                        path_trips,
                                                        path_routes,
                                                        path_temp_additional,
                                                        path_simulation_counts,
                                                        path_simulation_speeds,
                                                        rerouting_prob=rerouting_prob,
                                                        tls_tt_pen=tls_tt_pen, 
                                                        meso_minor_pen=meso_minor_pen, 
                                                        rerouting_period=rerouting_period,
                                                        rerouting_adaptation=rerouting_adaptation,
                                                        rerouting_adaptation_steps=rerouting_adaptation_steps,
                                                        meso_tls_flow_penalty=meso_tls_flow_penalty,
                                                        priority_factor = priority_factor)
    else:
        copy_additonal(PATH_ADDITIONAL[:-4]+"_"+which_perturb+path_temp_additional[-4:], 
                        path_temp_additional[:-4]+"_"+which_perturb+path_temp_additional[-4:])
        count_init, simulated_counts,\
            speed_init, simulated_speeds = od_to_counts(path_demand[:-4]+"_"+which_perturb+path_demand[-4:],
                                                        X,
                                                        path_trips[:-4]+"_"+which_perturb+path_trips[-4:],
                                                        path_routes[:-4]+"_"+which_perturb+path_routes[-4:],
                                                        path_temp_additional[:-4]+"_"+which_perturb+path_temp_additional[-4:],
                                                        path_simulation_counts[:-4]+"_"+which_perturb+path_simulation_counts[-4:],
                                                        path_simulation_speeds+"_"+which_perturb,
                                                        evaluation_run=eval_rmsn)
    
    # equal weights for counts and demand
    global weight_counts, weight_od, weight_speed

    if weighted == False:
        ### when objective function evaluation is to check the value
        count_gof = np.round(gof_eval(count_init.flatten(), simulated_counts.flatten(), estimator=estimator),4)
        speed_gof = np.round(gof_eval(speed_init.flatten(), simulated_speeds.flatten(), estimator=estimator),4)
        ### here its fine to use true ODs as we are trying to determine the distance
        ### of the estimates from the given OD
        rmsn = weight_counts*count_gof +\
                weight_speed*speed_gof +\
                weight_od*gof_eval(X_true, X, estimator=estimator)
        rmsn_c = count_gof
        rmsn_s = speed_gof
        rmsn_od = gof_eval(X_true, X, estimator=estimator)
    else:
        ### when objective function evaluation is to get the gradient
        if weight_counts!=0:
            sd_counts = weight_counts * squared_deviation(count_init.flatten(), simulated_counts.flatten())
        else:
            sd_counts = np.array([])
        if weight_od!=0:
            sd_od = weight_od * squared_deviation(X_prior, X)
        else:
            sd_od = np.array([])
        if weight_speed!=0:
            sd_speed = weight_speed * squared_deviation(speed_init.flatten(), simulated_speeds.flatten())
        else:
            sd_speed = np.array([])
        
        if sd_counts.size:
            if sd_od.size:
                if sd_speed.size:
                    rmsn = np.hstack((sd_counts, sd_od, sd_speed))
                else:
                    rmsn = np.hstack((sd_counts, sd_od))
            else:
                if sd_speed.size:
                    rmsn = np.hstack((sd_counts, sd_speed))
                else:
                    rmsn = sd_counts
        else:
            if sd_od.size:
                if sd_speed.size:
                    rmsn = np.hstack((sd_od, sd_speed))
                else:
                    rmsn = sd_od
            else:
                if sd_speed.size:
                    rmsn = sd_speed
                else:
                    raise("All weights cannot be zero")

    if eval_rmsn==True:
        global res_dict
        # stochastic averaging
        if len(res_dict["od_val"])>1:
            if np.std(res_dict["od_val"]) < 0.10:
                global stochastic_solution_counter
                stochastic_solutions = pd.DataFrame({"real": X_true.flatten(),"simulated": X.flatten()})
                stochastic_solutions.to_csv(pre_string+"/interim_"+str(stochastic_solution_counter)+"_"+"od.csv", index=None)
                save_counts = pd.DataFrame({"real": count_init.flatten(), 
                                            'simulated': simulated_counts.flatten()})
                save_counts.to_csv(pre_string+"/interim_"+str(stochastic_solution_counter)+"_counts.csv", index=None)
                stochastic_solution_counter+=1
        res_dict["f_val"].append(rmsn)
        res_dict["count_val"].append(rmsn_c)
        res_dict["speed_val"].append(rmsn_s)
        res_dict["od_val"].append(rmsn_od)
        global estimated
        estimated = X
        print(f"Weighted {estimator} = {rmsn}")
        print(f"Count {estimator} = {rmsn_c}")
        print(f"Speed {estimator} = {rmsn_s}")
        print(f"OD {estimator} = {rmsn_od}")
        
        global best_rmsn, best_simulated_counts, best_simulated_speeds
        if rmsn < best_rmsn:
            OD_dataframe_to_txt(pre_string +"/"+OD_FILE_IDENTIFIER+"_best.txt", X, PATH_DEMAND)
            best_simulated_speeds = simulated_speeds
            best_simulated_counts = simulated_counts
            best_rmsn = rmsn
            global best_od
            best_od = copy.deepcopy(X)
            res_dict['rerouting_prob'] = rerouting_prob
            res_dict['tls_tt_pen'] = tls_tt_pen
            res_dict['meso_minor_pen'] = meso_minor_pen
            res_dict['rerouting_period'] = rerouting_period
            res_dict['rerouting_adaptation'] = rerouting_adaptation
            res_dict['rerouting_adaptation_steps'] = rerouting_adaptation_steps
            res_dict['meso_tls_flow_penalty'] = meso_tls_flow_penalty
            res_dict['priority_factor'] = priority_factor

            ### save copies of the edge data
            copy_additonal(pre_string + '/edge_data_300', pre_string + '/best_edge_data_300')
            copy_additonal(pre_string + '/edge_data_900', pre_string + '/best_edge_data_900')
            copy_additonal(pre_string + '/edge_data_3600', pre_string + '/best_edge_data_3600')
            copy_additonal(pre_string + '/out.csv', pre_string + '/best_out.csv')

        if rmsn < 0.05:
            fig, ax = plt.subplots(1,1,figsize=(10,6))
            ax = utilities.plot_loss_curve(ax, res_dict, spsa_a, spsa_c)
            plt.savefig("../../images/loss_"+str(num_od)+"_"+\
                    str(n_iterations)+"_"+str(weight_od)+"_"+str(weight_counts)+"_"+\
                        str(weight_speed)+"_"+str(which_algo)+".png", dpi=300)
            plt.close()

            fig, ax = plt.subplots(3,2,figsize=(7,9))

            fig, ax = utilities.plot_45_degree_plots(fig, ax, X_OD, estimated, init,
                                                weight_counts, weight_od, weight_speed,
                                                true_counts, simulated_counts, initial_counts,
                                                true_speeds, simulated_speeds, speed_init, 
                                                spsa_a, spsa_c, noise_param, bias_param)

            plt.tight_layout()
            plt.savefig("../../images/results_"+str(num_od)+"_"+\
                    str(n_iterations)+"_"+str(weight_od)+"_"+str(weight_counts)+"_"+\
                        str(weight_speed)+"_"+str(which_algo)+".png", dpi=300)

            save_counts = pd.DataFrame({"real": true_counts, 'simulated': simulated_counts})
            save_counts.to_pickle(pre_string+"/counts.pkl")

            res_dict = save_params(res_dict, ['spsa_a', 'spsa_c', 'n_iterations',
                                    'noise_param', 'bias_param', 'spsa_reps',
                                    'weight_counts', 'weight_od', 'weight_speed', 'wspsa_thrshold'])

            with open(pre_string+'/results_'+timestr+'.json', 'w') as fp:
                print(res_dict)
                json.dump(res_dict, fp)
            sys.exit()
            # this is not good maybe change it
            # increasing the scale of RMSN as it is a small value
    return rmsn

def objective_function_without_simulator(X, 
                                        X_true, 
                                        X_prior,
                                        A, 
                                        count_init,
                                        speed_init,
                                        weighted=False,
                                        eval_rmsn=False,
                                        which_perturb='positive'):
    '''This is the objective function which estimates the rmsn between the 
    True and Simulaed detector counts
    TODO add a high cost when the demand value is negative'''
    
    X = np.array([int(i) for i in np.where(X<0, 0, X)])

    global simulated_counts, simulated_speeds
    # simulated_counts, simulated_speeds = synthetic_simulation(X, W, traffic_state, interval, num_detectors)
    # simulated_counts = np.dot(X.T, A) # disabled due to rounding off errors

    simulated_counts = np.zeros((A.shape[1], 1))
    for i in range(A.shape[1]):
        simulated_counts[i] = np.sum(np.multiply(X, A[:,i]).astype(int))

    simulated_counts = simulated_counts.reshape(-1, 1)
    # simulated_speeds = simulated_speeds.reshape(-1, 1)

    # equal weights for counts and demand
    global weight_counts, weight_od, weight_speed

    if weighted == False:
        count_rmsn = np.round(gof_eval(count_init.flatten(), simulated_counts.flatten(), estimator=estimator),4)
        rmsn = weight_counts*gof_eval(count_init.flatten(), simulated_counts.flatten(), estimator=estimator) \
                + weight_od*gof_eval(X_true, X, estimator=estimator)
    else:
        if weight_counts!=0:
            sd_counts = weight_counts * squared_deviation(count_init.flatten(), simulated_counts.flatten())
        else:
            sd_counts = np.array([])
        if weight_od!=0:
            sd_od = weight_od * squared_deviation(X_prior, X)
        else:
            sd_od = np.array([])
        
        if sd_counts.size:
            if sd_od.size:
                rmsn = np.hstack((sd_counts, sd_od))
            else:
                rmsn = sd_counts
        else:
            if sd_od.size:
                rmsn = sd_od
            else:
                raise("All weights cannot be zero")

    if eval_rmsn==True:
        global estimated, best_rmsn, best_od, best_simulated_counts
        estimated = X
        if rmsn<best_rmsn:
            global best_od
            best_od = copy.deepcopy(X)
            best_simulated_counts = simulated_counts.flatten()
            best_rmsn=rmsn

    return rmsn

def calibration_handler(obj_func,
                        x0,
                        x_true,
                        x_prior,
                        W,
                        true_counts,
                        true_speeds,
                        path_demand,
                        path_trips,
                        rerouting_prob,
                        tls_tt_pen, 
                        meso_minor_pen, 
                        rerouting_period,
                        rerouting_adaptation,
                        rerouting_adaptation_steps,
                        meso_tls_flow_penalty,
                        priority_factor,
                        n_iterations,
                        ak,
                        ck,
                        p_momentum,
                        bounds):
    '''Wrapper for calibration 
    TODO when the random pertubation lands in a negative range
    As expected, SUMO gives an errors and does not generates new trips
    So optimizer uses the same routes as last time for goodness evaluation
    Need to think a better way to handle this, so that negative values are totally
    avoided'''
    sf = SolutionFinder(obj_func,
                        bounds = bounds,
                        x0 = x0)
    # result = sf.bhop(path_args=(PATH_DEMAND, PATH_TRIPS))
    if which_algo=="spsa":
        result = sf.spsa(path_args=(path_demand, path_trips, x_true, 
                                    x_prior, true_counts, true_speeds),
                        paired=False,
                        a = ak,
                        c = ck,
                        reps = spsa_reps,
                        bounds = sf.bounds,
                        niter=n_iterations,
                        disp=True)

    elif which_algo == "wspsa":
        
        global weight_counts, weight_od, weight_speed
        weight_wspsa = prepare_weight_matrix(W, weight_counts, weight_od, weight_speed)
        result = sf.w_spsa(weights_wspsa= weight_wspsa,
                            path_args=(path_demand, path_trips, x_true, x_prior, true_counts, true_speeds,
                                        rerouting_prob, tls_tt_pen, meso_minor_pen, rerouting_period,
                                        rerouting_adaptation, rerouting_adaptation_steps, meso_tls_flow_penalty,
                                        priority_factor),
                            paired=False,
                            a = ak,
                            c = ck,
                            param_momentum = p_momentum,
                            reps = spsa_reps,
                            bounds = sf.bounds,
                            niter=n_iterations,
                            disp=True)

    else:
        result = sf.rps(path_args=(path_demand, path_trips, x_true, true_counts, true_speeds),
                        paired=False,
                        deltatol=10,
                        feps=0.005,
                        disp=True,
                        deltainit=n_iterations,
                        errorcontrol=False)
    return result

def calibration_handler_out_of_loop(obj_func,
                                    x0,
                                    x_true,
                                    x_prior,
                                    A,
                                    W,
                                    orig_count,
                                    orig_speed,
                                    n_iterations,
                                    ak,
                                    ck,
                                    bounds):
    '''Wrapper for calibration 
    TODO when the random pertubation lands in a negative range
    As expected, SUMO gives an errors and does not generates new trips
    So optimizer uses the same routes as last time for goodness evaluation
    Need to think a better way to handle this, so that negative values are totally
    avoided'''

    sf = SolutionFinder(obj_func,
                        bounds = bounds,
                        x0 = x0)
    global which_algo
    if which_algo == "wspsa":
        global weight_counts, weight_od, weight_speed
        global weight_counts, weight_od, weight_speed
        weight_wspsa = prepare_weight_matrix(W, weight_counts, weight_od, weight_speed = 0)
        # weight speed is zero since it is an analytical simulator and thus no need to use artificial speeds
        # for the calibration process
        result = sf.w_spsa(weights_wspsa = weight_wspsa,
                            path_args=(x_true, x_prior, A, orig_count, orig_speed),
                            paired=False,
                            a = ak,
                            c = ck,
                            gamma = 0.01,
                            alpha = 0.7,
                            reps = spsa_reps,
                            bounds = sf.bounds,
                            niter=n_iterations,
                            disp=True)
    return result			

def black_box_function(tls_tt_pen, 
                       meso_minor_pen, 
                       rerouting_probability,
                       rerouting_period,
                       rerouting_adaptation,
                       rerouting_adaptation_steps,
                       meso_tls_flow_penalty,
                       priority_factor):
    

    rerouting_probability = int(rerouting_probability*10)/10
    rerouting_period = 1 + int(rerouting_period/10)*10
    rerouting_adaptation = 1+ int((rerouting_adaptation)/10)*10
    rerouting_adaptation_steps = int(rerouting_adaptation_steps/10)*10
    meso_minor_pen = int(meso_minor_pen/10)*10
    tls_tt_pen = int(tls_tt_pen*10)/10
    meso_tls_flow_penalty = int(meso_tls_flow_penalty*10)/10
    priority_factor = float(priority_factor)

    true_counts, simulated_counts, \
        true_speeds, simulated_speeds = od_to_counts(pre_string +"/"+OD_FILE_IDENTIFIER+"_best.txt",
                                                    best_od,
                                                    path_trips,
                                                    path_routes,
                                                    path_temp_additional,
                                                    path_simulation_counts,
                                                    path_simulation_speeds,
                                                    rerouting_prob=rerouting_probability,
                                                    tls_tt_pen=tls_tt_pen, 
                                                    meso_minor_pen=meso_minor_pen, 
                                                    rerouting_period=rerouting_period,
                                                    rerouting_adaptation=rerouting_adaptation,
                                                    rerouting_adaptation_steps=rerouting_adaptation_steps,
                                                    meso_tls_flow_penalty=meso_tls_flow_penalty,
                                                    priority_factor = priority_factor)

    rmsn = gof_eval(true_counts, simulated_counts, estimator=estimator)
    # add speeds here?
    return -rmsn

def spsa_tune_function(log_spsa_a, 
                      log_spsa_c):
    spsa_a = np.power(10, log_spsa_a)
    spsa_c = np.power(10, log_spsa_c)
    res = 	calibration_handler_out_of_loop(objective_function_without_simulator,
                                    init_iter,
                                    X_OD,
                                    X_prior,
                                    A,
                                    W_out,
                                    true_counts,
                                    true_speeds,
                                    50,
                                    spsa_a,
                                    spsa_c,
                                    bounds= BOUNDS)

    X = res['x']
    X = np.array([int(i) for i in np.where(X<0, 0, X)])

    simulated_counts = np.zeros((A.shape[1], 1))
    for i in range(A.shape[1]):
        simulated_counts[i] = np.sum(np.multiply(X, A[:,i]).astype(int))
    simulated_counts = simulated_counts.reshape(-1, 1)

    global weight_counts, weight_od, weight_speed
    rmsn = weight_counts * gof_eval(count_init.flatten(), simulated_counts.flatten(), estimator=estimator) +\
           weight_od * gof_eval(X_OD, X, estimator=estimator)

    return -rmsn

if __name__=="__main__":

    execute_scenario = eval(run_scenario)
    synthetic_counts = eval(os.environ.get("synthetic_counts"))
    sim_in_loop = eval(os.environ.get("sim_in_loop"))
    sim_out_loop = eval(os.environ.get("sim_out_loop"))

    init_rerouting_prob = 0.9
    init_tls_tt_pen = 6
    init_meso_minor_pen = 10
    init_rerouting_period = 30
    init_rerouting_adaptation = 5
    init_rerouting_adaptation_steps = 157
    init_meso_tls_flow_penalty = 0.9
    init_priority_factor = 0.01

    if execute_scenario:
        create_scenario(path_trips, path_routes, PATH_DEMAND)
        trip_validator(path_trips, path_routes)

        # Enabling online routing
        copy_additonal(PATH_ADDITIONAL, path_temp_additional)
        run_simulation(path_trips, 
                        path_routes, 
                        path_temp_additional, 
                        routing_how = "reroute",
                        rerouting_prob = init_rerouting_prob,
                        tls_tt_pen = init_tls_tt_pen, 
                        meso_minor_pen =	init_meso_minor_pen, 
                        rerouting_period = init_rerouting_period,
                        rerouting_adaptation = init_rerouting_adaptation,
                        rerouting_adaptation_steps = init_rerouting_adaptation_steps,
                        meso_tls_flow_penalty = init_meso_tls_flow_penalty,
                        priority_factor = init_priority_factor)

    if synthetic_counts:
        # only one time for scenario creation with Synthetic demand
        create_synthetic_counts(path_additional = path_temp_additional,
                                path_output_counts = path_simulation_counts,
                                path_real_counts = PATH_REAL_COUNT,
                                sim_type = "meso",
                                count_noise = count_noise_param)
        create_synthetic_speeds(path_additional = path_temp_additional,
                                path_output_speeds = path_simulation_speeds,
                                path_real_speeds = PATH_REAL_SPEED)
        ### Add predicted counts here
        det_counts_for_weights = ""
    else:	
        # path to real - observed data
        det_counts_for_weights = ""
        counts_file = ""
        speed_file = ""

    if not synthetic_counts:
        df_real = pd.read_csv(counts_file)
        df_real.to_csv(PATH_REAL_COUNT, index=None)
        df_real_speed = pd.read_csv(speed_file)
        df_real_speed.to_csv(PATH_REAL_SPEED, index=None)

    true_counts, initial_counts = get_true_simulated(path_output_counts=path_simulation_counts, 
                                                    path_real_counts=PATH_REAL_COUNT,
                                                    # path_additional=path_temp_additional,
                                                    flow_col="count", 
                                                    detector_col="det_id",
                                                    time_col="hour",
                                                    sim_type = "meso")

    true_speeds, initial_speeds = get_true_simulated_speeds(path_output_speeds = path_simulation_speeds,
                                                            path_real_speeds = PATH_REAL_SPEED,
                                                            speed_col= "speed",#'q', 
                                                            detector_col= "det_id", #'iu_ac',
                                                            time_col= "hour")

    rmsn = gof_eval(true_counts, initial_counts, estimator=estimator)
    rmsn_speeds = gof_eval(true_speeds, initial_speeds, estimator=estimator)

    # When using synthetic demand, and counts, this is equal to 0
    print("Initial Weighted Count "+ estimator+ ": "+ str(np.round(rmsn*weight_counts, 4)))
    print("Initial Count " + estimator +": "+ str(rmsn))
    print("Initial Speeds " + estimator +": "+ str(rmsn_speeds))
    for seq in range(0, n_sequential):

        if eval(calibrate_demand): 

            if seq==0:
                X_OD = np.array(OD_txt_to_dataframe(path = PATH_DEMAND))
                X_base_true = copy.deepcopy(X_OD)
                # add artifical noise and bias to the  demand matrix which is zero and 1 in this case
                initial_solution = np.array(add_noise(X_OD, int(noise_param)/100, mu=bias_param))
                init = np.where(initial_solution<0, 0, initial_solution)
                init_seq = copy.deepcopy(init)
            else:
                X_OD = best_od
                # add artifical noise and bias to the  demand matrix which is zero and 1 in this case
                initial_solution = np.array(add_noise(X_OD, 0, mu=1))
                init = np.where(initial_solution<0, 0, initial_solution)

            count_init, sim_init_counts,\
                speeds_init, sim_init_speeds = od_to_counts(pre_string +"/"+OD_FILE_IDENTIFIER+"_init.txt",
                                                            init,
                                                            path_trips,
                                                            path_routes,
                                                            path_temp_additional,
                                                            path_simulation_counts,
                                                            path_simulation_speeds)
            
            W, A = wspsa_update_wrapper(true_count_file = PATH_REAL_COUNT, 
                                    det_counts = det_counts_for_weights,
                                    is_synthetic = synthetic_counts, 
                                    wscenario = SCENARIO, 
                                    threshold_val = wspsa_thrshold,
                                    binary_rounding = True)

            if which_algo=="wspsa":
                assert W.shape[0] == len(X_OD), "Shapes of Weight matrix: "+ str(W.shape[0]) +" and OD matrix: "+ str(len(X_OD))+" are not equal"
                assert W.shape[1] == len(true_counts), "Shapes of Weight matrix: "+ str(W.shape[1]) +" and Count matrix: "+ str(len(true_counts))+" are not equal"
                assert W.shape[1] == len(true_speeds), "Shapes of Weight matrix: "+ str(W.shape[1]) +" and Speed matrix: "+ str(len(true_speeds))+" are not equal"

            if weight_counts!=0:
                if bias_correction_method =="naive":
                    estimated_bias_factor = np.sum(sim_init_counts)/np.sum(count_init)
                elif bias_correction_method == "weighted":
                    estimated_bias_counts = sim_init_counts/(true_counts+1e-8)
                    estimated_bias_factor_hat = np.sum(sim_init_counts)/np.sum(count_init)
                    estimated_bias_factor = (W @ estimated_bias_counts).flatten()/W.sum(axis=1)
                    estimated_bias_factor = np.nan_to_num(estimated_bias_factor, nan=estimated_bias_factor_hat)
                    estimated_bias_factor = estimated_bias_factor.flatten()
                else:
                    raise("Please enter a valid bias correction method")
                domain_lower_bound = 0.1
                domain_upper_bound = (2/estimated_bias_factor) - domain_lower_bound
                # assuming spillback times are not very long such as demand of zones
                # is spilling onto the next calibration interval
                if correction_heuristic == False:
                    corrected_od = init
                else:
                    corrected_od = init/estimated_bias_factor        
            else:
                corrected_od = init
                domain_lower_bound = 0.1
                domain_upper_bound = 3

            rmsn_c = gof_eval(count_init, sim_init_counts, estimator=estimator)
            rmsn_s = gof_eval(speeds_init, sim_init_speeds, estimator=estimator)
            rmsn_od = gof_eval(X_base_true, init, estimator=estimator)

            if only_bias_correction:
                res_dict["count_val"].append(rmsn_c)
                res_dict["speed_val"].append(rmsn_s)
                res_dict["od_val"].append(rmsn_od)
                res_dict["f_val"].append(weight_od*rmsn_od + weight_counts*rmsn_c + weight_speed*rmsn_s)

                count_init, sim_corrected_counts,\
                speeds_init, sim_corrected_speeds = od_to_counts(pre_string +"/"+OD_FILE_IDENTIFIER+"_init.txt",
                                                            corrected_od,
                                                            path_trips,
                                                            path_routes,
                                                            path_temp_additional,
                                                            path_simulation_counts,
                                                            path_simulation_speeds)

                rmsn_c = gof_eval(count_init, sim_corrected_counts, estimator=estimator)
                rmsn_s = gof_eval(speeds_init, sim_corrected_speeds, estimator=estimator)
                rmsn_od = gof_eval(X_base_true, corrected_od, estimator=estimator)

                res_dict["count_val"].append(rmsn_c)
                res_dict["speed_val"].append(rmsn_s)
                res_dict["od_val"].append(rmsn_od)
                res_dict["f_val"].append(weight_od*rmsn_od + weight_counts*rmsn_c + weight_speed*rmsn_s)

                save_counts = pd.DataFrame({"real": count_init.flatten(), 
                                            'initial': sim_init_counts.flatten(),
                                            'simulated': sim_corrected_counts.flatten()})
                save_counts.to_csv(pre_string+"/"+"counts.csv")

                save_speeds = pd.DataFrame({"real": speeds_init.flatten(), 
                                            'initial': sim_init_speeds.flatten(),
                                            'simulated': sim_corrected_speeds.flatten()})
                save_speeds.to_csv(pre_string+"/"+"speeds.csv")

                save_od = pd.DataFrame({"real": X_base_true.flatten(), 
                                        'initial': init_seq.flatten(),
                                        'simulated': corrected_od.flatten()})
                save_od.to_csv(pre_string+"/mean_"+"od.csv", index=None)
        
                res_dict = save_params(res_dict, 
                                ['noise_param', 
                                'bias_param'                        
                                ])

                with open(pre_string+'/results_'+timestr+'.json', 'w') as fp:
                        json.dump(res_dict, fp)

            else:
                X_domain = copy.deepcopy(init)
                if bias_correction_method =="naive":
                    BOUNDS = np.array([[domain_lower_bound*i, domain_upper_bound*i] for i in X_domain])
                else:
                    BOUNDS = np.array([[domain_lower_bound*i, domain_upper_bound[i]*i] for i in X_domain])

                rmsn_od_bias_corrected = gof_eval(X_base_true, corrected_od, estimator=estimator)

                # When using synthetic demand, and counts, this is equal to 0
                print("B-N Weighted Count "+ estimator+ ": "+ str(np.round(rmsn_c*weight_counts, 4)))
                print("B-N Count "+ estimator+ ": "+ str(rmsn_c))
                print("B-N Speed "+ estimator+ ": "+ str(rmsn_s))
                print("B-N OD "+ estimator+ ": "+ str(rmsn_od))

                res_dict["count_val"].append(rmsn_c)
                res_dict["speed_val"].append(rmsn_s)
                res_dict["od_val"].append(rmsn_od)
                res_dict["f_val"].append(weight_od*rmsn_od + weight_counts*rmsn_c + weight_speed*rmsn_s)

                print("B-N Corrected OD "+ estimator+ ": "+ str(rmsn_od_bias_corrected))

                X_OD = np.array(X_OD)

                W_out = copy.deepcopy(W)
                W_in = copy.deepcopy(W)

                rerouting_probability = init_rerouting_prob
                rerouting_period = init_tls_tt_pen
                rerouting_adaptation = init_meso_minor_pen
                rerouting_adaptation_steps = init_rerouting_adaptation
                meso_minor_pen = init_meso_minor_pen
                tls_tt_pen = init_tls_tt_pen
                meso_tls_flow_penalty = init_meso_tls_flow_penalty
                priority_factor = init_priority_factor

                # SPSA parameter tuner
                if weight_counts!=0:
                    # spsa_autotune = True
                    if spsa_autotune == True:
                        init_iter = copy.deepcopy(corrected_od)
                        X_prior = copy.deepcopy(init_iter)

                        pbounds = {'log_spsa_a': (-7, -1),
                                'log_spsa_c': (-2, 1)
                                }

                        optimizer = BayesianOptimization(
                                    f = spsa_tune_function,
                                    pbounds=pbounds,
                                    random_state=1,
                                    allow_duplicate_points=True
                                    )
                        try:
                            ### previously seen points are used to explore the new points
                            load_logs(optimizer, logs=[pre_string+"/logs_spsa_tune.json"])
                        except FileNotFoundError:
                            logger = JSONLogger(path=pre_string+"/logs_spsa_tune.json")	

                        logger = JSONLogger(path=pre_string+"/logs_spsa_tune.json")
                        
                        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
                        optimizer.maximize(init_points=n_init_spsa_tune,
                                            n_iter=n_iterate_spsa_tune,
                                            )
                        
                        best_params = optimizer.max['params']
                        spsa_a = np.power(10, best_params['log_spsa_a'])
                        spsa_c = np.power(10, best_params['log_spsa_c'])
                else:
                    print("Using manual values of SPSA c and a")
                    print(spsa_a_init, spsa_c_init)

                od_bag_list = []
                param_momentum = momentum_beta
                for bagging in range(bagging_run):
                    # global best_rmsn

                    print(f"Run: {bagging}")
                    if not eval(set_spa):
                        init_iter = copy.deepcopy(corrected_od)
                    else:
                        if bagging==0:
                            init_iter = copy.deepcopy(corrected_od)
                        else:
                            init_iter = copy.deepcopy(best_od)

                    X_prior = copy.deepcopy(init_iter)
                    
                    noised_solution = np.array(add_noise(init_iter, int(exploration_noise)/100, mu=1))
                    init_iter = np.where(noised_solution<0, 0, noised_solution)

                    spsa_a = spsa_a_init
                    spsa_c = spsa_c_init
                    for counter_out in range(n_iterations):
                        ####### Demand Calibration ######
                        if sim_out_loop:
                            best_rmsn = 10000000
                            best_od = []
                            best_simulated_speeds = 0
                            best_simulated_counts = 0
                            ## with simulation out-of-loop
                            res = 	calibration_handler_out_of_loop(objective_function_without_simulator,
                                                                    init_iter,
                                                                    X_base_true,
                                                                    X_prior,
                                                                    A,
                                                                    W_out,
                                                                    true_counts,
                                                                    true_speeds,
                                                                    sim_out_iterations,
                                                                    spsa_a_out_sim,
                                                                    spsa_c_out_sim,
                                                                    bounds= BOUNDS)
                            spsa_a_out_sim = spsa_a_out_sim/(learning_decay_factor**sim_out_iterations)
                            spsa_c_out_sim = spsa_c_out_sim/(learning_decay_factor**sim_out_iterations)

                            init_iter = res['x']

                            OD_dataframe_to_txt(pre_string +"/"+OD_FILE_IDENTIFIER+"_n.txt", init_iter, path_base_file=PATH_DEMAND)

                            _, _, _, _ = 	od_to_counts(pre_string +"/"+OD_FILE_IDENTIFIER+"_n.txt",
                                                        init_iter,
                                                        path_trips,
                                                        path_routes,
                                                        path_temp_additional,
                                                        path_simulation_counts,
                                                        path_simulation_speeds,
                                                        rerouting_prob=rerouting_probability,
                                                        tls_tt_pen=tls_tt_pen, 
                                                        meso_minor_pen=meso_minor_pen, 
                                                        rerouting_period=rerouting_period,
                                                        rerouting_adaptation=rerouting_adaptation,
                                                        rerouting_adaptation_steps=rerouting_adaptation_steps,
                                                        meso_tls_flow_penalty=meso_tls_flow_penalty,
                                                        priority_factor = priority_factor)
                            W_prev = copy.deepcopy(W_out)
                            W_out, A = wspsa_update_wrapper(true_count_file=PATH_REAL_COUNT, 
                                                            det_counts=det_counts_for_weights,
                                                            is_synthetic= synthetic_counts,
                                                            wscenario=SCENARIO, 
                                                            threshold_val=wspsa_thrshold)
                            # W_out = (W_out+W_prev)/2

                            W_in = copy.deepcopy(W_out)

                            if which_algo=="wspsa":
                                assert W_out.shape[0] == len(X_OD), "Shapes of Weight matrix: "+ str(W_out.shape[0]) +" and OD matrix: "+ str(len(X_OD))+" are not equal"
                                assert W_out.shape[1] == len(true_counts), "Shapes of Weight matrix: "+ str(W_out.shape[1]) +" and Count matrix: "+ str(len(true_counts))+" are not equal"
                                assert W_out.shape[1] == len(true_speeds), "Shapes of Weight matrix: "+ str(W_out.shape[1]) +" and Count matrix: "+ str(len(true_speeds))+" are not equal"

                        if sim_in_loop:
                            best_rmsn = 10000000
                            best_od = []
                            best_simulated_speeds = 0
                            best_simulated_counts = 0
                            ### with simulation in loop
                            res = 	calibration_handler(objective_function,
                                                        init_iter,
                                                        X_base_true,
                                                        X_prior,
                                                        W_in,
                                                        true_counts,
                                                        true_speeds,
                                                        path_temp_demand,
                                                        path_trips,
                                                        rerouting_probability,
                                                        rerouting_period,
                                                        rerouting_adaptation,
                                                        rerouting_adaptation_steps,
                                                        meso_minor_pen,
                                                        tls_tt_pen,
                                                        meso_tls_flow_penalty,
                                                        priority_factor,
                                                        sim_in_iterations,
                                                        spsa_a,
                                                        spsa_c,
                                                        param_momentum,
                                                        bounds= BOUNDS) 

                            spsa_a = spsa_a/(learning_decay_factor**sim_in_iterations)
                            spsa_c = spsa_c/(learning_decay_factor**sim_in_iterations)

                            init_iter = res['x']

                            W_last = copy.deepcopy(W_in)
                            if counter_out<=n_iterations-1:
                                W_in, A = wspsa_update_wrapper(true_count_file=PATH_REAL_COUNT, 
                                                        det_counts=det_counts_for_weights,
                                                        is_synthetic= synthetic_counts, 
                                                        wscenario=SCENARIO, 
                                                        threshold_val=wspsa_thrshold)
                                
                                # W_out = (W_out+W_in)/2
                                W_in = (0.1*W_last + 0.9*W_in)

                            if which_algo=="wspsa":
                                assert W_in.shape[0] == len(X_OD), "Shapes of Weight matrix: "+ str(W_in.shape[0]) +" and OD matrix: "+ str(len(X_OD))+" are not equal"
                                assert W_in.shape[1] == len(true_counts), "Shapes of Weight matrix: "+ str(W_in.shape[1]) +" and Count matrix: "+ str(len(true_counts))+" are not equal"
                                assert W_in.shape[1] == len(true_speeds), "Shapes of Weight matrix: "+ str(W_in.shape[1]) +" and Count matrix: "+ str(len(true_speeds))+" are not equal"
                    od_bag_list.append(best_od)
                    save_od = pd.DataFrame({"real": X_base_true, 'simulated': best_od.flatten()})
                    save_od.to_csv(pre_string+"/"+str(bagging)+"_"+"od.csv", index=None)
            
                od_mean = np.mean(np.array(od_bag_list), axis=0).astype(int)
                best_od = copy.deepcopy(od_mean)
                
                rmsn_od = gof_eval(X_base_true, od_mean, estimator=estimator)
                print("Final bagged OD error "+ estimator+ ": "+ str(np.round(rmsn_od, 4)))

                _, sim_final_counts,\
                    _, sim_final_speeds = od_to_counts(pre_string +"/"+OD_FILE_IDENTIFIER+"_mean.txt",
                                                            od_mean,
                                                            path_trips,
                                                            path_routes,
                                                            path_temp_additional,
                                                            path_simulation_counts,
                                                            path_simulation_speeds)

                save_counts = pd.DataFrame({"real": true_counts.flatten(), 
                                            'initial': sim_init_counts.flatten(),
                                            'simulated': sim_final_counts.flatten()})

                save_speeds = pd.DataFrame({"real": true_speeds.flatten(), 
                                            'initial': sim_init_speeds.flatten(),
                                            'simulated': sim_final_speeds.flatten()})

                save_od = pd.DataFrame({"real": X_base_true.flatten(), 
                                        'initial': init_seq.flatten(),
                                        'simulated': od_mean.flatten()})

                save_counts.to_csv(pre_string+"/mean_"+str(seq)+"_"+str(bagging_run)+"_"+"counts.csv", index=None)
                save_speeds.to_csv(pre_string+"/mean_"+str(seq)+"_"+str(bagging_run)+"_"+"speeds.csv", index=None)
                save_od.to_csv(pre_string+"/mean_"+str(seq)+"_"+str(bagging_run)+"_"+"od.csv", index=None)
                # PATH_DEMAND = pre_string +"/"+OD_FILE_IDENTIFIER+"_mean.txt"
        
        od_mean = pd.read_csv(pre_string+"/mean_"+str(seq)+"_"+str(bagging_run)+"_"+"od.csv")
        od_mean = np.array(od_mean.simulated)
        best_od = copy.deepcopy(od_mean)

        ####### Supply Calibration ######
        if eval(calibrate_supply):

            pbounds = {'rerouting_probability': (0.1, 0.5),
                        'rerouting_period': (30, 90),
                        'rerouting_adaptation':(5, 120),
                        'rerouting_adaptation_steps': (120, 240),
                        'meso_minor_pen':(0, 60),
                        'tls_tt_pen':(0, 1),
                        'meso_tls_flow_penalty':(0, 1),
                        'priority_factor': (0.0001, 1)
                    }

            optimizer = BayesianOptimization(
                        f=black_box_function,
                        pbounds=pbounds,
                        random_state=1,
                        allow_duplicate_points=True
                        )
            try:
                ### previously seen points are used to explore the new points
                load_logs(optimizer, logs=[pre_string+"/logs_supply.json"])
            except FileNotFoundError:
                logger = JSONLogger(path=pre_string+"/logs_supply.json")	

            logger = JSONLogger(path=pre_string+"/logs_supply.json")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


            optimizer.maximize(
                            init_points=n_init_supply,
                            n_iter=n_iterate_supply,
                            )

            supply_res = optimizer.max['params']

            rerouting_probability = int(supply_res['rerouting_probability']*10)/10
            rerouting_period = 1 + int(supply_res['rerouting_period']/10)*10
            rerouting_adaptation = 1+ int((supply_res['rerouting_adaptation'])/10)*10
            rerouting_adaptation_steps = int(supply_res['rerouting_adaptation_steps']/10)*10
            meso_minor_pen = int(supply_res['meso_minor_pen']/10)*10
            tls_tt_pen = int(supply_res['tls_tt_pen']*10)/10
            meso_tls_flow_penalty = int(supply_res['meso_tls_flow_penalty']*10)/10
            priority_factor = float(supply_res['priority_factor'])

            bayes_optim_target_val = abs(optimizer.max['target'])

            if res_dict["best_supply_count_rmsn"] > bayes_optim_target_val:
                res_dict["best_supply_count_rmsn"] = bayes_optim_target_val

            res_dict["f_val_supply"].append(bayes_optim_target_val)
        

        _, sim_final_counts,\
        _, sim_final_speeds = od_to_counts(pre_string +"/"+OD_FILE_IDENTIFIER+"_mean.txt",
                                            od_mean,
                                            path_trips,
                                            path_routes,
                                            path_temp_additional,
                                            path_simulation_counts,
                                            path_simulation_speeds,                 
                                            rerouting_prob=rerouting_probability,
                                            tls_tt_pen=tls_tt_pen, 
                                            meso_minor_pen=meso_minor_pen, 
                                            rerouting_period=rerouting_period,
                                            rerouting_adaptation=rerouting_adaptation,
                                            rerouting_adaptation_steps=rerouting_adaptation_steps,
                                            meso_tls_flow_penalty=meso_tls_flow_penalty,
                                            priority_factor=priority_factor)

    res_dict = save_params(res_dict, ['spsa_a_init', 'spsa_c_init', 
                                    'n_iterations', 'sim_in_iterations',
                                    'noise_param', 'bias_param', 
                                    'spsa_reps',
                                    'weight_counts', 'weight_od', 'weight_speed', 
                                    'wspsa_thrshold',
                                    'rerouting_probability',
                                    'tls_tt_pen',
                                    'meso_minor_pen',
                                    'rerouting_period',
                                    'rerouting_adaptation',
                                    'rerouting_adaptation_steps',
                                    'meso_tls_flow_penalty',
                                    'priority_factor'
                                    # ,'noise_prior'
                                    ])

    with open(pre_string+'/results_'+timestr+'.json', 'w') as fp:
            print(res_dict)
            json.dump(res_dict, fp)

    plotting = True
    if plotting:
        
        fig, ax = plt.subplots(3,2,figsize=(7,9))

        fig, ax = utilities.plot_45_degree_plots(fig, ax, X_base_true, best_od, init_seq,
                                                weight_counts, weight_od, weight_speed,
                                                true_counts, best_simulated_counts, sim_init_counts,
                                                true_speeds, best_simulated_speeds, sim_init_speeds, 
                                                spsa_a_init, spsa_c_init, noise_param, bias_param)
        
        plt.tight_layout()
        plt.savefig(pre_string+"/img_results_"+str(bagging)+"_n_"+str(noise_param)+"_"+"b_"+str(int(100*bias_param))+"_"+\
                    "_"+str(n_iterations)+"_"+str(weight_od)+"_"+estimator+"_"+\
                    "_"+str(weight_counts)+"_"+str(weight_speed)+"_"+str(which_algo)+"_"+\
                    str(wspsa_thrshold)+".png", dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(1,2,figsize=(6,3))
        ax = utilities.plot_loss_curve(ax, res_dict, spsa_a_init, spsa_c_init, estimator)
        plt.suptitle("Result of OD Estimation", fontsize=16)

        plt.legend()
        plt.tight_layout()
        plt.savefig(pre_string+"/img_loss_"+timestr+"_n_"+str(noise_param)+"_"+"b_"+str(int(100*bias_param))+"_"+\
                    "_"+str(n_iterations)+"_"+estimator+"_"+\
                    str(weight_od)+"_"+str(weight_counts)+"_"+str(weight_speed)+"_"+\
                    str(which_algo)+"_"+str(wspsa_thrshold)+".png", dpi=300)
        plt.close()

        if sim_in_loop==False:
            if sim_out_loop==True:
                only_out_of_simulator=True
        else:
            only_out_of_simulator=False

        fig, ax = plt.subplots(3,2,figsize=(7,9))

        fig, ax = utilities.plot_45_degree_plots(fig, ax, X_base_true, od_mean, init_seq,
                                                weight_counts, weight_od, weight_speed,
                                                true_counts, sim_final_counts.flatten(), sim_init_counts,
                                                true_speeds, sim_final_speeds, sim_init_speeds, 
                                                spsa_a_init, spsa_c_init, noise_param, bias_param,
                                                only_out_of_simulator)
        
        plt.tight_layout()
        plt.savefig(pre_string+"/img_results_"+timestr+"_n_"+str(noise_param)+"_"+"b_"+str(int(100*bias_param))+"_"+\
                    "_"+str(n_iterations)+"_"+str(weight_od)+"_"+estimator+"_"+\
                    "_"+str(weight_counts)+"_"+str(weight_speed)+"_"+str(which_algo)+"_"+\
                    str(wspsa_thrshold)+".png", dpi=300)