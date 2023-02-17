import pandas as pd
import numpy as np
import time, subprocess, sys, os
from tqdm import tqdm
import seaborn as sns
from collections import Counter
import json
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import numpy as np
import re


def save_var_value(OD_sim, sim_counts, gof):
    '''we save the model outputs from every simulation
    to use later for a machine learning model. please 
    check that the savepath is out of the scenario folder so
    that the outputs are not deleted everytime a model is run'''
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    sim_output = dict()

    param = ['OD_sim', 'sim_counts', 'gof']

    for p in param:
        # print(eval(p))

        try:
            if len(eval(p))>1:
                sim_output[str(p)] = []
                for j in eval(p):
                    sim_output[str(p)].append(j)
        except:
            sim_output[str(p)] = eval(p)

    return sim_output

def plot_loss_curve(ax, results, a, c, estimator="smape"):
    # ax[0].set_title("Result of OD Estimation")
    ax[0].set_ylabel(estimator, fontsize=16)
    ax[1].set_ylabel(estimator, fontsize=16)
    ax[0].set_ylim([0,max(results["f_val"])])
    ax[1].set_ylim([0,max([max(results["speed_val"]), \
                           max(results['count_val']),\
                           max(results['od_val'])])])
    ax[0].plot(results["f_val"], label="Weighted loss")
    ax[1].plot(results["count_val"], label="Count loss")
    ax[1].plot(results["speed_val"], label="Speed loss")
    ax[1].plot(results["od_val"], label="OD loss")
    ax[0].set_title("a %0.5f , c %0.5f" %(a, c), fontsize=16)
    ax[1].set_title("a %0.5f , c %0.5f" %(a, c), fontsize=16)
    return ax

def plot_loss_curve_synthetic(ax, results, a, c, estimator="smape"):
    # ax[0].set_title("Result of OD Estimation")
    ax.set_ylabel(estimator, fontsize=16)
    ax.set_ylim([0,max(results["f_val"])])
    ax.plot(results["f_val"], label="Weighted loss")
    ax.set_title("a %0.5f , c %0.5f" %(a, c), fontsize=16)
    return ax

def plot_45_degree_plots(fig, ax, true_OD, estimated_OD, init_OD,
                            weight_counts, weight_od, weight_speed,
                            true_counts, simulated_counts, initial_counts,
                            true_speeds, simulated_speeds, initial_speed, 
                            spsa_a, spsa_c, noise, bias, only_out_of_simulator=False):
    
    ax[0,0].set_ylabel("Initial")
    ax[0,0].set_xlabel("True")
    ax[0,1].scatter(true_OD, estimated_OD, c='c', alpha=0.6, label="Estimated vs True", s=2)
    ax[0,0].scatter(true_OD, init_OD, c='r', alpha=0.6, label=f"Initial (Bias: {int(100*(bias-1))}%, Noise: ±{noise}%) vs true", s=2)

    sns.regplot(x=true_OD, y=estimated_OD, ax=ax[0,1], scatter=False, robust=False, ci=68, line_kws={'alpha':0.4})
    
    ax[0,1].set_ylabel("Estimated")
    ax[0,1].set_xlabel("True")
    ax[0,0].legend()
    ax[0,1].legend()
    
    OD_max = np.maximum(np.max(true_OD), np.max(estimated_OD))
    ax[0,0].plot([0,OD_max],[0,OD_max])
    ax[0,1].plot([0,OD_max],[0,OD_max])
    ax[0,0].set_xlim([-2, OD_max+5])
    ax[0,0].set_ylim([-2, OD_max+5])
    ax[0,1].set_xlim([-2, OD_max+5])
    ax[0,1].set_ylim([-2, OD_max+5])

    ax[0,0].set_title("ODs")
    ax[0,1].set_title("ODs")
    ax[2,0].set_title("Speed")
    ax[2,1].set_title("speed")
    ax[1,0].set_title("Counts")
    ax[1,1].set_title("Counts")
    ax[1,1].set_xlabel("True")
    ax[1,1].set_ylabel("Simulated")
    ax[1,0].set_xlabel("True")
    ax[1,0].set_ylabel("Initial")
    ax[2,1].set_xlabel("True")
    ax[2,1].set_ylabel("Simulated")
    ax[2,0].set_xlabel("True")
    ax[2,0].set_ylabel("Initial")
    
    ax[0,1].set_title('Weight = '+str(weight_od))
    ax[1,1].set_title('Weight = '+str(weight_counts))
    ax[2,1].set_title('Weight = '+str(weight_speed))

    count_max = np.max(np.maximum.reduce([initial_counts, simulated_counts, true_counts]))
    ax[1,1].scatter(true_counts, simulated_counts, c='b', label="Simulated vs True", s=2, alpha=0.5)
     
    ax[1,0].scatter(true_counts, initial_counts, c='b', alpha=0.6, label="Initial vs True", s=2)
    ax[1,1].set_ylim([-2, count_max+5])
    ax[1,1].set_xlim([-2, count_max+5])
    ax[1,0].set_ylim([-2, count_max+5])
    ax[1,0].set_xlim([-2, count_max+5])
    ax[1,1].plot([0,count_max],[0,count_max])
    ax[1,0].plot([0,count_max],[0,count_max])

    speed_max = np.maximum(np.max(true_speeds), np.max(simulated_speeds))
    ax[2,1].set_xlim([-2, speed_max+5])
    ax[2,1].set_ylim([-2, speed_max+5])
    ax[2,0].set_xlim([-2, speed_max+5])
    ax[2,0].set_ylim([-2, speed_max+5])
    ax[2,0].plot([0,speed_max],[0,speed_max])
    ax[2,1].plot([0,speed_max],[0,speed_max])
    ax[2,0].scatter(true_speeds, initial_speed, c='b', label="Initial vs True", s=2, alpha=0.5)

    if not only_out_of_simulator:
        sns.regplot(x=true_speeds, y=simulated_speeds, ax=ax[2,1], robust=False, scatter=False, line_kws={'alpha':0.4})
        ax[2,1].scatter(true_speeds, simulated_speeds, c='b', alpha=0.6, label="Simulated vs True", s=2)
    fig.suptitle("a %0.5f , c %0.5f" %(spsa_a, spsa_c))
    return fig, ax


def route_visualizer(scenario = "../../san_francisco/"):
    '''Function to visualize the routes to visualize the edge prevalence or
    which edges are used by how many vehicles. I created this to check the route choice and for
    debugging purposes'''
    SUMO_HOME = os.environ.get("SUMO_HOME")
    scenario_folder = sys.argv[1]

    list_edges = []
    Paths = {'routes_file' : scenario+scenario_folder+"/routes.rou.xml"}

    subprocess.run(SUMO_HOME+"/tools/xml/xml2csv.py " + Paths['routes_file'], shell=True)
    dfr = pd.read_csv(Paths['routes_file'][:-3]+"csv", sep=';')

    for i in tqdm(range(len(dfr))):
        list_edges.extend(dfr.loc[i, 'route_edges'].split(" "))
    geo_objects = json.load(open(scenario+ "network.geojson"))
    prevalence_edges = dict(Counter(list_edges))

    for i in tqdm(range(len(geo_objects['features']))):
        edge_id = geo_objects['features'][i]['properties']['id']
        try:
            geo_objects['features'][i]['properties']['edge_prevalence'] = prevalence_edges[edge_id]
        except KeyError:
            geo_objects['features'][i]['properties']['edge_prevalence'] = 0

    with open(scenario+scenario_folder+"/network_edge_prevalence.geojson", 'w') as f:
        json.dump(geo_objects, f)
    print("Done")

def get_errors(scenario='san_francisco',
            folder = "temp"):
    '''to visualzie the errors on the map'''

    root = ET.parse('../'+scenario+'/additional.add.xml').getroot()
    id = []
    lat = []
    lon  = []
    lane = []
    for children in root.getchildren():
        id.append(children.get('id'))
        lane.append(children.get('lane'))

        if len(children.getchildren()) ==1:
            lat.append(float(children.getchildren()[0].attrib['value'].split(",")[0]))
            lon.append(float(children.getchildren()[0].attrib['value'].split(",")[1]))
        else:
            lat.append(0)
            lon.append(0)
    
    df  = pd.DataFrame({'id': id, 'lane_id': lane, 'lat': lat, 'lon': lon})
    df_edge = df[df.lat!=0]
    df_edge = df_edge.groupby([s.split('_')[0] for s in df_edge.lane_id]).mean()
    df_edge = df_edge.reset_index()

    df_edge.to_csv("../data/"+scenario+"_detector_mapping.csv", index='edge_id')
    simulated = pd.read_csv("../"+scenario+"/"+folder+"/out.csv", sep=';')
    real = pd.read_csv("../"+scenario+"/"+folder+"/real_counts.csv")

    df_sim = []
    for interval in range(5,10,1):
        list_det = []
        list_count = []
        temp = simulated[(simulated.interval_begin>=(interval*3600)) & (simulated.interval_begin<3600*(interval+1))]
        temp = temp.copy()
        detector_ids = list(temp['interval_id'].apply(lambda x: x.split("_")[1]))
        temp.loc[:,'det_id']= detector_ids
        list_det.extend(list(temp.det_id))
        list_count.extend(list(temp.interval_entered))
        
        temp_real = real[real.hour==interval*3600]
        df_temp = pd.DataFrame({'det_id': list_det, 'count':list_count})
        df_temp['hour'] = interval*3600
        df_temp = df_temp.groupby(['det_id','hour']).sum().reset_index()
        
        temp_merge = pd.merge(temp_real, df_temp, left_on="det_id", right_on='det_id')
        df_sim.append(temp_merge)
    df_both = pd.concat(df_sim)

    df_both = pd.merge(df_both, df_edge, left_on="det_id", right_on = "index")
    df_both['pe']  = df_both.apply(lambda x: (x.count_x - x.count_y), axis=1)
    df_both.to_csv("../"+scenario+"/"+folder+"/errors.csv")


if __name__ == "__main__":
    # route_visualizer()
    get_errors()
