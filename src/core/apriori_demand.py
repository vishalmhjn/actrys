import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, WhiteKernel, RBF, RationalQuadratic


DEMAND_INTERVAL = 900/3600
TOD_START= 0
TOD_END= 24

def temporal_kernel(x, y, interval = 1):
    
    num_intervals = int(24/interval)
    kernel = Matern() + RationalQuadratic() # + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,
            random_state=0).fit(x, y)
    gpr.score(x, y)

    x_new = np.array(list(np.linspace(0,24,num_intervals))).reshape(-1,1)
    # print(x_new)
    y_fitted = gpr.predict(x_new, return_std=True)[0]


    return y_fitted, gpr

def OD_init_to_txt(df_demand, 
                    TOD,
                    kernel_shape,
                    path_sample,
                    path_od_save):

    '''Apply the temporal kernel to the spatial demand to
    get the synhtetic trips and format them as SUMO input'''

    assert len(TOD)==len(kernel_shape)

    total_flow = 0
    for interval, hour in enumerate(TOD):
        # print(interval)
        path_save = path_od_save[:-4] +\
                            "_" + str(float(hour)) + \
                            "_" + str(float(hour+DEMAND_INTERVAL)) +\
                            ".txt"
        path_base = path_sample
        text_file = open(path_save, "w+")
        base_file = open(path_base, "r")

        counter = 0

        df = pd.read_csv(path_base, sep=" ").reset_index()
        demand_factor = 1 # define a demand factor as integer to multiply demand


                
                
        text_file.write("$OR;D2"+'\n'+\
                        "* From-Time  To-Time"+'\n' + \
                        format((interval//4 + 0.6*DEMAND_INTERVAL*(interval%4)), '.2f') + " " + \
                        format(((interval+1)//4 + 0.6*DEMAND_INTERVAL*((interval+1)%4)),'.2f') +'\n' + \
                        "* Factor" + '\n' + \
                        "4.00" + '\n')
        for i in range(len(df_demand)):
                origin = str(df_demand['from'].iloc[i])
                destination = str(df_demand['to'].iloc[i])

                # Apply a demand temporal kernel to assign weights to the demand
                flow = str(int(df_demand.trips.iloc[i]*kernel_shape[interval]/max(kernel_shape)))

                total_flow+=int(flow)

                if i==len(df_demand):
                    text_file.write(origin+ " " + destination + " " + flow)
                else:
                    text_file.write(origin+ " " + destination + " " + flow  + '\n')
            
        text_file.close()
        base_file.close()
        print("Total trips in a day %d" %total_flow)


if __name__ == "__main__":

    pass
    print("Done!!!")


