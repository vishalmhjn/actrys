import numpy as np

def gof_eval(data, data_simulated, estimator="rmsn"):
    
    assert data.shape == data_simulated.shape

    if estimator=="rmsn":
        diff = (data - data_simulated)**2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(len(diff) * sum_diff)/data.sum(), 6)
    elif estimator=="rmse":
        diff = (data - data_simulated)**2
        sum_diff = diff.mean()
        gof_val = np.sqrt(sum_diff)
    elif estimator=="smape":
        diff = np.abs(data - data_simulated)/(0.5*(data+data_simulated+1e-8))
        sum_diff = diff.sum()
        gof_val = np.round(sum_diff/len(data), 6)
    elif estimator=="rmsp":
        diff = ((data - data_simulated)/(0.5*(data+data_simulated+1e-8)))**2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(sum_diff/len(diff)), 6)
    elif estimator=="wmape":
        gof_val = np.array(sum(abs(data - data_simulated))/sum(data))
    elif estimator=="composite":
        diff = (data - data_simulated)**2
        sum_diff = diff.sum()
        gof_rmsn = np.round(np.sqrt(len(diff) * sum_diff)/data.sum(), 6)
        diff = np.abs(data - data_simulated)/(0.5*(data+data_simulated+1e-8))
        sum_diff = diff.sum()
        gof_smape = np.round(sum_diff/len(data), 6)
        gof_val = gof_rmsn + gof_smape
    gof_val = gof_val.tolist()
    return gof_val

def median_gof_eval(data, data_simulated, estimator="rmsn"):
    
    assert data.shape == data_simulated.shape

    if estimator=="rmsn":
        diff = (data - data_simulated)**2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(sum_diff/ len(diff))/np.mean(data), 6)
    elif estimator=="smape":
        diff = np.abs(data - data_simulated)/(0.5*(data+data_simulated+1e-8))
        sum_diff = diff.sum()
        gof_val = np.round((np.median(diff)), 6)
    elif estimator=="rmsp":
        diff = ((data - data_simulated)/(0.5*(data+data_simulated+1e-8)))**2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(sum_diff/len(diff)), 6)
    elif estimator=="composite":
        diff = (data - data_simulated)**2
        sum_diff = diff.sum()
        gof_rmsn = np.round(np.sqrt(len(diff) * sum_diff)/data.sum(), 6)
        diff = np.abs(data - data_simulated)/(0.5*(data+data_simulated+1e-8))
        sum_diff = diff.sum()
        gof_smape = np.round(sum_diff/len(data), 6)
        gof_val = gof_rmsn + gof_smape
    gof_val = gof_val.tolist()
    return gof_val


def squared_deviation(data, data_simulated, which_metric="sq_smape"):
    '''Squared deviation for use in W-SPSA algorithm'''
    
    assert data.shape == data_simulated.shape
    ### Mean Squared Normalized Error: Almost Normalizing MSE reduces the noise the 
    ### final estimates. Some formulations can induce noise the final OD estimates:
    # second preferred one: induces small noise in small ODs but eliminates noise in large ODs
    # diff = ((data - data_simulated)**2)/(np.median(data))

    # preferred one using mean instead of median
    # learning rate needs to be higher here
    # diff = ((data - data_simulated)**2)/(np.mean(data))

    # Symmetric mean squared percentage error
    # multiplyting the SMAPE with the mean of data to avoid vanishing gradients
    # could we multiply instead with the cluster wise mean of the data
    if which_metric == "sq_smape":
        # pass
        # skip absolute as per original w-spsa implementation
        diff = (data + 1e-11) * (data - data_simulated)**2/(0.5 * np.abs(data + data_simulated + 1e-8))

        # diff = 0.3 * np.abs(data - data_simulated - np.mean(data - data_simulated )) \
        #     +  0.3 * np.mean((data - data_simulated )**2) \
        #     + 0.3 * (data + 1e-11) * (data - data_simulated)**2/(0.5 * np.abs(data + data_simulated + 1e-8))



        # diff = (np.abs(data - data_simulated))
    elif which_metric == "composite":
        pass

    # diff = ((data - data_simulated)**2)/(len(data))

    # induces constant noise in ODs of almost  all sizes
    # diff = ((data - data_simulated)**2)/(data+1)
    
    # induces noise and it does not corrects bias in the OD estimates well
    # diff = np.sqrt((data - data_simulated)**2/len(data))
    return diff

if __name__=="__main__":
    td = np.array([2,4,5,6,7])
    sd = np.array([2,4,5,8,9])
    print(gof_eval(td,sd))