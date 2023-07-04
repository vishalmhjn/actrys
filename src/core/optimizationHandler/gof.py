import numpy as np


def gof_eval(data, data_simulated, estimator="rmsn"):
    assert data.shape == data_simulated.shape

    if estimator == "rmsn":
        diff = (data - data_simulated) ** 2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(len(diff) * sum_diff) / data.sum(), 6)
    elif estimator == "rmse":
        diff = (data - data_simulated) ** 2
        sum_diff = diff.mean()
        gof_val = np.sqrt(sum_diff)
    elif estimator == "smape":
        diff = np.abs(data - data_simulated) / (0.5 * (data + data_simulated + 1e-8))
        sum_diff = diff.sum()
        gof_val = np.round(sum_diff / len(data), 6)
    elif estimator == "rmsp":
        diff = ((data - data_simulated) / (0.5 * (data + data_simulated + 1e-8))) ** 2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(sum_diff / len(diff)), 6)
    elif estimator == "wmape":
        gof_val = np.array(sum(abs(data - data_simulated)) / sum(data))
    elif estimator == "composite":
        diff = (data - data_simulated) ** 2
        sum_diff = diff.sum()
        gof_rmsn = np.round(np.sqrt(len(diff) * sum_diff) / data.sum(), 6)
        diff = np.abs(data - data_simulated) / (0.5 * (data + data_simulated + 1e-8))
        sum_diff = diff.sum()
        gof_smape = np.round(sum_diff / len(data), 6)
        gof_val = gof_rmsn + gof_smape
    gof_val = gof_val.tolist()
    return gof_val


def median_gof_eval(data, data_simulated, estimator="rmsn"):
    assert data.shape == data_simulated.shape

    if estimator == "rmsn":
        diff = (data - data_simulated) ** 2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(sum_diff / len(diff)) / np.mean(data), 6)
    elif estimator == "smape":
        diff = np.abs(data - data_simulated) / (0.5 * (data + data_simulated + 1e-8))
        sum_diff = diff.sum()
        gof_val = np.round((np.median(diff)), 6)
    elif estimator == "rmsp":
        diff = ((data - data_simulated) / (0.5 * (data + data_simulated + 1e-8))) ** 2
        sum_diff = diff.sum()
        gof_val = np.round(np.sqrt(sum_diff / len(diff)), 6)
    elif estimator == "composite":
        diff = (data - data_simulated) ** 2
        sum_diff = diff.sum()
        gof_rmsn = np.round(np.sqrt(len(diff) * sum_diff) / data.sum(), 6)
        diff = np.abs(data - data_simulated) / (0.5 * (data + data_simulated + 1e-8))
        sum_diff = diff.sum()
        gof_smape = np.round(sum_diff / len(data), 6)
        gof_val = gof_rmsn + gof_smape
    gof_val = gof_val.tolist()
    return gof_val


def squared_deviation(data, data_simulated, which_metric="sq_smape"):
    """Squared deviation for use in W-SPSA algorithm"""

    assert data.shape == data_simulated.shape
    if which_metric == "sq_smape":
        diff = (
            (data + 1e-11)
            * (data - data_simulated) ** 2
            / (0.5 * np.abs(data + data_simulated + 1e-8))
        )
    elif which_metric == "composite":
        pass
    return diff


if __name__ == "__main__":
    td = np.array([2, 4, 5, 6, 7])
    sd = np.array([2, 4, 5, 8, 9])
    print(gof_eval(td, sd))
