import numpy as np


def prepare_weight_matrix(W, weight_counts, weight_od, weight_speed):
    """Concatenate weight matrices as a preprocessing step for W-SPSA

    Parameters
    ----------
    W : numpy.ndarray, required
        link incidence matrix

    weight_counts : float, required
        weight of counts in the multi-objective optimization function

    weight_od : float, required
        weight of prior OD demand in the multi-objective optimization function

    weight_speed : float, required
        weight of speed in the multi-objective optimization function

    Returns
    ------
    numpy.ndarray
        weight matrix for W-SPSA
    """

    if weight_counts != 0:
        if weight_od != 0:
            if weight_speed != 0:
                ###### Warning: Change the dtype of the weight array to float32 (4 bytes)
                # if you want the correlation to be float values between 0 and 1. Now it is
                # configured to int8 which takes 1 byte of space
                weight_wspsa = np.zeros(
                    (W.shape[0], W.shape[1] + W.shape[0] + W.shape[1]), dtype="int8"
                )
                weight_wspsa[:, : W.shape[1]] = np.where(W > 0, 1, 0)
                weight_wspsa[:, W.shape[1] : W.shape[1] + W.shape[0]] = np.eye(
                    W.shape[0]
                )
                weight_wspsa[:, W.shape[1] + W.shape[0] :] = np.where(W > 0, 1, 0)
            else:
                weight_wspsa = np.zeros(
                    (W.shape[0], W.shape[1] + W.shape[0]), dtype="int8"
                )
                weight_wspsa[:, : W.shape[1]] = np.where(W > 0, 1, 0)
                weight_wspsa[:, W.shape[1] :] = np.eye(W.shape[0])
        else:
            if weight_speed != 0:
                weight_wspsa = np.zeros(
                    (W.shape[0], W.shape[1] + W.shape[1]), dtype="int8"
                )
                weight_wspsa[:, : W.shape[1]] = np.where(W > 0, 1, 0)
                weight_wspsa[:, W.shape[1] :] = np.where(W > 0, 1, 0)
            else:
                weight_wspsa = np.where(W > 0, 1, 0).astype("int8")
    else:
        if weight_od != 0:
            if weight_speed != 0:
                weight_wspsa = np.zeros(
                    (W.shape[0], W.shape[0] + W.shape[1]), dtype="int8"
                )
                weight_wspsa[:, : W.shape[0]] = np.eye(W.shape[0])
                weight_wspsa[:, W.shape[0] :] = np.where(W > 0, 1, 0)
            else:
                weight_wspsa = np.eye(W.shape[0], dtype="int8")
        else:
            if weight_speed != 0:
                weight_wspsa = np.where(W > 0, 1, 0).astype("int8")
            else:
                raise ("All the weights cannot be zero")
    return weight_wspsa


if __name__ == "__main__":
    print("Done")
