import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# def smape(y, yhat):

#     assert len(yhat) == len(y)
#     n = len(y)
#     err = np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)) * 100

#     return ((2 / n) * np.sum(err, axis=0)).mean()

def smape(x, y):
    """Symmetric mean absolute percentage error"""
    return 100 * np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y))) * 2

def horizoned_smape(x, xhat):
    """Given a horizoned forecast and ground truth, compute the SMAPE as a function of time"""
    nt = min(x.shape[0], xhat.shape[0])
    smape_vals = list()
    for i in range(1, nt+1):
        smape_vals.append(smape(x[:i], xhat[:i]))
    smape_vals = np.array(smape_vals)
    return smape_vals

def vpt_smape(x, xhat, threshold=30):
    """
    Find the first time index at which an array exceeds a threshold.

    Args:
        arr (np.ndarray): The array to search. The first dimension should be a horizoned
            smape series.
        threshold (float): The threshold to search for.

    Returns:
        int: The first time index at which the array exceeds the threshold.
    """
    arr = horizoned_smape(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]
    return tind

def rmse_at_t(x, x_hat, t):

    '''
    x: np.ndarray, shape (seq_len, input_dim)
    x_hat: np.ndarray, shape (seq_len, input_dim)
    '''

    x_t = x[t]
    x_hat_t = x_hat[t]

    std_x = np.std(x, axis=0)

    return np.sqrt(np.mean(((x_hat_t-x_t)/std_x)**2))

def horizoned_rmse(x, xhat):
    """Given a horizoned forecast and ground truth, compute the RMSE as a function of time"""
    nt = min(x.shape[0], xhat.shape[0])
    rmse_vals = list()

    for i in range(nt):
        rmse_vals.append(rmse_at_t(x, xhat, i))
    
    rmse_vals = np.array(rmse_vals)
    return rmse_vals

def vpt_rmse(x, xhat, threshold=0.01):

    """
    Find the first time index at which an array exceeds a threshold.
    """
    arr = horizoned_rmse(x, xhat)
    exceed_times = np.where(arr > threshold)[0]
    print(arr)

    if len(exceed_times) == 0:
        tind = len(arr)
    else:
        tind = exceed_times[0]

    return tind







