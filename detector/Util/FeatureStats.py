import numpy as np
import math

# This file contains helper functions for extracting statistical features
# from list or arrays of prediction features

def removeOutliers(preds_list, remove_n):
    """ This calls itself recursively until remove_n have been removed """
    if remove_n <= 0:
        return preds_list
    
    preds = np.array(preds_list)
    mean = np.mean(preds)

    # get distances from the mean value
    dist = np.abs(preds - mean)

    # remove a single outlier
    outlier = np.argmax(dist)
    preds_list.pop(outlier)
    
    # call recursively until n outliers are removed
    return removeOutliers(preds_list, remove_n - 1)

def getStatFeatNames(low_sample_count=False):
    if low_sample_count:
        return ['mean','max']
    return ['mean','max','std','median','Q3']

def preds2features(preds, weights=None, remove_n_outliers=0, low_sample_count=False, verbose=0):
    if verbose> 1:
        print("preds2features - preds before outlier removal {0}".format(preds))

    if remove_n_outliers > 0:
        preds_list = list(preds)
        preds_list = removeOutliers(preds_list, remove_n=remove_n_outliers)
        preds = np.array(preds_list)
    
    if verbose> 1:
        print("preds2features - preds after outlier removal {0}".format(preds))

    # Numpy weighted average, if weights is None, equal weights are assumed
    _mean = np.average(preds, weights=weights)
    # compute weighted standard deviation
    _variance = np.average((preds-_mean)**2, weights=weights)
    _std = math.sqrt(_variance) if _variance > 0 else 0
    _max = max(preds)
    _median = np.median(preds)
    _q3 = np.percentile(preds, 75)
    if low_sample_count:
        return [_mean, _max]
    return [_mean, _max, _std, _median]#, _q3]
