# Author: R Zinke
# March, 2025

import math
import warnings

import numpy as np
from scipy import stats

from solid_utils.variogram import remove_trend
from mintpy.utils import time_func, utils as ut


def model_timeseries(dates:np.ndarray, dis:np.ndarray, model:dict,
                    conf=95.45):
    """Model a displacement time-series.

    Parameters: dates     - np.ndarray, dates as Python datetime objects
                dis       - np.ndarray, displacements
                model     - dict, time function model
    Returns:    dis_hat   - np.ndarray, array of predicted displacement values
                mhat      - np.ndarray, model fit parameters
                mhat_se   - np.ndarray, standard errors for model fit params
    """
    # Construct design matrix from dates and model
    date_list = [date.strftime('%Y%m%d') for date in dates]
    G = time_func.get_design_matrix4time_func(date_list, model)

    # Invert for model parameters
    m_hat = np.linalg.pinv(G).dot(dis)

    # Predict displacements
    dis_hat = np.dot(G, m_hat)

    # Quantify error on model parameters
    resids = dis - dis_hat
    sse = np.sum(resids**2)
    n = len(dis_hat)
    dof = len(m_hat)
    pcov = sse/(n - dof) * np.linalg.inv(np.dot(G.T, G))
    mhat_se = np.sqrt(np.diag(pcov))

    # Propagate uncertainty
    dcov = G.dot(pcov).dot(G.T)
    derr = np.sqrt(np.diag(dcov))

    # Error envelope
    err_scale = stats.t.interval(conf/100, dof)
    err_lower = dis_hat + err_scale[0] * derr
    err_upper = dis_hat + err_scale[1] * derr
    err_envelope = [err_lower, err_upper]
    
    return dis_hat, m_hat, mhat_se, err_envelope


class IterativeOutlierFit:
    @staticmethod
    def outliers_zscore(dis:np.ndarray, dis_hat:np.ndarray, threshold:float):
        """Identify outliers using the z-score metric.
    
        Compute the number of standard deviations the data are from the mean
        and return the indices of values greater than the specified threshold.
    
        Parameters: dis          - np.ndarray, array of displacement values
                    dis_hat      - np.ndarray, array of predicted displacement
                                   values
                    threshold    - float, z-score value (standard deviation)
                                   beyond which to exclude data
        Returns:    outlier_ndxs - np.ndarray, boolean array where True
                                   indicates an outlier
                    n_outliers   - int, number of outliers
        """
        zscores = (dis - dis_hat) / np.std(dis - dis_hat)
        outlier_ndxs = np.abs(zscores) > threshold
        n_outliers = np.sum(outlier_ndxs)
    
        return outlier_ndxs, n_outliers
    
    def __init__(self, dates, dis, model, threshold=3, max_iter=2):
        """Determine which data points are outliers based on the z-score
        metric and remove those points.

        Parameters: dates     - np.ndarray, dates as Python datetime objects
                    dis       - np.ndarray, displacements
                    model     - dict, time function model
                    threshold - float, standard deviations beyond which values
                                are considered outliers
                    max_iter  - int, maximutm number of iterations before
                                stopping
        Returns:    dis_hat   - np.ndarray, array of predicted displacement
                                values
                    mhat      - np.ndarray, model fit parameters
                    mhat_se   - np.ndarray, standard errors for model fit
                                params
        """
        # Record parameters
        self.dates = dates
        self.dis = dis
        self.model = model
        self.outlier_threshold = threshold

        # Initialize outlier removal
        self.iters = 0
        self.outlier_dates = np.array([])
        self.outlier_dis = np.array([])

        # Initial fit to data
        (self.dis_hat,
         self.m_hat,
         self.mhat_se,
         self.err_envelope) = model_timeseries(self.dates, self.dis, self.model)

        # Determine outliers based on z-score
        outlier_ndxs, n_outliers = self.outliers_zscore(self.dis, self.dis_hat,
                                                        self.outlier_threshold)
        self.n_outliers = n_outliers

        # Remove outliers from data set
        while (n_outliers > 0) and (self.iters < max_iter):
            # Update time series
            self.outlier_dates = np.append(self.outlier_dates,
                                           self.dates[outlier_ndxs])
            self.outlier_dis = np.append(self.outlier_dis,
                                         self.dis[outlier_ndxs])

            self.dates = self.dates[~outlier_ndxs]
            self.dis = self.dis[~outlier_ndxs]

            # Update timeseries model
            (self.dis_hat,
             self.m_hat,
             self.mhat_se,
             self.err_envelope) = model_timeseries(self.dates, self.dis,
                                                   self.model)

            # Determine outliers based on z-score
            (outlier_ndxs,
             n_outliers) = self.outliers_zscore(self.dis, self.dis_hat,
                                                self.outlier_threshold)
            self.n_outliers += n_outliers

            # Update iteration counter
            self.iters += 1
