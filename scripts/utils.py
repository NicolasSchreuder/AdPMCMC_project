import numpy as np
import datetime
import scipy
from matplotlib.dates import date2num
from datetime import datetime

def process_date(float_date):
    """
    Processes float date format of the csv database to matplotlib date format
    """
    str_date = str(float_date)
    if len(str_date) < 7:
        str_date += '0'
    return  date2num(datetime.strptime(str_date, '%Y.%m'))

def gaussian_log_density(y, log_n, sigma_w):
    """
    Gaussian log-density for observation process
    """
    return -0.5*np.log(2*np.pi*sigma_w**2) - (y - np.exp(log_n))**2/(2*sigma_w**2)

def stratified_resampling(weights):
    """
    Stratified resampling
    Returns drawn indexes as a list
    weights needs to be a numpy array
    """
    n_samples = weights.shape[0]
        
    # Generate n_samples sorted uniforms with stratified sampling
    sorted_uniforms = np.zeros(n_samples)
    for n in range(n_samples):
        sorted_uniforms[n] = np.random.uniform(n/n_samples, (n+1)/n_samples)
        
    sampled_indexes = []
    j, partial_sum_weights = 0, weights[0]
    for n in range(n_samples):
        while sorted_uniforms[n] > partial_sum_weights:
            j += 1
            partial_sum_weights += weights[j]
        sampled_indexes += [j]
        
    return sampled_indexes

def gaussian_kernel_density(theta_prev, theta):
    """
    Density of non-adaptive proposal kernel
    """
    d = theta.shape[0]
    return scipy.stats.multivariate_normal.pdf(theta, theta_prev, 
                                               (0.1)**2/d * np.eye(d))