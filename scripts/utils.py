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

def equation(x, model, sigma_eps, b_0=0, b_1=0, b_2=0, b_3=0, b_4=0):
    """
    Defining equations for each model
    """   
    if model == 'M0':
        return x + b_0 + np.random.normal(0, sigma_eps)
    elif model == 'M1':
        return x + b_0 + b_1*np.exp(x) + np.random.normal(0, sigma_eps)
    elif model == 'M2':
        return x + b_0 + b_2*(np.exp(x)**b_3) + np.random.normal(0, sigma_eps)
    elif model == 'M3':
        return 2*x -np.log(b_4 + np.exp(x)) + b_0 + b_1*np.exp(x) + np.random.normal(0, sigma_eps)