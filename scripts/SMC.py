import numpy as np
import datetime
import scipy
from scipy.stats import norm
from matplotlib.dates import date2num
from datetime import datetime

from utils import process_date, stratified_resampling, gaussian_kernel_density, equation

def gaussian_log_pdf(y, log_n, sigma_w):
    """
    Gaussian log-density for observation process
    Reimplemented it because scipy.stats.norm.logpdf is too slow
    """
    return -0.5*np.log(2*np.pi*sigma_w**2) - (y - np.exp(log_n))**2/(2*sigma_w**2)


def SMC(Y, T, L, N_0, model, SigmaEps , SigmaW, B_0=0 , B_1=0 , B_2=0 , B_3=0 , B_4=0, B_5=0, B_6=0, B_7=0):
    """
    Particle filter for M0 model
    y : observation trajectory
    T : time horizon
    L : number of particles
    n_0 : initial value of latent variable
    b_0, sigma_eps, sigma_w : M0 parameters
    """
    
    # Initialization of log particles and log-weights matrix
    LogParticles = np.zeros((T, L))
    LogW = np.zeros((T, L))
    
    # Initialization of likelihood matrix 
    # Contains the obs likelihood for each particle
    w = np.zeros((T, L))
    
    # Initialization : check if OK for other models than M
    LogParticles[0, :] = np.log(N_0)
    
    # Log-weight computation
    for l in range(L):
        #w[0, l] = norm.logpdf(Y[0], loc=np.exp(LogParticles[0, l]), 
        #                                  scale=SigmaW)
        w[0, l] = gaussian_log_pdf(Y[0], LogParticles[0, l], SigmaW)
        LogW[0, l] = w[0, l]
    
    # Log-weight normalization
    LogW[0, :] -= np.max(LogW[0, :]) # scaling   
    LogSumOfWeights = np.log(sum(np.exp(LogW[0, :]))) 
    LogW[0, :] -= LogSumOfWeights
        
    for t in range(1, T):
        obs_noise = np.random.normal(0, SigmaEps, L)
        for l in range(L):
            # Propagation
            #LogParticles[t, l] = LogParticles[t-1, l] + B_0 + obs_noise[l]
            LogParticles[t, l] = equation(LogParticles[t-1, l], model, SigmaEps, 
                                          B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7)
        
            # Log-weight computation
            #w[t, l] = norm.logpdf(Y[t], loc=np.exp(LogParticles[t, l]), scale=SigmaW)
            w[t, l] = gaussian_log_pdf(Y[t], LogParticles[t, l], SigmaW)
            
            LogW[t, l] = LogW[t-1, l] + w[t, l]
                
        # Weight normalization (log scale)
        LogW[t, :] -= np.max(LogW[t, :]) # scaling   
        LogSumOfWeights = np.log(sum(np.exp(LogW[t, :]))) 
        LogW[t, :] -= LogSumOfWeights
                            
        # Adaptive resampling : sample if ESS < 80% of number of particles
        ESS = 1/sum(np.exp(LogW[t, :])**2)
                        
        if ESS < 0.8*L:
            # resample indexes according to the normalized importance weights
            ResampledIndexes = stratified_resampling(np.exp(LogW[t, :]))
            LogParticles[t, :] = LogParticles[t, ResampledIndexes]
            LogW[t, :] = np.log(np.ones(L)/L) # the new particles have equal weight
                
    # Evaluate marginal likelihood
    
    LogMarginalLikelihood = np.sum(np.log(np.sum(np.exp(w), axis=1)/L))     
        
    return LogParticles, LogW, LogMarginalLikelihood