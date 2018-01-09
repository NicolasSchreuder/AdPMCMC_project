import numpy as np

# g_t from the observation process is not defined in the paper.

# Since, in the paper, $p(y_t | n_t, \sigma^2_w)$ is assumed to be the density of a Gaussian distribution with mean $n_t$ and variance $\sigma^2_w$, I assume $g_t$ is the identity function.

def trajectory_simulation(model, T, n_0, sigma_eps, sigma_w, b_0=0, b_1=0, b_2=0, b_3=0, b4=0):
    """
    Simulates latent variable and observation trajectories of horizon T for the models M0 to M3
    """
    
    # Initialization of latent variable trajectory
    log_N = np.zeros(T+1)
    
    # Arbitrary initialization (found in the paper p. 12)
    log_N[0] = np.log(n_0)
    
    # Latent variable simulation
    
    if model == "M0":
        for t in range(T):
            log_N[t+1] = log_N[t] + b_0 + np.random.normal(0, sigma_eps)
            
    elif model == "M1":
        for t in range(T):
            log_N[t+1] = log_N[t] + b_0 + b_1 + np.random.normal(0, sigma_eps)
            
    elif model == "M2":
        for t in range(T):
            log_N[t+1] = log_N[t] + b_0 + b_2*np.exp(log_N[t])**b_3 + np.random.normal(0, sigma_eps)
            
    elif model == "M3":
        for t in range(T):
            log_N[t+1] = (2*log_N[t] -np.log(b_4 + np.exp(log_N[t])) + 
                          b0 + b1*np.exp(log_N[t]) + np.random.normal(0, sigma_eps))
    
    else:
        raise ValueError('Unknown model.')
    
    # Observation variable
    Y = np.exp(log_N) + np.random.normal(0, sigma_w, T+1)
    
    return log_N, Y