import numpy as np
from utils import equation
# g_t from the observation process is not defined in the paper.

# Since, in the paper, $p(y_t | n_t, \sigma^2_w)$ is assumed to be the density of a Gaussian distribution with mean $n_t$ and variance $\sigma^2_w$, I assume $g_t$ is the identity function.

def trajectory_simulation(model, T, n_0, sigma_eps, sigma_w, b_0=0, b_1=0, b_2=0, b_3=0, b_4=0):
    """
    Simulates latent variable and observation trajectories of horizon T for the models M0 to M3
    """
    
    # Initialization of latent variable trajectory
    log_N = np.zeros(T+1)
    
    # Arbitrary initialization (found in the paper p. 12)
    log_N[0] = np.log(n_0)
    
    # Latent variable simulation
    
    if model in ["M0","M1","M2","M3"]:
        for t in range(T):
            log_N[t+1] = equation(log_N[t],model,sigma_eps,b_0,b_1,b_2,b_3,b_4)
            
    else:
        raise ValueError('Unknown model.')
    
    # Observation variable
    Y = np.exp(log_N) + np.random.normal(0, sigma_w, T+1)
    
    return log_N, Y