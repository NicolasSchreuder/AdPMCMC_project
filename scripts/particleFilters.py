def SMC_M2(y, T, L, n_0, b_0, sigma_eps, sigma_w):
    """
    Particle filter for model M2
    """
    
    # L is the number of particles
    log_particles = np.zeros((T, L))
    
    w = np.zeros((T, L))
    
    # Weights matrix
    W = np.zeros((T, L))
        
    # Log-weights matrix
    log_W = np.zeros((T, L))
    
    # Initial propagation
    log_particles[0, :] = np.log(n_0) + b_0 + b_2*n_0**b_3 + np.random.normal(loc=0, scale=sigma_eps, size=L)
    
    # Not sure about W initialization
    for l in range(L):
        w[0, l] = g(y[0], log_particles[0, l], sigma_w)
        log_W[0, l] = w[0, l]
    
    # Weight normalization (log scale)
    log_W[0, :] -= np.max(log_W[0, :]) # scaling   
    log_sum_of_weights = np.log(sum(np.exp(log_W[0, :]))) 
    log_W[0, :] -= log_sum_of_weights
    
    W[0, :] = np.exp(W[0, :])
    
    for t in range(1, T):
        for l in range(L):
            # Propagation
            log_particles[t, l] = log_particles[t-1, l] + b_0 + b_2*np.exp(log_particles[t-1, l])**b_3 + np.random.normal(0, sigma_eps)
                        
            # Weight computation
            w[t, l] = g(y[t], log_particles[t, l], sigma_w)
            log_W[t, l] = log_W[t-1, l] + w[t, l]
                
        # Weight normalization (log scale)
        log_W[t, :] -= np.max(log_W[t, :]) # scaling   
        log_sum_of_weights = np.log(sum(np.exp(log_W[t, :]))) 
        log_W[t, :] -= log_sum_of_weights
                            
        # Adaptive resampling : sample if ESS < 80% of number of particles
        ESS = 1/sum(np.exp(log_W[t, :])**2)
        
        # Pb : ESS is always < 0.8*L 
                
        if ESS < 0.8*L:
            # resample indexes according to the normalized importance weights
            resampled_indexes = stratifiedResampling(np.exp(log_W[t, :]))
            log_particles[t, :] = log_particles[t, resampled_indexes]
            log_W[t, :] = np.log(np.ones(L)/L) # the new particles have equal weight
        
        W[t, :] = np.exp(log_W[t, :])
        
    # Evaluate marginal likelihood
        
    log_marginal_likelihood = np.sum(np.log(np.sum(np.exp(w), axis=1)/L))   
        
    return log_particles, W, log_marginal_likelihood