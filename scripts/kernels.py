import numpy as np

def nonAdaptiveThetaProposal(theta):
    """ 
    Non-adaptive Gaussian kernel for parameter proposal
    """
    d = theta.shape[0]
    return np.random.multivariate_normal(theta, (0.1)**2 / d * np.eye(d))

def adaptiveThetaProposal(thetas):
    """ 
    Adaptive Gaussian kernel for parameter proposal
    See algorithm 2 of paper
    """
    emp_cov = np.cov(thetas, rowvar=False)
    theta = thetas[-1]
    d = theta.shape[0]
    return np.random.multivariate_normal(theta, 2.38**2 / d * emp_cov)