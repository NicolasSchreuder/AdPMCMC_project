import numpy as np
import pandas as pd
import scipy
from scipy.stats import invgamma
from scipy.stats import norm
from tqdm import tqdm 

from kernels import non_adaptive_theta_proposal, adaptive_theta_proposal
from utils import gaussian_kernel_density
from SMC import SMC

def AdPMCMC_M0(NSteps, T, AdaptiveRate, Y, L, N_0):
    
    # draw uniforms for accept-reject procedure
    Uniforms = np.random.uniform(0, 1, NSteps)
    
    # inverse gamma prior hyperparameters
    AlphaEps = AlphaW =  T/2 
    BetaEps = BetaW = 2*(AlphaEps - 1)/10
    
    # trajectory history
    NHist = []

    # Parameters initialization
    B_0 = np.random.normal(0, 1)
    SigmaEps = np.sqrt(invgamma.rvs(a=AlphaEps, scale=BetaEps))
    SigmaW = np.sqrt(invgamma.rvs(a=AlphaW, scale=BetaW))
        
    Theta = np.array([B_0, SigmaEps, SigmaW])
    ThetaHist = [Theta]

    pb=0
    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_0, SigmaEps, SigmaW = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_0, SigmaEps, SigmaW])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0,'M0',SigmaEps, SigmaW, B_0)
        
        #print(LogMarginalLikelihood)
        if LogMarginalLikelihood == -np.inf:
            pb+=1
            #NSteps +=1 
            #print(NSteps)
        
        
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba

        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_0, PrevSigmaEps, PrevSigmaW = Theta

            # SMC marginal likelihood ratio
            MarginalLikelihoodRatio = np.exp(LogMarginalLikelihood - 
                                               PrevLogMarginalLikelihood)
                        
            # b_0 prior ratio
            MarginalB_0_ratio = np.exp(0.5*(PrevB_0**2 - B_0**2))

            # sigma_eps prior ratio
            MarginalEpsRatio = np.exp(invgamma.logpdf(x=SigmaEps**2, a=AlphaEps, scale=BetaEps) -
                                  invgamma.logpdf(x=PrevSigmaEps**2, a=AlphaEps, scale=BetaEps))

            # sigma_w prior ratio
            # change to log pdf
            MarginalWRatio = (invgamma.pdf(x=SigmaW**2, a=AlphaW, scale=BetaW) / 
                                invgamma.pdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            # theta proposal ratio (always equal to one because kernel is symmetric)
            ThetaPropRatio = (gaussian_kernel_density(ThetaProp, Theta) / 
                                    gaussian_kernel_density(Theta, ThetaProp))

            alpha = (MarginalLikelihoodRatio * MarginalB_0_ratio * 
                     MarginalEpsRatio * MarginalWRatio * ThetaPropRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j%500 == 0:
            print("Number of inf log-likelihood : "+str(pb)+" ; Number of accepted samples : "+str(len(ThetaHist)))

    print("Number of samples : {}".format(len(ThetaHist)))
    
    return ThetaHist


def AdPMCMC_M2(NSteps, T, AdaptiveRate, Y, L, N_0):
    
    # draw uniforms for accept-reject procedure
    Uniforms = np.random.uniform(0, 1, NSteps)
    
    # inverse gamma prior hyperparameters
    AlphaEps = AlphaW =  T/2 
    BetaEps = BetaW = 2*(AlphaEps - 1)/10
    
    # trajectory history
    NHist = []

    # Parameters initialization
    B_0 = B_2 = B_3 = np.random.normal(0, 1)
    SigmaEps = np.sqrt(invgamma.rvs(a=AlphaEps, scale=BetaEps))
    SigmaW = np.sqrt(invgamma.rvs(a=AlphaW, scale=BetaW))
        
    Theta = np.array([B_0, B_2, B_3, SigmaEps, SigmaW])
    ThetaHist = [Theta]

    pb=0
    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_0, B_2, B_3, SigmaEps, SigmaW = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_0, B_2, B_3, SigmaEps, SigmaW])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0,'M2',SigmaEps, SigmaW, B_0=B_0, B_2=B_2,B_3=B_3)
        
        #print(LogMarginalLikelihood)
        if LogMarginalLikelihood == -np.inf:
            pb+=1
            #NSteps +=1 
            #print(NSteps)
        
        
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba

        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_0, PrevB_2, PrevB_3, PrevSigmaEps, PrevSigmaW = Theta

            # SMC marginal likelihood ratio
            MarginalLikelihoodRatio = np.exp(LogMarginalLikelihood - 
                                               PrevLogMarginalLikelihood)
                        
            # b_0 prior ratio
            MarginalB_0_ratio = np.exp(0.5*(PrevB_0**2 - B_0**2))
            
            # b_2 prior ratio
            MarginalB_2_ratio = np.exp(0.5*(PrevB_2**2 - B_2**2))
            
            # b_3 prior ratio
            MarginalB_3_ratio = np.exp(0.5*(PrevB_3**2 - B_3**2))

            # sigma_eps prior ratio
            MarginalEpsRatio = np.exp(invgamma.logpdf(x=SigmaEps**2, a=AlphaEps, scale=BetaEps) -
                                  invgamma.logpdf(x=PrevSigmaEps**2, a=AlphaEps, scale=BetaEps))

            # sigma_w prior ratio
            # change to log pdf
            MarginalWRatio = (invgamma.pdf(x=SigmaW**2, a=AlphaW, scale=BetaW) / 
                                invgamma.pdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            # theta proposal ratio (always equal to one because kernel is symmetric)
            ThetaPropRatio = (gaussian_kernel_density(ThetaProp, Theta) / 
                                    gaussian_kernel_density(Theta, ThetaProp))

            alpha = (MarginalLikelihoodRatio * MarginalB_0_ratio *  MarginalB_2_ratio * MarginalB_3_ratio *
                     MarginalEpsRatio * MarginalWRatio * ThetaPropRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j%500 == 0:
            print("Number of inf log-likelihood : "+str(pb)+" ; Number of accepted samples : "+str(len(ThetaHist)))

    print("Number of samples : {}".format(len(ThetaHist)))
    
    return ThetaHist