import numpy as np
import pandas as pd
import scipy
from scipy.stats import invgamma
from scipy.stats import norm
from scipy.stats import gamma
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
    N_0 = N_0
    
    Theta = np.array([B_0, SigmaEps, SigmaW, N_0])
    ThetaHist = [Theta]
    
    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_0, SigmaEps, SigmaW, N_0 = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_0, SigmaEps, SigmaW, N_0])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0, 'M0', SigmaEps, SigmaW, B_0)
                
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba
        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_0, PrevSigmaEps, PrevSigmaW, PrevN_0 = Theta

            # SMC marginal likelihood ratio
            MarginalLikelihoodRatio = np.exp(LogMarginalLikelihood - 
                                               PrevLogMarginalLikelihood)
                        
            # b_0 prior ratio
            MarginalB_0_ratio = np.exp(0.5*(PrevB_0**2 - B_0**2))

            # sigma_eps prior ratio
            MarginalEpsRatio = np.exp(invgamma.logpdf(x=SigmaEps**2, a=AlphaEps, scale=BetaEps) -
                                  invgamma.logpdf(x=PrevSigmaEps**2, a=AlphaEps, scale=BetaEps))

            # sigma_w prior ratio            
            MarginalWRatio = np.exp(invgamma.logpdf(x=SigmaW**2, a=AlphaW, scale=BetaW) -
                                  invgamma.logpdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            alpha = (MarginalLikelihoodRatio * MarginalB_0_ratio * 
                     MarginalEpsRatio * MarginalWRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j==1000 and len(ThetaHist) < 5 :
            raise ValueError("Too few accepted samples")
            

    print("Number of samples : {}".format(len(ThetaHist)))
    
    return ThetaHist

def AdPMCMC_M1(NSteps, T, AdaptiveRate, Y, L, N_0):
    
    # draw uniforms for accept-reject procedure
    Uniforms = np.random.uniform(0, 1, NSteps)
    
    # inverse gamma prior hyperparameters
    AlphaEps = AlphaW =  T/2 
    BetaEps = BetaW = 2*(AlphaEps - 1)/10
    
    # trajectory history
    NHist = []

    # Parameters initialization
    B_0 = np.random.normal(0, 1)
    B_1 = np.random.normal(0, 1)
    SigmaEps = np.sqrt(invgamma.rvs(a=AlphaEps, scale=BetaEps))
    SigmaW = np.sqrt(invgamma.rvs(a=AlphaW, scale=BetaW))
    N_0 = N_0
    
    Theta = np.array([B_0, B_1, SigmaEps, SigmaW, N_0])
    ThetaHist = [Theta]
    
    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_0, B_1, SigmaEps, SigmaW, N_0 = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_0, B_1, SigmaEps, SigmaW, N_0])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0, 'M1', SigmaEps, SigmaW, B_0, B_1)
                
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba
        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_0, PrevB_1, PrevSigmaEps, PrevSigmaW, PrevN_0 = Theta

            # SMC marginal likelihood ratio
            MarginalLikelihoodRatio = np.exp(LogMarginalLikelihood - 
                                               PrevLogMarginalLikelihood)
                        
            # b_0 prior ratio
            MarginalB_0_ratio = np.exp(0.5*(PrevB_0**2 - B_0**2))
            
            # b_1 prior ratio
            MarginalB_1_ratio = np.exp(0.5*(PrevB_1**2 - B_1**2))

            # sigma_eps prior ratio
            MarginalEpsRatio = np.exp(invgamma.logpdf(x=SigmaEps**2, a=AlphaEps, scale=BetaEps) -
                                  invgamma.logpdf(x=PrevSigmaEps**2, a=AlphaEps, scale=BetaEps))

            # sigma_w prior ratio            
            MarginalWRatio = np.exp(invgamma.logpdf(x=SigmaW**2, a=AlphaW, scale=BetaW) -
                                  invgamma.logpdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            alpha = (MarginalLikelihoodRatio * MarginalB_0_ratio * MarginalB_1_ratio *
                     MarginalEpsRatio * MarginalWRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j==1000 and len(ThetaHist) < 5 :
            raise ValueError("Too few accepted samples")
            

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
    B_0 = np.random.normal(0, 1)
    B_2 = np.random.normal(0, 1)
    B_3 = np.random.normal(0, 1)
    SigmaEps = np.sqrt(invgamma.rvs(a=AlphaEps, scale=BetaEps))
    SigmaW = np.sqrt(invgamma.rvs(a=AlphaW, scale=BetaW))
    N_0 = N_0
        
    Theta = np.array([B_0, B_2, B_3, SigmaEps, SigmaW, N_0])
    ThetaHist = [Theta]

    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_0, B_2, B_3, SigmaEps, SigmaW, N_0 = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_0, B_2, B_3, SigmaEps, SigmaW, N_0])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0,'M2',SigmaEps, SigmaW, B_0=B_0, B_2=B_2,B_3=B_3)        
        
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba

        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_0, PrevB_2, PrevB_3, PrevSigmaEps, PrevSigmaW, PrevN_0 = Theta

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
            MarginalWRatio = np.exp(invgamma.logpdf(x=SigmaW**2, a=AlphaW, scale=BetaW) -
                                  invgamma.logpdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            # theta proposal ratio (always equal to one because kernel is symmetric)

            alpha = (MarginalLikelihoodRatio * MarginalB_0_ratio *  MarginalB_2_ratio 
                     * MarginalB_3_ratio * MarginalEpsRatio * MarginalWRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j==1000 and len(ThetaHist) < 5 :
            raise ValueError("Too few accepted samples")

    print("Number of samples : {}".format(len(ThetaHist)))
    
    return ThetaHist

def AdPMCMC_M3(NSteps, T, AdaptiveRate, Y, L, N_0):
    
    # draw uniforms for accept-reject procedure
    Uniforms = np.random.uniform(0, 1, NSteps)
    
    # inverse gamma prior hyperparameters
    AlphaEps = AlphaW =  T/2 
    BetaEps = BetaW = 2*(AlphaEps - 1)/10
    
    # trajectory history
    NHist = []

    # Parameters initialization
    B_0 = np.random.normal(0, 1)
    B_1 = np.random.normal(0, 1)
    B_4 = np.random.gamma(1,10)
    SigmaEps = np.sqrt(invgamma.rvs(a=AlphaEps, scale=BetaEps))
    SigmaW = np.sqrt(invgamma.rvs(a=AlphaW, scale=BetaW))
    N_0 = N_0
        
    Theta = np.array([B_0, B_1, B_4, SigmaEps, SigmaW, N_0])
    ThetaHist = [Theta]

    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_0, B_1, B_4, SigmaEps, SigmaW, N_0 = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_0, B_1, B_4, SigmaEps, SigmaW, N_0])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0,'M3',SigmaEps, SigmaW, B_0=B_0, B_1=B_1,B_4=B_4)        
        
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba

        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_0, PrevB_1, PrevB_4, PrevSigmaEps, PrevSigmaW, PrevN_0 = Theta

            # SMC marginal likelihood ratio
            MarginalLikelihoodRatio = np.exp(LogMarginalLikelihood - 
                                               PrevLogMarginalLikelihood)
                        
            # b_0 prior ratio
            MarginalB_0_ratio = np.exp(0.5*(PrevB_0**2 - B_0**2))
            
            # b_1 prior ratio
            MarginalB_1_ratio = np.exp(0.5*(PrevB_1**2 - B_1**2))
            
            # b_4 prior ratio
            MarginalB_4_ratio = np.exp(0.5*(PrevB_4**2 - B_4**2))

            # sigma_eps prior ratio
            MarginalEpsRatio = np.exp(invgamma.logpdf(x=SigmaEps**2, a=AlphaEps, scale=BetaEps) -
                                  invgamma.logpdf(x=PrevSigmaEps**2, a=AlphaEps, scale=BetaEps))

            # sigma_w prior ratio
            MarginalWRatio = np.exp(invgamma.logpdf(x=SigmaW**2, a=AlphaW, scale=BetaW) -
                                  invgamma.logpdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            # theta proposal ratio (always equal to one because kernel is symmetric)

            alpha = (MarginalLikelihoodRatio * MarginalB_0_ratio *  MarginalB_1_ratio 
                     * MarginalB_4_ratio * MarginalEpsRatio * MarginalWRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j==1000 and len(ThetaHist) < 5 :
            raise ValueError("Too few accepted samples")

    print("Number of samples : {}".format(len(ThetaHist)))
    
    return ThetaHist

def AdPMCMC_M4(NSteps, T, AdaptiveRate, Y, L, N_0):
    
    # draw uniforms for accept-reject procedure
    Uniforms = np.random.uniform(0, 1, NSteps)
    
    # inverse gamma prior hyperparameters
    AlphaEps = AlphaW =  T/2 
    BetaEps = BetaW = 2*(AlphaEps - 1)/10
    
    # trajectory history
    NHist = []

    # Parameters initialization
    B_5 = np.random.normal(0, 1)
    B_6 = np.random.normal(0, 1)
    B_7 = np.random.normal(0, 1)
    SigmaEps = np.sqrt(invgamma.rvs(a=AlphaEps, scale=BetaEps))
    SigmaW = np.sqrt(invgamma.rvs(a=AlphaW, scale=BetaW))
    N_0 = N_0
        
    Theta = np.array([B_5, B_6, B_7, SigmaEps, SigmaW, N_0])
    ThetaHist = [Theta]

    for j in tqdm(range(NSteps), desc="Running AdPMCMC"):
    
        # draw candidate theta wrt non adaptive proposal
        if np.random.rand() < 1 - AdaptiveRate or j < 500:
            ThetaProp = non_adaptive_theta_proposal(Theta)
        else:
            ThetaProp = adaptive_theta_proposal(ThetaHist)

        # unpack theta
        B_5, B_6, B_7, SigmaEps, SigmaW, N_0 = ThetaProp
                
        # To avoid negative std deviation errors 
        SigmaEps, SigmaW = np.abs(SigmaEps), np.abs(SigmaW)
        ThetaProp = np.array([B_5, B_6, B_7, SigmaEps, SigmaW, N_0])
        
        # run SMC
        LogParticles, LogW, LogMarginalLikelihood = SMC(
            Y, T, L, N_0,'M4',SigmaEps, SigmaW, B_5=B_5, B_6=B_6,B_7=B_7)        
        
        # sample a candidate path
        idx = np.random.choice(a=L, size=1, p=np.exp(LogW[-1,:]))[0]
        CandidatePath = LogParticles[:, idx]

        # compute acceptance proba

        if j == 0:
            # always accept first draw (initialization)
            alpha = 1
        else:
            # last accepted theta
            PrevB_5, PrevB_6, PrevB_7, PrevSigmaEps, PrevSigmaW, PrevN_0 = Theta

            # SMC marginal likelihood ratio
            MarginalLikelihoodRatio = np.exp(LogMarginalLikelihood - 
                                               PrevLogMarginalLikelihood)
                        
            # b_5 prior ratio
            MarginalB_5_ratio = np.exp(0.5*(PrevB_5**2 - B_5**2))
            
            # b_6 prior ratio
            MarginalB_6_ratio = np.exp(0.5*(PrevB_6**2 - B_6**2))
            
            # b_7 prior ratio
            MarginalB_7_ratio = np.exp(0.5*(PrevB_7**2 - B_7**2))

            # sigma_eps prior ratio
            MarginalEpsRatio = np.exp(invgamma.logpdf(x=SigmaEps**2, a=AlphaEps, scale=BetaEps) -
                                  invgamma.logpdf(x=PrevSigmaEps**2, a=AlphaEps, scale=BetaEps))

            # sigma_w prior ratio
            MarginalWRatio = np.exp(invgamma.logpdf(x=SigmaW**2, a=AlphaW, scale=BetaW) -
                                  invgamma.logpdf(x=PrevSigmaW**2, a=AlphaW, scale=BetaW))

            # theta proposal ratio (always equal to one because kernel is symmetric)

            alpha = (MarginalLikelihoodRatio * MarginalB_5_ratio *  MarginalB_6_ratio 
                     * MarginalB_7_ratio * MarginalEpsRatio * MarginalWRatio)
                                    
        # accept-reject 
        if Uniforms[j] < alpha:
            Theta = ThetaProp
            PrevLogMarginalLikelihood = LogMarginalLikelihood
            ThetaHist.append(Theta)
            NHist.append(CandidatePath)
            
        if j==1000 and len(ThetaHist) < 5 :
            raise ValueError("Too few accepted samples")

    print("Number of samples : {}".format(len(ThetaHist)))
    
    return ThetaHist


def AdPMCMC(model,NSteps, T, AdaptiveRate, Y, L, N_0):
    if model == 'M0':
        return AdPMCMC_M0(NSteps, T, AdaptiveRate, Y, L, N_0)
    elif model == 'M1':
        return AdPMCMC_M1(NSteps, T, AdaptiveRate, Y, L, N_0)
    elif model == 'M2':
        return AdPMCMC_M2(NSteps, T, AdaptiveRate, Y, L, N_0)
    elif model == 'M3':
        return AdPMCMC_M3(NSteps, T, AdaptiveRate, Y, L, N_0)
    elif model == 'M4':
        return AdPMCMC_M4(NSteps, T, AdaptiveRate, Y, L, N_0)
    