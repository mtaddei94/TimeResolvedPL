# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:50:34 2020

@author: sarth
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.special import gamma

def generate_single_exp(t, tau):
    I = np.exp(-t/tau) + 1e-4
    return I

def fun(y, t, k1, k2=8e-11):
    #y1, y2 = y #y1 is the carrier concentration and y2 is the photon concentration in the film
    dy1dt = -k1*y - k2*y**2 #+ (c/n)*Alpha_avg*y2 #change in the carrier population as function of time
    #dy2dt = -(c/n)*Alpha_avg*y2 + (k_b*y1**2*P_stay) #change in the photon density as function of time
    return dy1dt
    
# Solve the coupled ODE's
def ode_solve(N0, k1, k2=8e-11):#, k_A):
    sim, infod = odeint(fun, N0 , t, args=(k1, k2), full_output = 1, mxstep=5000000)
    return sim

def stretch_exp_fit(TRPL, t, Tc = (0,1e4*1e-9), Beta = (0,1), A = (0,1.5), noise=(0,1)):

    def exp_stretch(t, tc, beta, a, noise):
        return ((a * np.exp(-((1.0 / tc) * t) ** beta)) + noise) 
    
    def avg_tau_from_exp_stretch(tc, beta):
        return (tc / beta) * gamma(1.0 / beta)
    
    def Diff_Ev_Fit_SE(TRPL):
        
        def residuals(params):#params are the parameters to be adjusted by differential evolution or leastsq, interp is the data to compare to the model.
            #Variable Rates
            tc = params[0]
            beta = params[1]
            a = params[2]
            noise = params[3]
            
            PL_sim = exp_stretch(t,tc,beta,a,noise)
    
            Resid= (np.sum(((PL_sim-TRPL)**2)/(np.sqrt(PL_sim)**2)))
            return Resid #returns the difference between the PL data and simulated data
        
        bounds = [Tc, Beta, A, noise] 
    
        result = differential_evolution(residuals, bounds)
        return result.x
    
    p = Diff_Ev_Fit_SE(TRPL)

    tc = p[0]
    beta = p[1]
    a = p[2]
    noise = p[3]
    
    PL_fit = exp_stretch(t,tc,beta,a, noise)
    
    avg_tau = avg_tau_from_exp_stretch(tc,beta)
    
    return tc, beta, a, avg_tau, PL_fit, noise

def double_exp_fit(TRPL, t, tau1_bounds=(0,1000*1e-9), a1_bounds=(0,1), tau2_bounds=(0,10000*1e-9), a2_bounds=(0,1), noise=(0,1)):

    def single_exp(t, tau, a):
        return (a * np.exp(-((1.0 / tau)*t) ))
    
    def double_exp(t, tau1, a1, tau2, a2, noise):
        return ((single_exp(t, tau1, a1)) + (single_exp(t, tau2, a2)) + noise)
    
    def avg_tau_from_double_exp(tau1, a1, tau2, a2):
        return (((tau1*a1) + (tau2*a2))/(a1+a2))
    
    def Diff_Ev_Fit_DE(TRPL):
        
        def residuals(params):#params are the parameters to be adjusted by differential evolution or leastsq, interp is the data to compare to the model.
            #Variable Rates
            tau1 = params[0]
            a1 = params[1]
            tau2 = params[2]
            a2 = params[3]
            noise = params[4]
            
            PL_sim = double_exp(t,tau1, a1, tau2, a2, noise)
    
            Resid= (np.sum(((PL_sim-TRPL)**2)/(np.sqrt(PL_sim)**2)))
            return Resid #returns the difference between the PL data and simulated data
        
        bounds = [tau1_bounds, a1_bounds, tau2_bounds, a2_bounds, noise] 
    
        result = differential_evolution(residuals, bounds)
        return result.x
    
    p = Diff_Ev_Fit_DE(TRPL)

    tau1 = p[0]
    a1 = p[1]
    tau2 = p[2]
    a2 = p[3]
    noise = p[4]
    
    PL_fit = double_exp(t, tau1, a1, tau2, a2, noise)
    
    avg_tau = avg_tau_from_double_exp(tau1, a1, tau2, a2)
    
    return tau1, a1, tau2, a2, avg_tau, PL_fit, noise

def single_exp_fit(TRPL, t, tau_bounds=(0,1000*1e-9), a_bounds=(0,1), noise=(0,1)):

    def single_exp(t, tau, a, noise):
        return (a * np.exp(-((1.0 / tau)*t) ) + noise)
    
    def Diff_Ev_Fit_SiE(TRPL):
        
        def residuals(params):#params are the parameters to be adjusted by differential evolution or leastsq, interp is the data to compare to the model.
            #Variable Rates
            tau = params[0]
            a = params[1]
            noise = params[2]
            
            PL_sim = single_exp(t,tau, a, noise)
    
            Resid= (np.sum(((PL_sim-TRPL)**2)/(np.sqrt(PL_sim)**2)))
            return Resid #returns the difference between the PL data and simulated data
        
        bounds = [tau_bounds, a_bounds, noise] 
    
        result = differential_evolution(residuals, bounds)
        return result.x
    
    p = Diff_Ev_Fit_SiE(TRPL)

    tau = p[0]
    a = p[1]
    noise = p[2]
    
    PL_fit = single_exp(t, tau, a, noise)
    
    return tau, a, PL_fit, noise

    
