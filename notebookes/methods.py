# This file contains implementations of the methods used in the paper

import numpy as np
import cvxpy as cp

def DID_TWFE(Y,W):
    N, T = Y.shape
    unit_effects = cp.Variable((1,N))
    time_effects = cp.Variable((1,T))
    unit_factor = cp.kron(np.ones((T,1)),unit_effects).T
    time_factor = cp.kron(np.ones((N,1)),time_effects)
    mu = cp.Variable()
    tau = cp.Variable()

    # polishing effect of OSQP results in not truly weighting by zero; use cp.sum_squares instead
    #objective = cp.sum(cp.multiply(cp.square(Y-unit_factor-time_factor),W))
    
    objective = cp.sum_squares(Y-unit_factor-time_factor-mu-W*tau)  
    constraints = []

    prob = cp.Problem(cp.Minimize(objective),
                      constraints)
    prob.solve()

    return tau.value

def SC_TWFE(Y,W,treated_units,treated_periods = 10):

    #outcome matrix has dimension units*periods
    X = np.delete(Y,treated_units,axis=0)[:,:-treated_periods].T
    y = np.mean(Y[treated_units,:-treated_periods],axis=0).T

    _,N_control = X.shape

    unit_weights = cp.Variable((N_control,),nonneg=True)
    constraints = [cp.sum(unit_weights)==1]
    objective = cp.sum_squares(y-X@unit_weights)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.ECOS)

    #need to reshape features into 2D array
    X_predict = np.delete(Y,treated_units,axis=0)[:,-treated_periods:].T
    
    y_predict = X_predict.dot(unit_weights.value)

    return np.mean(Y[treated_units,-treated_periods:])-np.mean(y_predict)

def DIFP_TWFE(Y,W,treated_units,treated_periods = 10):

    #outcome matrix has dimension units*periods
    X = np.delete(Y,treated_units,axis=0)[:,:-treated_periods].T
    y = np.mean(Y[treated_units,:-treated_periods],axis=0).T

    _,N_control = X.shape

    unit_weights = cp.Variable((N_control,),nonneg=True)
    intercept = cp.Variable()
    constraints = [cp.sum(unit_weights)==1]
    objective = cp.sum_squares(y-X@unit_weights-intercept)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.ECOS)

    #need to reshape features into 2D array
    X_predict = np.delete(Y,treated_units,axis=0)[:,-treated_periods:].T
    
    y_predict = X_predict.dot(unit_weights.value)+intercept.value

    return np.mean(Y[treated_units,-treated_periods:])-np.mean(y_predict)

def TROP_TWFE_average(Y,W,treated_units,lambda_unit,lambda_time,lambda_nn,treated_periods = 10):
    
    N,T = Y.shape

    #dist_time
    dist_time = np.absolute(np.arange(T)-(T-treated_periods/2))

    #dist_unit
    average_treated = np.mean(Y[treated_units,:],axis=0)
    
    mask = np.ones((N, T))
    mask[:,-treated_periods:] = 0
    A = np.sum(np.multiply(np.square(average_treated-Y),mask),axis=1)
    B = np.sum(mask,axis=1)
    
    dist_unit = np.sqrt(A/B)
    
    #distance-based weights
    delta_unit = np.exp(-lambda_unit*dist_unit)
    delta_time = np.exp(-lambda_time*dist_time)
    delta = np.outer(delta_unit,delta_time)
    
    unit_effects = cp.Variable((1,N))
    time_effects = cp.Variable((1,T))
    unit_factor = cp.kron(np.ones((T,1)),unit_effects).T
    time_factor = cp.kron(np.ones((N,1)),time_effects)
    mu = cp.Variable()
    tau = cp.Variable()
    L = cp.Variable((N,T))
    
    if lambda_nn == np.inf:
        objective = cp.sum_squares(cp.multiply(Y-mu-unit_factor-time_factor-W*tau,delta))
        
    else:
        objective = cp.sum_squares(cp.multiply(Y-mu-unit_factor-time_factor-L-W*tau,delta)) + lambda_nn*cp.norm(L, "nuc")

    constraints = []

    prob = cp.Problem(cp.Minimize(objective),
                      constraints)
    prob.solve()
        
    return tau.value

def SDID_weights(Y, treated_units, treated_periods):
    
    N,T = Y.shape
    unit_weights_full = np.zeros((N,))
    time_weights_full = np.zeros((T,))
    
    control_units = ~np.isin(np.arange(N),treated_units)

    # unit weights
    X = Y[control_units,:-treated_periods].T
    y = np.mean(Y[treated_units,:-treated_periods].T,axis=1)
    unit_weights = cp.Variable((np.sum(control_units),),nonneg=True)
    constraints = [cp.sum(unit_weights)==1]
    
    # regularization (zeta^2)
    Delta = Y[control_units,:-treated_periods][:,1:]-Y[control_units,:-treated_periods][:,:-1]
    var = np.var(Delta)
    reg = np.sqrt(treated_units.shape[0]*treated_periods)*var

    mu = cp.Variable()
    objective = cp.sum_squares(y-X@unit_weights-mu) + reg*(T-treated_periods)*(cp.sum_squares(unit_weights))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.ECOS)
    unit_weights_full[control_units] = unit_weights.value
    unit_weights_full[treated_units] = 1/treated_units.shape[0]

    # time weights
    X = Y[control_units,:-treated_periods]
    y = np.mean(Y[control_units,-treated_periods:],axis=1)
    time_weights = cp.Variable((T-treated_periods,),nonneg=True)
    constraints = [cp.sum(time_weights)==1]
    
    mu = cp.Variable()
    objective = cp.sum_squares(y-X@time_weights-mu)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.ECOS)
    time_weights_full[:-treated_periods] = time_weights.value
    time_weights_full[-treated_periods:] = 1/treated_periods

    return unit_weights_full, time_weights_full

def SDID_TWFE(Y,W,treated_units,treated_periods=10):
    
    N,T = Y.shape

    #SDID weights
    delta_unit, delta_time = SDID_weights(Y, treated_units, treated_periods)
    delta = np.outer(delta_unit,delta_time)
    
    unit_effects = cp.Variable((1,N))
    time_effects = cp.Variable((1,T))
    unit_factor = cp.kron(np.ones((T,1)),unit_effects).T
    time_factor = cp.kron(np.ones((N,1)),time_effects)
    mu = cp.Variable()
    tau = cp.Variable()
    
    objective = cp.sum_squares(cp.multiply(Y-mu-unit_factor-time_factor-W*tau,delta))

    constraints = []

    prob = cp.Problem(cp.Minimize(objective),
                      constraints)
    prob.solve()
        
    return tau.value