import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from methods import DID_TWFE, SC_TWFE, DIFP_TWFE, TROP_TWFE_average, SDID_weights, SDID_TWFE

def load_CPS_data(outcome='log_wage', treatment='min_wage'):
    df = pd.read_csv('CPS.csv',sep=';')
    Y_true_full = np.reshape(df[outcome].values, (40,-1))
    Y_true_full = Y_true_full.T
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)
    N_total,T_total = Y_true_full.shape

    treatments = np.reshape(df[treatment].values, (40,-1)).T
    Ds = np.argwhere(treatments==True)[:,0]
    assignment_vector = np.zeros((N_total,))
    assignment_vector[Ds] = 1
    
    return [Y_true_full, assignment_vector]


def decompose_Y(Y,rank=4):
    N, T = Y.shape

    u,s,v = np.linalg.svd(Y)
    factor_unit = u[:,:rank]
    factor_time = v[:rank,:]
    L = np.dot(factor_unit*s[:rank],factor_time)
    E = Y - L
    F = np.add.outer(np.mean(L,axis=1),np.mean(L,axis=0)) - np.mean(L)
    M = L-F
    
    return F, M, E, factor_unit*np.sqrt(N)

def fit_ar2(E):
    
    T_full = E.shape[1]
    E_ts = E[:, 2:]
    E_lag_1 = E[:, 1:-1]
    E_lag_2 = E[:,:-2]
    
    a_1 = np.sum(np.diag(np.matmul(E_lag_1, E_lag_1.T)))
    a_2 = np.sum(np.diag(np.matmul(E_lag_2, E_lag_2.T)))
    a_3 = np.sum(np.diag(np.matmul(E_lag_1, E_lag_2.T)))
    
    matrix_factor = np.array([[a_1, a_3], 
                         [a_3, a_2]])
    
    b_1 = np.sum(np.diag(np.matmul(E_lag_1, E_ts.T)))
    b_2 = np.sum(np.diag(np.matmul(E_lag_2, E_ts.T)))
    
    ar_coef = np.linalg.inv(matrix_factor).dot(np.array([b_1, b_2]))

    return ar_coef

def ar2_correlation_matrix(ar_coef, T):
    
    result = np.zeros(T)
    result[0] = 1
    result[1] = ar_coef[0] / (1 - ar_coef[1])
    for t in range(2, T):
        result[t] = ar_coef[0] * result[t-1] + ar_coef[1] * result[t-2]
    
    index_matrix = np.abs(np.arange(T)[:, None] - np.arange(T))
    cor_matrix = result[index_matrix].reshape(T, T)
    
    return cor_matrix

def compute_pi_cov(E, unit_factors, assignment_vector):
    
    N_total, T_total = E.shape
    
    ar_coef = fit_ar2(E)
    
    cor_matrix = ar2_correlation_matrix(ar_coef, T_total)
    
    scaled_sd = np.linalg.norm(E.T.dot(E)/N_total,ord='fro')/np.linalg.norm(cor_matrix,ord='fro')
    
    cov_mat = cor_matrix*scaled_sd
    
    model = LogisticRegression(penalty=None).fit(unit_factors, assignment_vector)
    pi = model.predict_proba(unit_factors)[:,1]
    
    return pi, cov_mat

def generate_simulation_components(data):
    
    Y_true_full, assignment_vector = data
    
    F, M, E, unit_factors = decompose_Y(Y_true_full,rank=4)
    
    pi, cov_mat = compute_pi_cov(E, unit_factors, assignment_vector)
    
    return [F, M, cov_mat, pi]

def generate_data(F, M, cov_mat, pi, option=None, treated_periods = 10, treated_units = 10):
    
    N, T = F.shape
    
    # outcome model
    if option == 'No M':
        factor = M*0
        fixed = F
    elif option == 'No F':
        factor = M
        fixed = F*0
    elif option == 'Only Noise':
        factor = M*0
        fixed = F*0
    else:
        factor = M
        fixed = F
        
    if option == 'No Corr':
        noise = np.random.multivariate_normal(mean = np.zeros((T,)), cov = np.eye(cov_mat.shape[0]), size=N)
    elif option == 'No Noise':
        noise = 0
    else:
        noise = np.random.multivariate_normal(mean = np.zeros((T,)), cov = cov_mat, size=N)
    
    Y =  fixed + factor + noise
    
    W = np.zeros((N,T))
    
    if option in ['N_treated=1', 'T_post=N_tr=1']:
        treated_units = 1
                   
    # treatment assignment
    if option == 'Random':
        index = np.random.choice(np.arange(N),size=treated_units, replace=False)
    else:
        candidates = np.random.binomial(n=1,p=pi)

        treated_number = np.sum(candidates)
        
        if treated_number == 0:
            index = np.array(np.random.choice(N))
        elif treated_number == 1:
            index = np.array([np.squeeze(np.argwhere(candidates==1))])
        else:   
            index = np.squeeze(np.argwhere(candidates==1))
            if treated_number > treated_units:
                index = np.random.choice(index, size=treated_units, replace=False)
                    
    if option in ['T_post=1', 'T_post=N_tr=1']:
        treated_periods = 1
        
    W[index,-treated_periods:] = 1
    
    np.savetxt('treated_units.txt', index, fmt='%d')
                        
    return [Y, W, index, treated_periods]

def one_simulation(data, TROP_parameters=[0.01,0.2,0.2]):
    
    Y_true, W_true, treated_units, treated_periods = data
            
    lambda_unit, lambda_time, lambda_nn = TROP_parameters

    # TROP
    TROP_estimate = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=lambda_unit,lambda_time=lambda_time,lambda_nn=lambda_nn,treated_periods=treated_periods)

    # SDID
    SDID_estimate = SDID_TWFE(Y_true, W_true, treated_units, treated_periods)
    
    # SC
    SC_estimate = SC_TWFE(Y_true,W_true,treated_units,treated_periods)

    # DID 
    DID_estimate = DID_TWFE(Y_true,W_true)  
    
    # MC
    MC_estimate = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=0,lambda_time=0,lambda_nn=0.6)
    
    # DIFP
    DIFP_estimate = DIFP_TWFE(Y_true,W_true,treated_units,treated_periods)
    
    return np.array([TROP_estimate, SDID_estimate, SC_estimate, DID_estimate, MC_estimate, DIFP_estimate])

def parallel_experiments(num_cores, num_experiments, simulation_components, TROP_parameters, option=None):
    
    F, M, cov_mat, pi = simulation_components
         
    estimates = Parallel(n_jobs=num_cores, prefer='processes')(
                 delayed(one_simulation)(generate_data(F, M, cov_mat, pi, option), TROP_parameters)
                 for experiment in range(num_experiments))
    
    return [np.sqrt(np.mean(np.square(estimates),axis=0)), np.mean(estimates,axis=0)]
