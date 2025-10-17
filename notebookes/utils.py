import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
from methods import DID_TWFE, SC_TWFE, DIFP_TWFE, TROP_TWFE_average, SDID_weights, SDID_TWFE
import pyreadr

def load_boatlift_data(outcome='loguearnhre', treatment=None):

    df = pd.read_stata('aux_may-org.dta')
    df = df[['smsarank','year',outcome]]
    df = df.dropna()
    Y_true_full = np.reshape(df.copy().groupby(['smsarank','year'],observed=False).mean().values,(-1,19))
    #Y_true_full = np.delete(Y_true_full,25,0)
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)
    N_total,T_total = Y_true_full.shape

    assignment_vector = np.zeros((N_total,))
    assignment_vector[25] = 1
    
    return [Y_true_full, assignment_vector]

def load_smoking_data(outcome='PacksPerCapita', treatment=None):
    
    df = pd.read_csv('california_prop99.csv',sep=';')
    
    Y_true_full = df[outcome].copy().values
    Y_true_full = np.reshape(Y_true_full, (31,-1))
    #Y_true_full = Y_true_full.T[:-1,:]
    Y_true_full = Y_true_full.T
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)
    N_total,T_total = Y_true_full.shape

    assignment_vector = np.zeros((N_total,))
    assignment_vector[-1] = 1
    
    return [Y_true_full, assignment_vector]

def load_basque_data(outcome='gdpcap', treatment=None):
    
    result = pyreadr.read_r('basque.rda')
    df = result['basque']
    Y_true_full = df[outcome].copy().values
    Y_true_full = np.reshape(Y_true_full, (18,-1))
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)
    N_total,T_total = Y_true_full.shape

    assignment_vector = np.zeros((N_total,))
    assignment_vector[16] = 1
    
    return [Y_true_full, assignment_vector]

def load_Germany_data(outcome='gdp', treatment=None):
    df = pd.read_stata('germany.dta')
    Y_true_full = np.reshape(df[outcome].copy().values, (17,-1))
    Y_true_full = np.log(Y_true_full)
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)
    N_total,T_total = Y_true_full.shape
    
    assignment_vector = np.zeros((N_total,))
    assignment_vector[6] = 1
    
    return [Y_true_full, assignment_vector]


def load_PENN_data(outcome='log_gdp', treatment='dem', short_panel=None):
    df = pd.read_csv('PENN.csv',sep=';')
    Y_true_full = np.reshape(df[outcome].values, (48,-1))
    Y_true_full = Y_true_full.T
    if short_panel:
        Y_true_full = Y_true_full[:, -short_panel:]
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)
    N_total,T_total = Y_true_full.shape

    treatments = np.reshape(df[treatment].values, (48,-1)).T  
    Ds = np.argwhere(treatments==True)[:,0]
    assignment_vector = np.zeros((N_total,))
    assignment_vector[Ds] = 1
    
    return [Y_true_full, assignment_vector]

def load_PENN_data_subset(outcome='log_gdp', treatment='dem', unit_subset=None):
    df = pd.read_csv('PENN.csv',sep=';')
    Y_true_full = np.reshape(df[outcome].values, (48,-1))
    Y_true_full = Y_true_full.T
    
    N_total,T_total = Y_true_full.shape
    #selected_units = np.random.choice(N_total, unit_subset, replace=False)
    selected_units = np.arange(unit_subset)
    
    Y_true_full = Y_true_full[selected_units, :]
    Y_true_full /= np.std(Y_true_full)
    Y_true_full -= np.mean(Y_true_full)

    treatments = np.reshape(df[treatment].values, (48,-1)).T  
    Ds = np.argwhere(treatments==True)[:,0]
    assignment_vector = np.zeros((N_total,))
    assignment_vector[Ds] = 1
    
    return [Y_true_full, assignment_vector[selected_units]]

def load_CPS_data(outcome='log_wage', treatment='min_wage',short_panel=None):
    df = pd.read_csv('CPS.csv',sep=';')
    Y_true_full = np.reshape(df[outcome].values, (40,-1))
    Y_true_full = Y_true_full.T
    if short_panel:
        Y_true_full = Y_true_full[:, -short_panel:]
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

def compute_cov(E):
    
    N_total, T_total = E.shape
    
    ar_coef = fit_ar2(E)
    
    cor_matrix = ar2_correlation_matrix(ar_coef, T_total)
    
    scaled_sd = np.linalg.norm(E.T.dot(E)/N_total,ord='fro')/np.linalg.norm(cor_matrix,ord='fro')
    
    cov_mat = cor_matrix*scaled_sd
    
    return cov_mat

def compute_pi(unit_factors, assignment_vector):
    
    model = LogisticRegression(penalty=None).fit(unit_factors, assignment_vector)
    pi = model.predict_proba(unit_factors)[:,1]
    
    return pi

def generate_simulation_components(data, sc_weights=False):
    
    Y_true_full, assignment_vector = data
    
    F, M, E, unit_factors = decompose_Y(Y_true_full,rank=4)
    
    cov_mat = compute_cov(E)
    
    if sc_weights:   
        pi = SC_weights(data, treated_periods = 1)
    
    else:
        pi = compute_pi(unit_factors, assignment_vector)
        
    return [F, M, cov_mat, pi]

def generate_data(F, M, cov_mat, pi, option=None, seed=None, treated_periods = 10, treated_units = 10):
    
    if seed:
        np.random.seed(seed)
    
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
        noise = np.random.multivariate_normal(mean = np.zeros((T,)), cov = np.diag(np.diag(cov_mat)), size=N)
    elif option == 'No Noise':
        noise = 0
    else:
        noise = np.random.multivariate_normal(mean = np.zeros((T,)), cov = cov_mat, size=N)
    
    Y =  fixed + factor + noise
    
    W = np.zeros((N,T))
    
    if option in ['N_treated=1', 'T_post=N_tr=1']:
        treated_units = 1
        
    if option is not None and option.startswith('N_treated='):
        treated_units = int(option[len('N_treated='):])
                   
    # treatment assignment
    if option == 'Random':
        index = np.random.choice(np.arange(N),size=treated_units, replace=False)  
    else:
        candidates = np.random.binomial(n=1,p=pi)

        treated_number = np.sum(candidates)
        
        if treated_number == 0:
            index = np.array([np.random.choice(N)])
        elif treated_number == 1:
            index = np.array([np.squeeze(np.argwhere(candidates==1))])
        else:   
            index = np.squeeze(np.argwhere(candidates==1))
            if treated_number > treated_units:
                index = np.random.choice(index, size=treated_units, replace=False)
                    
    if option in ['T_post=1', 'T_post=N_tr=1']:
        treated_periods = 1
        
    if option is not None and option!='T_post=N_tr=1' and option.startswith('T_post='):
        treated_periods = int(option[len('T_post='):])
    
    W[index,-treated_periods:] = 1
    
    #np.savetxt('treated_units.txt', index, fmt='%d')
                        
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
    MC_estimate = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=0,lambda_time=0,lambda_nn=TROP_parameters[2])
    
    # DIFP
    DIFP_estimate = DIFP_TWFE(Y_true,W_true,treated_units,treated_periods)
    
    return np.array([TROP_estimate, SDID_estimate, SC_estimate, DID_estimate, MC_estimate, DIFP_estimate])

def parallel_experiments(num_cores, num_experiments, simulation_components, TROP_parameters, option=None):
    
    F, M, cov_mat, pi = simulation_components
         
    estimates = Parallel(n_jobs=num_cores, prefer='processes')(
                 delayed(one_simulation)(generate_data(F, M, cov_mat, pi, option), TROP_parameters)
                 for experiment in range(num_experiments))
    
    return [np.sqrt(np.mean(np.square(estimates),axis=0)), np.mean(estimates,axis=0)]

def one_simulation_variants(data, TROP_parameters):
    
    Y_true, W_true, treated_units, treated_periods = data
            
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[0]

    # TROP
    TROP_estimate = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=lambda_unit,lambda_time=lambda_time,lambda_nn=lambda_nn,treated_periods=treated_periods)
    
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[1]

    # lambda_nn=inf
    TROP_nn_inf = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=lambda_unit,lambda_time=lambda_time,lambda_nn=np.inf,treated_periods=treated_periods)
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[2]
    
    # lambda_unit = 0
    TROP_unit_0 = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=0,lambda_time=lambda_time,lambda_nn=lambda_nn,treated_periods=treated_periods)

    lambda_unit, lambda_time, lambda_nn = TROP_parameters[3]
    # lambda_time = 0
    TROP_time_0 = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=lambda_unit,lambda_time=0,lambda_nn=lambda_nn,treated_periods=treated_periods)
    
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[4]
    # lambda_unit_time = 0
    TROP_unit_time_0 = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=0,lambda_time=0,lambda_nn=lambda_nn,treated_periods=treated_periods)
    
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[5]
    # lambda_unit = 0, lambda_nn=inf
    TROP_unit_0_nn_inf = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=0,lambda_time=lambda_time,lambda_nn=np.inf,treated_periods=treated_periods)
    
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[6]
    # lambda_time = 0, lambda_nn=inf
    TROP_time_0_nn_inf = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=lambda_unit,lambda_time=0,lambda_nn=np.inf,treated_periods=treated_periods)
    
    lambda_unit, lambda_time, lambda_nn = TROP_parameters[7]
    # DID
    DID = TROP_TWFE_average(Y_true,W_true,treated_units,lambda_unit=0,lambda_time=0,lambda_nn=np.inf,treated_periods=treated_periods)
    
    return np.array([TROP_estimate, TROP_nn_inf, TROP_unit_0, TROP_time_0, TROP_unit_time_0, TROP_unit_0_nn_inf, TROP_time_0_nn_inf, DID])


def parallel_experiments_variants(num_cores, num_experiments, simulation_components, TROP_parameters, option=None):
    
    F, M, cov_mat, pi = simulation_components
         
    estimates = Parallel(n_jobs=num_cores, prefer='processes')(
                 delayed(one_simulation_variants)(generate_data(F, M, cov_mat, pi, option), TROP_parameters)
                 for experiment in range(num_experiments))
    
    return [np.sqrt(np.mean(np.square(estimates),axis=0)), np.mean(estimates,axis=0)]

def SC_weights(data, treated_periods = 10, reg = 0.1):
    
    [Y, assignment_vector] = data
    
    control_units = np.where(assignment_vector == 0)[0]
    
    treated_units = np.where(assignment_vector == 1)[0]

    #outcome matrix has dimension units*periods
    X = np.delete(Y,treated_units,axis=0)[:,:-treated_periods].T
    y = np.mean(Y[treated_units,:-treated_periods],axis=0).T

    _,N_control = X.shape

    unit_weights = cp.Variable((N_control,),nonneg=True)
    constraints = [cp.sum(unit_weights)==1]
    objective = cp.sum_squares(y-X@unit_weights) + reg*(cp.sum_squares(unit_weights))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.ECOS)

    sc_weights = np.zeros((Y.shape[0],))
    
    sc_weights[control_units] = unit_weights.value

    return sc_weights
