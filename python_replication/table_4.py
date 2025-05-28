import numpy as np
import pandas as pd
import pickle
from utils import load_CPS_data, generate_simulation_components, parallel_experiments

data_dict = {}
RMSE = {}
bias = {}

# set n_jobs to the number of cores
num_cores = 36
num_experiments = 1000

configs = {'Baseline': ['log_wage', 'min_wage', None],
            'Gun Law': ['log_wage', 'open_carry', None],
            'Abortion': ['log_wage', 'abort_ban', None],
            'Random': ['log_wage', 'min_wage', 'Random'],
            'Hours': ['hours', 'min_wage', None],
            'U-rate': ['urate', 'min_wage', None],
            'No Corr':['log_wage', 'min_wage', 'No Corr'],
            'No M': ['log_wage', 'min_wage', 'No M'],
            'No F': ['log_wage', 'min_wage', 'No F'],
            'Only Noise': ['log_wage', 'min_wage', 'Only Noise'],
            'No Noise': ['log_wage', 'min_wage', 'No Noise'],
            'T_post=1': ['log_wage', 'min_wage', 'T_post=1'],
            'N_tr=1': ['log_wage', 'min_wage', 'N_treated=1'],
            'T_post=N_tr=1': ['log_wage', 'min_wage', 'T_post=N_tr=1']}

TROP_dict = {'Baseline': [0.01, 0.2, 0.2],
            'Gun Law':  [0, 0.35, 0.041],
            'Abortion': [0, 0.2, 0.281],
            'Random': [0, 0.2, 0.21],
            'Hours': [1.8, 0.2, 0.031],
            'U-rate': [1.6, 0.35, 0.011],
            'No Corr':[0.7, 0.25, 0.6],
            'No M': [0.1, 0.025, 0.121],
            'No F': [1.4, 0.25, 0.301],
            'Only Noise': [1.8, 0.005, 0.4],
            'No Noise': [5.8, 1.2, 0.21],
            'T_post=1': [1.8, 0.5, 0.321],
            'N_treated=1': [4.5, 0.2, 0.011],
            'T_post=N_treated=1': [0.9, 0.04, 0.301]}

for setting, config in configs.items():
    
    print(setting)
    
    # load and process data for each setting
    outcome, treatment, option = config
    data = load_CPS_data(outcome, treatment)
    data_dict[setting] = data
    
    # run simulations
    simulation_components = generate_simulation_components(data)
    np.random.seed(0)
    RMSE[setting], bias[setting] = parallel_experiments(num_cores, num_experiments, simulation_components, TROP_dict[setting], option)

# save output to table
pd.DataFrame({'setting': RMSE.keys(), 'RMSE': RMSE.values()}).to_csv('RMSE.csv')
    
# save data and TROP parameters for reference
with open('table_4_processed_data.pkl', 'wb') as file:
    pickle.dump(data_dict, file)
with open('table_4_TROP_params.pkl', 'wb') as file:
    pickle.dump(TROP_dict, file)