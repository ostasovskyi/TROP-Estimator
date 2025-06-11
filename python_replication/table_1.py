import numpy as np
import pandas as pd
import pickle
from utils import load_CPS_data, load_PENN_data, decompose_Y, compute_pi_cov, generate_simulation_components, parallel_experiments

data_dict = {}
RMSE = {}
bias = {}

# set n_jobs to the number of cores
num_cores = 36
num_experiments = 1000

configs = {'Row 1': ['log_wage', 'min_wage', None],
           'Row 2': ['urate', 'min_wage', None],
           'Row 3': ['hours', 'min_wage', None],
            'Row 4': ['log_wage', 'open_carry', None],
            'Row 5': ['log_wage', 'abort_ban', None],
            'Row 6': ['log_wage', 'min_wage', 'Random'],
            'Row 7': ['log_gdp', 'dem', None],
            'Row 8': ['log_gdp', 'educ', None],
            'Row 9': ['log_gdp', 'dem', 'Random']}

TROP_dict = {'Row 1': [0.01, 0.2, 0.2],
             'Row 2': [1.6, 0.35, 0.011],
             'Row 3': [1.8, 0.2, 0.031],
            'Row 4':  [0, 0.35, 0.041],
            'Row 5': [0, 0.2, 0.281],
            'Row 6': [0, 0.2, 0.21],
            'Row 7': [0.3, 0.325, 0.016],
            'Row 8': [0.75, 0.275, 0.026],
            'Row 9': [0.4, 0.45, 0.003]}

for setting, config in list(configs.items())[:6]:
    
    print(setting)
    
    # load and process data for each setting
    outcome, treatment, option = config
    data = load_CPS_data(outcome, treatment)
    data_dict[setting] = data
    
    # run simulations
    simulation_components = generate_simulation_components(data)
    np.random.seed(0)
    RMSE[setting], bias[setting] = parallel_experiments(num_cores, num_experiments, simulation_components, TROP_dict[setting], option)

for setting, config in list(configs.items())[6:]:
    
    print(setting)
    
    # load and process data for each setting
    outcome, treatment, option = config
    data = load_PENN_data(outcome, treatment)
    data_dict[setting] = data
    
    # run simulations
    simulation_components = generate_simulation_components(data)
    np.random.seed(0)
    RMSE[setting], bias[setting] = parallel_experiments(num_cores, num_experiments, simulation_components, TROP_dict[setting], option)
    
normalized_df = pd.DataFrame(columns=['outcome','treatment','TROP','SDID','SC','DID','MC','DIFP'])
for setting, config in list(configs.items()):
    outcome, treatment, option = config
    RMSEs = RMSE[setting]/np.min(RMSE[setting])
    normalized_df.loc[len(normalized_df)] = [outcome, treatment] + list(RMSEs)
    
# save output to table
normalized_df.to_csv('table_1.csv')