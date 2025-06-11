import numpy as np
import pandas as pd
import pickle
from utils import load_PENN_data, generate_simulation_components, parallel_experiments

data_dict = {}
RMSE = {}
bias = {}

# set n_jobs to the number of cores
num_cores = 36
num_experiments = 1000

configs = {'Democracy': ['log_gdp', 'dem', None],
            'No AR': ['log_gdp', 'dem', 'No Corr'],
            'Education': ['log_gdp', 'educ', None],
            'Random': ['log_gdp', 'dem', 'Random']}

TROP_dict = {'Democracy': [0.3, 0.325, 0.016],
            'No AR':  [0.2, 0.3, 0.016],
            'Education': [0.75, 0.275, 0.026],
            'Random': [0.4, 0.45, 0.003]}

for setting, config in configs.items():
    
    print(setting)
    
    # load and process data for each setting
    outcome, treatment, option = config
    data = load_PENN_data(outcome, treatment)
    data_dict[setting] = data
    
    # run simulations
    simulation_components = generate_simulation_components(data)
    np.random.seed(0)
    RMSE[setting], bias[setting] = parallel_experiments(num_cores, num_experiments, simulation_components, TROP_dict[setting], option)

# save output to table
pd.DataFrame({'setting': RMSE.keys(), 'RMSE': RMSE.values()}).to_csv('RMSE_table_3.csv')
pd.DataFrame({'setting': bias.keys(), 'bias': bias.values()}).to_csv('bias_table_3.csv')    
# save data and TROP parameters for reference
with open('table_3_processed_data.pkl', 'wb') as file:
    pickle.dump(data_dict, file)
with open('table_3_TROP_params.pkl', 'wb') as file:
    pickle.dump(TROP_dict, file)