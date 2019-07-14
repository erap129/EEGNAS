import globals
from EEGNAS_experiment import get_configurations
from data_preprocessing import get_tuh_train_val_test

globals.init_config('configurations/config.ini')
configurations = get_configurations('TUH')
globals.set_config(configurations[0])
get_tuh_train_val_test('/data')