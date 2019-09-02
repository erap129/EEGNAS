import global_vars
from data.netflow.netflow_data_utils import get_whole_netflow_data
from utilities.config_utils import set_default_config, set_params_by_dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    set_default_config('../../configurations/config.ini')
    global_vars.set('dataset', 'netflow_asflow')
    set_params_by_dataset('../../configurations/dataset_params.ini')
    df = get_whole_netflow_data('akamai-dt-handovers_1.7.17-1.8.19.csv')
    df.plot()
    plt.show()


    dataset = get_dataset('all')
    dataset = unify_dataset(dataset)
    for threshold in range(0, 100000, 1000):
        num_overflows = count_overflows_in_data(dataset, threshold)
        num_overflows_15_19 = count_overflows_in_data(dataset, threshold, 15, 24)
        print(f'num overflows for threshold {threshold}: {num_overflows}/{len(dataset.y)}')
        print(f'num overflows for threshold {threshold} between 15:00 and 19:00: {num_overflows_15_19}/{len(dataset.y)}')