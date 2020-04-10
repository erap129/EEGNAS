from EEGNAS import global_vars
from EEGNAS.data.netflow.netflow_data_utils import get_whole_netflow_data
from EEGNAS.utilities.config_utils import set_default_config, set_params_by_dataset
import matplotlib.pyplot as plt
import os
import pandas as pd

if __name__ == '__main__':
    set_default_config('../../configurations/config.ini')
    global_vars.set('dataset', 'netflow_asflow')
    set_params_by_dataset('../../configurations/dataset_params.ini')
    all_as = [15169,16509,20940,202818,32934,6185,46489,2906,22822,32590,15133,0,20446,65013,3356,16625,5483,16276,60781,60068,8068,54113,24940,8972,49981,65505,9009,56467,65050,714,13335,39572,8302,29789,12876,174,8881,35402,8075,35415,203043,44178,51324,199156,6830,8560,43350,47764,12989,48345,6855,58073,61157,8403,24961,19679,14061,28753,199524,6724,3209,25291,12912,15555,41690,57976,49453,201011,29066,14618,31042,60558,203220,34086,62041,31334,9121,3223,12312,31615,196922,54994,60626,20773,65205,1299,3303,16097,41095,36351,12576,50389,6939,8708,44066,23393,47541,20948,36408]

    all_as_data = []
    for ass in all_as:
        df = get_whole_netflow_data(f'{os.path.dirname(os.path.abspath(__file__))}/top_99/{ass}_1.7.2017-25.12.2019.csv')
        handover_sums = df.sum(axis=0)
        total_volume = handover_sums.sum()
        if ass not in df.columns:
            pni_ratio = 0
        else:
            pni_ratio = handover_sums[ass] / total_volume
        all_as_data.append({'AS': ass, 'total volume': total_volume, 'PNI ratio': pni_ratio})
    pd.DataFrame(all_as_data).to_csv('pni_ratio.csv')









    # df.plot()
    # plt.show()
    #
    #
    # dataset = get_dataset('all')
    # dataset = unify_dataset(dataset)
    # for threshold in range(0, 100000, 1000):
    #     num_overflows = count_overflows_in_data(dataset, threshold)
    #     num_overflows_15_19 = count_overflows_in_data(dataset, threshold, 15, 24)
    #     print(f'num overflows for threshold {threshold}: {num_overflows}/{len(dataset.y)}')
    #     print(f'num overflows for threshold {threshold} between 15:00 and 19:00: {num_overflows_15_19}/{len(dataset.y)}')