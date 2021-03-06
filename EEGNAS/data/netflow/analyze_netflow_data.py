from EEGNAS import global_vars
from EEGNAS.data.netflow.netflow_data_utils import get_whole_netflow_data
from EEGNAS.data_preprocessing import get_dataset
from EEGNAS.utilities.config_utils import set_default_config, set_params_by_dataset
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


def get_pni_ratio(all_as):
    all_as_data = []
    for ass in all_as:
        df = get_whole_netflow_data(f'{os.path.dirname(os.path.abspath(__file__))}/top_10/{ass}_1.7.2017-31.12.2019.csv')
        handover_sums = df.sum(axis=0)
        total_volume = handover_sums.sum()
        if ass not in df.columns:
            pni_ratio = 0
        else:
            pni_ratio = handover_sums[ass] / total_volume
        all_as_data.append({'AS': ass, 'total volume': total_volume, 'PNI ratio': pni_ratio})
    pd.DataFrame(all_as_data).to_csv('pni_ratio.csv')


def get_data_samples(all_as):
    all_as_data = []
    for ass in all_as:
        global_vars.set('autonomous_systems', [ass])
        dataset = get_dataset('all')
        all_as_data.append(dataset['train'].X.mean(axis=0).swapaxes(0, 1))
    all_as_data_stacked = np.concatenate(all_as_data, axis=0)
    all_as_df = pd.DataFrame(data=all_as_data_stacked, columns=global_vars.get('netflow_handover_locations'))
    all_as_df['AS'] = [ass for ass in all_as for i in range(240)]
    all_as_df['time'] = [i for ass in all_as for i in range(240)]
    all_data_as = pd.melt(all_as_df, id_vars=['AS', 'time'], value_vars=global_vars.get('netflow_handover_locations'))
    all_data_as.to_csv('analysis_results/all_data_samples.csv')


def get_all_data(ass):
    df = get_whole_netflow_data(f'{os.path.dirname(os.path.abspath(__file__))}/top_10/{ass}_1.7.2017-31.12.2019.csv')
    df.to_csv(f'AS_{ass}_all_data.csv')


def get_corona_data(all_as):
    global_vars.set('date_range', "1.7.2017-8.5.2020")
    all_ass_dfs_with_self = []
    all_ass_dfs_no_self = []
    for ass in all_as:
        all_data = get_whole_netflow_data(
            f'top_10_corona/{ass}_{global_vars.get("date_range")}.csv')
        all_data_with_self = all_data.drop('sum', axis=1).sum(axis=1)
        all_data_no_self = all_data.drop(['sum', ass], axis=1).sum(axis=1)
        all_ass_dfs_with_self.append(all_data_with_self)
        all_ass_dfs_no_self.append(all_data_no_self)
    combined_df_with_self = pd.concat(all_ass_dfs_with_self, axis=1)
    combined_df_no_self = pd.concat(all_ass_dfs_no_self, axis=1)
    combined_df_with_self.columns = all_as
    combined_df_no_self.columns = all_as
    combined_df_with_self.iloc[5:].to_csv('corona_data_with_self.csv')
    combined_df_no_self.iloc[5:].to_csv('corona_data_no_self.csv')


if __name__ == '__main__':
    set_default_config('../../configurations/config.ini')
    global_vars.set('dataset', 'netflow_asflow')
    set_params_by_dataset('../../configurations/dataset_params.ini')
    global_vars.set('handovers',
                    [1299, 2914, 174, 3257, 6762, 6453, 3356, 33891, 6939, 6461, 1273, 5511, 15169, 20940, 2906, 16509,
                     32934, 15133, 22822, 202818, 16276, 46489])
    global_vars.set('netflow_handover_locations', [str(h) for h in global_vars.get('handovers')])
    global_vars.set('normalize_netflow_data', True)
    global_vars.set('date_range', "1.7.2017-31.12.2019")
    global_vars.set('start_hour', 15)
    global_vars.set('input_height', 240)
    global_vars.set('prediction_buffer', 2)
    global_vars.set('steps_ahead', 5)
    global_vars.set('jumps', 24)
    global_vars.set('netflow_subfolder', 'top_10')
    all_as = [15169, 20940, 2906, 16509, 32934, 15133, 22822, 202818, 16276, 46489]
    # all_as = [15169, 20940]

    get_corona_data(all_as)
    # get_all_data(20940)
    # get_data_samples(all_as)