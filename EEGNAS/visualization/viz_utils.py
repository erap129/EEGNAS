from copy import deepcopy
import numpy as np
from braindecode.torch_ext.util import np_to_var
import pandas as pd
from EEGNAS import global_vars
from EEGNAS_experiment import get_normal_settings
from EEGNAS.data_preprocessing import get_pure_cross_subject
from EEGNAS.evolution.nn_training import NN_Trainer
from EEGNAS.utilities.data_utils import get_dummy_input
from EEGNAS.visualization.cnn_layer_visualization import CNNLayerVisualization


def pretrain_model_on_filtered_data(pretrained_model, low_freq, high_freq):
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    pure_cross_subj_dataset = {}
    pure_cross_subj_dataset['train'], pure_cross_subj_dataset['valid'], \
    pure_cross_subj_dataset['test'] = get_pure_cross_subject(global_vars.get('data_folder'))
    freq_models = {}
    pure_cross_subj_dataset_copy = deepcopy(pure_cross_subj_dataset)
    for freq in range(low_freq, high_freq + 1):
        pretrained_model_copy = deepcopy(pretrained_model)
        for section in ['train', 'valid', 'test']:
            pure_cross_subj_dataset_copy[section].X = global_vars.get('band_filter')\
                (pure_cross_subj_dataset_copy[section].X, max(1, freq - 1), freq + 1, global_vars.get('frequency')).astype(np.float32)
        nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
        _, _, model, _, _ = nn_trainer.train_and_evaluate_model(pretrained_model_copy, pure_cross_subj_dataset_copy)
        freq_models[freq] = model
    return freq_models


def create_max_examples_per_channel(select_layer, model, steps=500):
    dummy_X = get_dummy_input().cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
        dummy_X = l(dummy_X)
    channels = dummy_X.shape[1]
    act_maps = []
    for c in range(channels):
        layer_vis = CNNLayerVisualization(model, select_layer, c)
        act_maps.append(layer_vis.visualise_layer_with_hooks(steps))
        print(f'created optimal example for layer {select_layer}, channel {c}')
    return act_maps


def get_max_examples_per_channel(data, select_layer, model):
    act_maps = {}
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for idx, example in enumerate(x):
        example_x = example[None, :, :, :]
        for l in modules[:select_layer + 1]:
            example_x = l(example_x)
        act_maps[idx] = example_x
    channels = act_maps[0].shape[1]
    selected_examples = np.zeros(channels)
    for c in range(channels):
        selected_examples[c]\
            = int(np.array([act_map.squeeze()[c].sum() for act_map in act_maps.values()]).argmax())
    return [int(x) for x in selected_examples]


def export_performance_frequency_to_csv(performances, retrained_performances, baselines, folder_name):
    df = pd.DataFrame()
    for subj_id, (performance, retrained_performance) in enumerate(zip(performances, retrained_performances)):
        for freq, (perf_freq, retrained_perf_freq) in enumerate(zip(performance, retrained_performance)):
            example_df = pd.DataFrame()
            if subj_id == len(performances) - 1:
                example_df['subject'] = ['average']
            else:
                example_df['subject'] = [subj_id + 1]
            example_df['baseline'] = baselines[subj_id]
            example_df['frequency'] = [freq + 1]
            example_df['performance'] = [perf_freq]
            example_df['retrained_performance'] = [retrained_perf_freq]
            df = df.append(example_df)
    df.to_csv(f'{folder_name}/performance_frequency_{global_vars.get("band_filter").__name__}.csv')
