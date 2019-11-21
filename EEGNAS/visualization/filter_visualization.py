import configparser
import os
import sys
from collections import defaultdict
from copy import deepcopy
from sacred import Experiment
from sacred.observers import MongoObserver
sys.path.append("..")
sys.path.append("../..")
from EEGNAS.visualization.external_models import MultivariateLSTM
from EEGNAS.evolution.nn_training import NN_Trainer
from EEGNAS.visualization import viz_reports
from EEGNAS.utilities.config_utils import set_default_config, update_global_vars_from_config_dict, get_configurations
from EEGNAS.utilities.misc import concat_train_val_sets
import logging
from EEGNAS.visualization.dsp_functions import butter_bandstop_filter, butter_bandpass_filter
from EEGNAS.visualization.signal_plotting import plot_one_tensor
import torch
from braindecode.torch_ext.util import np_to_var
from EEGNAS import global_vars
from EEGNAS.data_preprocessing import get_dataset
from EEGNAS_experiment import set_params_by_dataset, get_normal_settings
import matplotlib.pyplot as plt
from EEGNAS.utilities.misc import create_folder
from EEGNAS.visualization.pdf_utils import create_pdf
import numpy as np
from EEGNAS.visualization.wavelet_functions import subtract_frequency
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet
from EEGNAS.utilities.misc import label_by_idx
styles = getSampleStyleSheet()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
plt.interactive(False)
ex = Experiment()
SHAP_VALUES = {}


def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


@ex.main
def main():
    global SHAP_VALUES
    res = getattr(viz_reports, f'{global_vars.get("report")}_report')(model, dataset, folder_name)
    if global_vars.get('report') == 'shap':
        SHAP_VALUES[(global_vars.get('model_name'), global_vars.get('iteration'))] = res

if __name__ == '__main__':
    configs = configparser.ConfigParser()
    configs.read('visualization_configurations/viz_config.ini')
    configurations = get_configurations(sys.argv[1], configs, set_exp_name=False)
    global_vars.init_config('configurations/config.ini')
    set_default_config('../configurations/config.ini')
    global_vars.set('cuda', True)

    prev_dataset = None
    for configuration in configurations:
        update_global_vars_from_config_dict(configuration)
        global_vars.set('band_filter', {'pass': butter_bandpass_filter,
                                        'stop': butter_bandstop_filter}[global_vars.get('band_filter')])

        set_params_by_dataset('../configurations/dataset_params.ini')
        subject_id = global_vars.get('subject_id')
        dataset = get_dataset(subject_id)
        prev_dataset = global_vars.get('dataset')

        if global_vars.get('model_name') == 'rnn':
            model = MultivariateLSTM(dataset['train'].X.shape[1], 100, global_vars.get('batch_size'),
                                     global_vars.get('input_height'), global_vars.get('n_classes'), eegnas=True)
        else:
            model = torch.load(f'../models/{global_vars.get("models_dir")}/{global_vars.get("model_name")}')
        model.cuda()

        if global_vars.get('finetune_model'):
            stop_criterion, iterator, loss_function, monitors = get_normal_settings()
            trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
            model = trainer.train_model(model, dataset, final_evaluation=True)

        concat_train_val_sets(dataset)

        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y-%H:%M")
        folder_name = f'results/{date_time}_{global_vars.get("dataset")}_{global_vars.get("report")}'
        create_folder(folder_name)
        print(f'generating {global_vars.get("report")} report for model:')
        print(model)
        if global_vars.get('to_eeglab'):
            create_folder(f'{folder_name}/{global_vars.get("report")}')

        exp_name = f"{global_vars.get('dataset')}_{global_vars.get('report')}"
        ex.config = {}
        ex.add_config(configuration)
        if len(ex.observers) == 0 and len(sys.argv) <= 2:
            ex.observers.append(MongoObserver.create(url=f'mongodb://132.72.80.67/{global_vars.get("mongodb_name")}',
                                                     db_name=global_vars.get("mongodb_name")))
        global_vars.set('sacred_ex', ex)
        ex.run(options={'--name': exp_name})

    if len(SHAP_VALUES.keys()) > 0:
        models = list(set([i for (i,j) in list(SHAP_VALUES.keys())]))
        iterations = list(set([j for (i,j) in list(SHAP_VALUES.keys())]))

        for segment in ['train', 'test', 'both']:
            mse_avg_per_model = {}
            for model in models:
                model_avg = 0
                mse_calc = []
                for iteration in iterations:
                    mse_calc.append(SHAP_VALUES[(model, iteration)][segment])
                for shap_val_1 in mse_calc:
                    for shap_val_2 in mse_calc:
                        model_avg += (np.square(shap_val_1 - shap_val_2)).mean()
                model_avg = model_avg / len(mse_calc)
                with open(f"{folder_name}/shap_mse.txt", "a") as f:
                    f.write(f'{segment}: average MSE between iterations for model {model} is {model_avg}\n')
                mse_avg_per_model[model] = np.mean(mse_calc, axis=0)

            for model1 in models:
                for model2 in models:
                    with open(f"{folder_name}/shap_mse.txt", "a") as f:
                        f.write(f'{segment}: MSE between model {model1} and model {model2} is '
                                f'{(np.square(mse_avg_per_model[model1] - mse_avg_per_model[model2])).mean()}\n')


