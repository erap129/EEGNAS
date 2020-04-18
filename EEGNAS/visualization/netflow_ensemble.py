from datetime import datetime
import os
import pickle
import ast
import torch
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from torch import nn
import sys


import numpy as np
from EEGNAS import global_vars
from EEGNAS.data_preprocessing import makeDummySignalTargets, get_dataset
from EEGNAS.model_generation.custom_modules import BasicEnsemble, AveragingEnsemble
from EEGNAS.model_generation.simple_model_generation import create_ensemble_from_population_file
from EEGNAS.utilities.data_utils import unison_shuffled_copies
from EEGNAS.utilities.misc import reset_model_weights, concat_train_val_sets, unify_dataset
from EEGNAS.visualization.external_models import MultivariateLSTM, LSTNetModel
from nsga_net.models.micro_models import NetworkCIFAR
from nsga_net.models.micro_genotypes import NetflowMultiHandover
from nsga_net.search.micro_encoding import convert, decode

def train_model_for_netflow(model, dataset, trainer):
    print(f'start training model: {type(model).__name__}')
    if 'Ensemble' in type(model).__name__:
        for mod in model.models:
            mod.cuda()
            mod.train()
            train_model_for_netflow(mod, dataset, trainer)
        if type(model) == BasicEnsemble:
            model.freeze_child_models(False)
            trainer.train_model(model, dataset, final_evaluation=True)
            model.freeze_child_models(True)
    else:
        if global_vars.get('dataset') in ['solar', 'electricity', 'exchange_rate'] and global_vars.get('training_method') == 'LSTNet':
            data = Data_utility(f'../EEGNAS/EEGNAS/data/MTS_benchmarks/'
                                f'{global_vars.get("dataset")}.txt', 0.6, 0.2, device='cuda', window=24 * 7, horizon=12)
            optim = Optim(model.parameters(), 'adam', 0.001, 10.)
            criterion = torch.nn.MSELoss(size_average=False)
            MTS_train(model, data, criterion, optim, 100, 128)
        else:
            trainer.train_model(model, dataset, final_evaluation=True)


def filter_eegnas_population_files(filename):
    res = True
    if global_vars.get('normalize_netflow_data'):
        res = res and 'normalized' in filename
    if global_vars.get('per_handover_prediction'):
        res = res and 'per_handover' in filename
    else:
        res = res and 'per_handover' not in filename
    if global_vars.get('top_handovers'):
        res = res and f'top{global_vars.get("top_handovers")}' in filename
    else:
        res = res and 'top' not in filename
    if global_vars.get('same_handover_locations'):
        res = res and 'samelocs' in filename
    else:
        res = res and 'samelocs' not in filename
    if global_vars.get('current_fold_idx') is not None:
        res = res and f'fold{global_vars.get("current_fold_idx")}' in filename
    res = res and f'input_height_{global_vars.get("input_height")}' in filename
    return res


class MultivariateParallelMultistepLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim, batch_size, n_steps, num_layers=3, output_dim=5, eegnas=False):
        super(MultivariateParallelMultistepLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps
        self.output_dim = output_dim
        self.eegnas = eegnas

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim * self.n_features)

    def init_hidden(self, b_size=None):
        # This is what we'll initialise our hidden state as
        if b_size is None:
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda(),
                    torch.zeros(self.num_layers, b_size, self.hidden_dim).cuda())

    def forward(self, input, b_size=None):
        if self.eegnas:
            input = input.squeeze(dim=3)
            input = input.permute(0, 2, 1)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        if self.eegnas:
            y_pred = self.linear(lstm_out.contiguous().view(input.shape[0], -1))
        elif b_size is None:
            y_pred = self.linear(lstm_out.contiguous().view(self.batch_size, -1)).view(self.batch_size, self.output_dim,
                                                                                       self.n_features)
        else:
            y_pred = self.linear(lstm_out.contiguous().view(b_size, -1)).view(b_size, self.output_dim,
                                                                              self.n_features)
        return y_pred

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()


def get_evaluator(evaluator_type, fold_idx=None):
    if global_vars.get('top_handovers'):
        channels = global_vars.get('top_handovers')
    else:
        channels = global_vars.get('max_handovers')
    if global_vars.get('per_handover_prediction'):
        output_size = global_vars.get('steps_ahead') * channels
    else:
        output_size = global_vars.get('steps_ahead')
    if type(evaluator_type) == list:
        models = [get_evaluator(ev, fold_idx) for ev in global_vars.get('evaluator')]
        if not global_vars.get('true_ensemble_avg'):
            model = BasicEnsemble(models, output_size)
        else:
            model = AveragingEnsemble(models, global_vars.get('true_ensemble_avg'))
        model.cpu()
        return model
    if evaluator_type == 'cnn':
        if global_vars.get('cnn_ensemble'):
            all_population_files = os.listdir('eegnas_models')
            global_vars.set('current_fold_idx', fold_idx)
            pop_files = list(filter(filter_eegnas_population_files, all_population_files))
            assert len(pop_files) == 1
            pop_file = pop_files[0]
            model = create_ensemble_from_population_file(f'eegnas_models/{pop_file}', 5)
            if not global_vars.get('skip_cnn_training'):
                for mod in model.models:
                    reset_model_weights(mod)
            model.cpu()
        else:
            model = torch.load(
                f'../EEGNAS/EEGNAS/models/{global_vars.get("models_dir")}/{global_vars.get("model_file_name")}')
            reset_model_weights(model)
            model.cpu()
            load_values_from_config(f'../EEGNAS/EEGNAS/models/{global_vars.get("models_dir")}'
                                    f'/config_{global_vars.get("models_dir")}.ini',
                                    ['input_height', 'start_hour', 'start_point', 'date_range', 'prediction_buffer',
                                     'steps_ahead', 'jumps', 'normalize_netflow_data', 'per_handover_prediction'])
        return model
    elif evaluator_type == 'nsga':
        nsga_file = 'nsga_models/best_genome_normalized.pkl'
        if global_vars.get('top_handovers'):
            nsga_file = f'{nsga_file[:-4]}_top{global_vars.get("top_handovers")}.pkl'
        if global_vars.get('per_handover_prediction'):
            nsga_file = f'{nsga_file[:-4]}_samelocs.pkl'
        if global_vars.get('same_handover_locations'):
            nsga_file = f'{nsga_file[:-4]}_per_handover.pkl'
        if fold_idx is not None:
            nsga_file = f'{nsga_file[:-4]}_fold{fold_idx}.pkl'
        with open(nsga_file, 'rb') as f:
            genotype = pickle.load(f)
            genotype = decode(convert(genotype))
            model = NetworkCIFAR(24, global_vars.get('steps_ahead'), channels, 11, False, genotype)
        model.droprate = 0.0
        model.single_output = True
        return model
    elif evaluator_type == 'rnn':
        if global_vars.get('highest_handover_overflow'):
            model = LSTMMulticlassClassification(channels, 100, global_vars.get('batch_size'),
                                              global_vars.get('input_height'), num_layers=global_vars.get('lstm_layers'), eegnas=True)
        elif global_vars.get('per_handover_prediction'):
            model = MultivariateParallelMultistepLSTM(channels, 100, global_vars.get('batch_size'),
                                              global_vars.get('input_height'), num_layers=global_vars.get('lstm_layers'),
                                                      eegnas=True)

        else:
            model = MultivariateLSTM(channels, 100, global_vars.get('batch_size'),
                                 global_vars.get('input_height'), global_vars.get('steps_ahead'), num_layers=global_vars.get('lstm_layers'), eegnas=True)
        model.cpu()
        return model
    elif evaluator_type == 'LSTNet':
        model = LSTNetModel(channels, output_size, window=global_vars.get('input_height'))
        model.cpu()
        return model
    elif evaluator_type == 'MHANet':
        model = MHANetModel(channels, output_size)
        model.cpu()
        return model
    elif evaluator_type == 'WaveNet':
        model = WaveNet(input_channels=channels, output_channels=1, horizon=global_vars.get('steps_ahead'))
        model.cpu()
        return model
    elif evaluator_type == 'xgboost':
        model = MultiOutputRegressor(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
            max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
            n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
            silent=True, subsample=1, tree_method='gpu_hist', gpu_id=0))
        return model
    elif evaluator_type == 'autokeras':
        return ak.ImageRegressor(max_trials=3)


def get_fold_idxs(AS):
    if global_vars.get('k_fold_time'):
        kf = TimeSeriesSplit(n_splits=global_vars.get('n_folds'))
    else:
        kf = KFold(n_splits=global_vars.get('n_folds'), shuffle=True)
    prev_autonomous_systems = global_vars.get('autonomous_systems')
    global_vars.set('autonomous_systems', [AS])
    dataset = get_dataset('all')
    concat_train_val_sets(dataset)
    dataset = unify_dataset(dataset)
    fold_idxs = {i: {} for i in range(global_vars.get('n_folds'))}
    for fold_num, (train_index, test_index) in enumerate(kf.split(list(range(len(dataset.X))))):
        fold_idxs[fold_num]['train_idxs'] = train_index
        fold_idxs[fold_num]['test_idxs'] = test_index
    global_vars.set('autonomous_systems', prev_autonomous_systems)
    return fold_idxs


def get_data_by_balanced_folds(ASs, fold_idxs):
    prev_autonomous_systems = global_vars.get('autonomous_systems')
    folds = {i: {'X_train': [], 'X_test': [], 'y_train': [], 'y_test': []} for i in range(global_vars.get('n_folds'))}
    for AS in ASs:
        global_vars.set('autonomous_systems', [AS])
        dataset = get_dataset('all')
        concat_train_val_sets(dataset)
        dataset = unify_dataset(dataset)
        for fold_idx in range(global_vars.get('n_folds')):
            folds[fold_idx]['X_train'].extend(dataset.X[fold_idxs[fold_idx]['train_idxs']])
            folds[fold_idx]['X_test'].extend(dataset.X[fold_idxs[fold_idx]['test_idxs']])
            folds[fold_idx]['y_train'].extend(dataset.y[fold_idxs[fold_idx]['train_idxs']])
            folds[fold_idx]['y_test'].extend(dataset.y[fold_idxs[fold_idx]['test_idxs']])
    for key in folds.keys():
        for inner_key in folds[key].keys():
            folds[key][inner_key] = np.stack(folds[key][inner_key], axis=0)
    global_vars.set('autonomous_systems', prev_autonomous_systems)
    return folds


def get_dataset_from_folds(fold_samples):
    X_train, X_test = fold_samples['X_train'], fold_samples['X_test']
    y_train, y_test = fold_samples['y_train'], fold_samples['y_test']
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'),
                                                      shuffle=True)
    X_train, y_train = unison_shuffled_copies(X_train, y_train)
    dataset = {}
    dataset['train'], dataset['valid'], dataset['test'] = \
        makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
    return dataset


def get_pretrained_model(filename):
    filename_split = filename.split('_')
    ass = ast.literal_eval(filename_split[4])
    del filename_split[4]
    for file in [f'kfold_models/{f}' for f in os.listdir('kfold_models')]:
        file_split = file.split('_')
        curr_ass = ast.literal_eval(file_split[4])
        del file_split[4]
        if set(ass) == set(curr_ass) and file_split == filename_split:
            model = torch.load(file)
            print(f'loaded {file}')
            return model
    return None


def get_model_filename_kfold(type, fold_idx):
    unique_test_str, drop_others_str, top_pni_str, test_handover_str, as_str, samelocs_str, evaluator_str, seed_str, handover_str, interpolation_str = '', '', '', '', '', '', '', '', '', ''
    if global_vars.get('interpolate_netflow'):
        interpolation_str = '_interp'
    if global_vars.get('top_handovers'):
        handover_str = f'_top{global_vars.get("top_handovers")}'
    if global_vars.get('random_ho_permutations') and global_vars.get('permidx'):
        seed_str = f'_permidx_{global_vars.get("permidx")}'
    if set(global_vars.get('evaluator')) != set(["cnn", "rnn", "LSTNet", "nsga"]):
        evaluator_str = f'_{"_".join(global_vars.get("evaluator"))}'
    if global_vars.get('same_handover_locations'):
        samelocs_str = '_samelocs'
        if global_vars.get('test_handover_locs'):
            test_handover_str = '_testlocs'
    if global_vars.get('netflow_subfolder') == 'top_99':
        as_str = f'top{len(global_vars.get("autonomous_systems"))}'
    else:
        as_str = global_vars.get('autonomous_systems')
    if global_vars.get('top_pni'):
        top_pni_str = '_top_pni'
    if not global_vars.get('netflow_drop_others'):
        drop_others_str = '_others'
    if global_vars.get('unique_test_model'):
        unique_test_str = f'_unq_{global_vars.get("as_to_test")}'

    return f"{type}/{global_vars.get('dataset')}_{global_vars.get('date_range')}_{as_str}" \
            f"_{global_vars.get('per_handover_prediction')}_" \
            f"{global_vars.get('final_max_epochs')}_{global_vars.get('data_augmentation')}_" \
            f"{global_vars.get('n_folds')}_{fold_idx}_{global_vars.get('iteration')}{interpolation_str}" \
            f"{handover_str}{seed_str}{evaluator_str}{samelocs_str}{test_handover_str}{top_pni_str}{drop_others_str}{unique_test_str}.th"


