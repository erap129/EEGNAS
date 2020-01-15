from datetime import datetime
import os
import pickle

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


def get_evaluator(evaluator_type):
    if global_vars.get('per_handover_prediction'):
        output_size = global_vars.get('steps_ahead') * global_vars.get('max_handovers')
    else:
        output_size = global_vars.get('steps_ahead')
    if type(evaluator_type) == list:
        models = [get_evaluator(ev) for ev in global_vars.get('evaluator')]
        if not global_vars.get('true_ensemble_avg'):
            model = BasicEnsemble(models, output_size)
        else:
            model = AveragingEnsemble(models, global_vars.get('true_ensemble_avg'))
        model.cpu()
        return model
    if evaluator_type == 'cnn':
        if global_vars.get('cnn_ensemble'):
            all_population_files = os.listdir('eegnas_models')
            pop_files = list(filter(filter_eegnas_population_files, all_population_files))
            assert len(pop_files) == 1
            pop_file = pop_files[0]
            model = create_ensemble_from_population_file(f'eegnas_models/{pop_file}', 5)
            for mod in model.models:
                reset_model_weights(mod)
            model.cpu()
        return model
    elif evaluator_type == 'nsga':
        if global_vars.get('per_handover_prediction'):
            genotype = NetflowMultiHandover
            model = NetworkCIFAR(24, output_size, global_vars.get('max_handovers'), 11, False, genotype)
        else:
            with open('nsga_models/best_genome_normalized.pkl', 'rb') as f:
                genotype = pickle.load(f)
                genotype = decode(convert(genotype))
                model = NetworkCIFAR(24, global_vars.get('steps_ahead'), global_vars.get('max_handovers'), 11, False, genotype)
        model.droprate = 0.0
        model.single_output = True
        return model
    elif evaluator_type == 'rnn':
        if global_vars.get('per_handover_prediction'):
            model = MultivariateParallelMultistepLSTM(global_vars.get('max_handovers'), 100, global_vars.get('batch_size'),
                                              global_vars.get('input_height'), num_layers=global_vars.get('lstm_layers'),
                                                      eegnas=True)

        else:
            model = MultivariateLSTM(global_vars.get('max_handovers'), 100, global_vars.get('batch_size'),
                                 global_vars.get('input_height'), global_vars.get('steps_ahead'), num_layers=global_vars.get('lstm_layers'), eegnas=True)
        model.cpu()
        return model
    elif evaluator_type == 'LSTNet':
        model = LSTNetModel(global_vars.get('max_handovers'), output_size, window=global_vars.get('input_height'))
        model.cpu()
        return model


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
        start_date, end_date = global_vars.get('date_range').split('-')
        if len(dataset.X) != abs((datetime.strptime(start_date, "%d.%m.%Y") -
                                  datetime.strptime(end_date, "%d.%m.%Y")).days) - global_vars.get('input_height') / 24:
            print(f'dropped AS file: {filename}')
            continue
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
    if os.path.exists(filename):
        model = torch.load(filename)
        print(f'loaded pretrained model: {filename}')
        return model
    return None


def get_model_filename_kfold(type, fold_idx):
    return f"{type}/{global_vars.get('dataset')}_{global_vars.get('date_range')}_{global_vars.get('autonomous_systems')}" \
            f"_{global_vars.get('per_handover_prediction')}_" \
            f"{global_vars.get('final_max_epochs')}_{global_vars.get('data_augmentation')}_" \
            f"{global_vars.get('n_folds')}_{fold_idx}_{global_vars.get('iteration')}.th"

