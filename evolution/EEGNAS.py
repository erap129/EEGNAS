import csv
import platform
import time
from collections import OrderedDict

from braindecode.datautil.splitters import concatenate_sets
from braindecode.experiments.loggers import Printer
import globals
from data_preprocessing import get_train_val_test


class EEGNAS:
    def __init__(self, iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 subject_id, fieldnames, csv_file):
        global model_train_times
        model_train_times = []
        self.iterator = iterator
        self.exp_folder = exp_folder
        self.exp_name = exp_name
        self.monitors = monitors
        self.loss_function = loss_function
        self.stop_criterion = stop_criterion
        self.subject_id = subject_id
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', val_set), ('test', test_set))
        )
        self.cuda = globals.get('cuda')
        self.loggers = [Printer()]
        self.fieldnames = fieldnames
        self.models_set = []
        self.genome_set = []
        self.csv_file = csv_file
        self.current_model_index = -1
        if isinstance(self.subject_id, int):
            self.current_chosen_population_sample = [self.subject_id]
        else:
            self.current_chosen_population_sample = []
        self.mutation_rate = globals.get('mutation_rate')

    def write_to_csv(self, stats, generation, model='avg'):
        if self.csv_file is not None:
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if self.subject_id == 'all':
                    subject = ','.join(str(x) for x in self.current_chosen_population_sample)
                else:
                    subject = str(self.subject_id)
                for key, value in stats.items():
                    writer.writerow({'exp_name': self.exp_name, 'machine': platform.node(),
                                    'dataset': globals.get('dataset'), 'date': time.strftime("%d/%m/%Y"),
                                    'subject': subject, 'generation': str(generation), 'model': str(model),
                                    'param_name': key, 'param_value': value})

    def get_single_subj_dataset(self, subject=None, final_evaluation=False):
        if subject not in self.datasets['train'].keys():
            self.datasets['train'][subject], self.datasets['valid'][subject], self.datasets['test'][subject] = \
                get_train_val_test(globals.get('data_folder'), subject, globals.get('low_cut_hz'))
        single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject]),
                                           ('valid', self.datasets['valid'][subject]),
                                           ('test', self.datasets['test'][subject])))
        if final_evaluation:
            single_subj_dataset['train'] = concatenate_sets(
                [single_subj_dataset['train'], single_subj_dataset['valid']])
        return single_subj_dataset