from evolution.EEGNAS import EEGNAS
from data_preprocessing import get_pure_cross_subject
import NASUtils
import torch
import globals
from evolution.nn_training import NN_Trainer


class EEGNAS_from_file(EEGNAS):
    def __init__(self, iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 subject_id, fieldnames, csv_file, model_from_file=None):
        super(EEGNAS_from_file, self).__init__(iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 subject_id, fieldnames, csv_file)
        self.model_from_file = model_from_file

    def run_target_model(self):
        globals.set('max_epochs', globals.get('final_max_epochs'))
        globals.set('max_increase_epochs', globals.get('final_max_increase_epochs'))
        stats = {}
        if self.model_from_file is not None:
            if torch.cuda.is_available():
                model = torch.load(self.model_from_file)
            else:
                model = torch.load(self.model_from_file, map_location='cpu')
        else:
            model = target_model(globals.get('model_name'))
        if globals.get('target_pretrain'):
            self.datasets['train']['pretrain'], self.datasets['valid']['pretrain'], self.datasets['test']['pretrain'] = \
                get_pure_cross_subject(globals.get('data_folder'), globals.get('low_cut_hz'))
            nn_trainer = NN_Trainer(self.iterator, self.loss_function, self.stop_criterion, self.monitors)
            dataset = self.get_single_subj_dataset('pretrain', final_evaluation=False)
            _, _, model, _, _ = nn_trainer.evaluate_model(model, dataset)
        dataset = self.get_single_subj_dataset(self.subject_id, final_evaluation=True)
        nn_trainer = NN_Trainer(self.iterator, self.loss_function, self.stop_criterion, self.monitors)
        _, _, model, _, _ = nn_trainer.evaluate_model(model, dataset)
        final_time, evaluations, model, model_state, num_epochs =\
                    nn_trainer.evaluate_model(model, final_evaluation=True)
        stats['final_train_time'] = str(final_time)
        NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix="final_")
        self.write_to_csv(stats, generation=1)