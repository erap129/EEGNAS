import unittest
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from utils import GenericMonitor, acc_func
import numpy as np
from naiveNAS import NaiveNAS, finalize_model
from EEGNAS import NASUtils, global_vars
from EEGNAS.models_generation import uniform_model, breed_layers,\
    ConvLayer, ActivationLayer
from EEGNAS_experiment import get_configurations, parse_args, set_params_by_dataset, get_normal_settings
from EEGNAS.models_generation import target_model, random_model, random_grid_model
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from EEGNAS.global_vars import init_config
from EEGNAS.data_preprocessing import DummySignalTarget
import torch.nn.functional as F


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations(args.experiment)
        assert(len(configs) == 1)
        global_vars.set_config(configs[0])
        global_vars.set('eeg_chans', 22)
        global_vars.set('num_subjects', 9)
        global_vars.set('input_time_len', 1125)
        global_vars.set('n_classes', 4)
        set_params_by_dataset()
        input_shape = (50, global_vars.get('eeg_chans'), global_vars.get('input_time_len'))
        class Dummy:
            def __init__(self, X, y):
                self.X = X
                self.y = y
        dummy_data = Dummy(X=np.ones(input_shape, dtype=np.float32), y=np.ones(50, dtype=np.longlong))
        self.iterator = BalancedBatchSizeIterator(batch_size=global_vars.get('batch_size'))
        self.loss_function = F.nll_loss
        self.monitors = [LossMonitor(), MisclassMonitor(), GenericMonitor('accuracy', acc_func), RuntimeMonitor()]
        self.stop_criterion = Or([MaxEpochs(global_vars.get('max_epochs')),
                                  NoDecrease('valid_misclass', global_vars.get('max_increase_epochs'))])
        self.naiveNAS = NaiveNAS(iterator=self.iterator, exp_folder='../tests', exp_name='',
                                 train_set=dummy_data, val_set=dummy_data, test_set=dummy_data,
                                 stop_criterion=self.stop_criterion, monitors=self.monitors, loss_function=self.loss_function,
                                 config=global_vars.config, subject_id=1, fieldnames=None,
                                 model_from_file=None)

    @staticmethod
    def generate_dummy_data(batch_size):
        dummy_data = DummySignalTarget(np.random.rand(batch_size, global_vars.get('eeg_chans'), global_vars.get('input_time_len')),
                                       np.random.randint(0, global_vars.get('n_classes') - 1, batch_size))
        return dummy_data, dummy_data, dummy_data

    def test_hash_model_nochange(self):
        model1 = uniform_model(10, ActivationLayer)
        model2 = uniform_model(10, ActivationLayer)
        model_set = []
        genome_set = []
        NASUtils.hash_model(model1, model_set, genome_set)
        NASUtils.hash_model(model2, model_set, genome_set)
        assert(len(model_set)) == 1
        assert(len(genome_set)) == 1

    def test_hash_model(self):
        model1 = uniform_model(10, ActivationLayer)
        model2 = uniform_model(10, ActivationLayer)
        global_vars.set('mutation_rate', 1)
        model3, _ = breed_layers(1, model1, model2)
        model_set = []
        genome_set = []
        NASUtils.hash_model(model1, model_set, genome_set)
        NASUtils.hash_model(model2, model_set, genome_set)
        NASUtils.hash_model(model3, model_set, genome_set)
        assert(len(model_set)) == 1 or 2
        assert(len(genome_set)) == 1 or 2

    def test_hash_model_same_type(self):
        model1 = [ConvLayer(kernel_eeg_chan=1), ConvLayer(kernel_eeg_chan=2)]
        model2 = [ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1)]
        model_set = []
        genome_set = []
        NASUtils.hash_model(model1, model_set, genome_set)
        NASUtils.hash_model(model2, model_set, genome_set)
        assert (len(model_set)) == 2
        assert (len(genome_set)) >= 2

    def test_remove_from_hash(self):
        model1 = [ConvLayer(kernel_eeg_chan=1), ConvLayer(kernel_eeg_chan=2), ActivationLayer()]
        model2 = [ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1), ActivationLayer()]
        model_set = []
        genome_set = []
        NASUtils.hash_model(model1, model_set, genome_set)
        NASUtils.hash_model(model2, model_set, genome_set)
        assert(len(genome_set)) <= 5
        NASUtils.remove_from_models_hash(model1, model_set, genome_set)
        assert(len(genome_set)) >= 3
        NASUtils.remove_from_models_hash(model2, model_set, genome_set)
        assert(len(genome_set)) == 0

    def test_evaluation_target_model(self):
        model = target_model('deep')
        self.naiveNAS.train_and_evaluate_model(model)

    def test_save_best_model(self):
        weighted_population = [{'model': random_model(10)}]
        self.naiveNAS.save_best_model(weighted_population)

    def test_save_best_model_grid(self):
        global_vars.set('grid', True)
        global_vars.set('layer_num', [10, 10])
        weighted_population = [{'model': random_grid_model([10,10])}]
        self.naiveNAS.save_best_model(weighted_population)

    # def test_perm_ensembles(self):
    #     train_dummy, val_dummy, test_dummy = self.generate_dummy_data(10)
    #     stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    #     naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
    #                         train_set=train_dummy, val_set=val_dummy, test_set=test_dummy,
    #                         stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
    #                         config=globals.config, subject_id=1, fieldnames=None)
    #     naiveNAS.evolution_layers(None, None)

if __name__ == '__main__':
    unittest.main()
