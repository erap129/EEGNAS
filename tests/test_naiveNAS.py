import unittest
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from utils import GenericMonitor, acc_func
import numpy as np
from naiveNAS import NaiveNAS, finalize_model
import NASUtils
from models_generation import uniform_model, breed_layers,\
    ConvLayer, ActivationLayer
from BCI_IV_2a_experiment import get_configurations, parse_args
from models_generation import target_model, random_model, random_grid_model
import globals
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from globals import init_config
import torch.nn.functional as F


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations(args)
        assert(len(configs) == 1)
        globals.set_config(configs[0])
        globals.set('eeg_chans', 22)
        globals.set('num_subjects', 9)
        globals.set('input_time_len', 1125)
        globals.set('n_classes', 4)
        input_shape = (50, globals.get('eeg_chans'), globals.get('input_time_len'))
        class Dummy:
            def __init__(self, X, y):
                self.X = X
                self.y = y
        dummy_data = Dummy(X=np.ones(input_shape, dtype=np.float32), y=np.ones(50, dtype=np.longlong))
        self.iterator = BalancedBatchSizeIterator(batch_size=globals.get('batch_size'))
        self.loss_function = F.nll_loss
        self.monitors = [LossMonitor(), MisclassMonitor(), GenericMonitor('accuracy', acc_func), RuntimeMonitor()]
        self.stop_criterion = Or([MaxEpochs(globals.get('max_epochs')),
                             NoDecrease('valid_misclass', globals.get('max_increase_epochs'))])
        self.naiveNAS = NaiveNAS(iterator=self.iterator, exp_folder='../tests', exp_name='',
                            train_set=dummy_data, val_set=dummy_data, test_set=dummy_data,
                            stop_criterion=self.stop_criterion, monitors=self.monitors, loss_function=self.loss_function,
                            config=globals.config, subject_id=1, fieldnames=None,
                            model_from_file=None)

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
        globals.set('mutation_rate', 1)
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
        self.naiveNAS.evaluate_model(model)

    def test_save_best_model(self):
        weighted_population = [{'model': random_model(10)}]
        self.naiveNAS.save_best_model(weighted_population)

    def test_save_best_model_grid(self):
        globals.set('grid', True)
        weighted_population = [{'model': random_grid_model(10)}]
        self.naiveNAS.save_best_model(weighted_population)
    

if __name__ == '__main__':
    unittest.main()