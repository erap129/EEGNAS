import unittest
from naiveNAS import NaiveNAS
from models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, ConvLayer,\
    MyModel, ActivationLayer, random_model
from BCI_IV_2a_experiment import get_configurations
import globals
from globals import init_config


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        init_config('test_configs/single_subject_config.ini')
        configs = get_configurations()
        assert(len(configs) == 1)
        globals.set_config(configs[0])

    def test_hash_model_nochange(self):
        model1 = uniform_model(10, ActivationLayer)
        model2 = uniform_model(10, ActivationLayer)
        model_set = []
        genome_set = []
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        assert(len(model_set)) == 1
        assert(len(genome_set)) == 1

    def test_hash_model(self):
        model1 = uniform_model(10, ActivationLayer)
        model2 = uniform_model(10, ActivationLayer)
        globals.config['evolution']['mutation_rate'] = 1
        model3, _ = breed_layers(1, model1, model2)
        model_set = []
        genome_set = []
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        NaiveNAS.hash_model(model3, model_set, genome_set)
        assert(len(model_set)) == 1 or 2
        assert(len(genome_set)) == 1 or 2

    def test_hash_model_same_type(self):
        model1 = [ConvLayer(kernel_eeg_chan=1), ConvLayer(kernel_eeg_chan=2)]
        model2 = [ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1)]
        model_set = []
        genome_set = []
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        assert (len(model_set)) == 2
        assert (len(genome_set)) >= 2

    def test_remove_from_hash(self):
        model1 = [ConvLayer(kernel_eeg_chan=1), ConvLayer(kernel_eeg_chan=2), ActivationLayer()]
        model2 = [ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1), ActivationLayer()]
        model_set = []
        genome_set = []
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        assert(len(genome_set)) <= 5
        NaiveNAS.remove_from_models_hash(model1, model_set, genome_set)
        assert(len(genome_set)) >= 3
        NaiveNAS.remove_from_models_hash(model2, model_set, genome_set)
        assert(len(genome_set)) == 0

if __name__ == '__main__':
    unittest.main()