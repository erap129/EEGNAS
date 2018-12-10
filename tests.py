import unittest
from collections import OrderedDict
from naiveNAS import NaiveNAS
from models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, Layer, ConvLayer,\
    MyModel, ActivationLayer
import json
import pickle
import globals
from itertools import product

class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        globals.init_config()
        dummy_config = {'DEFAULT':{}, 'evolution':{}}
        globals.set_config(dummy_config)
        dummy_config['DEFAULT']['exp_type'] = "evolution_layers"
        dummy_config['DEFAULT']['data_folder'] = "data"
        dummy_config['DEFAULT']['valid_set_fraction'] = 0.2
        dummy_config['DEFAULT']['batch_norm_alpha'] = 0.1
        dummy_config['DEFAULT']['dropout_p'] = 0.5
        dummy_config['DEFAULT']['max_epochs'] = 50
        dummy_config['DEFAULT']['max_increase_epochs'] = 3
        dummy_config['DEFAULT']['batch_size'] = 32
        dummy_config['DEFAULT']['pin_memory'] = False
        dummy_config['DEFAULT']['do_early_stop'] = True
        dummy_config['DEFAULT']['remember_best'] = True
        dummy_config['DEFAULT']['remember_best_column'] = "valid_misclass"
        dummy_config['DEFAULT']['cuda'] = False
        dummy_config['DEFAULT']['cropping'] = False
        dummy_config['DEFAULT']['eeg_chans'] = 22
        dummy_config['DEFAULT']['spatial_to_channels'] = False
        dummy_config['DEFAULT']['input_time_len'] = 1125
        dummy_config['DEFAULT']['n_classes'] = 4
        dummy_config['DEFAULT']['split_final_conv'] = False
        dummy_config['DEFAULT']['channel_dim'] = "one"
        dummy_config['DEFAULT']['network_size'] = 10
        dummy_config['DEFAULT']['cross_subject'] = False
        dummy_config['DEFAULT']['num_subjects'] = 9
        dummy_config['evolution']['pop_size'] = 100
        dummy_config['evolution']['num_generations'] = 100
        dummy_config['evolution']['num_subjects'] = 3
        dummy_config['evolution']['mutation_rate'] = 0.1
        dummy_config['evolution']['breed_rate'] = 1
        dummy_config['evolution']['num_conv_blocks'] = 3
        dummy_config['evolution']['num_layers'] = 10
        dummy_config['evolution']['random_filter_range_min'] = 10
        dummy_config['evolution']['random_filter_range_max'] = 100

    def test_breed(self):
        model1 = uniform_model(10, BatchNormLayer)
        model2 = uniform_model(10, DropoutLayer)
        model3 = breed_layers(model1, model2, cut_point=4)
        for i in range(10):
            if i < 4:
                assert(type(model3[i]).__name__ == type(model1[i]).__name__)
            else:
                assert (type(model3[i]).__name__ == type(model2[i]).__name__)
        finalize_model(model3)
        pass

    def test_fix_model(self):
        model1 = uniform_model(10, ConvLayer)
        try:
            MyModel.new_model_from_structure_pytorch(model1)
            assert False
        except Exception:
            assert True
        try:
            MyModel.new_model_from_structure_pytorch(model1, applyFix=True)
            assert True
        except Exception:
            assert False

    def test_cartesian_product(self):
        default_config = globals.config._defaults
        for key in default_config.keys():
            default_config[key] = json.loads(default_config[key])
        b = list(product(*list(default_config.values())))
        new_dict = OrderedDict([])
        for i, key in enumerate(default_config.keys()):
            new_dict[key] = b[0][i]

    def test_hash_model_nochange(self):
        model1 = uniform_model(10, ActivationLayer)
        model2 = uniform_model(10, ActivationLayer)
        model_set = set()
        genome_set = set()
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        assert(len(model_set)) == 1
        assert(len(genome_set)) == 1

    def test_hash_model(self):
        model1 = uniform_model(10, ActivationLayer)
        model2 = uniform_model(10, ActivationLayer)
        globals.config['evolution']['mutation_rate'] = 1
        model3 = breed_layers(model1, model2)
        model_set = set()
        genome_set = set()
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        NaiveNAS.hash_model(model3, model_set, genome_set)
        assert(len(model_set)) == 2
        assert(len(genome_set)) == 1 or 2

    def test_hash_model_same_type(self):
        model1 = [ConvLayer(kernel_eeg_chan=1), ConvLayer(kernel_eeg_chan=2)]
        model2 = [ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1)]
        model_set = set()
        genome_set = set()
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        assert (len(model_set)) == 2
        assert (len(genome_set)) >= 2

    def test_remove_from_hash(self):
        model1 = [ConvLayer(kernel_eeg_chan=1), ConvLayer(kernel_eeg_chan=2), ActivationLayer()]
        model2 = [ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1), ActivationLayer()]
        model_set = set()
        genome_set = set()
        NaiveNAS.hash_model(model1, model_set, genome_set)
        NaiveNAS.hash_model(model2, model_set, genome_set)
        assert(len(genome_set)) <= 5
        NaiveNAS.remove_from_models_hash(model1, model_set, genome_set)
        assert(len(genome_set)) >= 3
        NaiveNAS.remove_from_models_hash(model2, model_set, genome_set)
        assert(len(genome_set)) == 0

    def test_check_pickle(self):
        test1 = pickle.dumps(ConvLayer(kernel_eeg_chan=2))
        test2 = pickle.dumps(ConvLayer(kernel_eeg_chan=1))
        test3 = pickle.dumps(ActivationLayer())
        test4 = pickle.dumps(ActivationLayer())
        assert(test3 == test4)
        assert(test1 != test2)

        model1 = pickle.dumps(uniform_model(10, ActivationLayer))
        model2 = pickle.dumps(uniform_model(10, ActivationLayer))
        model_set = set()
        model_set.add(model1)
        model_set.add(model2)
        assert(len(model_set) == 1)
        model3 = pickle.dumps([ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1), ActivationLayer()])
        model_set.add(model3)
        assert(len(model_set) == 2)

        model1 = pickle.loads(model1)
        for layer in model1:
            assert(pickle.dumps(layer) == pickle.dumps(ActivationLayer()))








if __name__ == '__main__':
    unittest.main()