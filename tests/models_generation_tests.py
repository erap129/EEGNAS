import unittest
from naiveNAS import NaiveNAS
from models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, ConvLayer,\
    MyModel, ActivationLayer, random_model
from BCI_IV_2a_experiment import get_configurations, get_multiple_values
import json
import random
import globals
from globals import init_config
from itertools import product

class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        init_config('test_configs/single_subject_config.ini')
        configs = get_configurations()
        assert(len(configs) == 1)
        globals.set_config(configs[0])

    def test_breed(self):
        model1 = uniform_model(10, BatchNormLayer)
        model2 = uniform_model(10, DropoutLayer)
        model3, _ = breed_layers(0, model1, model2, cut_point=4)
        for i in range(10):
            if i < 4:
                assert(type(model3[i]).__name__ == type(model1[i]).__name__)
            else:
                assert (type(model3[i]).__name__ == type(model2[i]).__name__)
        finalize_model(model3)
        pass

    def test_fix_model(self):
        model1 = uniform_model(3, ConvLayer)
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

    def test_layer_equality(self):
        assert(ActivationLayer() == ActivationLayer())
        check_list = [ActivationLayer()]
        assert(ActivationLayer() in check_list)
        check_list.remove(ActivationLayer())
        assert(len(check_list) == 0)

    def test_state_inheritance_breeding(self):
        globals.config['evolution']['inherit_breeding_weights'] = True
        globals.config['evolution']['num_layers'] = 4
        globals.config['evolution']['mutation_rate'] = 0
        model1 = uniform_model(4, ConvLayer)
        model1_state = finalize_model(model1).model.state_dict()
        model2 = uniform_model(4, ConvLayer)
        model2_state = finalize_model(model2).model.state_dict()
        model3, model3_state = breed_layers(0, model1, model2, model1_state, model2_state, 2)
        for s1, s3 in zip(list(model1_state.values())[:4], list(model3_state.values())[:4]):
            assert((s1==s3).all())
        for s2, s3 in zip(list(model2_state.values())[6:8], list(model3_state.values())[6:8]):
            assert((s2==s3).all())


if __name__ == '__main__':
    unittest.main()