import unittest
from collections import OrderedDict

from models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, Layer, ConvLayer,\
    MyModel
import json
import globals
from itertools import product

class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        globals.init_config()
        Layer.running_id = 0

    def test_breed(self):
        model1 = uniform_model(10, BatchNormLayer)
        model2 = uniform_model(10, DropoutLayer)
        model3 = breed_layers(model1, model2, mutation_rate=0, cut_point=4)
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
        print(new_dict)

if __name__ == '__main__':
    unittest.main()