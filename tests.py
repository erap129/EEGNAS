import unittest
from models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, Layer, create_topo_layers,\
    random_model
import globals

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


if __name__ == '__main__':
    unittest.main()