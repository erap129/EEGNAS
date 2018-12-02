import unittest
from models_generation import random_model, breed_layers
import globals

class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        globals.init_config()

    def test_breed(self):
        model1 = random_model(10)
        model2 = random_model(10)
        model3 = breed_layers(model1, model2, 0)
        pass

if __name__ == '__main__':
    unittest.main()