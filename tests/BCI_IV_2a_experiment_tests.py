import unittest
from BCI_IV_2a_experiment import get_configurations, get_multiple_values
import globals
from globals import init_config


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        init_config('test_configs/single_subject_config.ini')
        configs = get_configurations()
        assert(len(configs) == 1)
        globals.set_config(configs[0])

    def test_get_multiple_values(self):
        init_config('test_configs/multiple_values_test.ini')
        configs = get_configurations()
        assert(len(configs) == 12)
        for str_conf in ['cross_subject', 'num_conv_blocks', 'num_generations']:
            assert(str_conf in get_multiple_values(configs))
        globals.set_config(configs[0])

if __name__ == '__main__':
    unittest.main()