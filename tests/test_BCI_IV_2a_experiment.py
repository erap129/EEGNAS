import unittest
from EEGNAS_experiment import get_configurations, get_multiple_values, parse_args
from EEGNAS import global_vars
from EEGNAS.global_vars import init_config


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations(args)
        assert(len(configs) == 1)
        global_vars.set_config(configs[0])

    def test_get_multiple_values(self):
        args = parse_args(['-e', 'test_multiple_values', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations(args)
        assert(len(configs) == 12)
        for str_conf in ['cross_subject', 'num_conv_blocks', 'num_generations']:
            assert(str_conf in get_multiple_values(configs))
        global_vars.set_config(configs[0])

if __name__ == '__main__':
    unittest.main()