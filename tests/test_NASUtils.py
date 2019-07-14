import unittest
from EEGNAS_experiment import get_configurations, parse_args, set_params_by_dataset
import globals
from globals import init_config
import NASUtils


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations('tests')
        assert(len(configs) == 1)
        globals.set_config(configs[0])
        set_params_by_dataset()

    def test_perm_ensemble_fitness(self):
        globals.set('pop_size', 10)
        globals.set('ensemble_size', 2)
        globals.set('ga_objective', 'accuracy')
        globals.set('permanent_ensembles', True)
        dummy_weighted_pop = [{'val_raw': [[1-(1/i), 0, 0, 1/i]], 'val_target': [3]} for i in range(1, 11)]
        old_len = len(dummy_weighted_pop)
        NASUtils.permanent_ensemble_fitness(dummy_weighted_pop)
        NASUtils.sort_population(dummy_weighted_pop)
        assert len(dummy_weighted_pop) == old_len
        print(dummy_weighted_pop[-1])
