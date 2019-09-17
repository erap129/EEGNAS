import unittest
from EEGNAS_experiment import get_configurations, parse_args, set_params_by_dataset
from EEGNAS.global_vars import init_config
from EEGNAS import global_vars
from EEGNAS.utilities import NAS_utils


class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations('tests')
        assert(len(configs) == 1)
        global_vars.set_config(configs[0])
        set_params_by_dataset()

    def test_perm_ensemble_fitness(self):
        global_vars.set('pop_size', 10)
        global_vars.set('ensemble_size', 2)
        global_vars.set('ga_objective', 'accuracy')
        global_vars.set('permanent_ensembles', True)
        dummy_weighted_pop = [{'val_raw': [[1-(1/i), 0, 0, 1/i]], 'val_target': [3]} for i in range(1, 11)]
        old_len = len(dummy_weighted_pop)
        NAS_utils.permanent_ensemble_fitness(dummy_weighted_pop)
        NAS_utils.sort_population(dummy_weighted_pop)
        assert len(dummy_weighted_pop) == old_len
        print(dummy_weighted_pop[-1])
