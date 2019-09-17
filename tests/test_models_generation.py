import pickle
import unittest

from braindecode.torch_ext.util import np_to_var
from EEGNAS.models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, ConvLayer,\
    ActivationLayer, network_similarity, PoolingLayer, add_random_connection, IdentityLayer,\
    breed_grid
from EEGNAS_experiment import get_configurations, parse_args, set_params_by_dataset
from EEGNAS import global_vars, models_generation
from EEGNAS.utilities.NAS_utils import equal_grid_models
import networkx as nx
import numpy as np
from EEGNAS.global_vars import init_config
import matplotlib.pyplot as plt
from graphviz import Source
import torch
from networkx.classes.function import create_empty_copy

class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations('tests')
        assert(len(configs) == 1)
        global_vars.set_config(configs[0])
        set_params_by_dataset('../configurations/dataset_params.ini')

    def test_breed(self):
        model1 = uniform_model(10, BatchNormLayer)
        model2 = uniform_model(10, DropoutLayer)
        model3, _, _ = breed_layers(0, model1, model2, cut_point=4)
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
            models_generation.new_model_from_structure_pytorch(model1)
            assert False
        except Exception:
            assert True
        try:
            models_generation.new_model_from_structure_pytorch(model1, applyFix=True)
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
        global_vars.set('inherit_breeding_weights', True)
        global_vars.set('num_layers', 4)
        global_vars.set('mutation_rate', 0)
        model1 = uniform_model(4, ConvLayer)
        model1_state = finalize_model(model1).state_dict()
        model2 = uniform_model(4, ConvLayer)
        model2_state = finalize_model(model2).state_dict()
        model3, model3_state, _ = breed_layers(0, model1, model2, model1_state, model2_state, 2)
        for s1, s3 in zip(list(model1_state.values())[:4], list(model3_state.values())[:4]):
            assert((s1==s3).all())
        for s2, s3 in zip(list(model2_state.values())[6:8], list(model3_state.values())[6:8]):
            assert((s2==s3).all())

    def test_network_similarity(self):
        conv1 = ConvLayer(kernel_eeg_chan=3, kernel_time=10, filter_num=50)
        conv2 = ConvLayer(kernel_eeg_chan=5, kernel_time=20, filter_num=25)
        model1 = [conv1, DropoutLayer(), conv1]
        model2 = [conv2, DropoutLayer(), conv2]
        assert(network_similarity(model1, model1) > network_similarity(model1, model2))

        pool1 = PoolingLayer(stride_eeg_chan=2, stride_time=3)
        pool2 = PoolingLayer(stride_eeg_chan=10, stride_time=1)
        model1 = [conv1, pool2, conv1, pool2, conv1, pool2]
        model2 = [conv1, conv1, pool2, conv1, pool2, conv1, pool2]
        model3 = [conv2, pool1, conv2, pool1, conv2, pool1]
        assert(network_similarity(model1, model2) > network_similarity(model2, model3))

    def test_random_grid_model(self):
        model = models_generation.random_grid_model([10, 10])
        for i in range(100):
            add_random_connection(model)
        model.add_edge('input', (0, 5))
        real_model = models_generation.ModelFromGrid(model)
        input_shape = (2, global_vars.get('eeg_chans'), global_vars.get('input_time_len'), 1)
        out = real_model.forward(np_to_var(np.ones(input_shape, dtype=np.float32)))
        print(list(nx.topological_sort(model)))
        nx.draw(model, with_labels=True)
        plt.show()

    def test_remove_random_connection(self):
        model = models_generation.random_grid_model([10, 10])
        num_connections = len(list(model.edges))
        success = models_generation.remove_random_connection(model)
        assert(len(list(model.edges)) == num_connections - 1 or not success)

    def test_draw_grid_model(self):
        layer_grid = create_empty_copy(nx.to_directed(nx.grid_2d_graph(5, 5)))
        for node in layer_grid.nodes.values():
            node['layer'] = models_generation.random_layer()
        layer_grid.add_node('input')
        layer_grid.add_node('output_conv')
        layer_grid.nodes['output_conv']['layer'] = models_generation.IdentityLayer()
        layer_grid.nodes[(0,0)]['layer'] = ConvLayer(filter_num=50)
        layer_grid.nodes[(0,1)]['layer'] = ConvLayer(filter_num=50)
        layer_grid.nodes[(0,2)]['layer'] = DropoutLayer()
        layer_grid.add_edge('input', (0, 0))
        layer_grid.add_edge((0, 5 - 1), 'output_conv')
        for i in range(5 - 1):
            layer_grid.add_edge((0, i), (0, i + 1))
        layer_grid.graph['height'] = 5
        layer_grid.graph['width'] = 5
        if models_generation.check_legal_grid_model(layer_grid):
            print('legal model')
        else:
            print('illegal model')

        # model = models_generation.random_grid_model(10)
        layer_grid.add_edge((0,0), (0, 2))
        real_model = models_generation.ModelFromGrid(layer_grid)

        # model = models_generation.random_model(10)
        # real_model = finalize_model(model)

        # for i in range(100):
        #     add_random_connection(model)
        input_shape = (60, global_vars.get('eeg_chans'), global_vars.get('input_time_len'), 1)
        out = real_model(np_to_var(np.ones(input_shape, dtype=np.float32)))
        s = Source(make_dot(out), filename="test.gv", format="png")
        s.view()

    def test_breed_grid(self):
        global_vars.set('grid', True)
        layer_grid_1 = models_generation.random_grid_model([5, 5])
        layer_grid_2 = models_generation.random_grid_model([5, 5])
        for node in layer_grid_1.nodes.values():
            node['layer'] = models_generation.random_layer()
        for node in layer_grid_2.nodes.values():
            node['layer'] = models_generation.random_layer()
        layer_grid_1.nodes[(0,0)]['layer'] = ConvLayer()
        layer_grid_1.nodes[(0,1)]['layer'] = BatchNormLayer()
        layer_grid_1.nodes[(0,2)]['layer'] = PoolingLayer()
        layer_grid_1.nodes[(0,3)]['layer'] = BatchNormLayer()
        layer_grid_1.nodes[(0,4)]['layer'] = DropoutLayer()
        layer_grid_1.nodes[(1,1)]['layer'] = BatchNormLayer()
        layer_grid_1.nodes[(2,2)]['layer'] = ConvLayer()

        layer_grid_2.nodes[(0, 0)]['layer'] = BatchNormLayer()
        layer_grid_2.nodes[(0, 1)]['layer'] = PoolingLayer()
        layer_grid_2.nodes[(0, 2)]['layer'] = ConvLayer()
        layer_grid_2.nodes[(0, 3)]['layer'] = BatchNormLayer()
        layer_grid_2.nodes[(0, 4)]['layer'] = IdentityLayer()
        layer_grid_2.nodes[(1, 1)]['layer'] = ConvLayer()
        layer_grid_2.nodes[(2, 2)]['layer'] = IdentityLayer()
        layer_grid_2.nodes[(3, 3)]['layer'] = BatchNormLayer()
        layer_grid_2.nodes[(2, 4)]['layer'] = IdentityLayer()
        layer_grid_2.nodes[(2, 0)]['layer'] = PoolingLayer()

        layer_grid_1.add_edge((0, 1), (2, 1))
        layer_grid_1.add_edge((0, 1), (2, 2))
        layer_grid_1.add_edge((0, 2), (2, 0))
        layer_grid_1.add_edge((0, 2), (2, 3))
        layer_grid_1.add_edge((2, 2), (0, 4))
        layer_grid_1.add_edge((2, 3), (0, 3))
        layer_grid_1.add_edge((3, 1), (2, 3))

        layer_grid_2.add_edge((0, 0), (1, 1))
        layer_grid_2.add_edge((0, 1), (0, 2))
        layer_grid_2.add_edge((0, 1), (3, 3))
        layer_grid_2.add_edge((0, 3), (1, 1))
        layer_grid_2.add_edge((0, 3), (2, 4))
        layer_grid_2.add_edge((1, 1), (0, 4))
        layer_grid_2.add_edge((2, 2), (3, 3))
        layer_grid_2.add_edge((3, 3), (0, 3))

        model_1 = models_generation.ModelFromGrid(layer_grid_1)
        model_2 = models_generation.ModelFromGrid(layer_grid_2)

        child_model, child_model_state, _ = breed_grid(0, layer_grid_1, layer_grid_2, model_1.state_dict(), model_2.state_dict(),
                                 cut_point=2)

        assert (type(child_model.nodes[(0, 0)]['layer']) == ConvLayer)
        assert (type(child_model.nodes[(0, 1)]['layer']) == BatchNormLayer)
        assert (type(child_model.nodes[(0, 2)]['layer']) == ConvLayer)
        assert (type(child_model.nodes[(0, 3)]['layer']) == BatchNormLayer)
        assert (type(child_model.nodes[(0, 4)]['layer']) == IdentityLayer)
        assert (type(child_model.nodes[(1, 1)]['layer']) == BatchNormLayer)
        assert (type(child_model.nodes[(2, 2)]['layer']) == IdentityLayer)
        assert (type(child_model.nodes[(3, 3)]['layer']) == BatchNormLayer)
        assert (type(child_model.nodes[(2, 4)]['layer']) == IdentityLayer)

        finalized_child_model = finalize_model(child_model)
        finalized_child_model.load_state_dict(child_model_state)

        model_1_state = model_1.state_dict()
        child_state = finalized_child_model.state_dict()
        for key in child_state.keys():
            if '(0, 0)' in key:
                assert child_state[key].equal(model_1_state[key])
            if '(0, 1)' in key:
                assert child_state[key].equal(model_1_state[key])

    def test_grid_equality(self):
        global_vars.set('grid', True)
        layer_grid_1 = models_generation.random_grid_model([2, 2])
        layer_grid_2 = models_generation.random_grid_model([2, 2])
        layer_grid_3 = models_generation.random_grid_model([2, 2])

        layer_grid_1.nodes[(0,0)]['layer'] = ConvLayer(filter_num=10, kernel_time=10)
        layer_grid_1.nodes[(0,1)]['layer'] = BatchNormLayer()
        layer_grid_1.nodes[(1,0)]['layer'] = PoolingLayer(stride_time=3, pool_time=3)
        layer_grid_1.nodes[(1,1)]['layer'] = BatchNormLayer()

        layer_grid_2.nodes[(0,0)]['layer'] = ConvLayer(filter_num=10, kernel_time=10)
        layer_grid_2.nodes[(0,1)]['layer'] = BatchNormLayer()
        layer_grid_2.nodes[(1,0)]['layer'] = PoolingLayer(stride_time=3, pool_time=3)
        layer_grid_2.nodes[(1,1)]['layer'] = BatchNormLayer()

        layer_grid_3.nodes[(0, 0)]['layer'] = ConvLayer(filter_num=10, kernel_time=10)
        layer_grid_3.nodes[(0, 1)]['layer'] = BatchNormLayer()
        layer_grid_3.nodes[(1, 0)]['layer'] = PoolingLayer(stride_time=2, pool_time=3)
        layer_grid_3.nodes[(1, 1)]['layer'] = BatchNormLayer()

        assert(equal_grid_models(layer_grid_1, layer_grid_2))
        assert(not equal_grid_models(layer_grid_1, layer_grid_3))

        layer_grid_2.add_edge((0,0),(1,1))
        layer_grid_1.add_edge((0,0),(1,1))
        assert (equal_grid_models(layer_grid_1, layer_grid_2))

        layer_grid_2.add_edge((0,0),(1,0))
        assert (not equal_grid_models(layer_grid_1, layer_grid_2))

    def test_remove_all_edges_from_input(self):
        global_vars.set('grid', True)
        global_vars.set('parallel_paths_experiment', True)
        layer_grid_1 = models_generation.random_grid_model([5, 10])
        layer_grid_1.remove_edge('input', (0, 0))
        layer_grid_1.remove_edge('input', (1, 0))
        layer_grid_1.remove_edge('input', (2, 0))
        layer_grid_1.remove_edge('input', (3, 0))
        layer_grid_1.remove_edge('input', (4, 0))
        model = models_generation.ModelFromGrid(layer_grid_1)

    def test_pytorch_average(self):
        weighted_population = pickle.load(open('../weighted_populations/421_1_SO_pure_cross_subject_BCI_IV_2a.p', 'rb'))
        model_1_to_conv = torch.nn.Sequential(*list(weighted_population[0]['finalized_model'].children())[:11]).cpu()
        model_2_to_conv = torch.nn.Sequential(*list(weighted_population[1]['finalized_model'].children())[:11]).cpu()
        training_example = models_generation.get_dummy_input()
        model_1_output = model_1_to_conv(training_example)
        model_2_output = model_2_to_conv(training_example)
        regular_avg = (model_1_output + model_2_output) / 2

        linear_avg_layer = models_generation.LinearWeightedAvg(global_vars.get('n_classes'))
        linear_avg_layer.weight_inp1.data = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).view((1,4,1,1))
        linear_avg_layer.weight_inp2.data = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).view((1,4,1,1))
        pytorch_avg = linear_avg_layer.forward(model_1_output, model_2_output)

        assert torch.all(torch.eq(regular_avg, pytorch_avg))


if __name__ == '__main__':
    unittest.main()