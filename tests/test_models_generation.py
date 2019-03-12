import unittest

from braindecode.torch_ext.util import np_to_var

from models_generation import uniform_model, breed_layers,\
    finalize_model, DropoutLayer, BatchNormLayer, ConvLayer,\
    MyModel, ActivationLayer, network_similarity, PoolingLayer, add_random_connection, IdentityLayer,\
    breed_grid
import models_generation
from BCI_IV_2a_experiment import get_configurations, parse_args, set_params_by_dataset
import globals
import networkx as nx
import numpy as np
from globals import init_config
import matplotlib.pyplot as plt
from torchviz import make_dot
import torchvision.models as models
from copy import deepcopy
from graphviz import Source
import torch
from networkx.classes.function import create_empty_copy
from torchsummary import summary

class TestModelGeneration(unittest.TestCase):
    def setUp(self):
        args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
        init_config(args.config)
        configs = get_configurations(args)
        assert(len(configs) == 1)
        globals.set_config(configs[0])
        set_params_by_dataset()

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
        globals.set('inherit_breeding_weights', True)
        globals.set('num_layers', 4)
        globals.set('mutation_rate', 0)
        model1 = uniform_model(4, ConvLayer)
        model1_state = finalize_model(model1).model.state_dict()
        model2 = uniform_model(4, ConvLayer)
        model2_state = finalize_model(model2).model.state_dict()
        model3, model3_state = breed_layers(0, model1, model2, model1_state, model2_state, 2)
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
        model = models_generation.random_grid_model(10)
        for i in range(100):
            add_random_connection(model)
        model.add_edge('input', (0, 5))
        real_model = models_generation.ModelFromGrid(model)
        input_shape = (2, globals.get('eeg_chans'), globals.get('input_time_len'), 1)
        out = real_model.forward(np_to_var(np.ones(input_shape, dtype=np.float32)))
        print(list(nx.topological_sort(model)))
        nx.draw(model, with_labels=True)
        plt.show()

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


        summary(real_model, (globals.get('eeg_chans'), globals.get('input_time_len'), 1))
        input_shape = (60, globals.get('eeg_chans'), globals.get('input_time_len'), 1)
        out = real_model(np_to_var(np.ones(input_shape, dtype=np.float32)))
        s = Source(make_dot(out), filename="test.gv", format="png")
        s.view()

    def test_breed_grid(self):
        globals.set('grid', True)
        layer_grid_1 = models_generation.random_grid_model(5)
        layer_grid_2 = models_generation.random_grid_model(5)
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

        child_model, child_model_state = breed_grid(0, layer_grid_1, layer_grid_2, model_1.state_dict(), model_2.state_dict(),
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

        finalized_model = finalize_model(child_model)
        finalized_model.load_state_dict(child_model_state)

        child_state = child_model.state_dict()
        for key in child_state.keys():
            if '(0, 0)' in key:
                # assert child_state[key] ==
                pass
        assert (child_model)

        pass


if __name__ == '__main__':
    unittest.main()