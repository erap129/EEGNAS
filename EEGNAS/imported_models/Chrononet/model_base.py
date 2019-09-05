import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from EEGNAS.imported_models.Chrononet.base import ModelBase
from EEGNAS.imported_models.Chrononet.lasso_feature_selection import LassoFeatureSelection
import logging


logger = logging.getLogger(__name__)


class PytorchModelBase(nn.Module, ModelBase):
    Skip_None = 'none'
    Skip_Add = 'add'
    Skip_Concat = 'concat'

    Norm_None = 'none'
    Norm_Batch = 'batch_norm'
    Norm_Layer = 'layer_norm'

    def __init__(self, input_size, output_size, context_size, rnn_normalization, skip_mode, lasso_selection,
                 use_context, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.rnn_normalization = rnn_normalization
        self.skip_mode = skip_mode
        self.use_context = use_context
        self.state_tuple_dim = 1

        self.lasso_module = LassoFeatureSelection(input_size, lasso_selection)

    @staticmethod
    def add_arguments(parser):
        parser.section('model')
        parser.add_argument("rnn_normalization", type=str, default='none', choices=['none', 'batch_norm', 'layer_norm'],
                            help="Whether to use batch norm or not", )
        parser.add_argument("skip_mode", type=str, default='none',
                            choices=['none', 'add', 'concat'],
                            help="Whether to skip connections")
        parser.add_argument("lasso_selection", type=float, default=0.0, help="TODO")
        parser.add_argument("use_context", type=int, choices=[0, 1], default=0,
                            help="If 1 then context information will be used.")
        return parser

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def count_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    # Dummy method for non RNN models that do not require hidden state. Reimplemented in RnnBase.
    def initial_state(self):
        return 0

    # Dummy method for non RNN models that do not require hidden state. Reimplemented in RnnBase.
    def export_state(self, states):
        return [None for x in range(states[0])]

    # Dummy method for non RNN models that do not require hidden state. Reimplemented in RnnBase.
    def import_state(self, states):
        return [len(states)]


class RnnBase(PytorchModelBase):
    Initial_State_Random = 'random'
    Initial_State_Zero = 'zero'
    cell_mapper = {
        'LSTM': nn.LSTM,
        'GRU': nn.GRU,
        'RNN': nn.RNN,
        'IndRNN': IndRNN,
        'IndGRU': IndGRU
    }

    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("rnn_initial_state", type=str, default='random', choices=[RnnBase.Initial_State_Random,
                                                                                      RnnBase.Initial_State_Zero],
                            help="Initial state for RNN.")
        parser.add_argument("rnn_dilation", type=int, default=1,
                            help="Dilation applied to the RNN cells. Assumes that sequence can be split into equal "
                                 "dilation chunks in each layer.")
        parser.add_argument("rnn_hidden_size", type=int, default=128,
                            help="Number of neurons in the RNN layer.")
        parser.add_argument("rnn_num_layers", type=int, default=3,
                            help="Number of layers in the RNN network.")
        parser.add_argument("dropout_f", type=float, default=0.0,
                            help="Dropout value in the forward direction.")
        parser.add_argument("dropout_h", type=float, default=0.0,
                            help="Dropout value from hidden to hidden.")
        parser.add_argument("dropout_i", type=float, default=0.0,
                            help="Dropout value on the input.")
        parser.add_argument("rnn_cell_type", type=str, choices=RnnBase.cell_mapper.keys(),
                            default='GRU',
                            help="RNN cell type.")
        parser.add_argument("skip_first", type=int, choices=[0, 1], default=0,
                            help="If skip connection should be applied in the first RNN layer")
        parser.add_argument("skip_last", type=int, choices=[0, 1], default=0,
                            help="If skip connection should be applied in the last RNN layer")
        parser.add_argument("use_mc_dropout", type=int, choices=[0, 1], default=0,
                            help="If set to 1 then during testing and validation will apply random dropout instead of "
                                 "expected values to the weights")
        return parser

    def __init__(self, rnn_initial_state, rnn_dilation, rnn_hidden_size, rnn_num_layers, dropout_f, dropout_h, dropout_i,
                 rnn_cell_type, skip_first, skip_last, use_mc_dropout, **kwargs):
        super().__init__(**kwargs)
        self.rnn_initial_state = rnn_initial_state
        self.rnn_dilation = rnn_dilation
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.dropout_f = dropout_f
        self.dropout_h = dropout_h
        self.dropout_i = dropout_i
        self.rnn_cell_type = rnn_cell_type
        self.skip_first = bool(skip_first)
        self.skip_last = bool(skip_last)
        self.use_mc_dropout = use_mc_dropout

        if self.rnn_cell_type == "LSTM":
            self.state_tuple_dim = 2
        elif self.rnn_cell_type in ["GRU", "RNN", "IndRNN", "IndGRU"]:
            self.state_tuple_dim = 1
        else:
            raise NotImplementedError("Cell type %s not recognized" % self.rnn_cell_type)

    # Initial hidden state for the model. List of hidden_states for each layer. Layer hidden state represented as
    # a numpy array ([dilation, 1, hidden_size]) for GRU or as a tuple of numpy arrays for LSTM
    def initial_state(self):
        if self.state_tuple_dim == 1:
            return self._initial_state()
        else:
            return tuple(self._initial_state() for _ in range(self.state_tuple_dim))

    # Create a new hidden state for one sample.
    def _initial_state(self):
        layer_state_list = []
        for i_layer in range(self.rnn_num_layers):
            cumulative_dilation = self.rnn_dilation ** i_layer
            shape = (cumulative_dilation, 1, self.rnn_hidden_size)
            if self.rnn_initial_state == self.Initial_State_Random:
                h = np.array(np.random.normal(0, 1.0, shape), dtype=np.float32)
                h = np.clip(h, -1, 1).astype(dtype=np.float32)
            elif self.rnn_initial_state == self.Initial_State_Zero:
                h = np.zeros(shape, np.float32)
            else:
                raise NotImplementedError()

            layer_state_list.append(h)

        return layer_state_list

    # Zips together states from multiple samples
    def import_state(self, sample_state_list):
        if self.state_tuple_dim == 1:
            layer_batch_state_list = self._import_state(sample_state_list)
            return layer_batch_state_list
        else:
            layer_batch_state_list = []
            for state_index in range(self.state_tuple_dim):
                # Extract samples for specific state
                sample_state_i_list = [s[state_index] for s in sample_state_list]
                layer_batch_state_list.append(self._import_state(sample_state_i_list))

            # Convert it such that Layers are first and then tuples with states describe each layer state
            return [tuple(s) for s in zip(*layer_batch_state_list)]

    # States come from different samples, merge them into one single minibatch,
    # Each element from sample_state_list is a list of layer states of the shape [dilation, 1, hidden_size]
    # We need to transform it into a variable with format [1, batch_size*dilation, hidden_size]
    def _import_state(self, sample_state_list):
        layer_batch_state_list = []
        for i_layer in range(self.rnn_num_layers):
            layer_sample_state_list = [s[i_layer] for s in sample_state_list]
            # Concatenate samples for this layer,
            # the shape after this operation should be [batch_size*dilation, 1, hidden_size]
            layer_batch_state = np.concatenate(layer_sample_state_list)
            # Pytorch expects the shape to be [Num_layers=1, batch_size*dilation, hidden_size],
            # so we swap axes
            layer_batch_state = np.swapaxes(layer_batch_state, 1, 0)
            layer_batch_state = Variable(torch.from_numpy(layer_batch_state), requires_grad=False)
            layer_batch_state_list.append(layer_batch_state)
        return layer_batch_state_list

    # Converts PyTorch hidden state representation into numpy arrays that can be used by the data reader class
    def export_state(self, layer_batch_state_list):
        if self.state_tuple_dim == 1:
            sample_state_list = self._export_state(layer_batch_state_list)
            return sample_state_list
        else:
            sample_state_list = []
            for state_index in range(self.state_tuple_dim):
                layer_batch_state_i_list = [s[state_index] for s in layer_batch_state_list]
                sample_state_list.append(self._export_state(layer_batch_state_i_list))

            # Convert it such that samples are first then states and then layers
            return [tuple(s_l) for s_l in zip(*sample_state_list)]

    # As an input we have a list of states for each layer in the RNN
    # Each layer state will have the shape [1, batch_size*dilation, hidden_size]
    # We need to extract hidden state for each sample and output a list where each element
    # represents hidden state for one data sample, hidden state should be a list where each element
    # represents one layer hidden_state
    def _export_state(self, layer_batch_state_list):
        # Input is a list where each element is hidden state for a layer, shape [1, batch_size*dilation, hidden_size]
        # We need to make batch_size*dilation as a first dimension
        layer_batch_state_list = [torch.transpose(s, 1, 0) for s in layer_batch_state_list]

        # Because the first layer always has dilation of 1 we can find out batch_size
        batch_size = layer_batch_state_list[0].size(0)

        # Placeholder for samples
        # Here we will store for each sample list with hidden states for each RNN layer

        sample_state_list = [[] for _ in range(batch_size)]
        for i_layer in range(self.rnn_num_layers):
            dilation = self.rnn_dilation**i_layer
            layer_batch_state = layer_batch_state_list[i_layer]
            # layer_batch_state has dimension [batch_size*dilation, 1, hidden_size]
            # We split it into a list of layer_states for each sample,
            # those will have dimension [dilation, 1, hidden_size]
            layer_sample_state_list = torch.split(layer_batch_state, dilation)

            # We save hidden states from each sample into the correct place
            for s, st in zip(sample_state_list, layer_sample_state_list):
                s.append(st.cpu().data.numpy())

        # At the end samples should be a list where each element represents hidden state for a given sample
        # This hidden state should be a list where each element represents hidden state for a given layer
        # Hidden state for one layer should have dimensions [dilation, 1, hidden_size]
        return sample_state_list

    def offset_size(self, sequence_size):
        return 0


# Some simple tests
if __name__ == '__main__':
    for cell_type in ['LSTM']:
        for dilation in [2]:
            rnn = RnnBase(rnn_initial_state='random',
                          rnn_dilation=dilation,
                          rnn_hidden_size=2,
                          rnn_num_layers=2,
                          dropout_f=0.0,
                          dropout_h=0.0,
                          dropout_i=0.0,
                          rnn_cell_type=cell_type,
                          skip_first=0,
                          skip_last=0,
                          use_mc_dropout=0,
                          input_size=5,
                          output_size=2,
                          context_size=0,
                          rnn_normalization='none',
                          skip_mode='none',
                          lasso_selection=0.0,
                          use_context=0)

            initial_states = [rnn.initial_state() for s in range(2)]
            print('Initial states %d' % len(initial_states))
            for i_s in initial_states:
                if isinstance(i_s, tuple):
                    for i_s_i in i_s:
                        for i_l in range(rnn.rnn_num_layers):
                            print(i_s_i[i_l])
                else:
                    for i_l in range(rnn.rnn_num_layers):
                        print(i_s[i_l])

            imported_states = rnn.import_state(initial_states)

            for layer_i in range(rnn.rnn_num_layers):
                print('States imported to pytorch ( layer_i %d)' % layer_i)
                print(imported_states[layer_i])

            exported_states = rnn.export_state(imported_states)

            print('States exported from pytroch')
            for e_s in exported_states:
                if isinstance(e_s, tuple):
                    for e_s_i in e_s:
                        for i_l in range(rnn.rnn_num_layers):
                            print(e_s_i[i_l])
                else:
                    for i_l in range(rnn.rnn_num_layers):
                        print(e_s[i_l])

            for i_s, e_s in zip(initial_states, exported_states):
                print('Compare sample')
                if isinstance(i_s, tuple):
                    for i_s_i, e_s_i in zip(i_s, e_s):

                        for i_l in range(rnn.rnn_num_layers):

                            print(i_s_i[i_l])
                            print(e_s_i[i_l])
                            assert np.array_equal(i_s_i[i_l], e_s_i[i_l])
                else:
                    for i_l in range(rnn.rnn_num_layers):
                        print(i_s[i_l])
                        print(e_s[i_l])
                        assert np.array_equal(i_s[i_l], e_s[i_l])
