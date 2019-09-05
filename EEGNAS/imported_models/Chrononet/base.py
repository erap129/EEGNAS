class ModelBase:

    cell_types = ['LSTM', 'PhasedLSTM', 'GRU', 'QRNN', 'LayerNormLSTM', 'BNLSTM', 'BNGRU']

    def offset_size(self, sequence_size):
        raise NotImplementedError('You need to implement offset_size function for your model. If it does not use '
                                  'convolutions in the first layers, it should most likely return 0.')
