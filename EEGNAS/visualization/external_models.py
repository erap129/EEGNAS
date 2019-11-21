from torch import nn
import torch

# Here we define our model as a class
class MultivariateLSTM(nn.Module):

    def __init__(self, n_features, hidden_dim, batch_size, n_steps, output_dim=1,
                 num_layers=2, eegnas=False):
        super(MultivariateLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim  # number of hidden states
        self.batch_size = batch_size
        self.num_layers = num_layers  # number of LSTM layers (stacked)
        self.n_steps = n_steps
        self.eegnas = eegnas

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.n_features, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * self.n_steps, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())

    def forward(self, input):
        if self.eegnas:
            input = input.squeeze(dim=3)
            input = input.permute(0, 2, 1)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out.contiguous().view(input.shape[0], -1))
        return y_pred

    def predict(self, X_test):
        y_hat = self.forward(X_test)
        return y_hat.tolist()