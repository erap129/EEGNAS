from torch import nn
import torch
import numpy as np

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


class LSTNetModel(nn.Module):
    def __init__(self, num_channels, steps_ahead, window=24*10, hidRNN=100, hidCNN=100, hidSkip=5, CNN_kernel=6, skip=24, highway_window=24, dropout=0.2, output_fun='sigmoid'):
        super(LSTNetModel, self).__init__()
        self.P = window
        self.m = num_channels
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip

        self.steps_ahead = steps_ahead
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if (self.skip > 0):
            self.pt = (self.P - self.Ck) / self.skip
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, steps_ahead)
        else:
            self.linear1 = nn.Linear(self.hidR, steps_ahead)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
            self.highway_linear = nn.Linear(num_channels, steps_ahead)
        self.output = None
        if (output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (output_fun == 'tanh'):
            self.output = torch.tanh

    def forward(self, x):
        x = x.squeeze(dim=3)
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))



        # skip-rnn

        if (self.skip > 0):
            self.pt=int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(x.shape[0], -1)
            z = self.highway_linear(z)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res


class MHANetModel(nn.Module):
    def __init__(self, num_channels, steps_ahead, window=24*10, hidRNN=100, hidCNN=100, d_k=64, d_v=64, CNN_kernel=6, highway_window=24, n_head=8, dropout=0.2, rnn_layers=1, output_fun='sigmoid'):
        super(MHANetModel, self).__init__()
        self.window = window
        self.variables = num_channels
        self.hidC = hidCNN
        self.hidR = hidRNN
        self.hw=highway_window

        self.d_v=d_v
        self.d_k=d_k
        self.Ck = CNN_kernel
        self.GRU = nn.GRU(self.variables, self.hidR, num_layers=rnn_layers)
        # self.Conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.variables))

        self.slf_attn = MultiHeadAttentionModel(n_head, self.variables, self.d_k,self.d_v , dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.linear_out=nn.Linear(self.hidR, steps_ahead)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
            self.highway_linear = nn.Linear(num_channels, steps_ahead)
        self.output = None
        if (output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (output_fun == 'tanh'):
            self.output = torch.tanh


    def forward(self, x):
        x = x.squeeze(dim=3)
        x = x.permute(0, 2, 1)

        attn_output, slf_attn=self.slf_attn(x,x,x,mask=None)

        r=attn_output.permute(1,0,2).contiguous()
        _,r=self.GRU(r)
        r = self.dropout(torch.squeeze(r[-1:, :, :], 0))
        out = self.linear_out(r)

        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            z = self.highway_linear(z)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out


class ScaledDotProductAttention(nn.Module):

    # Scaled Dot-Product Attention

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttentionModel(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.2))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
