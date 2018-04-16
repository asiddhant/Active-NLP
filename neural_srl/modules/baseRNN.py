import torch.nn as nn

class baseRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, input_dropout_p, output_dropout_p, n_layers, rnn_cell, max_len=25):
        super(baseRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len
        
        self.input_dropout_p = input_dropout_p
        self.output_dropout_p = output_dropout_p
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()