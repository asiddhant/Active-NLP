import torch.nn as nn

from .baseRNN import baseRNN

class CharEncoderRNN(baseRNN):

    def __init__(self, vocab_size, embedding_size ,hidden_size, input_dropout_p=0, output_dropout_p=0,
                 n_layers=1, bidirectional=True, rnn_cell='lstm'):
        
        super(CharEncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, 
                                             output_dropout_p, n_layers, rnn_cell)

        self.embedding = nn.Embedding(vocab_size,  embedding_size)
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=output_dropout_p)

    def forward(self, input_var, input_lengths=None):
        
        embedded = self.embedding(input_var).transpose(0,1)
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        
        output, _ = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output.transpose(0,1)
        
        return output