import torch
import torch.nn as nn

from .baseRNN import baseRNN

class EncoderRNN(baseRNN):

    def __init__(self, vocab_size, embedding_size ,hidden_size= 200, input_dropout_p=0, 
                 output_dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='lstm'):
        
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, 
                                             output_dropout_p, n_layers, rnn_cell)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 bidirectional=bidirectional, dropout=output_dropout_p,
                                 batch_first=True)

    def forward(self, words, input_lengths):
        
        batch_size = words.size()[0]
        embedded = self.embedding(words)
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first= True)
        _, output = self.rnn(embedded)
        output = output[0].transpose(0,1).contiguous().view(batch_size, -1)
        
        return output