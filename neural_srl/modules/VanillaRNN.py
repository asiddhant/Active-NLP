import torch
import torch.nn as nn

from .baseRNN import baseRNN

class VanillaRNN(baseRNN):

    def __init__(self, vocab_size, embedding_size ,hidden_size, verb_size, cap_size, input_dropout_p=0, 
                 output_dropout_p=0.1, n_layers=4, bidirectional=True, rnn_cell='lstm'):
        
        super(VanillaRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, 
                                             output_dropout_p, n_layers, rnn_cell)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        augmented_embedding_size = embedding_size + verb_size + cap_size
        self.rnn = self.rnn_cell(augmented_embedding_size, hidden_size, n_layers,
                                 bidirectional=bidirectional, dropout=output_dropout_p,
                                 batch_first=True)

    def forward(self, words, verb_embedding, cap_embedding, input_lengths):
        
        embedded = self.embedding(words)
        if cap_embedding is not None:
            embedded = torch.cat((embedded,verb_embedding,cap_embedding),2)  
        else:
            embedded = torch.cat((embedded,verb_embedding),2)
    
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first= True)
        output, _ = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first= True)
        
        return output