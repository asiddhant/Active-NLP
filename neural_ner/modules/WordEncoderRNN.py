import torch
import torch.nn as nn

from .baseRNN import baseRNN

class WordEncoderRNN(baseRNN):

    def __init__(self, vocab_size, embedding_size ,hidden_size, char_size, cap_size=0, input_dropout_p=0.5, 
                 output_dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='lstm'):
        
        super(WordEncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, 
                                             output_dropout_p, n_layers, rnn_cell)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        augmented_embedding_size = embedding_size + char_size + cap_size
        self.rnn = self.rnn_cell(augmented_embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=output_dropout_p)

    def forward(self, sentence, char_embedding, cap_embedding=None ,input_lengths=None):
        
        embedded = self.embedding(sentence)
        if cap_embedding:
            embedded = torch.cat((embedded,char_embedding,cap_embedding),1)  
        else:
            embedded = torch.cat((embedded,char_embedding),1)
        embedded = embedded.unsqueeze(1)
        embedded = self.input_dropout(embedded)
        
        output, _ = self.rnn(embedded)
        output = output.view(len(sentence), self.hidden_size*2)
        
        return output