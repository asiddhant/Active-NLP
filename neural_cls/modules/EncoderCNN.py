import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, out_channels = 100, dropout_p=0):
        
        super(EncoderCNN, self).__init__()
        
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.cnn1 = nn.Conv1d(embedding_size, out_channels, kernel_size=3,
                             padding = 1)
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel_size=4,
                             padding = 1)
        self.cnn3 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                             padding = 1)

    def forward(self, word_embeddings, input_lengths=None):

        word_embeddings = self.dropout(word_embeddings)
        word_embeddings = word_embeddings.transpose(1,2)
        output1 = F.relu(self.cnn1(word_embeddings))       
        output2 = F.relu(self.cnn2(output1))
        output3 = F.relu(self.cnn3(output2))
        output = nn.functional.max_pool1d(output3, kernel_size=output3.size(2))
        output = output.squeeze(2)
        
        return output