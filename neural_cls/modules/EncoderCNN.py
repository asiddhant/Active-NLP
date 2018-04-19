import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, out_channels = 100, in_channels=1, dropout_p=0):
        
        super(EncoderCNN, self).__init__()
        
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, embedding_size),
                             padding = (1,0))
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1),
                             padding = (1,0))
        self.cnn3 = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1),
                             padding = (1,0))

    def forward(self, words, input_lengths=None):
        
        batch_size, _ = words.size()
        
        embedded = self.embedding(words).unsqueeze(1)
        embedded = self.dropout(embedded)
                        
        output1 = F.relu(self.cnn1(embedded))       
        output2 = F.relu(self.cnn2(output1))
        output3 = F.relu(self.cnn3(output2))
        output = nn.functional.max_pool2d(output3, kernel_size=(output3.size(2), 1))
        output = output.squeeze(3).squeeze(2)
        
        return output