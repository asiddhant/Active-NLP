import torch
import torch.nn as nn

class WordEncoderCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, char_size, kernel_width = 5, pad_width = 2, 
                 out_channels=200 , cap_size=0, input_dropout_p=0.5, output_dropout_p=0.5):
        
        super(WordEncoderCNN, self).__init__()
        
        self.kernel_width = kernel_width
        self.out_channels = out_channels
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.output_dropout = nn.Dropout(p=output_dropout_p)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        in_channels = embedding_size + char_size + cap_size
        
        self.cnn1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_width,
                             padding = pad_width)
        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_width,
                             padding = pad_width)
        self.cnn3 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_width,
                             padding = pad_width)
        self.cnn4 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_width,
                             padding = pad_width)

    def forward(self, words, char_embedding, cap_embedding=None ,input_lengths=None):
        
        embedded = self.embedding(words)
        
        if cap_embedding:
            embedded = torch.cat((embedded,char_embedding,cap_embedding),2)  
        else:
            embedded = torch.cat((embedded,char_embedding),2)
        
        embedded1 = self.input_dropout(embedded)
        embedded1 = embedded1.transpose(1,2)
        
        output1 = self.cnn1(embedded1)
        output2 = self.cnn2(output1)
        output3 = self.cnn3(output2)
        output4 = self.cnn4(output3)
        output4 = output4.transpose(1,2)
        
        return output4, embedded