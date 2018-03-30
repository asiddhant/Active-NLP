import torch
import torch.nn as nn

class WordEncoderCNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, char_size, kernel_width = 5, pad_width = 4, 
                 in_channels=1, out1_channels=800, out2_channels=800, cap_size=0, input_dropout_p=0.5, 
                 output_dropout_p=0):
        
        super(WordEncoderCNN, self).__init__()
        
        self.kernel_width = kernel_width
        self.out2_channels = out2_channels
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        new_embedding_size = embedding_size + char_size
        self.cnn1 = nn.Conv2d(in_channels, out1_channels, kernel_size=(kernel_width, new_embedding_size),
                             padding = (pad_width,0))
        self.cnn2 = nn.Conv2d(out1_channels, out2_channels, kernel_size=(kernel_width, 1),
                             padding = (pad_width,0))

    def forward(self, sentence, char_embedding, cap_embedding=None ,input_lengths=None):
        
        embedded = self.embedding(sentence)
        if cap_embedding:
            embedded = torch.cat((embedded,char_embedding,cap_embedding),1)  
        else:
            embedded = torch.cat((embedded,char_embedding),1)
        
        embedded1 = embedded.unsqueeze(0).unsqueeze(0)
        embedded1 = self.input_dropout(embedded1)
        
        output1 = self.cnn1(embedded1)
        output1 = nn.functional.max_pool2d(output1, kernel_size=(self.kernel_width, 1), stride = 1)
        
        output2 = self.cnn2(output1)
        output2 = nn.functional.max_pool2d(output2, kernel_size=(self.kernel_width, 1), stride = 1)
        output2 = output2.squeeze(3).squeeze(0).t()
        
        return output2, embedded