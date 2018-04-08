import torch.nn as nn


class CharEncoderCNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_size ,out_channels, kernel_width, pad_width, 
                 input_dropout_p=0, output_dropout_p=0, in_channels=1):
        
        super(CharEncoderCNN, self).__init__()
        
        self.out_channels = out_channels
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.output_dropout = nn.Dropout(output_dropout_p)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size = (kernel_width, embedding_size),
                             padding = (pad_width,0))

    def forward(self, input_var, input_lengths=None):
        
        embedded = self.embedding(input_var).unsqueeze(1)
        embedded = self.input_dropout(embedded)
        output = self.cnn(embedded)
        output = nn.functional.max_pool2d(output, kernel_size=(output.size(2), 1))
        output = output.squeeze(3).squeeze(2)
        
        return output