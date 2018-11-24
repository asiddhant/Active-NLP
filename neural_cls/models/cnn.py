import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import neural_cls
from neural_cls.modules import EncoderCNN, Embedder

class CNN(nn.Module):
    
    def __init__(self, embedding_type, word_embedding_dim, word_vocab_size, 
                 embedding_path, mappings, word_out_channels, output_size, 
                 dropout_p = 0.5):
        
        super(CNN, self).__init__()
        
        self.embedding_type = embedding_type
        self.word_embedding_dim = word_embedding_dim
        self.word_vocab_size = word_vocab_size
        self.word_out_channels = word_out_channels

        self.word_embedder = Embedder(word_vocab_size, word_embedding_dim, 
                                      mappings)
        self.word_encoder = EncoderCNN(word_embedding_dim, word_out_channels)
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(word_out_channels, output_size)
        self.lossfunc = nn.CrossEntropyLoss()
        
    def forward(self, words, strwords, tags, wordslen=None, nsamples=1):
        
        batch_size, max_len = words.size()
        word_embeddings = self.word_embedder(words, strwords)
        word_features = self.word_encoder(word_embeddings)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        loss = self.lossfunc(output, tags)
        
        return loss
    
    def predict(self, words, strwords, wordslen=None, scoreonly=False):
        
        batch_size, max_len = words.size()
        word_embeddings = self.word_embedder(words, strwords)
        word_features = self.word_encoder(word_embeddings)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        
        scores = torch.max(F.softmax(output, dim =1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores
        
        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction