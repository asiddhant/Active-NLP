import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


import neural_ner
from neural_ner.util import Initializer
from neural_ner.util import Loader
from neural_ner.modules import CharEncoderCNN
from neural_ner.modules import WordEncoderRNN
from neural_ner.modules import DecoderCRF

class CNN_BiLSTM_CRF_MC(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                 char_embedding_dim, char_out_channels, tag_to_id, cap_input_dim=4 ,
                 cap_embedding_dim=0, pretrained=None):
        
        super(CNN_BiLSTM_CRF_MC, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_out_channels = char_out_channels
        
        self.cap_input_dim = cap_input_dim
        self.cap_embedding_dim = cap_embedding_dim
        
        self.tag_to_ix = tag_to_id
        self.tagset_size = len(tag_to_id)
        
        self.initializer = Initializer()
        self.loader = Loader()
        
        if self.cap_embedding_dim:
            self.cap_embedder = nn.Embedding(self.cap_input_dim, self.cap_embedding_dim)
            self.initializer.init_embedding(self.cap_embedder.weight)
        
        self.char_encoder = CharEncoderCNN(char_vocab_size, char_embedding_dim, char_out_channels, 
                                           kernel_width=3, pad_width=1)
        
        self.initializer.init_embedding(self.char_encoder.embedding.weight)
        
        self.word_encoder = WordEncoderRNN(word_vocab_size, word_embedding_dim ,word_hidden_dim, 
                                           char_out_channels, cap_embedding_dim, input_dropout_p=0.5)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
            
        self.initializer.init_lstm(self.word_encoder.rnn)
        
        self.decoder = DecoderCRF(word_hidden_dim*2, self.tag_to_ix, input_dropout_p=0.5)
        self.initializer.init_linear(self.decoder.hidden2tag)
        
    def forward(self, words, tags, chars, caps, wordslen, charslen, tagsmask, usecuda=True):
        
        batch_size, max_len = words.size()
        
        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None
        
        char_features = self.char_encoder(chars)
        char_features = char_features.view(batch_size, max_len, -1)
        
        word_features = self.word_encoder(words, char_features, cap_features, wordslen)
        
        score = self.decoder(word_features, tags, tagsmask, usecuda=usecuda)
        
        return score
    
    def decode(self, words, chars, caps, wordslen, charslen, tagsmask, usecuda=True,
               score_only = False):
        
        batch_size, max_len = words.size()
        
        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None
        
        char_features = self.char_encoder(chars)
        char_features = char_features.view(batch_size, max_len, -1)
        
        word_features = self.word_encoder(words, char_features, cap_features, wordslen)
        
        if score_only:
            score = self.decoder.decode(word_features, tagsmask, usecuda=usecuda, 
                                        score_only = True)
            return score
        score, tag_seq = self.decoder.decode(word_features, tagsmask, usecuda=usecuda)
        return score, tag_seq