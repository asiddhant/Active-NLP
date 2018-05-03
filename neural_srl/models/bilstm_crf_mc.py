import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


import neural_srl
from neural_srl.util import Initializer
from neural_srl.util import Loader
from neural_srl.modules import VanillaRNN
from neural_srl.modules import DecoderCRF


class BiLSTM_CRF_MC(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_hidden_dim, tag_to_id, 
                 verb_embedding_dim, cap_embedding_dim, verb_input_dim = 2, cap_input_dim=4, 
                 pretrained=None):
        
        super(BiLSTM_CRF_MC, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        self.verb_input_dim = verb_input_dim 
        self.verb_embedding_dim = verb_embedding_dim
        
        self.cap_input_dim = cap_input_dim
        self.cap_embedding_dim = cap_embedding_dim
        
        self.tag_to_ix = tag_to_id
        self.tagset_size = len(tag_to_id)
        
        self.initializer = Initializer()
        
        if self.cap_embedding_dim:
            self.cap_embedder = nn.Embedding(self.cap_input_dim, self.cap_embedding_dim)
            self.initializer.init_embedding(self.cap_embedder.weight)
            
        self.verb_embedder = nn.Embedding(self.verb_input_dim, self.verb_embedding_dim)
        self.initializer.init_embedding(self.verb_embedder.weight)
        
        self.word_encoder = VanillaRNN(word_vocab_size, word_embedding_dim ,word_hidden_dim, 
                                     verb_embedding_dim, cap_embedding_dim, input_dropout_p=0)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
            
        self.initializer.init_lstm(self.word_encoder.rnn)
        
        self.decoder = DecoderCRF(word_hidden_dim*2, self.tag_to_ix, input_dropout_p=0)
        self.initializer.init_linear(self.decoder.hidden2tag)
        
    def forward(self, words, tags, verbs, caps, wordslen, tagsmask, usecuda=True):
        
        batch_size, max_len = words.size()
        
        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None
        verb_features = self.verb_embedder(verbs)
        word_features = self.word_encoder(words, verb_features ,cap_features, wordslen)
        score = self.decoder(word_features, tags, tagsmask, usecuda=usecuda)
        
        return score
    
    def decode(self, words, verbs, caps, wordslen, tagsmask, usecuda=True, score_only = False):
        
        batch_size, max_len = words.size()
        
        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None
        verb_features = self.verb_embedder(verbs)
        word_features = self.word_encoder(words, verb_features ,cap_features, wordslen)
        if score_only:
            score = self.decoder.decode(word_features, tagsmask, usecuda=usecuda, 
                                        score_only = True)
            return score
        score, tag_seq = self.decoder.decode(word_features, tagsmask, usecuda=usecuda)
        
        return score, tag_seq