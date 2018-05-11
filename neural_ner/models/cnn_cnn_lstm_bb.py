import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


import neural_ner
from neural_ner.util import Initializer
from neural_ner.util import Loader
from neural_ner.modules import CharEncoderCNN
from neural_ner.modules import WordEncoderCNN_BB
from neural_ner.modules import DecoderRNN

from neural_ner.util.utils import *

class CNN_CNN_LSTM_BB(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size, 
                 char_embedding_dim, char_out_channels, decoder_hidden_units, tag_to_id, sigma_prior, 
                 cap_input_dim=4, cap_embedding_dim=0, pretrained=None):
        
        super(CNN_CNN_LSTM_BB, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels
        
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_out_channels = char_out_channels
        
        self.cap_input_dim = cap_input_dim
        self.cap_embedding_dim = cap_embedding_dim
        self.sigma_prior = sigma_prior
        
        self.tag_to_ix = tag_to_id
        self.tagset_size = len(tag_to_id)
        
        self.initializer = Initializer()
        self.loader = Loader()
        
        if self.cap_input_dim and self.cap_embedding_dim:
            self.cap_embedder = nn.Embedding(self.cap_input_dim, self.cap_embedding_dim)
            self.initializer.init_embedding(self.cap_embedder.weight)
        
        self.char_encoder = CharEncoderCNN(char_vocab_size, char_embedding_dim, char_out_channels, 
                                           kernel_width=3, pad_width=1)
        
        self.initializer.init_embedding(self.char_encoder.embedding.weight)
        
        self.word_encoder = WordEncoderCNN_BB(word_vocab_size, word_embedding_dim, char_out_channels, 
                                              sigma_prior= sigma_prior, kernel_width = 3, pad_width = 1, 
                                              input_dropout_p=0.5, out_channels=word_out_channels)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        
        augmented_decoder_inp_size = (word_out_channels + word_embedding_dim + 
                                      char_out_channels + cap_embedding_dim)
        self.decoder = DecoderRNN(augmented_decoder_inp_size, decoder_hidden_units, self.tagset_size, 
                                  self.tag_to_ix, input_dropout_p=0.5)
        
    def forward_pass(self, words, tags, chars, caps, wordslen, charslen, tagsmask, usecuda=True):
        
        batch_size, max_len = words.size()
        
        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None
        
        char_features = self.char_encoder(chars)
        char_features = char_features.view(batch_size, max_len, -1)
        
        word_features, word_input_feats = self.word_encoder(words, char_features, cap_features)
        
        new_word_features = torch.cat((word_features,word_input_feats),2)
        loss = self.decoder(new_word_features, tags, tagsmask, usecuda=usecuda)
        
        return loss
    
    def forward(self, words, tags, chars, caps, wordslen, charslen, tagsmask, n_batches, n_samples = 3, usecuda=True):
        batch_size, max_len = words.size()
        s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
        
        for _ in xrange(n_samples):
            sample_log_likelihood = -1. * self.forward_pass(words, tags, chars, caps, wordslen, charslen, tagsmask, 
                                                            usecuda = usecuda)
            sample_log_pw, sample_log_qw = self.word_encoder.get_lpw_lqw()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood
        
        log_pw, log_qw, log_llh = s_log_pw/n_samples, s_log_qw/n_samples, s_log_likelihood/n_samples
        loss = bayes_loss_function(log_pw, log_qw, log_llh, n_batches, batch_size)
        
        return loss
    
    def decode(self, words, chars, caps, wordslen, charslen, tagsmask, usecuda=True, 
               score_only = False):
        
        batch_size, max_len = words.size()
        
        cap_features = self.cap_embedder(caps) if self.cap_embedding_dim else None
        
        char_features = self.char_encoder(chars)
        char_features = char_features.view(batch_size, max_len, -1)
        
        word_features, word_input_feats = self.word_encoder(words, char_features, cap_features)
        
        new_word_features = torch.cat((word_features,word_input_feats),2)
        
        if score_only:
            score, _ = self.decoder.decode(new_word_features, tagsmask, wordslen, usecuda=usecuda)
            return score
        
        score, tag_seq = self.decoder.decode(new_word_features, tagsmask, wordslen, usecuda=usecuda)
        return score, tag_seq