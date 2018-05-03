import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import neural_cls
from neural_cls.util import Initializer
from neural_cls.util import Loader
from neural_cls.modules import EncoderCNN_BB
from neural_cls.util.utils import *

class CNN_BB(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, output_size, 
                 sigma_prior, dropout_p = 0.5, pretrained=None):
        
        super(CNN_BB, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels
        self.sigma_prior = sigma_prior
        
        self.initializer = Initializer()
        self.loader = Loader()
        
        self.word_encoder = EncoderCNN_BB(word_vocab_size, word_embedding_dim, out_channels= word_out_channels, 
                                          sigma_prior = sigma_prior)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = word_out_channels
        self.linear = nn.Linear(hidden_size, output_size)
                
    def forward_pass(self, words, wordslen, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen, usecuda=usecuda)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        
        return output
        
    def forward(self, words, tags, tagset_size, wordslen, n_batches, n_samples=3, usecuda=True):
        
        batch_size, max_len = words.size()
        s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
                
        if usecuda:
            onehottags = Variable(torch.zeros(batch_size, tagset_size)).cuda()
        else:
            onehottags = Variable(torch.zeros(batch_size, tagset_size))
        onehottags.scatter_(1, tags.unsqueeze(1), 1)
                
        for _ in xrange(n_samples):
            output = self.forward_pass(words, wordslen, usecuda = usecuda)
            sample_log_pw, sample_log_qw = self.word_encoder.get_lpw_lqw()
            sample_log_likelihood = log_gaussian(onehottags, output, self.sigma_prior).sum()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood
        
        log_pw, log_qw, log_llh = s_log_pw/n_samples, s_log_qw/n_samples, s_log_likelihood/n_samples
        loss = bayes_loss_function(log_pw, log_qw, log_llh, n_batches, batch_size)
        
        return loss
    
    def predict(self, words, wordslen, scoreonly=False, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        
        scores = torch.max(F.softmax(output, dim =1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores
        
        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction