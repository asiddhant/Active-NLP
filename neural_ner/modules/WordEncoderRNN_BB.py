import torch
import torch.nn as nn
import math

from torch.autograd import Variable
import neural_ner
from neural_ner.util.utils import *
from torch.nn.utils.rnn import PackedSequence
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class RNNBase_BB(nn.Module):

    def __init__(self, mode, input_size, hidden_size, sigma_prior,
                 num_layers=1, batch_first=False,
                 dropout=0, bidirectional=True):
        
        super(RNNBase_BB, self).__init__()
        
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.sampled_weights = []
        self.sigma_prior = sigma_prior

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size
        
        self.means = []
        self.logvars = []
                
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih_mu = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh_mu = Parameter(torch.Tensor(gate_size, hidden_size))                
                w_ih_logvar = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh_logvar = Parameter(torch.Tensor(gate_size, hidden_size))
                                
                b_ih_mu = Parameter(torch.Tensor(gate_size))
                b_hh_mu = Parameter(torch.Tensor(gate_size))
                b_ih_logvar = Parameter(torch.Tensor(gate_size))
                b_hh_logvar = Parameter(torch.Tensor(gate_size))
                
                self.means += [w_ih_mu, w_hh_mu, b_ih_mu, b_hh_mu]
                self.logvars += [w_ih_logvar, w_hh_logvar, b_ih_logvar, b_hh_logvar]
                
                layer_params = (w_ih_mu,  w_ih_logvar, w_hh_mu, w_hh_logvar, b_ih_mu, b_ih_logvar, b_hh_mu, b_hh_logvar)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l_mu{}{}', 'weight_ih_l_logvar{}{}', 'weight_hh_l_mu{}{}', 'weight_hh_l_logvar{}{}']
                param_names += ['bias_ih_l_mu{}{}', 'bias_ih_l_logvar{}{}', 'bias_hh_l_mu{}{}', 'bias_hh_l_logvar{}{}']
                
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

        self.reset_parameters()
        self.lpw = 0
        self.lqw = 0

    def _apply(self, fn):
        ret = super(RNNBase_BB, self)._apply(fn)
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        logvar_init = math.log(stdv) * 2
        for mean in self.means:
            mean.data.uniform_(-stdv, stdv)
        for logvar in self.logvars:
            logvar.data.fill_(logvar_init)
            
    def get_all_weights(self, weights):
        
        start = 0
        all_weights = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                w_ih = weights[start]
                w_hh = weights[start+1]
                b_ih = weights[start+2]
                b_hh = weights[start+3]
                start += 4
                all_weights.append([w_ih, w_hh, b_ih, b_hh])

        return all_weights
    
    def sample(self, usecuda = True):
        self.sampled_weights = []
        for i in range(len(self.means)):
            mean = self.means[i]
            logvar = self.logvars[i]
            eps = torch.zeros(mean.size())
            if usecuda:
                eps = eps.cuda()

            eps.normal_(0, self.sigma_prior)
            std = logvar.mul(0.5).exp()
            weight = mean + Variable(eps) * std
            self.sampled_weights.append(weight)
            
    def _calculate_prior(self, weights):
        lpw = 0.
        for w in weights:
            lpw += log_gaussian(w, 0, self.sigma_prior).sum()
        return lpw
    
    def _calculate_posterior(self, weights):
        lqw = 0.
        for i,w in enumerate(weights):
            lqw += log_gaussian_logsigma(w, self.means[i], 0.5*self.logvars[i]).sum()
        return lqw

    def forward(self, input, hx=None, usecuda = True):
        if self.training:
            self.sample(usecuda = usecuda)
            weights = self.sampled_weights
            self.lpw = self._calculate_prior(weights)
            self.lqw = self._calculate_posterior(weights)
        else:
            weights = self.means

        self.all_weights = self.get_all_weights(weights)
        
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)

        func = self._backend.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=None
        )
        # change this line
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden


class LSTM_BB(RNNBase_BB):

    def __init__(self, *args, **kwargs):
        super(LSTM_BB, self).__init__('LSTM', *args, **kwargs)

class baseRNN_BB(nn.Module):

    def __init__(self, vocab_size, hidden_size, input_dropout_p, output_dropout_p, n_layers, rnn_cell, 
                 max_len=25):
        
        super(baseRNN_BB, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len
        
        self.input_dropout_p = input_dropout_p
        self.output_dropout_p = output_dropout_p
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = LSTM_BB
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class WordEncoderRNN_BB(baseRNN_BB):

    def __init__(self, vocab_size, embedding_size ,hidden_size, char_size, cap_size, sigma_prior, input_dropout_p=0.5, 
                 output_dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='lstm'):
        
        super(WordEncoderRNN_BB, self).__init__(vocab_size, hidden_size, input_dropout_p, 
                                             output_dropout_p, n_layers, rnn_cell)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        augmented_embedding_size = embedding_size + char_size + cap_size
        self.rnn = self.rnn_cell(augmented_embedding_size, hidden_size, n_layers,
                                 bidirectional=bidirectional, dropout=output_dropout_p,
                                 batch_first=True)

    def forward(self, words, char_embedding, cap_embedding, input_lengths):
        
        embedded = self.embedding(words)
        if cap_embedding is not None:
            embedded = torch.cat((embedded,char_embedding,cap_embedding),2)  
        else:
            embedded = torch.cat((embedded,char_embedding),2)
    
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first= True)
        output, _ = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first= True)
        
        return output
    
    def get_lpw_lqw(self):
        
        lpw = self.rnn.lpw
        lqw = self.rnn.lqw
        return lpw, lqw