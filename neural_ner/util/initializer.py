import torch.nn as nn
from torch.nn import init
import numpy as np

class Initializer(object):
    
    def __init__(self):
        pass
    
    def init_embedding(self, input_embedding):
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)
    
    def init_linear(self, input_linear):
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()
    
    def init_lstm(self, input_lstm):
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
        
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform(weight, -bias, bias)
                weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform(weight, -bias, bias)
        
        if input_lstm.bias:
            
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            
            if input_lstm.bidirectional:
                for ind in range(0, input_lstm.num_layers):
                    weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                    weight.data.zero_()
                    weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                    weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                    weight.data.zero_()
                    weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1