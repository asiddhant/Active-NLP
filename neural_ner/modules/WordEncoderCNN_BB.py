import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable
import neural_ner
from neural_ner.util.utils import *

class bb_ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(bb_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.W_mu = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).normal_(0, 0.01))
            self.W_logsigma = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size).normal_(0, 0.01))
        else:
            self.W_mu = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).normal_(0, 0.01))
            self.W_logsigma = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size).normal_(0, 0.01))
        if bias:
            self.b_mu = Parameter(torch.Tensor(out_channels).uniform_(-0.01, 0.01))
            self.b_logsigma = Parameter(torch.Tensor(out_channels).uniform_(-0.01, 0.01))
        else:
            self.register_parameter('bias', None)
            

class bb_Conv1d(bb_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, sigma_prior, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(bb_Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)
        self.lpw = 0
        self.lqw = 0
        self.sigma_prior = sigma_prior

    def forward(self, input, usecuda=True):
        if not self.training:
            return F.conv1d(input, self.W_mu, self.b_mu, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.transposed:
            if usecuda:
                epsilon_W = Variable(torch.Tensor(
                    self.in_channels, self.out_channels // self.groups, *self.kernel_size).normal_(0, self.sigma_prior)).cuda()
                epsilon_b = Variable(torch.Tensor(self.out_channels).normal_(0, self.sigma_prior)).cuda()
            else:
                epsilon_W = Variable(torch.Tensor(
                    self.in_channels, self.out_channels // self.groups, *self.kernel_size).normal_(0, self.sigma_prior))
                epsilon_b = Variable(torch.Tensor(self.out_channels).normal_(0, self.sigma_prior))
        else:
            if usecuda:
                epsilon_W = Variable(torch.Tensor(
                    self.out_channels, self.in_channels // self.groups, *self.kernel_size).normal_(0, self.sigma_prior)).cuda()
                epsilon_b = Variable(torch.Tensor(self.out_channels).normal_(0, self.sigma_prior)).cuda()
            else:
                epsilon_W = Variable(torch.Tensor(
                    self.out_channels, self.in_channels // self.groups, *self.kernel_size).normal_(0, self.sigma_prior))
                epsilon_b = Variable(torch.Tensor(self.out_channels).normal_(0, self.sigma_prior))
        
        self.weight = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        self.bias = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        
        self.lpw = log_gaussian(self.weight, 0, self.sigma_prior).sum() + log_gaussian(self.bias, 0, self.sigma_prior).sum()
        self.lqw = (log_gaussian_logsigma(self.weight, self.W_mu, self.W_logsigma).sum() + 
                   log_gaussian_logsigma(self.bias, self.b_mu, self.b_logsigma).sum())
        
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
class WordEncoderCNN_BB(nn.Module):

    def __init__(self, vocab_size, embedding_size, char_size, sigma_prior, kernel_width = 5, pad_width = 2, 
                 out_channels=200 , cap_size=0, input_dropout_p=0.5, output_dropout_p=0.5):
        
        super(WordEncoderCNN_BB, self).__init__()
        
        self.kernel_width = kernel_width
        self.out_channels = out_channels
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.output_dropout = nn.Dropout(p=output_dropout_p)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        in_channels = embedding_size + char_size + cap_size
        
        self.cnn1 = bb_Conv1d(in_channels, out_channels, kernel_size=kernel_width, sigma_prior = sigma_prior,
                             padding = pad_width)
        self.cnn2 = bb_Conv1d(out_channels, out_channels, kernel_size=kernel_width, sigma_prior = sigma_prior,
                             padding = pad_width)
        self.cnn3 = bb_Conv1d(out_channels, out_channels, kernel_size=kernel_width, sigma_prior = sigma_prior,
                             padding = pad_width)
        self.cnn4 = bb_Conv1d(out_channels, out_channels, kernel_size=kernel_width, sigma_prior = sigma_prior,
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
    
    def get_lpw_lqw(self):
        
        lpw = self.cnn1.lpw + self.cnn2.lpw + self.cnn3.lpw + self.cnn4.lpw
        lqw = self.cnn1.lqw + self.cnn2.lqw + self.cnn3.lqw + self.cnn4.lqw
        return lpw, lqw