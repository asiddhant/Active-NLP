import torch
import torch.nn as nn
from torch.autograd import Variable

from neural_ner.util.utils import *

class DecoderRNN(nn.Module):
    def __init__(self, input_size ,hidden_size, tag_size, tag_to_ix, input_dropout_p=0.5, 
                 output_dropout_p=0, n_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.input_dropout_p = input_dropout_p
        self.output_dropout_p = output_dropout_p
        
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.dropout = nn.Dropout(input_dropout_p)
        
        self.rnn = nn.LSTM(input_size + tag_size, hidden_size, n_layers, bidirectional=False)
        self.linear = nn.Linear(input_size, tag_size)
        self.ignore = -1
        self.lossfunc = nn.CrossEntropyLoss(ignore_index= self.ignore, size_average=False)
        
    def forward_step(self, input_var, prev_tag, hidden ,usecuda=True):
        
        prev_tag_onehot = torch.eye(self.tagset_size)
        prev_tag_onehot = prev_tag_onehot.index_select(0,torch.LongTensor(prev_tag))
        
        if usecuda:
            prev_tag_onehot = Variable(prev_tag_onehot).cuda()
        else:
            prev_tag_onehot = Variable(prev_tag_onehot)
        
        decoder_input = torch.cat([input_var, prev_tag_onehot],1).unsqueeze(0)
        output, hidden = self.rnn(decoder_input, hidden)
        output = self.linear(output.squeeze(0))
        output_tag = output.max(1)[1].data.cpu().numpy().tolist()

        return output, output_tag, hidden
        
    def forward(self, input_var, tags, mask, usecuda=True):
        
        batch_size, sequence_len, _ = input_var.size()
        
        input_var = self.dropout(input_var)
        
        input_var = input_var.transpose(0, 1).contiguous()
        
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        
        maskedtags = tags.clone()
        maskedtags[mask==0] = -1
        
        loss = 0.0
        prev_tag = [self.tag_to_ix[START_TAG]]*batch_size
        hidden = None
        
        for i in range(sequence_len):
            output, prev_tag, hidden=self.forward_step(input_var[i], prev_tag, hidden, 
                                                       usecuda=usecuda)
            loss += self.lossfunc(output, maskedtags[i])
        return loss
    
    def decode(self, input_var, mask, wordslen, usecuda=True):
        
        batch_size, sequence_len, _ = input_var.size()
        
        input_var = self.dropout(input_var)
        input_var = input_var.transpose(0, 1).contiguous()
        
        loss = 0.0
        prev_tag = [self.tag_to_ix[START_TAG]]*batch_size
        hidden = None
        
        tag_seq = []
        probs= []
        for i in range(sequence_len):
            output, prev_tag, hidden=self.forward_step(input_var[i], prev_tag, hidden, 
                                                       usecuda=usecuda)
            tag_seq.append(prev_tag)
            pb = nn.functional.log_softmax(output, dim = 1).data.cpu().numpy()
            probs.append(pb)
        
        probs = np.array(probs).transpose(1,0,2).max(2)
        probs = probs * mask.cpu().data.numpy()
        probs = probs.sum(1)
        
        tag_seq = np.array(tag_seq).transpose().tolist()
        tag_seq = [ts[:wordslen[i]] for i,ts in enumerate(tag_seq)]
        
        return probs, tag_seq
