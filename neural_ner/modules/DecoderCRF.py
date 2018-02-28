import torch
import torch.nn as nn
from torch.autograd import Variable

from neural_ner.util.utils import *

class DecoderCRF(nn.Module):

    def __init__(self, input_dimension, tag_to_ix, input_dropout_p=0.5):
        
        super(DecoderCRF, self).__init__()
        
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.dropout = nn.Dropout(input_dropout_p)
        self.hidden2tag = nn.Linear(input_dimension, self.tagset_size)
        
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
    
    def viterbi_decode(self, features):
        
        backpointers = []
        
        init_vars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = Variable(init_vars).cuda()
        
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, 
                                      self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t)).cuda()
            
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        
        return path_score, best_path
    
    def crf_forward(self, feats):
        
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = Variable(init_alphas).cuda()
        
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)
            
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        
        return alpha
    
    def score_sentence(self, features, tags):
        
        r = torch.LongTensor(range(features.size()[0])).cuda()
        pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])
        return score
    
    def decode(self, input_var, tags, input_lengths=None):
        
        input_var = self.dropout(input_var)
        features = self.hidden2tag(input_var)
        score, tag_seq = self.viterbi_decode(features)
        
        return score, tag_seq
    
    def forward(self, input_var, tags, input_lengths=None):
        
        input_var = self.dropout(input_var)
        features = self.hidden2tag(input_var)
        forward_score = self.crf_forward(features)
        ground_score = self.score_sentence(features, tags)
        
        return forward_score-ground_score