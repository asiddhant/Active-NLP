import torch
import torch.nn as nn
from torch.autograd import Variable

from neural_srl.util.utils import *

class DecoderCRF(nn.Module):

    def __init__(self, input_dimension, tag_to_ix, input_dropout_p=0):
        
        super(DecoderCRF, self).__init__()
        
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.dropout = nn.Dropout(input_dropout_p)
        self.hidden2tag = nn.Linear(input_dimension, self.tagset_size)
        
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
    
    def viterbi_decode(self, feats, mask ,usecuda = True, score_only= False):
    
        batch_size, sequence_len, num_tags = feats.size()
        
        assert num_tags == self.tagset_size
        
        mask = mask.transpose(0, 1).contiguous()
        feats = feats.transpose(0, 1).contiguous()
        
        backpointers = []
        
        all_forward_vars = Variable(torch.Tensor(sequence_len, 
                                    batch_size, num_tags).fill_(0.)).cuda()
        sum_all_forward_vars = Variable(torch.Tensor(sequence_len, 
                                    batch_size, num_tags).fill_(0.)).cuda()
        
        init_vars = torch.Tensor(batch_size, num_tags).fill_(-10000.)
        init_vars[:,self.tag_to_ix[START_TAG]] = 0.
        if usecuda:
            forward_var = Variable(init_vars).cuda()
            sum_forward_var = Variable(init_vars).cuda()
        else:
            forward_var = Variable(init_vars)
            sum_forward_var = Variable(init_vars)
        
        for i in range(sequence_len):
            
            broadcast_forward = forward_var.view(batch_size, 1, num_tags)
            sum_broadcast_forward = sum_forward_var.view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            
            next_tag_var = broadcast_forward + transition_scores
            sum_next_tag_var = sum_broadcast_forward + transition_scores
            
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=2)
            sum_viterbivars_t = log_sum_exp(sum_next_tag_var, dim=2)
            
            forward_var = viterbivars_t + feats[i]
            sum_forward_var = sum_viterbivars_t + feats[i]
            
            all_forward_vars[i,:,:] = forward_var
            sum_all_forward_vars[i,:,:] = sum_forward_var

            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            backpointers.append(bptrs_t)
        
        mask_sum = torch.sum(mask, dim = 0, keepdim =True) - 1
        mask_sum_ex = mask_sum.view(1, batch_size, 1).expand(1, batch_size, num_tags)
        final_forward_var = all_forward_vars.gather(0, mask_sum_ex).squeeze(0)
        sum_final_forward_var = sum_all_forward_vars.gather(0, mask_sum_ex).squeeze(0)
        
        terminal_var = final_forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1, num_tags)
        sum_terminal_var = sum_final_forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1, num_tags)
        terminal_var.data[:,self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[:,self.tag_to_ix[START_TAG]] = -10000.
        sum_terminal_var.data[:,self.tag_to_ix[STOP_TAG]] = -10000.
        sum_terminal_var.data[:,self.tag_to_ix[START_TAG]] = -10000.
        
        path_score, best_tag_id = torch.max(terminal_var, dim = 1)
        sum_path_score = log_sum_exp(sum_terminal_var, dim = 1)
        
        probs_score = path_score - sum_path_score
                
        if score_only:
            return probs_score.data.cpu().numpy()
        
        n_mask_sum = mask_sum.squeeze().data.cpu().numpy() + 1
        best_tag_id = best_tag_id.data.cpu().numpy()
        decoded_tags = []
        for i in range(batch_size):
            best_path = [best_tag_id[i]]
            bp_list = reversed([itm[i] for itm in backpointers[:n_mask_sum[i]]])
            for bptrs_t in bp_list:
                best_tag_id[i] = bptrs_t[best_tag_id[i]]
                best_path.append(best_tag_id[i])
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]
            best_path.reverse()
            decoded_tags.append(best_path)
        
        return probs_score.data.cpu().numpy(), decoded_tags
    
    def crf_forward(self, feats, mask, usecuda=True):
        
        batch_size, sequence_length, num_tags = feats.size()
        
        mask = mask.float().transpose(0, 1).contiguous()
        feats = feats.transpose(0, 1).contiguous()
        
        init_alphas = torch.Tensor(batch_size, num_tags).fill_(-10000.)
        init_alphas[:,self.tag_to_ix[START_TAG]] = 0.
        if usecuda:
            forward_var = Variable(init_alphas).cuda()
        else:
            forward_var = Variable(init_alphas)
        
        for i in range(sequence_length):
            emit_score = feats[i].view(batch_size, num_tags, 1)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_forward = forward_var.view(batch_size, 1, num_tags)
            tag_var = broadcast_forward + transition_scores + emit_score 
            
            forward_var = (log_sum_exp(tag_var, dim = 2) * mask[i].view(batch_size, 1) +
                            forward_var * (1 - mask[i]).view(batch_size, 1))
            
        terminal_var = (forward_var + (self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1))
        alpha = log_sum_exp(terminal_var, dim = 1)
        
        return alpha
        
    
    def score_sentence(self, feats, tags, mask, usecuda=True):
                
        batch_size, sequence_length, num_tags = feats.size()
        
        feats = feats.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
                
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags)
        
        score = self.transitions[:,self.tag_to_ix[START_TAG]].index_select(0, tags[0])
        
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i+1]
            
            transition_score = (
                     broadcast_transitions
                    .gather(1, next_tag.view(batch_size, 1, 1).expand(batch_size, 1, num_tags))
                    .squeeze(1)
                    .gather(1, current_tag.view(batch_size, 1))
                    .squeeze(1)
                    )

            emit_score = feats[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            score = score + transition_score* mask[i + 1] + emit_score * mask[i]  
        last_tag_index = mask.sum(0).long() - 1

        last_tags = tags.gather(0, last_tag_index.view(1, batch_size).expand(sequence_length, batch_size))
        last_tags = last_tags[0]

        last_transition_score = self.transitions[self.tag_to_ix[STOP_TAG]].index_select(0, last_tags)
        
        last_inputs = feats[-1]                                     
        last_input_score = last_inputs.gather(1, last_tags.view(batch_size, 1))
        last_input_score = last_input_score.squeeze(1)
        
        score = score + last_transition_score + last_input_score * mask[-1]
        
        return score
    
    def decode(self, input_var, mask, usecuda=True, score_only= False):
        
        input_var = self.dropout(input_var)
        features = self.hidden2tag(input_var)
        if score_only:
            score = self.viterbi_decode(features, mask, usecuda=usecuda, score_only=True)
            return score
        score, tag_seq = self.viterbi_decode(features, mask, usecuda=usecuda)
        return score, tag_seq
    
    def forward(self, input_var, tags, mask=None, usecuda=True):
        
        if mask is None:
            mask = Variable(torch.ones(*tags.size()).long())
        
        input_var = self.dropout(input_var)
        features = self.hidden2tag(input_var)
        forward_score = self.crf_forward(features, mask, usecuda=usecuda)
        ground_score = self.score_sentence(features, tags, mask, usecuda=usecuda)
        
        return torch.sum(forward_score-ground_score)