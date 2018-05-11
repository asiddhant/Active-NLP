import torch
torch.manual_seed(0)
from torch.autograd import Variable
import numpy as np
from collections import Counter
import time
from scipy import stats
from neural_ner.util.utils import *
import pandas as pd

class Acquisition(object):
    
    def __init__(self, train_data, acq_mode='d', init_percent=2, seed=0, usecuda = True):
        self.tokenlen = sum([len(x['words']) for x in train_data])
        self.train_index = set()
        self.npr = np.random.RandomState(seed)
        self.obtain_data(train_data, acquire = init_percent)
        self.acq_mode = acq_mode
        self.usecuda = usecuda
        
    def get_random(self, data, num_tokens):
        test_indices = self.npr.permutation(len(data))
        cur_tokens=0
        cur_indices = set()
        i = 0
        while cur_tokens<num_tokens:
            if test_indices[i] not in self.train_index:
                cur_indices.add(test_indices[i])
                cur_tokens += len(data[test_indices[i]]['words'])
            i+=1
        self.train_index.update(cur_indices)
                 
    def get_mnlp(self, dataset, model_path, decoder, num_tokens, batch_size = 50):
        
        model = torch.load(model_path)
        model.train(False)
        tm = time.time()
        probs = np.ones(len(dataset))*float('Inf')
        
        new_dataset = [datapoint for j,datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]
        
        data_batches = create_batches(new_dataset, batch_size = batch_size, str_words = True,
                                      tag_padded = False)
        probscores = []
        for data in data_batches:

            words = data['words']
            chars = data['chars']
            caps = data['caps']
            mask = data['tagsmask']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
                chars = Variable(torch.LongTensor(chars)).cuda()
                caps = Variable(torch.LongTensor(caps)).cuda()
                mask = Variable(torch.LongTensor(mask)).cuda()
            else:
                words = Variable(torch.LongTensor(words))
                chars = Variable(torch.LongTensor(chars))
                caps = Variable(torch.LongTensor(caps))
                mask = Variable(torch.LongTensor(mask))

            wordslen = data['wordslen']
            charslen = data['charslen']
            sort_info = data['sort_info']
            
            score = model.decode(words, chars, caps, wordslen, charslen, mask, usecuda = self.usecuda,
                                 score_only = True)
            
            norm_scores = score/np.array(wordslen)
            assert len(norm_scores) == len(words)
            probscores.extend(list(norm_scores[np.array(sort_info)]))
            
        assert len(new_datapoints) == len(probscores)
        probs[new_datapoints] = np.array(probscores)
        
        test_indices = np.argsort(probs)
        cur_tokens=0
        cur_indices = set()
        i = 0
        while cur_tokens<num_tokens:
            cur_indices.add(test_indices[i])
            cur_tokens += len(dataset[test_indices[i]]['words'])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('D Acquisition took %d seconds:' %(time.time()-tm))
        
    def get_mnlp_mc(self, dataset, model_path, decoder, num_tokens, nsamp=100, batch_size = 50):
        
        model = torch.load(model_path)
        model.train(True)
        tm = time.time()
        
        probs = np.ones((len(dataset),nsamp))*float('Inf')
        varsc = np.ones(len(dataset))*float('Inf')
        
        new_dataset = [datapoint for j,datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]
        
        data_batches = create_batches(new_dataset, batch_size = batch_size, str_words = True,
                                      tag_padded = False)
        
        varsc_outer_list = []
        probs_outer_list = []
        for data in data_batches:

            words = data['words']
            chars = data['chars']
            caps = data['caps']
            mask = data['tagsmask']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
                chars = Variable(torch.LongTensor(chars)).cuda()
                caps = Variable(torch.LongTensor(caps)).cuda()
                mask = Variable(torch.LongTensor(mask)).cuda()
            else:
                words = Variable(torch.LongTensor(words))
                chars = Variable(torch.LongTensor(chars))
                caps = Variable(torch.LongTensor(caps))
                mask = Variable(torch.LongTensor(mask))

            wordslen = data['wordslen']
            charslen = data['charslen']
            sort_info = data['sort_info']
            
            tag_seq_list = []
            probs_list = []
            for itr in range(nsamp):
                score, tag_seq = model.decode(words, chars, caps, wordslen, charslen, mask, 
                                              usecuda = self.usecuda, score_only = False)
                tag_seq = [[str(tg) for tg in one_tag_seq] for one_tag_seq in tag_seq]
                tag_seq = np.array(['_'.join(one_tag_seq) for one_tag_seq in tag_seq])
                tag_seq_new = tag_seq[np.array(sort_info)]
                assert len(tag_seq_new) == len(words)
                tag_seq_list.append(tag_seq_new)
                norm_scores = score/np.array(wordslen)
                probs_list.append(norm_scores[np.array(sort_info)])
            
            tag_seq_list = np.array(tag_seq_list)
            probs_list = np.array(probs_list).transpose()
            _, tag_seq_count = stats.mode(tag_seq_list)
            tag_seq_count = tag_seq_count.squeeze(0)
            assert len(tag_seq_count) == len(words)
            varsc_outer_list.extend(list(tag_seq_count))
            probs_outer_list.extend(list(probs_list))
           
        assert len(new_datapoints) == len(varsc_outer_list)
        varsc[new_datapoints] = np.array(varsc_outer_list)
        assert len(new_datapoints) == len(probs_outer_list)
        probs[new_datapoints,:] = np.array(probs_outer_list)
        probsmean = np.mean(probs, axis = 1)
        test_indices = np.lexsort((probsmean, varsc))
                
        cur_tokens=0
        cur_indices = set()
        i = 0
        while cur_tokens<num_tokens:
            cur_indices.add(test_indices[i])
            cur_tokens += len(dataset[test_indices[i]]['words'])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('*'*80)
        print ('MC Acquisition took %d seconds:' %(time.time()-tm))
        print ('*'*80)
        
    def get_mnlp_bb(self, dataset, model_path, decoder, num_tokens, nsamp=100, batch_size = 50):
        
        model = torch.load(model_path)
        model.train(True)
        tm = time.time()
        
        probs = np.ones((len(dataset),nsamp))*float('Inf')
        varsc = np.ones(len(dataset))*float('Inf')
        
        new_dataset = [datapoint for j,datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]
        
        data_batches = create_batches(new_dataset, batch_size = batch_size, str_words = True,
                                      tag_padded = False)
        
        varsc_outer_list = []
        probs_outer_list = []
        for data in data_batches:

            words = data['words']
            chars = data['chars']
            caps = data['caps']
            mask = data['tagsmask']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
                chars = Variable(torch.LongTensor(chars)).cuda()
                caps = Variable(torch.LongTensor(caps)).cuda()
                mask = Variable(torch.LongTensor(mask)).cuda()
            else:
                words = Variable(torch.LongTensor(words))
                chars = Variable(torch.LongTensor(chars))
                caps = Variable(torch.LongTensor(caps))
                mask = Variable(torch.LongTensor(mask))

            wordslen = data['wordslen']
            charslen = data['charslen']
            sort_info = data['sort_info']
            
            tag_seq_list = []
            probs_list = []
            for itr in range(nsamp):
                score, tag_seq = model.decode(words, chars, caps, wordslen, charslen, mask, 
                                              usecuda = self.usecuda, score_only = False)
                tag_seq = [[str(tg) for tg in one_tag_seq] for one_tag_seq in tag_seq]
                tag_seq = np.array(['_'.join(one_tag_seq) for one_tag_seq in tag_seq])
                tag_seq_new = tag_seq[np.array(sort_info)]
                assert len(tag_seq_new) == len(words)
                tag_seq_list.append(tag_seq_new)
                norm_scores = score/np.array(wordslen)
                probs_list.append(norm_scores[np.array(sort_info)])
            
            tag_seq_list = np.array(tag_seq_list)
            probs_list = np.array(probs_list).transpose()
            _, tag_seq_count = stats.mode(tag_seq_list)
            tag_seq_count = tag_seq_count.squeeze(0)
            assert len(tag_seq_count) == len(words)
            varsc_outer_list.extend(list(tag_seq_count))
            probs_outer_list.extend(list(probs_list))
           
        assert len(new_datapoints) == len(varsc_outer_list)
        varsc[new_datapoints] = np.array(varsc_outer_list)
        assert len(new_datapoints) == len(probs_outer_list)
        probs[new_datapoints,:] = np.array(probs_outer_list)
        probsmean = np.mean(probs, axis = 1)
        test_indices = np.lexsort((probsmean, varsc))
                
        cur_tokens=0
        cur_indices = set()
        i = 0
        while cur_tokens<num_tokens:
            cur_indices.add(test_indices[i])
            cur_tokens += len(dataset[test_indices[i]]['words'])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('*'*80)
        print ('MC Acquisition took %d seconds:' %(time.time()-tm))
        print ('*'*80)
        
    def obtain_data(self, data, model_path=None, model_name=None, acquire=2, method='random', num_samples=100):
        
        num_tokens = (acquire*self.tokenlen)/100
        
        if model_path is None or model_name is None:
            method = 'random'
        
        if method=='random':
            self.get_random(data, num_tokens)
        else:
            decoder = model_name.split('_')[2]
            if self.acq_mode == 'd':
                if method=='mnlp':
                    self.get_mnlp(data, model_path, decoder, num_tokens)
                else:
                    raise NotImplementedError()
            elif self.acq_mode == 'm':
                if method=='mnlp':
                    self.get_mnlp_mc(data, model_path, decoder, num_tokens, nsamp = num_samples)
                else:
                    raise NotImplementedError()
            elif self.acq_mode == 'b':
                if method=='mnlp':
                    self.get_mnlp_bb(data, model_path, decoder, num_tokens, nsamp = num_samples)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()