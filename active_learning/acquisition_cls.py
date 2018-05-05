import torch
torch.manual_seed(0)
from torch.autograd import Variable
import numpy as np
from collections import Counter
import time
from scipy import stats
from neural_cls.util.utils import *
import pandas as pd

class Acquisition_CLS(object):
    
    def __init__(self, train_data, acq_mode='d', init_percent=2, seed=9, usecuda = True):
        self.sentencelen = len(train_data)
        self.train_index = set()
        self.npr = np.random.RandomState(seed)
        self.obtain_data(train_data, acquire = init_percent)
        self.acq_mode = acq_mode
        self.usecuda = usecuda
        
    def get_random(self, data, num_sentences):
        test_indices = self.npr.permutation(len(data))
        cur_indices = set()
        i = 0
        while len(cur_indices)<num_sentences:
            if test_indices[i] not in self.train_index:
                cur_indices.add(test_indices[i])
            i+=1
        self.train_index.update(cur_indices)
                 
    def get_mnlp(self, dataset, model_path, num_sentences, batch_size = 32):
        
        model = torch.load(model_path)
        model.train(False)
        tm = time.time()
        probs = np.ones(len(dataset))*float('Inf')
        
        new_dataset = [datapoint for j,datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]
        
        data_batches = create_batches(new_dataset, batch_size = batch_size)
        probscores = []
        
        for data in data_batches:
            
            words = data['words']
            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
            else:
                words = Variable(torch.LongTensor(words))

            wordslen = data['wordslen']
            sort_info = data['sort_info']
            
            score = model.predict(words, wordslen, usecuda = self.usecuda, scoreonly = True)
            probscores.extend(list(score[np.array(sort_info)]))
            
        assert len(new_datapoints) == len(probscores)
        probs[new_datapoints] = np.array(probscores)
        
        test_indices = np.argsort(probs)
        cur_indices = set()
        i = 0
        while len(cur_indices)<num_sentences:
            cur_indices.add(test_indices[i])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('D Acquisition took %d seconds:' %(time.time()-tm))
        
    def get_mnlp_mc(self, dataset, model_path, num_sentences, nsamp=100, batch_size = 50):
        
        model = torch.load(model_path)
        model.train(True)
        tm = time.time()
        
        probs = np.ones((len(dataset),nsamp))*float('Inf')
        varsc = np.ones(len(dataset))*float('Inf')
        
        new_dataset = [datapoint for j,datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]
        
        data_batches = create_batches(new_dataset, batch_size = batch_size)
        
        varsc_outer_list = []
        probs_outer_list = []
        
        for data in data_batches:

            words = data['words']
            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
            else:
                words = Variable(torch.LongTensor(words))

            wordslen = data['wordslen']
            sort_info = data['sort_info']
            
            tag_seq_list = []
            probs_list = []
            for itr in range(nsamp):
                score, tag_seq = model.predict(words, wordslen, usecuda = self.usecuda, 
                                               scoreonly = False)
                tag_seq_new = np.array(tag_seq)[np.array(sort_info)]
                assert len(tag_seq_new) == len(words)
                tag_seq_list.append(tag_seq_new)
                probs_list.append(score[np.array(sort_info)])
            
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
                
        cur_indices = set()
        i = 0
        while len(cur_indices)<num_sentences:
            cur_indices.add(test_indices[i])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('*'*80)
        print ('MC Acquisition took %d seconds:' %(time.time()-tm))
        print ('*'*80)
        
    def get_mnlp_bb(self, dataset, model_path, num_sentences, nsamp=100, batch_size = 50):
        
        model = torch.load(model_path)
        model.train(True)
        tm = time.time()
        
        probs = np.ones((len(dataset),nsamp))*float('Inf')
        varsc = np.ones(len(dataset))*float('Inf')
        
        new_dataset = [datapoint for j,datapoint in enumerate(dataset) if j not in self.train_index]
        new_datapoints = [j for j in range(len(dataset)) if j not in self.train_index]
        
        data_batches = create_batches(new_dataset, batch_size = batch_size)
        
        varsc_outer_list = []
        probs_outer_list = []
        
        for data in data_batches:

            words = data['words']
            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
            else:
                words = Variable(torch.LongTensor(words))

            wordslen = data['wordslen']
            sort_info = data['sort_info']
            
            tag_seq_list = []
            probs_list = []
            for itr in range(nsamp):
                score, tag_seq = model.predict(words, wordslen, usecuda = self.usecuda, 
                                               scoreonly = False)
                tag_seq_new = np.array(tag_seq)[np.array(sort_info)]
                assert len(tag_seq_new) == len(words)
                tag_seq_list.append(tag_seq_new)
                probs_list.append(score[np.array(sort_info)])
            
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
        test_indices = np.lexsort((varsc, probsmean))
                
        cur_indices = set()
        i = 0
        while len(cur_indices)<num_sentences:
            cur_indices.add(test_indices[i])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('*'*80)
        print ('BB Acquisition took %d seconds:' %(time.time()-tm))
        print ('*'*80)
        
    def obtain_data(self, data, model_path=None, model_name=None, acquire=2, method='random', num_samples=100):
        
        num_sentences = (acquire*self.sentencelen)/100
        
        if model_path is None or model_name is None:
            method = 'random'
        
        if method=='random':
            self.get_random(data, num_sentences)
        else:
            if self.acq_mode == 'd':
                if method=='mnlp':
                    self.get_mnlp(data, model_path, num_sentences)
                else:
                    raise NotImplementedError()
            elif self.acq_mode == 'm':
                if method=='mnlp':
                    self.get_mnlp_mc(data, model_path, num_sentences, nsamp = num_samples)
                else:
                    raise NotImplementedError()
            elif self.acq_mode == 'b':
                if method=='mnlp':
                    self.get_mnlp_bb(data, model_path, num_sentences, nsamp = num_samples)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()