import torch
torch.manual_seed(0)
import numpy as np
from collections import Counter
import time

class Acquisition(object):
    
    def __init__(self, train_data, acq_mode='d', init_percent=2, seed=0):
        self.tokenlen = sum([len(x['words']) for x in train_data])
        self.train_index = set()
        self.npr = np.random.RandomState(seed)
        self.obtain_data(train_data, acquire = init_percent)
        self.acq_mode = acq_mode
        
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
                 
    def get_mnlp(self, dataset, model_path, decoder, num_tokens):
        model = torch.load(model_path)
        model.train(False)
        probs = np.ones(len(dataset))*float('Inf')
        for j, data in enumerate(dataset):
            if j not in self.train_index:
                sentence = data['words']
                tags = data['tags']
                chars = data['chars']
                caps = data['caps']
                if decoder=='CRF':
                    score, _ = model.decode(sentence, tags, chars, caps)
                elif decoder=='LSTM':
                    raise NotImplementedError()
                probs[j] = score/len(sentence)
        test_indices = np.argsort(probs)
        cur_tokens=0
        cur_indices = set()
        i = 0
        while cur_tokens<num_tokens:
            cur_indices.add(test_indices[i])
            cur_tokens += len(dataset[test_indices[i]]['words'])
            i+=1
        self.train_index.update(cur_indices)
        
    def get_mnlp_mc(self, dataset, model_path, decoder, num_tokens, nsamp=100):
        model = torch.load(model_path)
        model.train(True)
        tm = time.time()
        probs = np.ones((len(dataset),nsamp))*float('Inf')
        varsc = np.ones(len(dataset))*float('Inf')
        for j, data in enumerate(dataset):
            if j not in self.train_index:
                sentence = data['words']
                tags = data['tags']
                chars = data['chars']
                caps = data['caps']
                tag_seq_list = []
                for itr in range(nsamp):
                    if decoder=='CRF':
                        score, tag_seq = model.decode(sentence, tags, chars, caps)
                    elif decoder=='LSTM':
                        raise NotImplementedError()
                    tag_seq = [str(tg) for tg in tag_seq]
                    tag_seq_list.append('_'.join(tag_seq))
                    probs[j][itr] = score/len(sentence)
                varsc[j] = Counter(tag_seq_list).most_common(1)[0][1]
        
        probsmean = np.mean(probs, axis=1)
        test_indices = np.argsort(varsc)
        #test_indices = np.lexsort((varsc, probsmean))
        cur_tokens=0
        cur_indices = set()
        i = 0
        while cur_tokens<num_tokens:
            cur_indices.add(test_indices[i])
            cur_tokens += len(dataset[test_indices[i]]['words'])
            i+=1
        self.train_index.update(cur_indices)
        
        print ('Acquisition took %d seconds:' %(time.time()-tm))
        
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
            else:
                raise NotImplementedError()