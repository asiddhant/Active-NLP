import torch
import numpy as np

class Acquisition(object):
    def __init__(self, train_data, init_percent=2, seed=0):
        self.tokenlen = sum([len(x['words']) for x in train_data])
        self.train_index = set()
        self.npr = np.random.RandomState(seed)
        self.obtain_data(train_data, acquire = init_percent)
        
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
                else:
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
        
    def get_bald(self, data, model_path, num_tokens, num_samples=10):
        raise NotImplementedError()
        
    def obtain_data(self, data, model_path=None, model_name=None, acquire=2, method='random', num_samples=10):
        num_tokens = (acquire*self.tokenlen)/100
        if model_path is None or model_name is None:
            method = 'random'
        
        if method=='random':
            self.get_random(data, num_tokens)
        else:
            decoder = model_name.split('_')[2]
            if method=='mnlp':
                self.get_mnlp(data, model_path, decoder, num_tokens)
            elif method=='bald':
                self.get_bald(data, model_path, decoder, num_tokens, num_samples)
            else:
                raise NotImplementedError()
            
        
        