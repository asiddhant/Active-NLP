from __future__ import print_function
import os
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from utils import *
import codecs
import cPickle
import itertools
from collections import Counter

class Loader(object):
    
    def load_mareview(self, datapath, pretrained, word_dim = 100):
        
        trainpospath = os.path.join(datapath, 'train-rt-polarity.pos')
        trainnegpath = os.path.join(datapath, 'train-rt-polarity.neg')
        
        testpospath = os.path.join(datapath, 'test-rt-polarity.pos')
        testnegpath = os.path.join(datapath, 'test-rt-polarity.neg')
        
        train_pos_data = []
        with open (trainpospath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 1
                train_pos_data.append((sentence, tag))        
        
        train_neg_data = []
        with open (trainnegpath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 0
                train_neg_data.append((sentence, tag))
                
        test_pos_data = []
        with open (testpospath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 1
                test_pos_data.append((sentence, tag))        
        
        test_neg_data = []
        with open (testnegpath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 0
                test_neg_data.append((sentence, tag))
                
        train_data = train_pos_data + train_neg_data
        test_data = test_pos_data + test_neg_data
        
        dico_words_train = word_mapping(train_data)[0]
                
        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in test_data])
                ) if not all_embedding else None)
        
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data)
        
        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)

        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag
        }
                
        return train_data_final, test_data_final, mappings