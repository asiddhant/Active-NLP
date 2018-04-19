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
    
    def __init__(self):
        pass
    
    def word_mapping(self, dataset):
        
        words = [[x.lower() for x in s[0].split()] for s in dataset]
        dico = create_dico(words)

        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000
        dico = {k:v for k,v in dico.items() if v>=2}
        word_to_id, id_to_word = create_mapping(dico)

        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        return dico, word_to_id, id_to_word
    
    def load_trec(self, datapath, pretrained, word_dim = 100):
        
        trainpath = os.path.join(datapath, 'train_5500.label')
        testpath = os.path.join(datapath, 'TREC_10.label')
        
        train_data = []
        with open (trainpath) as f:
            for line in f:
                content = line.strip().split(' ')
                sentence = ' '.join(content[1:])
                tag = content[0].split(':')[0]
                train_data.append((sentence, tag))        
        
        test_data = []
        with open (testpath) as f:
            for line in f:
                content = line.strip().split(' ')
                sentence = ' '.join(content[1:])
                tag = content[0].split(':')[0]
                test_data.append((sentence, tag))  
                
        dico_words_train = self.word_mapping(train_data)[0]
        
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
        
           
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                     
        return train_data_final, test_data_final, mappings
    
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
        
        dico_words_train = self.word_mapping(train_data)[0]
                
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
        
        
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                
        return train_data_final, test_data_final, mappings