from __future__ import print_function
import os
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from .utils import *
import codecs
import pickle as pkl
import itertools

class Loader(object):
    
    def __init__(self):
        pass
    
    def load_conll12srl(self, dataset, parameters):
        
        word_dim = parameters['wrdim']
        pretrained = parameters['ptrnd']
        
        srl_data = pkl.load(open(os.path.join(dataset,'srl_data.p'),'rb'))
        
        srl_train_data = srl_data[:148357]
        srl_val_data = srl_data[297357:317241]
        srl_test_data = srl_data[338241:354000]
        
        dico_words_train, _, _ = word_mapping(srl_train_data)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(srl_train_data+srl_val_data+srl_test_data)

        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                                             dico_words_train.copy(), pretrained,
                                             list(itertools.chain.from_iterable(
                                             [[str(x).lower() for x in s[0]] for s in srl_val_data+srl_test_data])))

        train_data = prepare_dataset(srl_train_data, word_to_id, tag_to_id)
        dev_data = prepare_dataset(srl_val_data, word_to_id, tag_to_id)
        test_data = prepare_dataset(srl_test_data, word_to_id, tag_to_id)

        print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))
        
        mapping_file = os.path.join(dataset,'mapping1.pkl')
        
        if not os.path.isfile(mapping_file):
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

            with open(mapping_file, 'wb') as f:
                mappings = {
                    'word_to_id': word_to_id,
                    'id_to_word': id_to_word,
                    'tag_to_id': tag_to_id,
                    'id_to_tag': id_to_tag,
                    'word_embeds': word_embeds
                    }
                pkl.dump(mappings, f)
        else:
            mappings = pkl.load(open(mapping_file,'rb'))

        return train_data, dev_data, test_data, mappings
    
    def load_conll05srl(self, dataset, parameters):
        
        word_dim = parameters['wrdim']
        pretrained = parameters['ptrnd']
        
        srl_train_data = pkl.load(open(os.path.join(dataset,'train_data.pkl'),'rb'),errors='ignore')
        srl_val_data = pkl.load(open(os.path.join(dataset,'dev_data.pkl'),'rb'),errors='ignore')
        srl_test_data = pkl.load(open(os.path.join(dataset,'test_data.pkl'),'rb'),errors='ignore')
        
        for i in range(len(srl_train_data)):
            srl_train_data[i][2][-1] = srl_train_data[i][2][-1].strip()
        for i in range(len(srl_val_data)):
            srl_val_data[i][2][-1] = srl_val_data[i][2][-1].strip()
        for i in range(len(srl_test_data)):
            srl_test_data[i][2][-1] = srl_test_data[i][2][-1].strip()
        
        dico_words_train, _, _ = word_mapping(srl_train_data)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(srl_train_data+srl_val_data+srl_test_data)

        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                                             dico_words_train.copy(), pretrained,
                                             list(itertools.chain.from_iterable(
                                             [[str(x).lower() for x in s[0]] for s in srl_val_data+srl_test_data])))

        train_data = prepare_dataset(srl_train_data, word_to_id, tag_to_id)
        dev_data = prepare_dataset(srl_val_data, word_to_id, tag_to_id)
        test_data = prepare_dataset(srl_test_data, word_to_id, tag_to_id)

        print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))
        
        mapping_file = os.path.join(dataset,'mapping.pkl')
        
        if not os.path.isfile(mapping_file):
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

            with open(mapping_file, 'wb') as f:
                mappings = {
                    'word_to_id': word_to_id,
                    'id_to_word': id_to_word,
                    'tag_to_id': tag_to_id,
                    'id_to_tag': id_to_tag,
                    'word_embeds': word_embeds
                    }
                pkl.dump(mappings, f)
        else:
            mappings = pkl.load(open(mapping_file,'rb'))

        return train_data, dev_data, test_data, mappings
