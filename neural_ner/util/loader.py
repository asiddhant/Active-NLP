from __future__ import print_function
import os
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from .utils import *
import codecs
import cPickle
import itertools

class Loader(object):
    
    def __init__(self):
        pass
    
    def pad_sequence_cnn(self, chars):
        d = {}
        chars_length = [len(c) for c in chars]
        chars_maxlen = max(chars_length)
        chars_mask = np.zeros((len(chars_length), chars_maxlen), dtype='int')
        for i, c in enumerate(chars):
            chars_mask[i, :chars_length[i]] = c
        return chars_mask, chars_length, d
    
    
    def pad_sequence_rnn(self, chars):
        chars_sorted = sorted(chars, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars):
            for j, cj in enumerate(chars_sorted):
                if ci == cj and not j in d and not i in d.values():
                    d[j] = i
                    continue
        chars_length = [len(c) for c in chars_sorted]
        chars_maxlen = max(chars_length)
        chars_mask = np.zeros((len(chars_sorted), char_maxlen), dtype='int')
        for i, c in enumerate(chars_sorted):
            chars_mask[i, :chars_length[i]] = c
        return chars_mask, chars_length, d
    
    def update_tag_scheme(self, sentences, tag_scheme):
        
        for i, s in enumerate(sentences):
            tags = [w[-1] for w in s]
            if not iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'iob':
                for word, new_tag in zip(s, tags):
                    word[-1] = new_tag
            elif tag_scheme == 'iobes':
                new_tags = iob_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[-1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')
                
    def word_mapping(self, sentences, lower):
        
        words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
        dico = create_dico(words)

        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000
        dico = {k:v for k,v in dico.items() if v>=3}
        word_to_id, id_to_word = create_mapping(dico)

        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        return dico, word_to_id, id_to_word
    
    def load_conll_sentences(self, path, lower, zeros):
        
        sentences = []
        sentence = []
        for line in codecs.open(path, 'r', 'utf-8'):
            line = zero_digits(line.rstrip()) if zeros else line.rstrip()
            if not line:
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:
                        sentences.append(sentence)
                    sentence = []
            else:
                word = line.split()
                assert len(word) >= 2
                sentence.append(word)
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
        return sentences
    
    def load_conll(self, dataset ,parameters):
        
        zeros = parameters['zeros']
        lower = parameters['lower']
        word_dim = parameters['wrdim']
        pretrained = parameters['ptrnd']
        tag_scheme = parameters['tgsch']
        
        train_path = os.path.join(dataset,'eng.train')
        dev_path = os.path.join(dataset,'eng.testa')
        test_path = os.path.join(dataset,'eng.testb')
        test_train_path = os.path.join(dataset,'eng.train54019')
        
        train_sentences = self.load_conll_sentences(train_path, lower, zeros)
        dev_sentences = self.load_conll_sentences(dev_path, lower, zeros)
        test_sentences = self.load_conll_sentences(test_path, lower, zeros)
        test_train_sentences = self.load_conll_sentences(test_train_path, lower, zeros)
        
        self.update_tag_scheme(train_sentences, tag_scheme)
        self.update_tag_scheme(dev_sentences, tag_scheme)
        self.update_tag_scheme(test_sentences, tag_scheme)
        self.update_tag_scheme(test_train_sentences, tag_scheme)
        
        dico_words_train = self.word_mapping(train_sentences, lower)[0]
        
        all_embedding = 1
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in dev_sentences + test_sentences])
                ) if not all_embedding else None)

        dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        
        train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower)
        dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower)
        test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)
        test_train_data = prepare_dataset(test_train_sentences, word_to_id, char_to_id, tag_to_id, lower)
        
        print("%i / %i / %i sentences in train / dev / test." % (
              len(train_data), len(dev_data), len(test_data)))
        
        mapping_file = os.path.join(dataset,'mapping_'+ str(tag_scheme) +'.pkl')
        
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
                    'tag_to_id': tag_to_id,
                    'id_to_tag': id_to_tag,
                    'char_to_id': char_to_id,
                    'parameters': parameters,
                    'word_embeds': word_embeds
                }
                cPickle.dump(mappings, f)
        else:
            mappings = cPickle.load(open(mapping_file,'rb'))
            
        return train_data, dev_data, test_data, test_train_data, mappings
        
    def load_ontonotes(self, dataset ,parameters):
        
        zeros = parameters['zeros']
        lower = parameters['lower']
        word_dim = parameters['wrdim']
        pretrained = parameters['ptrnd']
        tag_scheme = parameters['tgsch']
        
        train_path = os.path.join(dataset,'eng.train')
        dev_path = os.path.join(dataset,'eng.testa')
        test_path = os.path.join(dataset,'eng.testb')
        
        train_sentences = self.load_conll_sentences(train_path, lower, zeros)
        dev_sentences = self.load_conll_sentences(dev_path, lower, zeros)
        test_sentences = self.load_conll_sentences(test_path, lower, zeros)
        
        self.update_tag_scheme(train_sentences, tag_scheme)
        self.update_tag_scheme(dev_sentences, tag_scheme)
        self.update_tag_scheme(test_sentences, tag_scheme)
        
        dico_words_train = self.word_mapping(train_sentences, lower)[0]
        
        all_embedding = 1
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in dev_sentences + test_sentences])
                ) if not all_embedding else None)

        dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        
        train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower)
        dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower)
        test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)
        
        print("%i / %i / %i sentences in train / dev / test." % (
              len(train_data), len(dev_data), len(test_data)))
        
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
                    'tag_to_id': tag_to_id,
                    'id_to_tag': id_to_tag,
                    'char_to_id': char_to_id,
                    'parameters': parameters,
                    'word_embeds': word_embeds
                }
                cPickle.dump(mappings, f)
        else:
            mappings = cPickle.load(open(mapping_file,'rb'))
            
        return train_data, dev_data, test_data, mappings
        