from __future__ import print_function
import os
import re
import codecs
import torch
import copy
import numpy as np
np.random.seed(0)
import random
random.seed(0)

START_TAG = '<START>'
STOP_TAG = '<STOP>'

def pad_seq(seq, max_length, PAD_token=0):
    
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [s[2] for s in sentences]
    dico = create_dico(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def word_mapping(sentences):

    words = [[str(x).lower() for x in s[0]] for s in sentences]
    dico = create_dico(words)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3}
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def prepare_dataset(sentences, word_to_id, tag_to_id):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower()
    data = []
    for s in sentences:
        str_words = [str(x) for x in s[0]]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[t] for t in s[2]]
        verbs = [int(v) for v in s[1]]
        data.append({
            'str_words': str_words,
            'words': words,
            'caps': caps,
            'tags': tags,
            'verbs': verbs
        })
    return data

def log_sum_exp(vec, dim=-1, keepdim = False):
    max_score, _ = vec.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = vec - max_score
    else:
        stable_vec = vec - max_score.unsqueeze(dim)
    output = max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()
    return output

def create_batches(dataset, batch_size, order='keep', str_words=False, tag_padded= True):

    newdata = copy.deepcopy(dataset)
    
    if order=='sort':
        newdata.sort(key = lambda x:len(x['words']))
    elif order=='random':
        random.shuffle(newdata)

    newdata = np.array(newdata)  
    batches = []
    num_batches = np.ceil(len(dataset)/float(batch_size)).astype('int')

    for i in range(num_batches):
        batch_data = newdata[(i*batch_size):min(len(dataset),(i+1)*batch_size)]

        words_seqs = [itm['words'] for itm in batch_data]
        caps_seqs = [itm['caps'] for itm in batch_data]
        verbs_seqs = [itm['verbs'] for itm in batch_data]
        target_seqs = [itm['tags'] for itm in batch_data]
        str_words_seqs = [itm['str_words'] for itm in batch_data]

        seq_pairs = sorted(zip(words_seqs, caps_seqs, target_seqs, verbs_seqs, str_words_seqs,
                            range(len(words_seqs))), key=lambda p: len(p[0]), reverse=True)

        words_seqs, caps_seqs, target_seqs, verbs_seqs, str_words_seqs, sort_info = zip(*seq_pairs)
        words_lengths = np.array([len(s) for s in words_seqs])

        words_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in words_seqs])
        caps_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in caps_seqs])
        verbs_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in verbs_seqs])

        if tag_padded:
            target_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in target_seqs])
        else:
            target_padded = target_seqs

        words_mask = (words_padded!=0).astype('int')

        if str_words:
            outputdict = {'words':words_padded, 'caps':caps_padded, 'tags': target_padded, 
                          'verbs': verbs_padded, 'wordslen': words_lengths, 'tagsmask':words_mask, 
                          'str_words': str_words_seqs, 'sort_info': sort_info}
        else:
            outputdict = {'words':words_padded, 'caps':caps_padded, 'tags': target_padded, 
                          'verbs': verbs_padded, 'wordslen': words_lengths, 'tagsmask':words_mask,
                          'sort_info': sort_info}

        batches.append(outputdict)

    return batches
