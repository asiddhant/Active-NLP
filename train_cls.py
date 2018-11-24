from __future__ import print_function
from collections import OrderedDict
import os
import neural_cls
from neural_cls.util import Trainer, Loader
from neural_cls.models import CNN
import torch
import numpy as np
import copy

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='subj', 
                    type=str, help='Dataset to be Used')
parser.add_argument('--usemodel', default='CNN', type=str, dest='usemodel',
                    help='Model to Use')
parser.add_argument('--embeddings', type=str, dest='embeddings',
                    help='Model to Use')
parser.add_argument('--worddim', default=300, type=int, dest='worddim',
                    help="Word Embedding Dimension")
parser.add_argument('--pretrained', type=str, dest='pretrnd',
                    help="Location of pretrained embeddings")
parser.add_argument('--result_path', action='store', dest='result_path', 
                    default='results/neural_cls/',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--restart', default=0, type=int, dest='restart',
                    help="Reload from last state in training")
parser.add_argument('--use_cuda', default=1, type=int, dest='use_cuda',
                    help="Whether to use_cuda")
parser.add_argument('--checkpoint', default=".", type=str, dest='checkpoint',
                    help="Location of trained model")

opt = parser.parse_args()

parameters = OrderedDict()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['embeddings'] = opt.embeddings
parameters['pretrained'] = opt.pretrained


if opt.usemodel == 'CNN' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 20
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
else:
    raise NotImplementedError()

use_dataset = opt.dataset
dataset_path = os.path.join('datasets', use_dataset)
result_path = os.path.join(opt.result_path, use_dataset)
model_name = opt.usemodel
model_load = opt.reload
loader = Loader()

print('Model:', model_name)
print('Dataset:', use_dataset)

if not os.path.exists(result_path):
    os.makedirs(result_path)
    
if not os.path.exists(os.path.join(result_path,model_name)):
    os.makedirs(os.path.join(result_path,model_name))

if opt.dataset == 'mareview':
    train_data, test_data, mappings = loader.load_mareview(
                                      dataset_path, 
                                      parameters['ptrnd'], 
                                      parameters['wrdim'])
    valid_data = copy.deepcopy(test_data)
else:
    raise NotImplementedError()
    
word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']

print('Load Complete')

if model_load:
    print ('Loading Saved Weights............................')
    model_path = os.path.join(result_path, model_name, opt.checkpoint, 
                              'modelweights')
    model = torch.load(model_path)
else:
    print('Building Model.....................................')
    if (model_name == 'CNN'):
        print ('CNN with Embeddings %s' %(opt.embeddings))
        embeddings_type = parameters['embeddings']
        word_embedding_dim = parameters['wrdim']
        word_vocab_size = len(word_to_id)
        word_out_channels = parameters['wlchl']
        output_size = parameters['opsiz']
        
        model = CNN(embedding_type, word_embedding_dim, word_vocab_size,
                    word_out_channels, output_size)
    
use_cuda = torch.cuda.is_available() and opt.use_cuda
if use_cuda:
    model.cuda()

learning_rate = parameters['lrate']
num_epochs = parameters['nepch']
print('Initial learning rate is: %s' %(learning_rate))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer(model, optimizer, result_path, model_name, tag_to_id, 
                  usedataset=opt.dataset, usecuda = use_cuda) 
losses, all_F = trainer.train_model(num_epochs, train_data, valid_data, 
                                    test_data, learning_rate,
                                    batch_size = parameters['batch_size'])

print ('Training Complete.......................................')