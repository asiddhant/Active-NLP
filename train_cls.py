from __future__ import print_function
from collections import OrderedDict
import os
import neural_cls
from neural_cls.util import Trainer, Loader
from neural_cls.models import BiLSTM
from neural_cls.models import CNN
from neural_cls.models import CNN_BB
import matplotlib.pyplot as plt
import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='mareview', type=str,
                    help='Dataset to be Used')
parser.add_argument('--result_path', action='store', dest='result_path', default='neural_cls/results/',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--usemodel', default='CNN', type=str, dest='usemodel',
                    help='Model to Use')
parser.add_argument('--worddim', default=300, type=int, dest='worddim',
                    help="Word Embedding Dimension")
parser.add_argument('--pretrnd', default="wordvectors/glove.6B.300d.txt", type=str, dest='pretrnd',
                    help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int, dest='reload',
                    help="Reload the last saved model")
parser.add_argument('--checkpoint', default=".", type=str, dest='checkpoint',
                    help="Location of trained Model")

opt = parser.parse_args()

parameters = OrderedDict()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['ptrnd'] = opt.pretrnd

if opt.usemodel == 'BiLSTM' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6

elif opt.usemodel == 'BiLSTM' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 5
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    
elif opt.usemodel == 'CNN' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    
elif opt.usemodel == 'CNN' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 5
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    
elif opt.usemodel == 'CNN_BB' and opt.dataset == 'trec':
    parameters['wlchl'] = 100
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['sigmp'] = float(np.exp(-3))
    
elif opt.usemodel == 'CNN_BB' and opt.dataset == 'mareview':
    parameters['wlchl'] = 100
    parameters['nepch'] = 5
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['sigmp'] = float(np.exp(-3))
    
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

if opt.dataset == 'trec':
    train_data, test_data, mappings = loader.load_trec(dataset_path, parameters['ptrnd'], 
                                                       parameters['wrdim'])
elif opt.dataset == 'mareview':
    train_data, test_data, mappings = loader.load_mareview(dataset_path, parameters['ptrnd'], 
                                                       parameters['wrdim'])
else:
    raise NotImplementedError()
    
word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
word_embeds = mappings['word_embeds']

print('Load Complete')

if model_load:
    print ('Loading Saved Weights....................................................................')
    model_path = os.path.join(result_path, model_name, opt.checkpoint, 'modelweights')
    model = torch.load(model_path)
else:
    print('Building Model............................................................................')
    if (model_name == 'BiLSTM'):
        print ('BiLSTM')
        word_vocab_size = len(word_to_id)
        word_embedding_dim = parameters['wrdim']
        word_hidden_dim = parameters['wldim']
        output_size = parameters['opsiz']
        
        model = BiLSTM(word_vocab_size, word_embedding_dim, word_hidden_dim,
                       output_size, pretrained = word_embeds)
        
    elif (model_name == 'CNN'):
        print ('CNN')
        word_vocab_size = len(word_to_id)
        word_embedding_dim = parameters['wrdim']
        word_out_channels = parameters['wlchl']
        output_size = parameters['opsiz']
        
        model = CNN(word_vocab_size, word_embedding_dim, word_out_channels, 
                    output_size, pretrained = word_embeds)
        
    elif (model_name == 'CNN_BB'):
        print ('CNN_BB')
        word_vocab_size = len(word_to_id)
        word_embedding_dim = parameters['wrdim']
        word_out_channels = parameters['wlchl']
        output_size = parameters['opsiz']
        sigma_prior = parameters['sigmp']
        
        model = CNN_BB(word_vocab_size, word_embedding_dim, word_out_channels, 
                       output_size, sigma_prior=sigma_prior, pretrained = word_embeds)
    
    
model.cuda()
learning_rate = parameters['lrate']
num_epochs = parameters['nepch']
print('Initial learning rate is: %s' %(learning_rate))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer(model, optimizer, result_path, model_name, tag_to_id, usedataset=opt.dataset) 
losses, all_F = trainer.train_model(num_epochs, train_data, test_data, learning_rate,
                                    batch_size = parameters['batch_size'])
    
plt.plot(losses)
plt.savefig(os.path.join(result_path, model_name, 'lossplot.png'))
