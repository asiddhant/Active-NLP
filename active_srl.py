from __future__ import print_function
from collections import OrderedDict
import os
import neural_srl
from neural_srl.util import Trainer, Loader
from neural_srl.models import BiLSTM_CRF
from neural_srl.models import BiLSTM_CRF_MC
import matplotlib.pyplot as plt
import torch
from active_learning import Acquisition_SRL
import pickle as pkl
import numpy as np
import random
random.seed(0)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='conll12srl', type=str,
                    help='Dataset to be Used')
parser.add_argument('--result_path', action='store', dest='result_path', default='neural_srl/results/',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--usemodel', default='BiLSTM_CRF', type=str, dest='usemodel',
                    help='Model to Use')
parser.add_argument('--worddim', default=100, type=int, dest='worddim',
                    help="Word Embedding Dimension")
parser.add_argument('--pretrnd', default="wordvectors/glove.6B.100d.txt", type=str, dest='pretrnd',
                    help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int, dest='reload',
                    help="Reload the last saved model")
parser.add_argument('--checkpoint', default=".", type=str, dest='checkpoint',
                    help="Location of trained Model")
parser.add_argument('--num_epochs', default=10, type=int, dest='num_epochs',
                    help="Reload the last saved model")
parser.add_argument('--initdata', default=2, type=int, dest='initdata',
                    help="Percentage of Data to being with")
parser.add_argument('--acquiremethod', default='random', type=str, dest='acquiremethod',
                    help="Percentage of Data to Acquire from Rest of Training Set")

parameters = OrderedDict()

opt = parser.parse_args()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['ptrnd'] = opt.pretrnd

if opt.usemodel == 'BiLSTM_CRF' and opt.dataset == 'conll12srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 60
    
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM_CRF' and opt.dataset == 'conll05srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 80
    
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM_CRF_MC' and opt.dataset == 'conll12srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 60
    
    parameters['acqmd'] = 'm'
    
elif opt.usemodel == 'BiLSTM_CRF_MC' and opt.dataset == 'conll05srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 80
    
    parameters['acqmd'] = 'm'
    
else:
    raise NotImplementedError()
    

use_dataset = opt.dataset
dataset_path = os.path.join('datasets', use_dataset)
result_path = os.path.join(opt.result_path, use_dataset)
model_name = opt.usemodel
model_load = opt.reload
checkpoint = opt.checkpoint
init_percent = opt.initdata
acquire_method = opt.acquiremethod
loader = Loader()

print('Model:', model_name)
print('Dataset:', use_dataset)
print('Acquisition:', acquire_method)

if not os.path.exists(result_path):
    os.makedirs(result_path)
    
if not os.path.exists(os.path.join(result_path, model_name)):
    os.makedirs(os.path.join(result_path, model_name))

if not os.path.exists(os.path.join(result_path, model_name, 'active_checkpoint', acquire_method)):
    os.makedirs(os.path.join(result_path, model_name, 'active_checkpoint', acquire_method))

if opt.dataset == 'conll12srl':
    train_data, dev_data, test_data, mappings = loader.load_conll12srl(dataset_path, parameters)
    test_train_data = random.sample(train_data, 20000)
elif opt.dataset == 'conll05srl':
    train_data, dev_data, test_data, mappings = loader.load_conll05srl(dataset_path, parameters)
    test_train_data = random.sample(train_data, 10000)
else:
    raise NotImplementedError()

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
word_embeds = mappings['word_embeds']

print('Load Complete')

total_tokens = sum([len(x['words']) for x in train_data])
avail_budget = total_tokens

print('Building Model............................................................................')
if (model_name == 'BiLSTM_CRF'):
    print ('BiLSTM_CRF')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    verb_embedding_dim = parameters['vbdim']
    cap_embedding_dim = parameters['cpdim']

    model = BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, tag_to_id, 
                       verb_embedding_dim, cap_embedding_dim, pretrained = word_embeds)

elif (model_name == 'BiLSTM_CRF_MC'):
    print ('BiLSTM_CRF_MC')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    verb_embedding_dim = parameters['vbdim']
    cap_embedding_dim = parameters['cpdim']

    model = BiLSTM_CRF_MC(word_vocab_size, word_embedding_dim, word_hidden_dim, tag_to_id, 
                       verb_embedding_dim, cap_embedding_dim, pretrained = word_embeds)
if model_load:
    print ('Loading Saved Data points....................................................................')
    acquisition_path = os.path.join(result_path, model_name, 'active_checkpoint', acquire_method,
                                    checkpoint, 'acquisition2.p')
    acquisition_function = pkl.load(open(acquisition_path,'rb'))
    
else:       
    acquisition_function = Acquisition_SRL(train_data, init_percent=init_percent, seed=0, 
                                           acq_mode = parameters['acqmd'])
    
model.cuda()
learning_rate = parameters['lrate']
print('Initial learning rate is: %s' %(learning_rate))
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-6)

active_train_data = [train_data[i] for i in acquisition_function.train_index]
tokens_acquired = sum([len(x['words']) for x in active_train_data])

num_acquisitions_required = 25
acquisition_strat_all = [2]*24 + [5]*10 + [0]
acquisition_strat = acquisition_strat_all[:num_acquisitions_required]

for acquire_percent in acquisition_strat:
    
    checkpoint_folder = os.path.join('active_checkpoint',acquire_method, str(tokens_acquired).zfill(8))
    checkpoint_path = os.path.join(result_path, model_name, checkpoint_folder)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    acq_plot_every = max(len(acquisition_function.train_index)/(5*parameters['batch_size']),1)
    trainer = Trainer(model, optimizer, result_path, model_name, usedataset=opt.dataset, mappings= mappings)
    losses, all_F = trainer.train_model(opt.num_epochs, active_train_data, dev_data, test_train_data, test_data,
                                        learning_rate = learning_rate, checkpoint_folder = checkpoint_folder,
                                        batch_size = min(parameters['batch_size'],int(len(acquisition_function.train_index)/200)),
                                        eval_test_train=False, plot_every = int(acq_plot_every), adjust_lr=False)
    
    pkl.dump(acquisition_function, open(os.path.join(checkpoint_path,'acquisition1.p'),'wb'))
    
    acquisition_function.obtain_data(model_path = os.path.join(checkpoint_path ,'modelweights'), model_name = model_name,
                                     data = train_data, acquire = acquire_percent, method=acquire_method)
    
    pkl.dump(acquisition_function, open(os.path.join(checkpoint_path,'acquisition2.p'),'wb'))
    
    print ('*'*80)
    saved_epoch = np.argmax(np.array([item[1] for item in all_F]))
    print ('Budget Exhausted: %d, Best F on Validation %.3f, Best F on Test %.3f' %(tokens_acquired,
                                        all_F[saved_epoch][1], all_F[saved_epoch][2]))
    print ('*'*80)
    
    active_train_data = [train_data[i] for i in acquisition_function.train_index]
    tokens_acquired = sum([len(x['words']) for x in active_train_data])
    
    plt.clf()
    plt.plot(losses)
    plt.savefig(os.path.join(checkpoint_path,'lossplot.png'))