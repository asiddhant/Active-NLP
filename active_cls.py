from __future__ import print_function
from collections import OrderedDict
import os
import neural_cls
from neural_cls.util import Trainer, Loader
from neural_cls.models import BiLSTM
from neural_cls.models import CNN
from neural_cls.models import BiLSTM_MC
from neural_cls.models import BiLSTM_BB
from neural_cls.models import CNN_MC
from neural_cls.models import CNN_BB
from word_language_model.lmapi import lmAPI
#import matplotlib.pyplot as plt
import torch
from active_learning import Acquisition_CLS
import cPickle as pkl
import numpy as np
import copy
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='subj', type=str,
                    help='Dataset to be Used')
parser.add_argument('--result_path', action='store', dest='result_path', default='results/neural_cls',
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
parser.add_argument('--initdata', default=1, type=int, dest='initdata',
                    help="Percentage of Data to being with")
parser.add_argument('--acquiremethod', default='random', type=str, dest='acquiremethod',
                    help="Percentage of Data to Acquire from Rest of Training Set")

parameters = OrderedDict()

opt = parser.parse_args()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['ptrnd'] = opt.pretrnd

if opt.usemodel == 'CNN' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 20
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 32
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'd'
    
elif opt.usemodel == 'CNN' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 20
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 32
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'CNN' and opt.dataset == 'subj':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 20
    
    parameters['lrate'] = 0.0001
    parameters['batch_size'] = 16
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'd'

elif opt.usemodel == 'BiLSTM' and opt.dataset == 'subj':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 20
    
    parameters['lrate'] = 0.0001
    parameters['batch_size'] = 16
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'd'
    
elif opt.usemodel == 'CNN_MC' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'm'
    
elif opt.usemodel == 'CNN_MC' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wlchl'] = 100
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'm'
    
elif opt.usemodel == 'BiLSTM_MC' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['acqmd'] = 'm'

elif opt.usemodel == 'BiLSTM_MC' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 10
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['acqmd'] = 'm'
    
elif opt.usemodel == 'CNN_BB' and opt.dataset == 'trec':
    parameters['wlchl'] = 100
    parameters['nepch'] = 25
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'
    
elif opt.usemodel == 'CNN_BB' and opt.dataset == 'mareview':
    parameters['wlchl'] = 100
    parameters['nepch'] = 25
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'
    
elif opt.usemodel == 'BiLSTM_BB' and opt.dataset == 'trec':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 25
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 6
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'
    
elif opt.usemodel == 'BiLSTM_BB' and opt.dataset == 'mareview':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 200
    parameters['nepch'] = 25
    
    parameters['lrate'] = 0.001
    parameters['batch_size'] = 50
    parameters['opsiz'] = 2
    parameters['sigmp'] = float(np.exp(-3))

    parameters['acqmd'] = 'b'
    
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

if opt.dataset == 'trec':
    train_data, test_data, mappings = loader.load_trec(dataset_path, parameters['ptrnd'], 
                                                       parameters['wrdim'])
    valid_data = copy.deepcopy(test_data)
elif opt.dataset == 'mareview':
    train_data, test_data, mappings = loader.load_mareview(dataset_path, parameters['ptrnd'], 
                                                       parameters['wrdim'])
    valid_data = copy.deepcopy(test_data)
elif opt.dataset == 'subj':
    train_data, valid_data, test_data, mappings = loader.load_subj(dataset_path, parameters['ptrnd'], 
                                                       parameters['wrdim'])
else:
    raise NotImplementedError()

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
word_embeds = mappings['word_embeds']

use_cuda = torch.cuda.is_available()

print('Load Complete')

total_sentences = len(train_data)
avail_budget = total_sentences

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

elif (model_name == 'BiLSTM_MC'):
    print ('BiLSTM_MC')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    output_size = parameters['opsiz']

    model = BiLSTM_MC(word_vocab_size, word_embedding_dim, word_hidden_dim,
                   output_size, pretrained = word_embeds)

elif (model_name == 'CNN_MC'):
    print ('CNN_MC')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_out_channels = parameters['wlchl']
    output_size = parameters['opsiz']

    model = CNN_MC(word_vocab_size, word_embedding_dim, word_out_channels, 
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
        
elif (model_name == 'BiLSTM_BB'):
    print ('BiLSTM_BB')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    output_size = parameters['opsiz']
    sigma_prior = parameters['sigmp']

    model = BiLSTM_BB(word_vocab_size, word_embedding_dim, word_hidden_dim,
                   output_size, sigma_prior=sigma_prior, pretrained = word_embeds)

if model_load:
    print ('Loading Saved Weights....................................................................')
    acquisition_path = os.path.join(result_path, model_name, 'active_checkpoint', acquire_method,
                                    checkpoint, 'acquisition2.p')
    acquisition_function = pkl.load(open(acquisition_path,'rb'))
    
else:
        
    acquisition_function = Acquisition_CLS(train_data, init_percent=init_percent, seed=0, 
                                           acq_mode = parameters['acqmd'], usecuda = use_cuda)
    
if use_cuda: model.cuda()
learning_rate = parameters['lrate']
num_epochs = parameters['nepch']
print('Initial learning rate is: %s' %(learning_rate))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if acquire_method == 'lm_parallel':
    lmapi = lmAPI(mappings)
else:
    lmapi = None

active_train_data = [train_data[i] for i in acquisition_function.train_index]
sentences_acquired = len(acquisition_function.train_index)

num_acquisitions_required = 50
acquisition_strat_all = [1]*49 + [0]
acquisition_strat = acquisition_strat_all[:num_acquisitions_required]

for acquire_percent in acquisition_strat:
    
    checkpoint_folder = os.path.join('active_checkpoint',acquire_method, str(sentences_acquired).zfill(8))
    checkpoint_path = os.path.join(result_path, model_name, checkpoint_folder)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    acq_plot_every = max(len(acquisition_function.train_index)/(5*parameters['batch_size']),1)
    adj_batch_size = min(parameters['batch_size'], max(2,sentences_acquired/10))
    trainer = Trainer(model, optimizer, result_path, model_name, tag_to_id, usedataset=opt.dataset,
                      usecuda = use_cuda) 
    losses, all_F = trainer.train_model(num_epochs, active_train_data, valid_data, test_data, learning_rate,
                                        batch_size = adj_batch_size, checkpoint_folder = checkpoint_folder,
                                        plot_every = acq_plot_every)

    if acquire_method == 'lm_parallel':
        lmapi.train_lm(active_train_data, checkpoint_folder = os.path.join(checkpoint_path,'lmweights'))
    
    pkl.dump(acquisition_function, open(os.path.join(checkpoint_path,'acquisition1.p'),'wb'))
    
    saved_file_name = 'modelweights' if acquire_method != 'lm_parallel' else 'lmweights'
    acquisition_function.obtain_data(model_path = os.path.join(checkpoint_path ,saved_file_name),
                                     model_name = model_name, data = train_data, acquire = acquire_percent, 
                                     method =acquire_method, lm_api = lmapi)
    
    pkl.dump(acquisition_function, open(os.path.join(checkpoint_path,'acquisition2.p'),'wb'))
    
    print ('*'*80)
    saved_epoch = np.argmax(np.array([item[1] for item in all_F]))
    print ('Budget Exhausted: %d, Best F on Train %.3f, Best F on Dev %.3f, Best F on Test %.3f' 
            %(sentences_acquired, all_F[saved_epoch][0], all_F[saved_epoch][1], all_F[saved_epoch][2]))
    print ('*'*80)
    
    active_train_data = [train_data[i] for i in acquisition_function.train_index]
    sentences_acquired = len(acquisition_function.train_index)
    
    #plt.clf()
    #plt.plot(losses)
    #plt.savefig(os.path.join(checkpoint_path,'lossplot.png'))
