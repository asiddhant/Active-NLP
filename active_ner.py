from __future__ import print_function
from collections import OrderedDict
import os
import neural_ner
from neural_ner.util import Trainer, Loader
from neural_ner.models import CNN_BiLSTM_CRF
from neural_ner.models import CNN_BiLSTM_CRF_MC
from neural_ner.models import CNN_BiLSTM_CRF_BB
from neural_ner.models import CNN_CNN_LSTM
from neural_ner.models import CNN_CNN_LSTM_MC
from neural_ner.models import CNN_CNN_LSTM_BB
import matplotlib.pyplot as plt
import torch
from active_learning import Acquisition
import cPickle as pkl
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='conll', type=str,
                    help='Dataset to be Used')
parser.add_argument('--result_path', action='store', dest='result_path', default='neural_ner/results/',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--usemodel', default='CNN_BiLSTM_CRF', type=str, dest='usemodel',
                    help='Model to Use')
parser.add_argument('--worddim', default=100, type=int, dest='worddim',
                    help="Word Embedding Dimension")
parser.add_argument('--pretrnd', default="wordvectors/glove.6B.100d.txt", type=str, dest='pretrnd',
                    help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int, dest='reload',
                    help="Reload the last saved model")
parser.add_argument('--checkpoint', default=".", type=str, dest='checkpoint',
                    help="Location of trained Model")
parser.add_argument('--num_epochs', default=20, type=int, dest='num_epochs',
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

if opt.usemodel == 'CNN_BiLSTM_CRF':
    parameters['lower'] = 1
    parameters['zeros'] = 0
    parameters['cpdim'] = 0
    parameters['dpout'] = 0.5
    parameters['chdim'] = 25
    parameters['tgsch'] = 'iobes'

    parameters['wldim'] = 200
    parameters['cldim'] = 25
    parameters['cnchl'] = 25
    
    parameters['lrate'] = 0.015
    parameters['batch_size'] = 16
    parameters['acqmd'] = 'd'
    
elif opt.usemodel == 'CNN_BiLSTM_CRF_MC':
    parameters['lower'] = 1
    parameters['zeros'] = 0
    parameters['cpdim'] = 0
    parameters['dpout'] = 0.5
    parameters['chdim'] = 25
    parameters['tgsch'] = 'iobes'

    parameters['wldim'] = 200
    parameters['cldim'] = 25
    parameters['cnchl'] = 25
    
    parameters['lrate'] = 0.015
    parameters['batch_size'] = 16
    parameters['acqmd'] = 'm'
    
elif opt.usemodel == 'CNN_BiLSTM_CRF_BB':
    parameters['lower'] = 1
    parameters['zeros'] = 0
    parameters['cpdim'] = 0
    parameters['dpout'] = 0.5
    parameters['chdim'] = 25
    parameters['tgsch'] = 'iobes'

    parameters['wldim'] = 200
    parameters['cldim'] = 25
    parameters['cnchl'] = 25
    
    parameters['lrate'] = 0.015
    parameters['batch_size'] = 16
    parameters['sigmp'] = float(np.exp(-3))
    parameters['acqmd'] = 'b'

elif opt.usemodel == 'CNN_CNN_LSTM':
    parameters['lower'] = 1
    parameters['zeros'] = 0
    parameters['cpdim'] = 0
    parameters['dpout'] = 0.5
    parameters['chdim'] = 25
    parameters['tgsch'] = 'iobes'
    
    parameters['wdchl'] = 200
    parameters['cldim'] = 25
    parameters['cnchl'] = 50
    parameters['dchid'] = 50
    
    parameters['lrate'] = 0.01
    parameters['batch_size'] = 16
    parameters['acqmd'] = 'd'
    
elif opt.usemodel == 'CNN_CNN_LSTM_MC':
    parameters['lower'] = 1
    parameters['zeros'] = 0
    parameters['cpdim'] = 0
    parameters['dpout'] = 0.5
    parameters['chdim'] = 25
    parameters['tgsch'] = 'iobes'
    
    parameters['wdchl'] = 200
    parameters['cldim'] = 25
    parameters['cnchl'] = 50
    parameters['dchid'] = 50
    
    parameters['lrate'] = 0.01
    parameters['batch_size'] = 16
    parameters['acqmd'] = 'm'

elif opt.usemodel == 'CNN_CNN_LSTM_BB':
    parameters['lower'] = 1
    parameters['zeros'] = 0
    parameters['cpdim'] = 0
    parameters['dpout'] = 0.5
    parameters['chdim'] = 25
    parameters['tgsch'] = 'iobes'
    
    parameters['wdchl'] = 125
    parameters['cldim'] = 25
    parameters['cnchl'] = 50
    parameters['dchid'] = 50
    
    parameters['lrate'] = 0.01
    parameters['batch_size'] = 10
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

if opt.dataset == 'conll':
    train_data, dev_data, test_data, test_train_data, mappings = loader.load_conll(dataset_path, parameters)
elif opt.dataset == 'ontonotes':
    train_data, dev_data, test_data, mappings = loader.load_ontonotes(dataset_path, parameters)
    test_train_data = train_data[-10000:]
else:
    raise NotImplementedError()

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
char_to_id = mappings['char_to_id']
word_embeds = mappings['word_embeds']

print('Load Complete')

total_tokens = sum([len(x['words']) for x in train_data])
avail_budget = total_tokens

print('Building Model............................................................................')
if (model_name == 'CNN_BiLSTM_CRF'):
    print ('CNN_BiLSTM_CRF')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    char_vocab_size = len(char_to_id)
    char_embedding_dim = parameters['chdim']
    char_out_channels = parameters['cnchl']

    model = CNN_BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                           char_embedding_dim, char_out_channels, tag_to_id, pretrained = word_embeds)

elif (model_name == 'CNN_BiLSTM_CRF_MC'):
    print ('CNN_BiLSTM_CRF_MC')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    char_vocab_size = len(char_to_id)
    char_embedding_dim = parameters['chdim']
    char_out_channels = parameters['cnchl']

    model = CNN_BiLSTM_CRF_MC(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                           char_embedding_dim, char_out_channels, tag_to_id, pretrained = word_embeds)

elif (model_name == 'CNN_BiLSTM_CRF_BB'):
        print ('CNN_BiLSTM_CRF_BB')
        word_vocab_size = len(word_to_id)
        word_embedding_dim = parameters['wrdim']
        word_hidden_dim = parameters['wldim']
        char_vocab_size = len(char_to_id)
        char_embedding_dim = parameters['chdim']
        char_out_channels = parameters['cnchl']
        sigma_prior = parameters['sigmp']

        model = CNN_BiLSTM_CRF_BB(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                               char_embedding_dim, char_out_channels, tag_to_id, sigma_prior=sigma_prior,
                                  pretrained = word_embeds)

elif (model_name == 'CNN_CNN_LSTM'):
    print ('CNN_CNN_LSTM')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_out_channels = parameters['wdchl']
    char_vocab_size = len(char_to_id)
    char_embedding_dim = parameters['chdim']
    char_out_channels = parameters['cnchl']
    decoder_hidden_units = parameters['dchid']

    model = CNN_CNN_LSTM(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size, 
                         char_embedding_dim, char_out_channels, decoder_hidden_units,
                         tag_to_id, pretrained = word_embeds)

elif (model_name == 'CNN_CNN_LSTM_MC'):
    print ('CNN_CNN_LSTM_MC')
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_out_channels = parameters['wdchl']
    char_vocab_size = len(char_to_id)
    char_embedding_dim = parameters['chdim']
    char_out_channels = parameters['cnchl']
    decoder_hidden_units = parameters['dchid']

    model = CNN_CNN_LSTM_MC(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size, 
                            char_embedding_dim, char_out_channels, decoder_hidden_units,
                            tag_to_id, pretrained = word_embeds)
    
elif (model_name == 'CNN_CNN_LSTM_BB'):
        print ('CNN_CNN_LSTM_BB')
        word_vocab_size = len(word_to_id)
        word_embedding_dim = parameters['wrdim']
        word_out_channels = parameters['wdchl']
        char_vocab_size = len(char_to_id)
        char_embedding_dim = parameters['chdim']
        char_out_channels = parameters['cnchl']
        decoder_hidden_units = parameters['dchid']
        sigma_prior = parameters['sigmp']

        model = CNN_CNN_LSTM_BB(word_vocab_size, word_embedding_dim, word_out_channels, char_vocab_size, 
                                char_embedding_dim, char_out_channels, decoder_hidden_units,
                                tag_to_id, sigma_prior = sigma_prior, pretrained = word_embeds)

if model_load:
    print ('Loading Saved Data points....................................................................')
    acquisition_path = os.path.join(result_path, model_name, 'active_checkpoint', acquire_method,
                                    checkpoint, 'acquisition2.p')
    acquisition_function = pkl.load(open(acquisition_path,'rb'))
    
else:       
    acquisition_function = Acquisition(train_data, init_percent=init_percent, seed=0, 
                                           acq_mode = parameters['acqmd'])
    
model.cuda()
learning_rate = parameters['lrate']
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

trainer = Trainer(model, optimizer, result_path, model_name, usedataset=opt.dataset, mappings= mappings)

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
    losses, all_F = trainer.train_model(opt.num_epochs, active_train_data, dev_data, test_train_data, test_data,
                                        learning_rate = learning_rate, checkpoint_folder = checkpoint_folder,
                                        batch_size = min(parameters['batch_size'],len(acquisition_function.train_index)/100),
                                        eval_test_train=False, plot_every = acq_plot_every, lr_decay = 0.05)
    
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