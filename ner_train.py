from __future__ import print_function
from collections import OrderedDict
import os
import neural_ner
from neural_ner.util import Trainer, Loader
from neural_ner.models import CNN_BiLSTM_CRF
from neural_ner.models import CNN_CNN_LSTM
import matplotlib.pyplot as plt
import torch

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
parser.add_argument('--num_epochs', default=10, type=int, dest='num_epochs',
                    help="Reload the last saved model")

parameters = OrderedDict()

opt = parser.parse_args()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['ptrnd'] = opt.pretrnd

parameters['lower'] = 1
parameters['zeros'] = 0
parameters['cpdim'] = 0
parameters['dpout'] = 0.5
parameters['chdim'] = 25
parameters['tgsch'] = 'iobes'

parameters['wldim'] = 200
parameters['cldim'] = 25

parameters['w1chl'] = 50
parameters['w2chl'] = 200
parameters['cnchl'] = 25
parameters['dchid'] = 200

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

if opt.dataset == 'conll':
    train_data, dev_data, test_data, test_train_data, mappings = loader.load_conll(dataset_path, parameters)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

print('Load Complete')

if model_load:
    print ('Loading Saved Weights....................................................................')
    model_path = os.path.join(result_path, model_name, opt.checkpoint, 'modelweights')
    model = torch.load(model_path)
else:
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

    elif (model_name == 'CNN_CNN_LSTM'):
        print ('CNN_CNN_LSTM')
        word_vocab_size = len(word_to_id)
        word_embedding_dim = parameters['wrdim']
        word_out1_channels = parameters['w1chl']
        word_out2_channels = parameters['w2chl']
        char_vocab_size = len(char_to_id)
        char_embedding_dim = parameters['chdim']
        char_out_channels = parameters['cnchl']
        decoder_hidden_units = parameters['dchid']

        model = CNN_CNN_LSTM(word_vocab_size, word_embedding_dim, word_out1_channels, word_out2_channels,
                             char_vocab_size, char_embedding_dim, char_out_channels, decoder_hidden_units,
                             tag_to_id, pretrained = word_embeds)
    
model.cuda()
learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

trainer = Trainer(model, optimizer, result_path, model_name, usedataset=opt.dataset, mappings= mappings) 
losses, all_F = trainer.train_single(opt.num_epochs, train_data, dev_data, test_train_data, test_data,
                                    learning_rate = learning_rate)
    
plt.plot(losses)
plt.savefig(os.path.join(result_path, model_name, 'lossplot.png'))