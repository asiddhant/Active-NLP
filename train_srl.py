from __future__ import print_function
from collections import OrderedDict
import os
import neural_srl
from neural_srl.util import Trainer, Loader
from neural_srl.models import BiLSTM_CRF
import matplotlib.pyplot as plt
import torch
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
parser.add_argument('--num_epochs', default=20, type=int, dest='num_epochs',
                    help="Reload the last saved model")

opt = parser.parse_args()

parameters = OrderedDict()

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

elif opt.usemodel == 'BiLSTM_CRF' and opt.dataset == 'conll05srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 80

elif opt.usemodel == 'BiLSTM_CRF_MC' and opt.dataset == 'conll12srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 60
    
elif opt.usemodel == 'BiLSTM_CRF_MC' and opt.dataset == 'conll05srl':
    parameters['dpout'] = 0.5
    parameters['wldim'] = 300
    parameters['vbdim'] = 100
    parameters['cpdim'] = 0
    
    parameters['lrate'] = 1.0
    parameters['batch_size'] = 80
    
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

if opt.dataset == 'conll12srl':
    train_data, dev_data, test_data, mappings = loader.load_conll12srl(dataset_path, parameters)
    test_train_data = random.sample(train_data, 20000)
elif opt.dataset == 'conll05srl':
    train_data, dev_data, test_data, mappings = loader.load_conll05srl(dataset_path, parameters)
    test_train_data = random.sample(train_data, 20000)
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
    
    
model.cuda()
learning_rate = parameters['lrate']
print('Initial learning rate is: %s' %(learning_rate))
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.95, eps=1e-6)

trainer = Trainer(model, optimizer, result_path, model_name, usedataset=opt.dataset, mappings= mappings) 
losses, all_F = trainer.train_model(opt.num_epochs, train_data, dev_data, test_data, test_train_data,
                                    learning_rate = learning_rate, batch_size = parameters['batch_size'])
    
plt.plot(losses)
plt.savefig(os.path.join(result_path, model_name, 'lossplot.png'))
