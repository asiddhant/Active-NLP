from __future__ import print_function
from collections import OrderedDict
import os
import neural_ner
from neural_ner.util import Trainer, Loader
from neural_ner.models import CNN_BiLSTM_CRF
import matplotlib.pyplot as plt
import torch

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='conll', type=str,
                    help='Dataset to be Used')
parser.add_argument('--result_path', action='store', dest='result_path', default='neural_ner/results',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--usemodel', default='CNN_BiLSTM_CRF', type=str,
                    help='Model to Use')
parser.add_argument('--worddim', default=100, type=int,
                    help="Word Embedding Dimension")
parser.add_argument('--pretrnd', default="wordvectors/glove.6B.100d.txt", type=str,
                    help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int,
                    help="Reload the last saved model")
parser.add_argument('--num_epochs', default=10, type=int,
                    help="Reload the last saved model")

parameters = OrderedDict()

opt = parser.parse_args()

parameters['model'] = opt.usemodel
parameters['wrdim'] = opt.worddim
parameters['ptrnd'] = opt.pretrnd
parameters['rload'] = opt.reload

parameters['lower'] = 1
parameters['zeros'] = 0
parameters['cpdim'] = 0
parameters['dpout'] = 0.5
parameters['chdim'] = 25
parameters['tgsch'] = 'iobes'

parameters['wldim'] = 200
parameters['cldim'] = 25

parameters['wnchl'] = 200
parameters['cnchl'] = 25

dataset_path = os.path.join('datasets',opt.dataset)
result_path = opt.result_path
model_name = opt.usemodel
loader = Loader()

if opt.dataset == 'conll':
    train_data, dev_data, test_data, test_train_data, mappings = loader.load_conll(dataset_path, parameters, result_path)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

print('Load Complete')

if model_name == 'CNN_BiLSTM_CRF':
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    char_vocab_size = len(char_to_id)
    char_embedding_dim = parameters['chdim']
    char_out_channels = parameters['cnchl']
    
    model = CNN_BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                           char_embedding_dim, char_out_channels, tag_to_id, pretrained = word_embeds)
    
if parameters['rload']:
    model_path = os.path.join(result_path, model_name)
    model.load_state_dict(torch.load(model_path))
    
model.cuda()
learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

trainer = Trainer(model, optimizer, result_path, model_name, usedataset=opt.dataset, mappings= mappings) 
losses, all_F = trainer.train_single(opt.num_epochs, train_data, dev_data, test_train_data, test_data,
                                    learning_rate = learning_rate)
    
plt.plot(losses)
plt.savefig(os.path.join(result_path,'_'.join([model_name, opt.dataset, 'loss.png'])))
