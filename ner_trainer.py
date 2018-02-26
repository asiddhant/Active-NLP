from __future__ import print_function
from collections import OrderedDict
import os
import neural_ner
from neural_ner.util import Trainer, Loader
from neural_ner.models import CNN_BiLSTM_CRF
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset_path', default='conll'
                    help='Dataset to be Used')
parser.add_argument('--result_path', action='store', dest='result_path', default='neural_ner/results'
                    help='Path to Save/Load Result')
parser.add_argument('--usemodel', default='CNN_BiLSTM_CRF', 
                    help='Model to Use')
parser.add_argument('--pretrnd', default="wordvectors/glove.6B.100d.txt",
                    help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, 
                    help="Reload the last saved model")
parser.add_argument('--num_epochs', default=100, 
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

dataset_path = os.path.join('dataset',opt.dataset)
result_path = opt.result_path
loader = Loader()

if opt.dataset == 'conll':
    train_data, dev_data, test_data, test_train_data, mappings = loader.load_conll(dataset_path, parameters, result_path)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
char_to_id = mappings['char_to_id']
parameters = mappings['parameters']
word_embeds = mappings['word_embeds']

if opt.usemodel == 'CNN_BiLSTM_CRF':
    
    word_vocab_size = len(word_to_id)
    word_embedding_dim = parameters['wrdim']
    word_hidden_dim = parameters['wldim']
    char_vocab_size = len(char_to_id)
    char_embedding_dim = parameters['chdim']
    char_out_channels = parameters['cnchl']
    
    model = CNN_BiLSTM_CRF(word_vocab_size, word_embedding_dim, word_hidden_dim, char_vocab_size,
                           char_embedding_dim, char_out_channels, tag_to_id, pretrained = word_embeds)
    
    if parameters['rload']:
        model_path = os.path.join(result_path,opt.usemodel)
        model.load_state_dict(torch.load(model_path))
    
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015, momentum=0.9)

trainer = Trainer(model, optimizer, result_path, model_name, dataset=opt.dataset) 
losses, all_F = trainer.train_single(opt.num_epochs, train_data, dev_data, test_train_data)
    
plt.plot(losses)
plt.savefig(os.path.join(result_path,model_name+'_'+opt.dataset+'_'+'loss.png'))
