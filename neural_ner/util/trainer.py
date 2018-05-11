from __future__ import print_function
from torch.autograd import Variable
import time
from .evaluator import Evaluator
import sys
import os
import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from .utils import *


class Trainer(object):
    
    def __init__(self, model, optimizer, result_path, model_name, usedataset, mappings, 
                 eval_every=1, usecuda = True):
        self.model = model
        self.optimizer = optimizer
        self.eval_every = eval_every
        self.model_name = os.path.join(result_path, model_name)
        self.usecuda = usecuda
        
        self.evaluator = Evaluator(result_path, model_name, mappings).evaluate_conll
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_model(self, num_epochs, train_data, dev_data, test_train_data, test_data, learning_rate,
                    checkpoint_folder='.', eval_test_train=True, plot_every=50, adjust_lr=True,
                    batch_size = 16, lr_decay = 0.05):

        losses = []
        loss = 0.0
        best_dev_F = -1.0
        best_test_F = -1.0
        best_train_F = -1.0
        all_F=[[0,0,0]]
        count = 0
        word_count = 0
        
        self.model.train(True)
        for epoch in range(1, num_epochs+1):
            t=time.time()
            
            train_batches = create_batches(train_data, batch_size= batch_size, order='random')
            n_batches = len(train_batches)
            
            for i, index in enumerate(np.random.permutation(len(train_batches))): 
                
                data = train_batches[index]
                self.model.zero_grad()

                words = data['words']
                tags = data['tags']
                chars = data['chars']
                caps = data['caps']
                mask = data['tagsmask']
                
                if self.usecuda:
                    words = Variable(torch.LongTensor(words)).cuda()
                    chars = Variable(torch.LongTensor(chars)).cuda()
                    caps = Variable(torch.LongTensor(caps)).cuda()
                    mask = Variable(torch.LongTensor(mask)).cuda()
                    tags = Variable(torch.LongTensor(tags)).cuda()
                else:
                    words = Variable(torch.LongTensor(words))
                    chars = Variable(torch.LongTensor(chars))
                    caps = Variable(torch.LongTensor(caps))
                    mask = Variable(torch.LongTensor(mask))
                    tags = Variable(torch.LongTensor(tags))
                
                wordslen = data['wordslen']
                charslen = data['charslen']
                
                score = self.model(words, tags, chars, caps, wordslen, charslen, mask, n_batches,
                                         usecuda=self.usecuda)
                
                loss += score.data[0]/np.sum(data['wordslen'])
                score.backward()
                
                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                count += 1
                word_count += batch_size
                
                if count % plot_every == 0:
                    loss /= plot_every
                    print(word_count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0
                                        
            if adjust_lr:
                self.adjust_learning_rate(self.optimizer, lr=learning_rate/(1+lr_decay*float(word_count)/len(train_data)))
            
            if epoch%self.eval_every==0:
                
                self.model.train(False)
                
                if eval_test_train:
                    best_train_F, new_train_F, _ = self.evaluator(self.model, test_train_data, best_train_F, 
                                                                  checkpoint_folder=checkpoint_folder)
                else:
                    best_train_F, new_train_F, _ = 0, 0, 0
                best_dev_F, new_dev_F, save = self.evaluator(self.model, dev_data, best_dev_F,
                                                             checkpoint_folder=checkpoint_folder)
                if save:
                    torch.save(self.model, os.path.join(self.model_name, checkpoint_folder, 'modelweights'))
                best_test_F, new_test_F, _ = self.evaluator(self.model, test_data, best_test_F,
                                                            checkpoint_folder=checkpoint_folder)
                sys.stdout.flush()

                all_F.append([new_train_F, new_dev_F, new_test_F])
                
                self.model.train(True)

            print('*'*80)
            print('Epoch %d Complete: Time Taken %d' %(epoch ,time.time() - t))

        return losses, all_F