from __future__ import print_function
from torch.autograd import Variable
import time
from evaluator import Evaluator
import sys
import os
import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn

class Trainer(object):
    
    def __init__(self, model, optimizer, result_path, model_name, usedataset, mappings, eval_every=1):
        self.model = model
        self.optimizer = optimizer
        self.eval_every = eval_every
        self.model_name = os.path.join(result_path, model_name)
        
        if usedataset=='conll':
            self.evaluator = Evaluator(result_path, model_name, mappings).evaluate_conll
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_single(self, num_epochs, train_data, dev_data, test_train_data, test_data, learning_rate,
                     checkpoint_folder='.', eval_test_train=True, plot_every=1000):
        
        losses = []
        loss = 0.0
        best_dev_F = -1.0
        best_test_F = -1.0
        best_train_F = -1.0
        all_F=[[0,0,0]]
        count = 0
        
        self.model.train(True)
        for epoch in range(1, num_epochs+1):
            t=time.time()
            for i, index in enumerate(np.random.permutation(len(train_data))):
                
                data = train_data[index]
                self.model.zero_grad()

                sentence = data['words']
                tags = data['tags']
                chars = data['chars']
                caps = data['caps']

                score = self.model(sentence, tags, chars, caps)
                loss += score.data[0]/len(data['words'])
                score.backward()
                
                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                count += 1
                
                if count % plot_every == 0:
                    loss /= plot_every
                    print(count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0
                    
                if count % len(train_data) == 0:
                    self.adjust_learning_rate(self.optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))
            
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