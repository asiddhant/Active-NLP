from __future__ import print_function
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import time
from evaluator import Evaluator
import sys
import os

class Trainer(object):
    
    def __init__(self, model, optimizer, result_path, model_name,  usedataset
                 plot_every=500, eval_every=1):
        self.model = model
        self.optimizer = optimizer
        self.plot_every = plot_every
        self.eval_every = eval_every
        self.model_name = os.path.join('../',result_path, model_name)
        
        if usedataset=='conll':
            self.evaluator = Evaluator(result_path, model_name).evaluate_conll
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_single(self, num_epochs, train_data, dev_data, test_train_data):
        
        losses = []
        loss = 0.0
        best_dev_F = -1.0
        best_test_F = -1.0
        best_train_F = -1.0
        all_F=[[0,0,0]]
        count = 0
        
        model.train(True)
        for epoch in range(1, num_epochs+1):
            for i, index in enumerate(np.random.permutation(len(train_data))):
                
                data = train_data[index]
                self.model.zero_grad()

                sentence = data['words']
                tags = data['tags']
                chars = data['chars']
                caps = data['caps']

                score = self.model.forward(sentence, tags, chars, caps)
                loss += score.data[0]/len(data['words'])
                score.backward()
                
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                count += 1
                
                if count % self.plot_every == 0:
                    loss /= plot_every
                    print(count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0
                    
                if count % len(train_data) == 0:
                    self.adjust_learning_rate(self.optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))
            
                #if epoch%self.eval_every==0: ##Isko Left Lena hai
                
                model.train(False)
                
                best_train_F, new_train_F, _ = self.evaluator(model, test_train_data, best_train_F)
                best_dev_F, new_dev_F, save = self.evaluator(model, dev_data, best_dev_F)
                if save:
                    torch.save(model, self.model_name)
                best_test_F, new_test_F, _ = self.evaluator(model, test_data, best_test_F)
                sys.stdout.flush()

                all_F.append([new_train_F, new_dev_F, new_test_F])
                model.train(True)

                print('*'*80)
                print('Epoch %d Complete: Time Taken %d' %(epoch ,time.time() - t))

        return losses, all_F