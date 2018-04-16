import os
import codecs
import torch
from .utils import *
import torch
from torch.autograd import Variable

class Evaluator(object):
    def __init__(self, result_path, model_name, mappings, usecuda=True):
        self.result_path = result_path
        self.model_name = model_name
        self.tag_to_id = mappings['tag_to_id']
        self.id_to_tag = mappings['id_to_tag']
        self.usecuda = usecuda

    def evaluate_consrl(self, model, dataset, best_F, eval_script='./datasets/conll/conlleval',
                          checkpoint_folder='.', record_confmat = False, batch_size = 80):
        
        prediction = []
        save = False
        new_F = 0.0
        confusion_matrix = torch.zeros((len(self.tag_to_id) - 2, len(self.tag_to_id) - 2))
    
        data_batches = create_batches(dataset, batch_size = batch_size, str_words = True,
                                      tag_padded = False)

        for data in data_batches:

            words = data['words']
            verbs = data['verbs']
            caps = data['caps']
            mask = data['tagsmask']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
                verbs = Variable(torch.LongTensor(verbs)).cuda()
                caps = Variable(torch.LongTensor(caps)).cuda()
                mask = Variable(torch.LongTensor(mask)).cuda()
            else:
                words = Variable(torch.LongTensor(words))
                verbs = Variable(torch.LongTensor(verbs))
                caps = Variable(torch.LongTensor(caps))
                mask = Variable(torch.LongTensor(mask))

            wordslen = data['wordslen']
            str_words = data['str_words']
            
            _, out = model.decode(words, verbs, caps, wordslen, mask, usecuda = self.usecuda)
                                
            ground_truth_id = data['tags']
            predicted_id = out            
            
            for (swords, sground_truth_id, spredicted_id) in zip(str_words, ground_truth_id, predicted_id):
                for (word, true_id, pred_id) in zip(swords, sground_truth_id, spredicted_id):
                    if self.id_to_tag[true_id]!='B-V':
                        line = ' '.join([word, self.id_to_tag[true_id], self.id_to_tag[pred_id]])
                        prediction.append(line)
                        confusion_matrix[true_id, pred_id] += 1
                prediction.append('')
        
        predf = os.path.join(self.result_path, self.model_name, checkpoint_folder ,'pred.txt')
        scoref = os.path.join(self.result_path, self.model_name, checkpoint_folder ,'score.txt')

        with open(predf, 'w+') as f:
            f.write('\n'.join(prediction))

        os.system('%s < %s > %s' % (eval_script, predf, scoref))

        eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

        for i, line in enumerate(eval_lines):
            print(line)
            if i == 1:
                new_F = float(line.strip().split()[-1])
                if new_F > best_F:
                    best_F = new_F
                    save = True
                    print('the best F is ', new_F)
        
        return best_F, new_F, save