import os

class Evaluator(object):
    def __init__(self, result_path, model_name):
        self.result_path = os.path.join('..',result_path)
        self.model_name = model_name

    def evaluate_conll(self, model, dataset, best_F, eval_script='../../datasets/conll/conlleval'):
        
        prediction = []
        save = False
        new_F = 0.0
        confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
        for data in dataset:
            
            sentence = data['words']
            tags = data['tags']
            chars = data['chars']
            caps = data['caps']

            words = data['str_words']
            
            val, out = model.decode(sentence, tags, chars, caps) 
            
            predicted_id = out
            ground_truth_id = tags
            for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
                line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
                prediction.append(line)
                confusion_matrix[true_id, pred_id] += 1
            prediction.append('')
        
        predf = os.path.join(result_path, 'pred.' + self.model_name)
        scoref = os.path.join(result_path, 'score.' + self.model_name)

        with open(predf, 'wb') as f:
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

        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            "ID", "NE", "Total",
            *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
        ))
        for i in range(confusion_matrix.size(0)):
            print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
                str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
                *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
                  ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
            ))
        return best_F, new_F, save

