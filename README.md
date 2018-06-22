# Active-NLP

Implementation of different acquisition functions for NER, classification and SRL task. (Machine Translation in progress). I ll soon add a proper readme file (this weekend). For now, You can peek into train_ner.py/active_ner.py and figure out arguments according to help in argparse.

For example to run a CNN_BiLSTM_CRF model on Conll dataset on full dataset, you can run

     $ python train_ner.py --usemodel CNN_BiLSTM_CRF --dataset conll

and to run active learning for CNN_BiLSTM_CRF model on  Conll dataset with "MNLP" acquisition function, you can run

     $ python active_ner.py --usemodel CNN_BiLSTM_CRF --dataset conll --acquiremethod mnlp
