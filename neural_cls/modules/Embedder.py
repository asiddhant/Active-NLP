import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_cls
from neural_cls.util.utils import *

class Embedder(nn.Module):

    def __init__(self, embedding_type, word_embedding_dim, embedding_path,
                 mappings):
        super(Embedder, self).__init__()
        self.embedding_type = embedding_type
        if self.embedding_type == 'noemb':
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        elif self.embedding_type == 'glove':
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            pretrained = load_glove_embeddings(embedding_path,
                                               word_embedding_dim,
                                               mappings)
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        else:
            raise NotImplementedError()
        

    def forward(self, words, strwords=None):
        if self.embedding_type in ['noemb','glove']:
            word_embeddings = self.embedding(words)
            return word_embeddings
        elif self.embedding_type == "elmo":
            assert strwords is not None, "ELMo embeddings needs string input"


        