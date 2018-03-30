import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


import neural_ner
from neural_ner.util import Initializer
from neural_ner.util import Loader
from neural_ner.modules import CharEncoderCNN
from neural_ner.modules import WordEncoderCNN
from neural_ner.modules import DecoderRNN

class CNN_CNN_LSTM(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_out1_channels, word_out2_channels,
                 char_vocab_size, char_embedding_dim, char_out_channels, decoder_hidden_units,
                 tag_to_id, cap_input_dim=0, cap_embedding_dim=0, pretrained=None):
        
        super(CNN_CNN_LSTM, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out1_channels = word_out1_channels
        self.word_out2_channels = word_out2_channels
        
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_out_channels = char_out_channels
        
        self.cap_input_dim = cap_input_dim
        self.cap_embedding_dim = cap_embedding_dim
        
        self.tag_to_ix = tag_to_id
        self.tagset_size = len(tag_to_id)
        
        self.initializer = Initializer()
        self.loader = Loader()
        
        if self.cap_input_dim and self.cap_embedding_dim:
            self.cap_embedder = nn.Embedding(self.cap_input_dim, self.cap_embedding_dim)
            self.initializer.init_embedding(self.cap_embedder.weight)
        
        self.char_encoder = CharEncoderCNN(char_vocab_size, char_embedding_dim, char_out_channels, 
                                           kernel_width=3, pad_width=2)
        
        self.initializer.init_embedding(self.char_encoder.embedding.weight)
        
        self.word_encoder = WordEncoderCNN(word_vocab_size, word_embedding_dim, char_out_channels,
                                           kernel_width = 3, pad_width = 2, input_dropout_p=0.5,
                                           out1_channels=word_out1_channels, out2_channels=word_out2_channels)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        
        augmented_decoder_inp_size = (word_out2_channels + word_embedding_dim + 
                                      char_out_channels + cap_embedding_dim)
        self.decoder = DecoderRNN(augmented_decoder_inp_size, decoder_hidden_units, self.tagset_size, 
                                  self.tag_to_ix, input_dropout_p=0.5)
        
    def forward(self, sentence, tags, chars, caps, usecuda=True):
        
        if usecuda:
            sentence = Variable(torch.LongTensor(sentence)).cuda()
            tags = torch.LongTensor(tags).cuda()
            caps = Variable(torch.LongTensor(caps)).cuda()
        else:
            sentence = Variable(torch.LongTensor(sentence))
            tags = torch.LongTensor(tags)
            caps = Variable(torch.LongTensor(caps))
            
        chars_mask, _, _ = self.loader.pad_sequence_cnn(chars)
        
        if usecuda:
            chars_mask = Variable(torch.LongTensor(chars_mask)).cuda()
        else:
            chars_mask = Variable(torch.LongTensor(chars_mask))
        
        if self.cap_input_dim and self.cap_embedding_dim:
            cap_features = self.cap_embedder(caps)
        else:
            cap_features = None
            
        char_features = self.char_encoder(chars_mask) 
        word_features, word_input_feats = self.word_encoder(sentence, char_features, cap_features)
        
        new_word_features = torch.cat((word_features,word_input_feats),1)
        score, tag_seq = self.decoder(new_word_features, tags, usecuda=usecuda)
        
        return score
    
    def decode(self, sentence, tags, chars, caps, usecuda=True):
        
        if usecuda:
            sentence = Variable(torch.LongTensor(sentence)).cuda()
            tags = torch.LongTensor(tags).cuda()
            caps = Variable(torch.LongTensor(caps)).cuda()
        else:
            sentence = Variable(torch.LongTensor(sentence))
            tags = torch.LongTensor(tags)
            caps = Variable(torch.LongTensor(caps))
            
        chars_mask, _, _ = self.loader.pad_sequence_cnn(chars)
        
        if usecuda:
            chars_mask = Variable(torch.LongTensor(chars_mask)).cuda()
        else:
            chars_mask = Variable(torch.LongTensor(chars_mask))
        
        if self.cap_input_dim and self.cap_embedding_dim:
            cap_features = self.cap_embedder(caps)
        else:
            cap_features = None
            
        char_features = self.char_encoder(chars_mask) 
        word_features, word_input_feats = self.word_encoder(sentence, char_features, cap_features)
        
        new_word_features = torch.cat((word_features,word_input_feats),1)
        score, tag_seq = self.decoder(new_word_features, tags, usecuda=usecuda)
        
        return score, tag_seq
        