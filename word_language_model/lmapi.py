import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert2tensor(data, eos_token):
    all_data = []
    for x in data:
        vx = x['words'] + [eos_token]
        all_data.extend(vx)
    return torch.LongTensor(all_data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

class lmAPI():
    def __init__(self, mapping, verbose = False):
        self.mapping = mapping
        self.eos_token = len(mapping['word_to_id'])
        self.ntokens = len(mapping['word_to_id']) + 1
        self.verbose = verbose

    def train_epoch(self, train_data, epoch):

        self.lm.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = self.ntokens
        hidden = self.lm.init_hidden(self.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(train_data, i, self.bptt)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            self.lm.zero_grad()
            output, hidden = self.lm(data, hidden)
            loss = self.criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.lm.parameters(), self.clip)
            for p in self.lm.parameters():
                p.data.add_(-self.lr, p.grad.data)

            total_loss += loss.item()

            if batch % self.log_interval == 0 and batch > 0:
                cur_loss = total_loss / self.log_interval
                elapsed = time.time() - start_time
                if self.verbose:
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // self.bptt, self.lr,
                        elapsed * 1000 / self.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def train_lm(self, acq_data, batch_size = 20, celltype='LSTM', embedding_size = 100, 
                 hidden_size = 100, num_layers = 1, dropout = 0.2, tied = True, bptt = 25,
                 log_interval= 10, lrate = 20.0, num_epochs = 20, clip = 0.25,
                 checkpoint_folder = 'lmweights'):
        
        corpus_train = convert2tensor(acq_data, self.eos_token)
        train_data = batchify(corpus_train, bsz=batch_size)

        self.bptt = bptt
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.clip = clip
        self.lm = model.RNNModel(celltype, self.ntokens, embedding_size, 
                            hidden_size, num_layers, dropout, tied).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lrate
        best_val_loss = None

        for epoch in range(1, num_epochs+1):
            epoch_start_time = time.time()
            self.train_epoch(train_data, epoch = epoch)
            self.lr /= 4.0

        torch.save(self.lm, checkpoint_folder)
