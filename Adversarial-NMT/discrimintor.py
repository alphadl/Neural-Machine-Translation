# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# discrimintor.py 30/11/18 16:19

# torch全家桶
import torch
import torch.nn as nn
import torch.nn.funcitonal as F
from torch.autograd import Variable

import numpy as np


class Discriminator(nn.Module):
    def __init__(self, src_vocab_size,
                 tgt_vocab_size,
                 word_emb_size,
                 src_vocab,
                 tgt_vocab,
                 use_cuda=False):
        super(Discriminator, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.word_emb_size = word_emb_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.use_cuda = use_cuda

        self.embedding_s = nn.Embedding(src_vocab_size, word_emb_size)
        self.embedding_t = nn.Embedding(tgt_vocab_size, word_emb_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel=word_emb_size*2,
                      out_channel=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel=64,
                      out_channel=20,
                      kernel_size=3,
                      stride=1,
                      padding=1,),
            nn.BatchNorm2d(20),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #why 1280
        self.mlp = nn.Linear(1280,20)
        self.ll = nn.Linear(20,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,src_batch,tgt_batch,is_train=False):
        #src_batch: (src_seq_len, batch_size)  (35,80)
        src_embeded = self.embedding_s(src_batch)
        tgt_embeded = self.embedding_t(tgt_batch)

        #padding
        src_padded = np.zero((35,src_batch.size(-1),self.word_emb_size))
        tgt_padded = np.zero((35,src_batch.size(-1),self.word_emb_size))
        src_padded[:src_embeded.size(0),:src_embeded.size(1), :src_embeded.size(2)] = src_embeded.data
        tgt_padded[:tgt_embeded.size(0),:tgt_embeded.size(1), :tgt_embeded.size(2)] = tgt_embeded.data

        src_padded = np.transpose(np.expand_dims(src_padded,2),(1,3,2,0))
        src_padded = np.concatenate([src_padded]*35, axis=2)
        tgt_padded = np.transpose(np.expand_dims(tgt_padded,2),(1,3,2,0))
        tgt_padded = np.concatenate([tgt_padded]*35, axis=3)

        input = Variable(torch.from_numpy(np.concatenate([src_padded,tgt_padded],axis=1)).float())

        if self.use_cuda == True:
            input = input.cuda()

        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.size(0),1280)
        output = F.relu(self.mlp(output))
        output = self.ll(output)
        output = self.sigmoid(output)

        return output