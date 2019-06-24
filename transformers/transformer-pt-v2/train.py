# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# train.py 2019-06-24 12:26

'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data

import transformer.Constant as Constant
from dataset import TranslationDateset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


def cal_loss(pred, gold, smoothing):
    '''Calculate cross entropy loss, apply label smoothing if needed'''

    gold = gold.contiguous().view(-1)  # * -> n

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter_(1, gold.view(-1, 1),
                                                  1)  # example. [1,2,0]->[[0,1,0,0],[0,0,1,0],[1,0,0,0]]
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constant.PAD)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later

    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constant.PAD, reduction='sum')

    return loss


def cal_performance(pred, gold, smoothing=False):
    # TODO
    return NotImplementedError
