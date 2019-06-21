# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# SubLayers.py 2019-06-21 19:52

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformer.Modules import ScaledDotProductionAttention


class MultiHeadAttention(nn.Module):
    '''Multi-Head Attention module'''

    # def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)




