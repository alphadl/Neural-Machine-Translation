# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# SubLayers.py 2019-06-21 19:52

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

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / self.d_model + self.d_k))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / self.d_model + self.d_k))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / self.d_model + self.d_k))

        self.attention = ScaledDotProductionAttention(temperature=np.power(self.d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_v, d_model)
        nn.init.xavier_norm_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_k)  # (n*b) x lq x dk

        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, mask=mask)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    '''A two-feed-forward-layer module'''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # method1: Conv1d
        # self.Conv1_trans = nn.Conv1d(d_in, d_hid, 1)
        # self.Conv2_trans = nn.Conv1d(d_hid, d_in, 1)

        # method2: Linear
        self.Linear1_trans = nn.Linear(d_in, d_hid, bias=False)
        self.Linear2_trans = nn.Linear(d_hid, d_in, bias=False)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # output = x.transpose(1, 2)
        output = self.Linear2_trans(F.relu(self.Linear1_trans(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        # output = output.transpose(1, 2)
        return output
