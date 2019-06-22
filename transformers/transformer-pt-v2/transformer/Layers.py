# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# Layers.py 2019-06-22 18:30

import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    '''Compose with two layers: slf_attn + pos_ffn'''

    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    '''Compose with three layers: self_attn + enc_attn + pos_ffn'''

    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, dropout=dropout
        )
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # component 1
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        dec_output *= non_pad_mask
        # component 2
        dec_output, dec_enc_attn = self.slf_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output *= non_pad_mask
        # component 3
        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
