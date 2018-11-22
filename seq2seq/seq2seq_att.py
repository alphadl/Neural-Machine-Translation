# -*-coding:utf-8-*-
# Author: alphadl
# seq2seq_att.py 21/11/18 17:52
from __future__ import unicode_literals, print_function, division

"""
Translation with a Sequence to Sequence Network and Attention

This implementation is heavily based on NMT tutorial of Pytorch
"""

"""
Requirements
"""
import unicodedata
import string
import re
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from logger import Logger

logger = Logger("./runs")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Load the data
"""
SOS_token = 0
EOS_token = 1


class Langugae_Process:
    def __init__(self, name):
        self.name = name
        self.w2i = {}  # word to index
        self.w2c = {}  # word to count
        self.i2w = {0: "SOS", 1: "EOS"}  # index to word
        self.n_words = 2  # count SOS and EOS

    def addSent(self, sent):
        for w in sent.split(' '):
            self.addWord(w)

    def addWord(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.n_words
            self.w2c[word] = 1
            self.i2w[self.n_words] = word
            self.n_words += 1
        else:
            self.w2c[word] += 1


# Turn a Unicode string to plain ASCII
def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


# Lowercase trim and remove the non-letter characters
def normlizaString(s):
    s = unicode2ascii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?，。！？]+", r" ", s)
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5 ,.!?，。！？、-]")
    s = rule.sub('', s)
    return s


"""
To read the data file we will split the file into lines, and then split lines into pairs.
"""


def readLangs(srcLang, tgtLang):
    print("Reading lines...")

    # Read file and split into lines
    with open("../data/train.%s" % (srcLang), encoding="utf-8") as srcT, \
            open("../data/train.%s" % (tgtLang), encoding="utf-8") as tgtT, \
            open("../data/val.%s" % (srcLang), encoding="utf-8") as srcV, \
            open("../data/val.%s" % (tgtLang), encoding="utf-8") as tgtV:
        srcT = [line.strip() for line in srcT.readlines()]
        srcV = [line.strip() for line in srcV.readlines()]
        tgtT = [line.strip() for line in tgtT.readlines()]
        tgtV = [line.strip() for line in tgtV.readlines()]
        lines = np.array([srcT + srcV, tgtT + tgtV]).transpose()
        # print("raw input>>>>",len(lines),lines[0])
        pairs = [[normlizaString(ll) for ll in l] for l in lines]
        # print("normlized>>>>",len(pairs),pairs[0])
        # make Lang instances
        input_lang = Langugae_Process(srcLang)
        output_lang = Langugae_Process(tgtLang)

        return input_lang, output_lang, pairs


"""
Filtering seq length to simple the model 
"""
MAX_LENGTH = 80


def filterPair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and \
           len(p[1].split(" ")) < MAX_LENGTH


def fileterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


"""
Full process for preparing the data is :
1) Read text file and split into lines , split lines into pairs
2) Normalize text, filter by length and content
3) Make Vocab from sentences in pairs
"""


def preprocess(srcLang, tgtLang):
    src_Lang, tgt_Lang, pairs = readLangs(srcLang, tgtLang)
    print("Totally read %s sentences pairs" % len(pairs))
    pairs = fileterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        src_Lang.addSent(pair[0])
        tgt_Lang.addSent(pair[1])
    print("Counted words:")
    print("vocab scale of", src_Lang.name, ":", src_Lang.n_words)
    print("vocab scale of", tgt_Lang.name, ":", tgt_Lang.n_words)
    return src_Lang, tgt_Lang, pairs


print("-" * 20 + "starting pre-process" + "-" * 20)

src_lang, tgt_lang, pairs = preprocess('en', 'cn')
# print(random.choice(pairs))

"""
define encoder
"""


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.h_size = hidden_size
        self.i_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # print(hidden[0].size(),"+++",hidden[1].size())
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        # return (torch.zeros(1, 1, self.h_size, device=device), \
        #        torch.zeros(1, 1, self.h_size, device=device))
        return torch.zeros(1, 1, self.h_size, device=device)

"""
define decoder
"""

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""
Training
"""


# preparing
def s2i(lang, sentence):  # sentence to index
    return [lang.w2i[word] for word in sentence.split(" ")]


def s2t(lang, sentence):  # sentence to tensor
    indexes = s2i(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def p2t(pair):  # pair to tensors
    src_tensor = s2t(src_lang, pair[0])
    tgt_tensor = s2t(tgt_lang, pair[1])
    return (src_tensor, tgt_tensor)


# training
teacher_forcing_ratio = 1.0


def train(src_tensor, tgt_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    src_len = src_tensor.size(0)
    tgt_len = tgt_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.h_size, device=device)

    loss = 0

    for ei in range(src_len):
        encoder_output, encoder_hidden = encoder(src_tensor[ei], encoder_hidden)
        # encoder_output[ei] = encoder_output[0, 0]
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Feed the target as the next input
        for di in range(tgt_len):
            decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input, decoder_hidden,\
                                                     encoder_outputs)
            loss += criterion(decoder_output, tgt_tensor[di])
            decoder_input = tgt_tensor[di]
    else:
        # without teach forcing: use its own predictions as the next input
        for di in range(tgt_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,\
                                                     encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, tgt_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / tgt_len


# helper function to print the time
import time
import math


def toMintutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (toMintutes(s), toMintutes(rs))


"""
Full process for training the model is :
1) Start a timer
2) Initialize Optimizers and criterion 
3) Create set of training pairs
4) Start empty losses array for plotting
"""


def trainIters(encoder, decoder, n_iters, print_every=100, learning_rate=0.01, save_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    # plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [p2t(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        # plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (Now is:%d,Finished %d%%) //average loss:%.4f' % (timeSince(start, iter / n_iters),
                                                                        iter, (iter / n_iters) * 100, print_loss_avg))
        # plot the loss
        # if iter % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0

        # save the model checkpoint
        if iter % 10000 == 0:
            torch.save(decoder.state_dict(), os.path.join("./model", 'decoder-{}.pt'.format(iter + 1)))
            torch.save(encoder.state_dict(), os.path.join("./model", 'encoder-{}.pt'.format(iter + 1)))
            print("Save the %dth step checkpoint at ./model" % (iter + 1))

        # (1) log the scalar values
        info = {'loss_+att': loss}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, iter)


            # showPlot(plot_losses) #this is not enable in the server


# plot function
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    print("in plot")
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("./loss.jpg")


"""
parameters for training
"""
print("-" * 20 + "starting training" + "-" * 20)
hidden_size = 256

encoder1 = Encoder(src_lang.n_words, hidden_size).to(device)
decoder1 = AttnDecoder(hidden_size, tgt_lang.n_words,dropout_p=0.1).to(device)

trainIters(encoder1, decoder1, 120000, print_every=50)
