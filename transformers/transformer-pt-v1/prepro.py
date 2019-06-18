# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# prepro.py 2019-06-18 23:20


from __future__ import print_function
from hparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter


def make_vocab(fpath, fname):
    '''Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    '''
    text = codecs.open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    make_vocab(hp.source_train, "de.vocab.tsv")
    make_vocab(hp.target_train, "en.vocab.tsv")
    print("Done")
