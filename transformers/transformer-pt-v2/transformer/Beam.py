# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# Beam.py 2019-06-22 23:20

""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""
import torch

import transformer.Constant as Constant


class Beam():
    '''Beam search'''

    def __init__(self, beam_size, device=False):
        self.size = beam_size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((self.size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((self.size,), Constant.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = Constant.BOS

    def get_current_state(self):
        '''Get the outputs for the current timestep.'''
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        '''Get the backpointers for the current timestep.'''
        return self.prev_ksp[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        '''Update beam status and check if finished or not.'''
        num_words = word_prob.size(1)
