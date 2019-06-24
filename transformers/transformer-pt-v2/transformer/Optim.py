# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# Optim.py 2019-06-24 10:30

import numpy as np


class ScheduledOptim():
    '''A wrapper class for Noam learning scheduling.'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.warmup_steps = n_warmup_steps
        self.current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        '''Step with the inner optimzer'''
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        '''Zero out the gradients by the inner optimizer'''
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.current_steps, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_steps
        ])

    def _update_learning_rate(self):
        '''Learning rate scheduling per step.'''
        self.current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr