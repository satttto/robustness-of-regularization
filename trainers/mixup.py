
# https://arxiv.org/abs/1905.04899
# https://github.com/hongyi-zhang/mixup

import pathlib
import torch
import pandas as pd
import numpy as np
from .base import Trainer

class MixUpTrainer(Trainer):
    def __init__(self, model, layer_mix=None, mixup_alpha=1., mixup_prob=1., **kwargs):
        '''
        Args:
            layer_mix (string) - The layer on which mixup is applied. If None, a number is randomly picked between 0 and 2
        '''
        super().__init__(model, **kwargs)
        self._param_dict['mixup_alpha'] = mixup_alpha
        self._param_dict['mixup_prob'] = mixup_prob
        self._param_dict['layer_mix'] = layer_mix
    
    def _train(self, train_loader):
        ''' One epoch of training '''
        self._model.train()
        running_loss = 0. # total loss
        for images, labels in train_loader:
            images, labels = images.to(self._param_dict['device']), labels.to(self._param_dict['device'])
            # MixUp
            r = np.random.rand()
            if self._param_dict['mixup_alpha'] > 0 and r < self._param_dict['mixup_prob']:
                # Compute output 
                predictions, labels_a, labels_b, lam = self._model(images, labels, True, self._param_dict['mixup_alpha'], self._param_dict['layer_mix'])
                # Compute loss
                loss = lam * self._param_dict['criterion'](predictions, labels_a) + (1. - lam) * self._param_dict['criterion'](predictions, labels_b)
            else:
                # Compute output 
                predictions = self._model(images)
                 # Compute loss
                loss = self._param_dict['criterion'](predictions, labels)

            running_loss += loss.item()
            # compute gradient and do SGD 
            self._param_dict['optimizer'].zero_grad()
            loss.backward()
            self._param_dict['optimizer'].step()
        return running_loss / len(train_loader) # train_loss of the epoch 


    def _get_filename(self):
        ''' Get the filename
        '''
        if self._param_dict['layer_mix'] == 0:
            return 'lr{:.2f}_da_mu'.format(self._lr_orig)
        return 'lr{:.2f}_mu_p{:.2f}_alpha{:.1f}'.format(self._lr_orig, self._param_dict['mixup_prob'], self._param_dict['mixup_alpha'])
