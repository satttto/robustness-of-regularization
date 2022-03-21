
# https://arxiv.org/abs/1905.04899
# https://github.com/clovaai/CutMix-PyTorch

import pathlib
import torch
import pandas as pd
import numpy as np
from .base import Trainer

class CutMixTrainer(Trainer):

    def __init__(self, model, cutmix_alpha=1, cutmix_prob=0.5, **kwargs):
        super().__init__(model, **kwargs)
        self._param_dict['cutmix_alpha'] = cutmix_alpha
        self._param_dict['cutmix_prob'] = cutmix_prob
    
    def _process_train_data(self, images, labels):
        ''' cutmix
        '''
        # generate mixed sample
        lam = np.random.beta(self._param_dict['cutmix_alpha'], self._param_dict['cutmix_alpha'])
        rand_index = torch.randperm(images.size()[0])
        labels_a = labels
        labels_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        return images, labels_a, labels_b, lam

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def _train(self, train_loader):
        ''' One epoch of training '''
        self._model.train()
        running_loss = 0. # total loss
        for images, labels in train_loader:
            images, labels = images.to(self._param_dict['device']), labels.to(self._param_dict['device'])
            # CutMix
            r = np.random.rand()
            if self._param_dict['cutmix_alpha'] > 0 and r < self._param_dict['cutmix_prob']:
                images, labels_a, labels_b, lam = self._process_train_data(images, labels)
                # Compute output 
                predictions = self._model(images)
                # Compute loss
                loss = lam * self._param_dict['criterion'](predictions, labels_a) + (1. - lam) * self._param_dict['criterion'](predictions, labels_b) 
            # No CutMix
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
        return 'lr{:.2f}_da_cm'.format(self._lr_orig)

    
            