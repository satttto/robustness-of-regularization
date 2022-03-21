import pathlib
import torch
import pandas as pd
import numpy as np
from .base import Trainer

class DropBlockTrainer(Trainer):

    def __init__(self, model, drop_prob=0.9, block_size=5, **kwargs):
        super().__init__(model, **kwargs)
        self._param_dict['drop_prob'] = drop_prob
        self._param_dict['block_size'] = block_size

    def _get_filename(self):
        ''' Get the filename
        '''
        return 'lr{:.2f}_dr_p{:.2f}_s{}'.format(self._lr_orig, self._param_dict['drop_prob'], self._param_dict['block_size'])