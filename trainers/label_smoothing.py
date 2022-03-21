from .base import Trainer

class LabelSmoothingTrainer(Trainer):
    '''Training using label smoothing regularization
    '''
    def __init__(self, model, label_smoothing=0.0, **kwargs):
        '''
        Args:
            label_smoothing (float)
        '''
        super().__init__(model, **kwargs)
        self._param_dict['label_smoothing'] = label_smoothing
            

    def _get_filename(self):
        ''' Get the filename without extension 
        
        Overide this method to change the name of result and model file.
        '''
        ls = self._param_dict['label_smoothing']
        return 'lr{:.2f}_ls{:.1f}'.format(self._lr_orig, ls)