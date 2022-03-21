from .base import Trainer

class L2Trainer(Trainer):
    '''Training using L2 regularization
    '''
    def __init__(self, model, l2_lambda, **kwargs):
        '''
        Args:
            l2_lambda (float) - L2 regulaization parameter. It is used like loss + l2_lambda * l2_norm 
        '''
        super().__init__(model, **kwargs)
        self._param_dict['l2_lambda']   = l2_lambda
            

    def _get_filename(self):
        ''' Get the filename without extension 
        
        Overide this method to change the name of result and model file.
        '''
        l2_lambda = self._param_dict['l2_lambda']
        return 'lr{:.2f}_l2_wd{:.6f}'.format(self._lr_orig, l2_lambda)