from .base import Trainer


class DATrainer(Trainer):

    def __init__(self, model, methods=[], **kwargs):
        super().__init__(model, **kwargs)
        self._param_dict['methods'] = methods

    def _get_filename(self):
        ''' Get the filename without extension 
        crop: random crop
        hflip: horizontal flip
        rotate: rotation
        erase: random erasing
        '''
        methods = '_'.join(self._param_dict['methods'])
        print(methods)
        suffix = '_' + methods if methods != '' else ''
        return 'lr{:.2f}_da{}'.format(self._lr_orig, suffix) 
