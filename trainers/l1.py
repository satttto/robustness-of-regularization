from .base import Trainer
import torch

class L1Trainer(Trainer):
    '''Training using L1 regularization
    '''
    def __init__(self, model, l1_lambda, **kwargs):
        '''
        Args:
            l1_lambda (float) - L1 regulaization parameter. It is used like loss + l1_lambda * l1_norm 
        '''
        super().__init__(model, **kwargs)
        self._param_dict['l1_lambda'] = l1_lambda

    def _train(self, train_loader):
        ''' One epoch of training '''
        self._model.train()
        running_loss = 0. # total loss
        for images, labels in train_loader:
            # Process images, labels
            images, labels = self._process_train_data(images, labels)
            images = images.to(self._param_dict['device'])
            labels = labels.to(self._param_dict['device'])
            # Compute output 
            predictions = self._model(images)
            # Compute loss
            l1_reg = torch.tensor(0., requires_grad=True)
            for name, param in self._model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            loss = self._param_dict['criterion'](predictions, labels) + self._param_dict['l1_lambda'] * l1_reg
            running_loss += loss.item()
            # compute gradient and do SGD 
            self._param_dict['optimizer'].zero_grad()
            loss.backward()
            self._param_dict['optimizer'].step()
        return running_loss / len(train_loader) # train_loss of the epoch
            

    def _get_filename(self):
        ''' Get the filename without extension 
        
        Overide this method to change the name of result and model file.
        '''
        l1_lambda = self._param_dict['l1_lambda']
        return 'lr{:.2f}_l1_lambda{:.6f}'.format(self._lr_orig, l1_lambda)