from .base import Trainer
import pathlib
import torch
import pandas as pd


class ESTrainer(Trainer):

    def __init__(self, model, metric, patience, **kwargs):
        '''
        Args:
            metric (string)    - Based on what you want to early stop 
            patience (integer) - The maximum number of continuation
        '''
        super().__init__(model, **kwargs)
        self._latest_progressed_model = self._model
        self._param_dict['metric']   = metric
        self._param_dict['patience'] = patience

    def _fresh_caches(self):
        '''
        _train_losses - save all the training lossess throughtout epochs (len=#epochs)
        _val_losses   - save all the validation lossess throughout epochs (len=#epochs)
        _val_accs     - save all the accuracy on validation dataset throught epochs (len=#epochs)
        _count        - The number of continuation. If it reaches the num of patience, stop learning.
        '''
        super()._fresh_caches()
        self._count = 0
 
    @Trainer._training_decorator
    def train_model(self, train_loader, test_loader, ):
        for epoch in range(self._param_dict['num_epochs']):
            # Training 
            train_loss = self._train(train_loader)
            # Validation
            val_loss, val_acc = self._test(test_loader)
            # Print result of the epoch
            print('epoch %d, train_loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch+1, train_loss, val_loss, val_acc))
            # Save result of the epoch
            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)
            self._val_accs.append(val_acc)
            # Check if the learning progresses. Count up if not from last training
            if epoch > 1:
                if self._is_progressed():
                    self._latest_progressed_model = self._model
                    self._count = 0
                else:
                    self._count += 1
            # Early stop
            if self._should_early_stop():
                print('Early Stopped')
                break
            # update learning rate
            self._param_dict['lr_scheduler'].step()

    def _is_progressed(self):
        ''' Determine if the learning progress '''
        if self._param_dict['metric'] == 'vloss':
            return self._val_losses[-2] > self._val_losses[-1]
        elif self._param_dict['metric'] == 'tloss':
            return self._train_losses[-2] > self._train_losses[-1]
        elif self._param_dict['metric'] == 'vacc':
            return self._val_accs[-2] < self._val_accs[-1]
        return True
    
    def _should_early_stop(self):
        ''' Check if the count reaches the patience '''
        return self._count >= self._param_dict['patience']

    def save_model_in(self, base, is_latest_progressed_model=False):
        ''' Save model.

        Args:
            is_latest_progressed_model (boolean) - save the best model in the last "patience" epocks.
        '''
        if not self._is_trained:
            raise ValueError('No model is available. You may train the model first.')
        base = self._remove_last_slash(base) + '/models'
        print('Save the model to ', base)
        pathlib.Path(base).mkdir(parents=True, exist_ok=True)
        filename = self._get_model_file()
        if is_latest_progressed_model and self._latest_progressed_model.state_dict() is not None:
            print('save the latest progressed model')
            model = self._latest_progressed_model
        else:
            print('save the last model')
            model = self._model
        torch.save(model.state_dict(), f'{base}/{filename}')

    def _get_filename(self):
        ''' Get the filename without extension '''
        mode = self._param_dict['metric']
        patience = self._param_dict['patience'] 
        return 'lr{:.2f}_es_{}_p{}'.format(self._lr_orig, mode, patience)
        

