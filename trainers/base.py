import pathlib
import torch
import pandas as pd


class Trainer:
    ''' Base tainer class

    All the trainers must be based on this class.
    Possible subclasses include regularization, attacks, etc.
    '''
    def __init__(self, model, criterion, optimizer, lr_scheduler, num_epochs, device, *args, **kwargs):
        self._model = model
        self._best_result_model = model
        self._param_dict = {
            'device': device,
            'criterion': criterion,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'num_epochs': num_epochs,
        }
        self._is_trained = False
        self._lr_orig = optimizer.param_groups[0]['lr']
    
    def _fresh_caches(self):
        '''
        _train_losses - save all the training lossess (len=#epochs)
        _val_losses   - save all the validation lossess (len=#epochs)
        _val_accs     - save all the accuracy in validation (len=#epochs)
        _is_trained   - true if training is already done.
        _best_acc     - The best accuracy on test set
        '''
        self._train_losses = []
        self._val_losses = []
        self._val_accs   = []
        self._is_trained = False
        self._best_acc = 0
        self._best_acc_epoch = 0

    def print_params(self):
        ''' Print out training params '''
        print()
        print('######### Training Parameters Info ###########')
        for key, val in self._param_dict.items():
            print(f'{key}: {val}')
        print('######### Training Parameters Info ###########')
        print()
    
    @property
    def model(self):
        if not self._is_trained:
            raise ValueError('No trained model')
        return self._model

    def _training_decorator(func):
        ''' Preprocess and Postprocess of training 
        
        1. prepare cache.
        2. execute training
        3. set _is_trained done.
        ''' 
        def train(self, *args, **kwargs):
            self._fresh_caches()
            print()
            print('########### Start Training ###########')
            func(self, *args, **kwargs)
            self._is_trained = True
            print('########### End Training ###########')
            print()
        return train

    @_training_decorator
    def train_model(self, train_loader, test_loader):
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
            # update the model that records the best accuracy
            self._update_best()
            # update learning rate
            self._param_dict['lr_scheduler'].step()

    def _update_best(self):
        if self._val_accs[-1] > self._best_acc:
            self._best_acc = self._val_accs[-1]
            self._best_acc_epoch = len(self._val_accs)
            self._best_result_model = self._model
            

    def _process_train_data(self, images, labels):
        ''' Process training data during training
        
        e.g. adversarial example
        '''
        return images, labels

    def _process_test_data(self, images, labels):
        ''' Process validation data during training '''
        return images, labels
    
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
            loss = self._param_dict['criterion'](predictions, labels)
            running_loss += loss.item()
            # compute gradient and do SGD 
            self._param_dict['optimizer'].zero_grad()
            loss.backward()
            total_norm = 0
            for p in self._model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(total_norm)
            self._param_dict['optimizer'].step()
        return running_loss / len(train_loader) # train_loss of the epoch

    def _test(self, test_loader):
        ''' One epoch of validation '''
        self._model.eval()
        running_loss = 0. # accumulated loss through the validation step 
        correct = 0       # the number of correct prediction
        total = 0         # total number of test data
        with torch.no_grad():
            for images, labels in test_loader:
                # Process images, labels
                images, labels = self._process_test_data(images, labels)
                images = images.to(self._param_dict['device'])
                labels = labels.to(self._param_dict['device'])
                # Compute output
                predictions = self._model(images)
                # Compute loss
                loss = self._param_dict['criterion'](predictions, labels)
                running_loss += loss.item()
                # Compute accuracy
                _, predicted = torch.max(predictions, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss = running_loss / len(test_loader) 
        val_acc  = float(correct) / total
        return val_loss, val_acc

    def restore_best_model(self):
        ''' Restore the model that recorded the best accuracy on test set '''
        if not self._is_trained:
            raise ValueError('Not trained  yet. You may train the model first.')
        print('Restored the best model: acc {} at the epoch {}'.format(self._best_acc, self._best_acc_epoch))
        self._model = self._best_result_model 

    
    def report_result(self):
        ''' Return the result as a dictionary 
        
        - key: 'train_loss', 'val_loss', 'acc'
        '''
        if not self._is_trained:
            raise ValueError('No result available. You may train the model first.')
        return { 'train_loss': self._train_losses, 'val_loss': self._val_losses, 'acc': self._val_accs }    
    
    def save_model_in(self, base):
        ''' Save model
        
        The variable "base" is the directory in which file is going to be placed.
        If the directory does not exist, make it, then save the trained params
        '''
        if not self._is_trained:
            raise ValueError('No model is available. You may train the model first.')
        base = self._remove_last_slash(base) + '/models'
        print('Save the model to ', base)
        pathlib.Path(base).mkdir(parents=True, exist_ok=True)
        filename = self._get_model_file()
        torch.save(self._model.state_dict(), f'{base}/{filename}')

    def save_result_in(self, base):
        ''' Save result
        
        The variable "base" is the directory in which file is going to be placed.
        If the directory does not exist, make it, then save the result.
        '''
        if not self._is_trained:
            raise ValueError('No result is available. You may train the model first.')
        base = self._remove_last_slash(base) + '/results'
        print('Save the result to ', base)
        pathlib.Path(base).mkdir(parents=True, exist_ok=True)
        filename = self._get_result_file()
        df = pd.DataFrame(self.report_result())
        df.to_csv(f'{base}/{filename}')

    def _remove_last_slash(self, base):
        ''' Remove slash at the end of "base" if exists '''
        return base[:-1] if base.endswith('/') else base

    def _get_filename(self):
        ''' Get the filename without extension 
        
        Overide this method to change the name of result and model file.
        '''
        return 'lr{:.2f}'.format(self._lr_orig)

    def _get_model_file(self):
        ''' Get the name of model file to be saved '''
        filename = self._get_filename()
        return f'{filename}.pth'

    def _get_result_file(self):
        ''' Get the name of result file to be saved '''
        filename = self._get_filename()
        return f'{filename}.csv'

    
            