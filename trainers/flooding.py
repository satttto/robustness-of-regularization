from .base import Trainer

class FloodTrainer(Trainer):
    ''' Training with flooding regularizer '''
    def __init__(self, model, flood_level, *args, **kwargs):
        super().__init__(model, **kwargs)
        self._param_dict['flood_level'] = flood_level

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
            flood = (loss - self._param_dict['flood_level']).abs() + self._param_dict['flood_level'] 
            # compute gradient and do SGD 
            self._param_dict['optimizer'].zero_grad()
            flood.backward()
            self._param_dict['optimizer'].step()
        return running_loss / len(train_loader) # train_loss of the epoch

    def _get_filename(self):
        ''' Get the filename without extension '''
        fl = self._param_dict['flood_level']
        return 'lr{:.2f}_fl_{:.2f}'.format(self._lr_orig, fl)

            