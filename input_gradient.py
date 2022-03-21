import argparse
import glob
from collections import defaultdict
import torch
from auto_attack import CustomAutoAttack
from dataset.mnist import MNIST
from dataset.cifar10 import CIFAR10
from dataset.svhn import SVHN
from model_maker import ModelMaker
from utils import get_base_path, is_mlp, model_path_to_label
import json


def main(base_path, architecture, dataset_type, batch_size, is_l2, is_l1, is_early_stop, is_flooding): 

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Prepare test set 
    require_flatten = is_mlp(architecture)
    if dataset_type == 'cifar10':
        dataset = CIFAR10(256, require_flatten) # Not specify batch_size
    elif dataset_type == 'svhn':
        dataset = SVHN(256, require_flatten) # Not specify batch_size  
    else:
        raise ValueError('This dataset is not supported yet.') 
    dataset.fetch()
    test_iter = iter(dataset.test_loader)
    images, labels = test_iter.next()

    # parameters of model
    model_params = {
        'num_classes': dataset.num_classes,
        'input_size': dataset.input_size,
    }
    # make model
    option = None
    maker = ModelMaker(architecture, dataset_type, option, **model_params)
    model = maker.model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # set path to models folder
    if base_path.endswith('/'):
        base_path = base_path[:-1]
    path = f'{base_path}/{dataset_type}/{architecture}/models/*.pth'

    # extract needed file paths
    model_param_paths = glob.glob(path)
    if not is_l2:
        model_param_paths = list(filter(lambda s: 'l2' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_l1:
        model_param_paths = list(filter(lambda s: 'l1' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_early_stop:
        # remove files that have 'es'(early stopping) in its name
        model_param_paths = list(filter(lambda s: 'es' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_flooding:
        # remove files that have 'fl'(flooding) in its name
        model_param_paths = list(filter(lambda s: 'fl' not in s.rsplit('/', 1)[1], model_param_paths))
    # remove files that have 'ls'(label smoothing) in its name　
    model_param_paths = list(filter(lambda s: 'ls' not in s.rsplit('/', 1)[1], model_param_paths))
    # remove files that have 'da'(data augmentation) in its name　
    model_param_paths = list(filter(lambda s: 'da_' not in s.rsplit('/', 1)[1], model_param_paths))
    # remove files that have 'dr'(drop block) in its name　
    model_param_paths = list(filter(lambda s: 'dr' not in s.rsplit('/', 1)[1], model_param_paths))
    # remove files that have 'mu'(mixup) in its name　
    model_param_paths = list(filter(lambda s: 'mu' not in s.rsplit('/', 1)[1], model_param_paths))
    if len(model_param_paths) == 0:
        raise FileNotFoundError('No such model. You may train models first')
    model_param_paths.sort()

    result_dict = {} # result_dict[model name] = {'l1': [1, 2, 1, 2, ...], 'l2': [4, 4, 2, ...]}
    for path in model_param_paths:
        i = 0
        # modelの学習済みパラメータのロード
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.to(device)
        print('model path: {}'.format(path))
        
        target_model = model_path_to_label(path)
        result_dict[target_model] = []
        for images, labels in test_iter:
            # to device
            images = images.to(device)
            labels = labels.to(device)
            images.requires_grad = True

            # calc L2 norm wrt input 
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()

            # output
            norm = torch.linalg.norm(images.grad, dim=(1,2,3))
            result_dict[target_model].extend(norm.tolist())
            i += 1
            if i == 5:
                break
        print(len(result_dict[target_model]))


    # save the result as json
    t = ''
    if is_l2: t = 'l2'
    elif is_l1: t = 'l1'
    elif is_early_stop: t = 'es'
    elif is_flooding: t = 'fl'
    file = 'results/cifar10/resnet18/input_gradient/{}.json'.format(t)
    j = json.dumps(result_dict, indent=4)
    with open(file,"w") as f:
        f.write(j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line')
    parser.add_argument('--platform', default='', help='colab or not specifid if it is local')
    parser.add_argument('-p', '--path', default='results',
                    help='Folder where model is stored. Specify the folder name after "/content/drive/My Drive/"')
    parser.add_argument('-a', '--architecture', default='mlp',
                    help='Model type. Available optins: mlp, mlp-bn, resnet18. Default is mlp')
    parser.add_argument('-d', '--dataset', default='mnist',
                    help='Dataset. Available options: mnist, cifar10')
    parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size. Default is 128')
    # Regularization params
    parser.add_argument('--l2', action='store_true',
                    help='Compare results that use l2 regularization')
    parser.add_argument('--l1', action='store_true',
                    help='Compare results that use l1 regularization')
    parser.add_argument('--early-stop', action='store_true',
                    help='Compare results that use early stop')
    parser.add_argument('--flooding', action='store_true',
                    help='Compare reusults that use flooding.')

    args = parser.parse_args()
    base_path = get_base_path(args.path, args.platform) 

    # keyword arguments
    kwargs = {
        'base_path': base_path,
        'architecture': args.architecture, 
        'dataset_type': args.dataset,
        'batch_size': args.batch_size,
        'is_l2': args.l2,
        'is_l1': args.l1,
        'is_early_stop': args.early_stop,
        'is_flooding': args.flooding,
    }
    
    main(**kwargs)