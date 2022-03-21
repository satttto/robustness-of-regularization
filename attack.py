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


def main(base_path, architecture, dataset_type, batch_size, \
    is_l2, is_l1, is_early_stop, is_label_smoothing, is_flooding, is_data_augment, is_hidden_mixup, is_drop_block, \
    min_eps_nu=0, max_eps_nu=5, step=1): # eps_nu / 255 is used as eps

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Prepare test set 
    require_flatten = is_mlp(architecture)
    if dataset_type == 'mnist':
        dataset = MNIST(None, require_flatten) # Not specify batch_size 
    elif dataset_type == 'cifar10':
        dataset = CIFAR10(None, require_flatten) # Not specify batch_size
    elif dataset_type == 'svhn':
        dataset = SVHN(5000, require_flatten) # Not specify batch_size  
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
    if is_hidden_mixup:
        option = 'mixup'
    maker = ModelMaker(architecture, dataset_type, option, **model_params)
    model = maker.model.to(device)

    # set path to models folder
    if base_path.endswith('/'):
        base_path = base_path[:-1]
    path = f'{base_path}/{dataset_type}/{architecture}/models/*.pth'

    # extract needed file paths
    model_param_paths = glob.glob(path)
    if not is_l2:
        model_param_paths = list(filter(lambda s: 'l2_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_l1:
        model_param_paths = list(filter(lambda s: 'l1_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_early_stop:
        # remove files that have 'es_'(early stopping) in its name
        model_param_paths = list(filter(lambda s: 'es_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_label_smoothing:
        # remove files that have 'ls_'(label smoothing) in its name　
        model_param_paths = list(filter(lambda s: 'ls_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_flooding:
        # remove files that have 'fl_'(flooding) in its name
        model_param_paths = list(filter(lambda s: 'fl_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_data_augment:
        # remove files that have 'da_'(data augmentation) in its name　
        model_param_paths = list(filter(lambda s: 'da_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_drop_block:
        # remove files that have 'dr_'(drop block) in its name　
        model_param_paths = list(filter(lambda s: 'dr_' not in s.rsplit('/', 1)[1], model_param_paths))
    if not is_hidden_mixup:
        # remove files that have 'mu_'(hidden mixup) in its name　
        model_param_paths = list(filter(lambda s: 'mu_' not in s.rsplit('/', 1)[1], model_param_paths))
    if len(model_param_paths) == 0:
        raise FileNotFoundError('No such model. You may train models first')
    model_param_paths.sort()    

    eps_numerators = []
    result_dict = {} # result_dict[model_param][eps] = {'attack-1': acc1, 'attack-2': acc2, ...}
    for path in model_param_paths:
        # modelの学習済みパラメータのロード
        model.load_state_dict(torch.load(path))
        model.to(device)
        print('model path: {}'.format(path))
        
        target_model = model_path_to_label(path) 
        result_dict[target_model] = {}

        curr_eps_nu = min_eps_nu
        while curr_eps_nu <= max_eps_nu:
            eps_numerators.append(curr_eps_nu)
            eps = curr_eps_nu / 255
            print(f'attack eps: {curr_eps_nu} / 255')
            # prepare attack
            adversary = CustomAutoAttack(model, norm='Linf', eps=eps, version='standard')
            # ミニバッチごとに各種AAが実施される
            adv, result_dict[target_model][curr_eps_nu] = adversary.run_standard_evaluation_individual(images, labels, bs=128)
            # update 
            curr_eps_nu += step

    # save the result as json
    # first, check if the target dir exists.
    d = f'{base_path}/{dataset_type}/{architecture}/attacks'
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    # second, create the result file.
    t = ''
    if is_l2: t = 'l2'
    elif is_l1: t = 'l1'
    elif is_early_stop: t = 'es'
    elif is_flooding: t = 'fl'
    elif is_data_augment: t = 'da'
    elif is_drop_block: t = 'db'
    elif is_hidden_mixup: t = 'hidden_mu'
    elif is_label_smoothing: t = 'ls'
    file = f'{d}/attack_result_{t}.json'
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
    # regularization params
    parser.add_argument('--l2', action='store_true',
                    help='Compare results that use l2 regularization')
    parser.add_argument('--l1', action='store_true',
                    help='Compare results that use l1 regularization')
    parser.add_argument('--early-stop', action='store_true',
                    help='Compare results that use early stop')
    parser.add_argument('--label-smoothing', action='store_true',
                    help='Compare results that use label smoothing')
    parser.add_argument('--flooding', action='store_true',
                    help='Compare reusults that use flooding.')
    parser.add_argument('--data-augment', action='store_true',
                    help='Compare reusults that use data augmentation including cutmix and mixup')
    parser.add_argument('--hidden-mixup', action='store_true',
                    help='Compare reusults that use mixup on hidden layers')
    parser.add_argument('--drop-block', action='store_true',
                    help='Compare reusults that use drop block')

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
        'is_label_smoothing': args.label_smoothing,
        'is_flooding': args.flooding,
        'is_data_augment': args.data_augment,
        'is_hidden_mixup': args.hidden_mixup,
        'is_drop_block': args.drop_block,
    }
    
    main(**kwargs)