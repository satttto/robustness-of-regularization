import argparse
import glob
import pandas as pd
from utils import get_base_path

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_columns(metric):
    d = {
        'all': ['acc', 'train_loss', 'val_loss'],
        'acc' : ['acc'],
        'tloss': ['train_loss'],
        'vloss': ['val_loss'],
    }
    return d[metric]


def main(base_path, architecture, dataset_type, metric, \
        lr, \
        is_l2, is_early_stop, is_flooding, is_data_augment):

    # set path to results folder
    if base_path.endswith('/'):
        base_path = base_path[:-1]
    path = f'{base_path}/{dataset_type}/{architecture}/results/*.csv'

    # extract needed file paths
    results_paths = glob.glob(path)
    if not is_l2:
        # remove files that have 'es' in its name
        results_paths = list(filter(lambda s: 'l2' not in s.rsplit('/', 1)[1], results_paths))
    if not is_early_stop:
        # remove files that have 'es'(early stopping) in its name
        results_paths = list(filter(lambda s: 'es' not in s.rsplit('/', 1)[1], results_paths))
    if not is_flooding:
        # remove files that have 'fl'(flooding) in its name
        results_paths = list(filter(lambda s: 'fl' not in s.rsplit('/', 1)[1], results_paths))
    if not is_data_augment:
        # remove files that have 'da'(data augmentation) in its name　
        results_paths = list(filter(lambda s: 'da' not in s.rsplit('/', 1)[1], results_paths))
    if not is_fgsm:
        # remove files that have 'fgsm' in its name
        results_paths = list(filter(lambda s: 'fgsm' not in s.rsplit('/', 1)[1], results_paths))
    if len(results_paths) == 0:
        raise FileNotFoundError('No such result. You may train models first')
    results_paths.sort()
    print(results_paths)

    # read results
    results = []
    cols = get_columns(metric)
    for file_path in results_paths:
        csv = pd.read_csv(filepath_or_buffer=file_path, sep=',')
        results.append(csv[cols])
    
    # visualize
    for col in cols:
        plt.figure()
        for i in range(len(results)):
            label = results_paths[i].rsplit('/', 1)[1].replace('.csv', '')
            plt.plot(range(1, len(results[i])+1), results[i][col].values, color=cm.jet(1-i/20.0), label=label)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel(col)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line')
    parser.add_argument('-p', '--path', default='2.Lab/experiment/',
                    help='Folder where model is stored. Specify the folder name after "/content/drive/My Drive/"')
    parser.add_argument('-a', '--architecture', default='mlp',
                    help='Model type. Available optins: mlp, mlp-bn, resnet18. Default is mlp')
    parser.add_argument('-d', '--dataset', default='mnist',
                    help='Dataset. Available options: mnist, cifar10')
    parser.add_argument('--metric', default='all',
                    help='What graphs do you need? Options: all, acc, tloss as in train loss, vloss as in validation loss')
    # optimization params
    parser.add_argument('--lr', default=0.1, type=float,
                    help='Learning rate. Default is 0.1')
    # regularization params
    parser.add_argument('--l2', action='store_true',
                    help='Compare results that use l2 regularization')
    parser.add_argument('--early-stop', action='store_true',
                    help='Compare results that use early stop')
    parser.add_argument('--flooding', action='store_true',
                    help='Compare reusults that use flooding.')
    parser.add_argument('--data-augment', action='store_true',
                    help='Compare reusults that use data augmentation')

    args = parser.parse_args()
    base_path = get_base_path(args.path)

    kwargs = {
        'base_path': base_path,
        'architecture': args.architecture, 
        'dataset_type': args.dataset,
        'metric': args.metric,
        'lr': 0.07,
        'is_l2': args.l2,
        'is_early_stop': args.early_stop,
        'is_flooding': args.flooding,
        'is_data_augment': args.data_augment,
    }

    main(**kwargs)