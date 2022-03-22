import argparse
import torch
import torch.nn as nn
from dataset.mnist import MNIST
from dataset.cifar10 import CIFAR10
from dataset.cifar10_aug import CIFAR10Aug
from dataset.svhn import SVHN
from dataset.svhn_aug import SVHNAug
from model_maker import ModelMaker
from trainers.base import Trainer
from trainers.l2 import L2Trainer
from trainers.l1 import L1Trainer
from trainers.flooding import FloodTrainer
from trainers.early_stopping import ESTrainer
from trainers.label_smoothing import LabelSmoothingTrainer
from trainers.data_augment import DATrainer
from trainers.mixup import MixUpTrainer
from trainers.cutmix import CutMixTrainer
from trainers.dropblock import DropBlockTrainer
from utils import get_base_path, is_mlp


def main(base_path, architecture, dataset_type, batch_size, restore_best, \
        num_epochs, lr, momentum, \
        l2_lambda, l1_lambda, es_metric, es_patience, label_smoothing, flood_level, aug_methods, \
        mixup, mixup_hidden, mixup_alpha, mixup_prob, \
        cutmix, cutmix_alpha, cutmix_prob, \
        drop_block, drop_prob, block_size):

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # remove randomness
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
    # prepare dataset based on database type
    require_flatten = is_mlp(architecture)
    if dataset_type == 'mnist':
        dataset = MNIST(batch_size, require_flatten)
    elif dataset_type == 'cifar10':
        if len(aug_methods) != 0:
            dataset = CIFAR10Aug(batch_size, aug_methods)
        else: 
            dataset = CIFAR10(batch_size, require_flatten)
    elif dataset_type == 'svhn':
        if len(aug_methods) != 0:
            dataset = SVHNAug(batch_size, aug_methods)
        else: 
            dataset = SVHN(batch_size, require_flatten)
    else:
        raise ValueError('This dataset is not supported yet.') 
    dataset.fetch()

    # parameters of model
    model_params = {
        'num_classes': dataset.num_classes,
        'input_size': dataset.input_size,
    }
    # model option
    option = None
    if mixup:
        model_params['mixup_hidden'] = mixup_hidden
        option = 'mixup'
    elif drop_block:
        model_params['drop_prob'] = drop_prob
        model_params['block_size'] = block_size
        option = 'dropblock'
    # make model based on architecture name
    maker = ModelMaker(architecture, dataset_type, option, **model_params)
    model = maker.model.to(device)
    
    # parameters for training
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_lambda)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 105], gamma=0.1) # 100 epoch: lr*0.1, 150 epoch: lr*0.1*0.1  
    train_params = {
        'device': device,
        'num_epochs': num_epochs,
        'criterion': criterion,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
    }
    if flood_level:
        train_params['flood_level'] = flood_level
        trainer = FloodTrainer(model, **train_params)
    elif es_metric:
        train_params['metric'] = es_metric
        train_params['patience'] = es_patience
        trainer = ESTrainer(model, **train_params)
    elif l2_lambda:
        train_params['l2_lambda'] = l2_lambda # pass this param for result  file names (model, result).
        trainer = L2Trainer(model, **train_params)
    elif l1_lambda:
        train_params['l1_lambda'] = l1_lambda
        trainer = L1Trainer(model, **train_params)
    elif len(aug_methods) != 0:
        train_params['methods'] = aug_methods
        trainer = DATrainer(model, **train_params)
    elif mixup:
        if not mixup_hidden:
            train_params['layer_mix'] = 0
        train_params['mixup_alpha'] = mixup_alpha
        train_params['mixup_prob'] = mixup_prob
        trainer = MixUpTrainer(model, **train_params)
    elif cutmix:
        train_params['cutmix_alpha'] = cutmix_alpha
        train_params['cutmix_prob'] = cutmix_prob
        trainer = CutMixTrainer(model, **train_params)
    elif drop_block:
        train_params['drop_prob'] = drop_prob
        train_params['block_size'] = block_size
        trainer = DropBlockTrainer(model, **train_params)
    elif label_smoothing:
        train_params['label_smoothing'] = label_smoothing
        trainer = LabelSmoothingTrainer(model, **train_params)
    else:
        trainer = Trainer(model, **train_params)
    trainer.print_params()

    # training 
    trainer.train_model(dataset.train_loader, dataset.test_loader)
    if restore_best:
        trainer.restore_best_model()

    # save model and result 
    # base has to be like {dataset}/{architecture}/
    if base_path.endswith('/'):
        base_path = base_path[:-1]
    base_path += f'/{dataset_type}/{architecture}/'
    trainer.save_model_in(base_path)
    trainer.save_result_in(base_path)


if __name__ == '__main__':
    # Parse command line 
    parser = argparse.ArgumentParser(description='Command Line')
    # Basic params
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--platform', default='', help='colab or not specifid if it is local')
    parser.add_argument('-p', '--path', default='results',
                    help='Folder to store the result. Specify the folder name after "/content/drive/My Drive/"')
    parser.add_argument('-a', '--architecture', default='mlp',
                    help='Architecture type. Available options: mlp, mlp-bn, resnet18 and vgg13. Default is mlp')
    parser.add_argument('-d', '--dataset', default='mnist',
                    help='Dataset. Available options: mnist, cifar10')
    parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size. Default is 128')
    parser.add_argument('-b', '--restore-best', action='store_true',
                    help='Do you want the model that records the best test accuracy? If so, use this flag')
    # Optimization params
    parser.add_argument('-n', '--epochs', default=110, type=int,
                    help='The number of epochs, Default is 200')
    parser.add_argument('--lr', default=0.1, type=float,
                    help='Learning rate. Default is 0.1')
    parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum. Default is 0.9 (official default is 0)')
    # learning-restriction regularizaion 
    parser.add_argument('--l2', default=0, type=float,
                    help='L2 regularization lambda.')
    parser.add_argument('--l1', default=0, type=float,
                    help='L1 regularization lambda.')
    parser.add_argument('--early-stop', 
                    help='Early stop metric. Available options are tloss, vloss, vacc')
    parser.add_argument('--patience', default=5, type=int,
                    help='Early stop patience. Must be int. Default is 5')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                    help='Label smoothing param')
    # data-augment regularization
    parser.add_argument('--flood-level', type=float,
                    help='Flooding level.')
    parser.add_argument('--crop', action='store_true',
                    help='Whether use random crop technique')
    parser.add_argument('--hflip', action='store_true',
                    help='Whether use hflip technique')
    parser.add_argument('--rotate', action='store_true',
                    help='Whether use random rotate technique')
    parser.add_argument('--erase', action='store_true',
                    help='Whether use random erase technique')
    parser.add_argument('--mixup', action='store_true',
                    help='Whether use MixUp technique')
    parser.add_argument('--mixup-hidden', action='store_true',
                    help='Whether appling MixUp technique to hidden layers')
    parser.add_argument('--mixup-alpha', default=1., type=float,
                    help='Mixup alpha. Default is 1. Change it to 2 if you want to mixup hidden layers')
    parser.add_argument('--mixup-prob', default=1., type=float,
                    help='Mixup alpha. Default is 1')       
    parser.add_argument('--cutmix', action='store_true',
                    help='Whether use CutMix technique')
    parser.add_argument('--cutmix-alpha', default=1., type=float,
                    help='Cutmix alpha. Default is 1')
    parser.add_argument('--cutmix-prob', default=0.5, type=float,
                    help='Cutmix probability. Default is 0.5')
    # model-based regularization
    parser.add_argument('--drop-block', action='store_true',
                    help='Whether use Drop Block technique')
    parser.add_argument('--drop-prob', type=float, default=0.,
                    help='dropblock dropout probability')
    parser.add_argument('--block-size', type=int, default=5,
                    help='dropblock block size')
    
    args = parser.parse_args()

    # Change the directory to store results when it's debug mode
    if args.debug:
        args.path += '/debug'    
    base_path = get_base_path(args.path, args.platform)

    # data augmentation option
    aug_methods = []
    if args.crop:
        aug_methods.append('crop')
    if args.hflip:
        aug_methods.append('hflip')
    if args.rotate:
        aug_methods.append('rotate')
    if args.erase:
        aug_methods.append('erase')

    # keyword arguments
    kwargs = {
        'base_path': base_path,
        'architecture': args.architecture, 
        'dataset_type': args.dataset,
        'batch_size': args.batch_size,
        'restore_best': args.restore_best,
        'num_epochs': args.epochs,
        'lr': args.lr,
        'momentum': args.momentum,
        'l2_lambda': args.l2,
        'l1_lambda': args.l1,
        'es_metric': args.early_stop,
        'es_patience': args.patience,
        'label_smoothing': args.label_smoothing,
        'flood_level': args.flood_level,
        'aug_methods': aug_methods,
        'mixup': args.mixup,
        'mixup_hidden': args.mixup_hidden,
        'mixup_alpha': args.mixup_alpha,
        'mixup_prob': args.mixup_prob,
        'cutmix': args.cutmix,
        'cutmix_alpha': args.cutmix_alpha,
        'cutmix_prob': args.cutmix_prob,
        'drop_block': args.drop_block,
        'drop_prob': args.drop_prob,
        'block_size': args.block_size,
    }
    
    main(**kwargs)
