#!/bin/bash
# python3 main.py -d cifar10 -a resnet18 --early-stop vacc --patience 1
# python3 main.py -d cifar10 -a resnet18 --early-stop vacc --patience 2
# python3 main.py -d cifar10 -a resnet18 --early-stop vacc --patience 3
# python3 main.py -d cifar10 -a resnet18 --early-stop vacc --patience 4
# python3 main.py -d cifar10 -a resnet18 --early-stop vacc --patience 5
# python3 main.py -d svhn -a resnet18 --early-stop vacc --patience 1
# python3 main.py -d svhn -a resnet18 --early-stop vacc --patience 2
# python3 main.py -d svhn -a resnet18 --early-stop vacc --patience 3
# python3 main.py -d svhn -a resnet18 --early-stop vacc --patience 4
# python3 main.py -d svhn -a resnet18 --early-stop vacc --patience 5
python3 main.py -d cifar10 -a vgg13 --early-stop vacc --patience 1
python3 main.py -d cifar10 -a vgg13 --early-stop vacc --patience 2
python3 main.py -d cifar10 -a vgg13 --early-stop vacc --patience 3
python3 main.py -d cifar10 -a vgg13 --early-stop vacc --patience 4
python3 main.py -d cifar10 -a vgg13 --early-stop vacc --patience 5