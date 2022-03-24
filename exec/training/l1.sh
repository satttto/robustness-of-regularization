#!/bin/bash
# python3 main.py -d cifar10 -a resnet18 --l1 1e-5
# python3 main.py -d cifar10 -a resnet18 --l1 3e-5
# python3 main.py -d cifar10 -a resnet18 --l1 5e-5
# python3 main.py -d cifar10 -a resnet18 --l1 7e-5
# python3 main.py -d cifar10 -a resnet18 --l1 9e-5
# python3 main.py -d svhn -a resnet18 --l1 1e-6
# python3 main.py -d svhn -a resnet18 --l1 3e-6
# python3 main.py -d svhn -a resnet18 --l1 5e-6
# python3 main.py -d svhn -a resnet18 --l1 7e-6
# python3 main.py -d svhn -a resnet18 --l1 9e-6
python3 main.py -d cifar10 -a vgg13 --l1 1e-5
python3 main.py -d cifar10 -a vgg13 --l1 3e-5
python3 main.py -d cifar10 -a vgg13 --l1 5e-5
python3 main.py -d cifar10 -a vgg13 --l1 7e-5
python3 main.py -d cifar10 -a vgg13 --l1 9e-5