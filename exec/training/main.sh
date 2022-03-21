#!/bin/bash
sh exec/training/cutmix.sh
sh exec/training/data_augment.sh
sh exec/training/drop_block.sh
sh exec/training/early_stop.sh
sh exec/training/flooding.sh
sh exec/training/l1.sh
sh exec/training/l2.sh
sh exec/training/label_smoothing.sh
sh exec/training/mixup.sh