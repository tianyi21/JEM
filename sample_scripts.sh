#! /bin/sh
#
# sample_scripts.sh
# Copyright (C) 2021 tianyi <tianyi@Tianyis-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.
#


# Training
## n_classes:   number of classes before dropping
## backbone:    either mlp or resnet
## arch:        MLP: dimension of each layer
##              resnet: dimension of each residual layer
## num_block:   only for resnet, number of residual block per residual layer
## class_drop:  the index of class dropped, -1 stands for no dropping 

## mlp example
python train_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 11 --backbone mlp --arch 500 200  --print_every 20 --batch_size 128 --class_drop 1

## resnet example
python train_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 11 --backbone resnet --arch 75 50 --num_block 2 1  --print_every 20 --batch_size 128 --class_drop 1

# OOD
## eval:        logp_hist, ood: see corresponding documentation below
##              in case of 
##                      logp_hist:
##                              plot histogram for a given set of data
##                                    specify --logpset
##                      ood:         
##                              plot histogram for comparison against rset sets
##                                    specify --rset for train set
##                                            --fset for "fake", i.e., list of sets to be compared to --rset 
## split_dict:  set split dict in training
## set n_classes, backbone, arch, num_block accordingly to the training params

## logp_hist example
python eval_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 11 --backbone mlp --arch 500 200 --logpset test+ood --split_dict ./experiment/set_split_idx_ood_1.pkl --eval OOD --load_path PATH_TO_best_valid_ckpt_ood_[0-10].pt

## ood example
## default --eval is ood, default --rset is train, default --fset is test+ood
"""
python eval_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 11 --backbone mlp --arch 500 200 --rset train --fset test ood --split_dict ./experiment/set_split_idx_ood_1.pkl --load_path PATH_TO_best_valid_ckpt_ood_[0-10].pt
"""
python eval_wrn_ebm2.py --dataset ./data/pbmc_filtered.pkl --n_classes 11 --backbone resnet --arch 800 200 --rset train --fset test ood --split_dict models/res-d0/experiment/set_split_idx_ood_0.pkl --load_path ./models/res-d0/experiment/best_valid_ckpt_ood_0.pt --num_block 2 4