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
python train_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 10 --backbone mlp --arch 500 200  --print_every 20 --batch_size 128 --class_drop 1

## resnet example
python train_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 10 --backbone resnet --arch 75 50 --num_block 2 1  --print_every 20 --batch_size 128 --class_drop 1

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
python eval_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 10 --backbone mlp --arch 500 200 --logpset test+ood --split_dict ./experiment/set_split_idx_ood_1.pkl --eval OOD --load_path PATH_TO_best_valid_ckpt_ood_[0-10].pt

## ood example
## default --eval is ood, default --rset is train, default --fset is test+ood
"""
python eval_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 10 --backbone mlp --arch 500 200 --rset train --fset test ood --split_dict ./experiment/set_split_idx_ood_1.pkl --load_path PATH_TO_best_valid_ckpt_ood_[0-10].pt
"""
# Resnet example

for i in {0..10}
do
   python eval_wrn_ebm.py --dataset /scratch/gobi2/phil/data/pbmc_filtered.pkl --n_classes 10 --backbone resnet --arch 800 200 --rset test --fset ood --split_dict /scratch/gobi2/phil/JEM_experiment/resnet/res-d${i}/experiment/set_split_idx_ood_${i}.pkl --class_drop ${i} --load_path /scratch/gobi2/phil/JEM_experiment/resnet/res-d${i}/experiment/best_valid_ckpt_ood_${i}.pt --num_block 2 4 --score_fn pxy px py pxgrad pxygrad svm_cal --act_func elu --save_dir ./plots/resnet_${i}_img --svm_cal_path /scratch/gobi2/phil/JEM_experiment/svm_cal/

   python eval_wrn_ebm.py --dataset /scratch/gobi2/phil/data/pbmc_filtered.pkl --n_classes 10 --backbone mlp --arch 2000 1200 600 200 --rset test --fset ood --split_dict /scratch/gobi2/phil/JEM_experiment/mlp/mlp-d${i}/experiment/set_split_idx_ood_${i}.pkl --class_drop ${i} --load_path /scratch/gobi2/phil/JEM_experiment/mlp/mlp-d${i}/experiment/best_valid_ckpt_ood_${i}.pt --score_fn pxy px py pxgrad pxygrad svm_cal --save_dir ./plots/mlp_${i}_img --svm_cal_path /scratch/gobi2/phil/JEM_experiment/svm_cal/
done

python eval_wrn_ebm.py --dataset /scratch/gobi2/phil/data/pbmc_filtered.pkl --n_classes 10 --backbone resnet --arch 800 200 --rset test --fset ood --split_dict /scratch/gobi2/phil/JEM_experiment/resnet/res-d0/experiment/set_split_idx_ood_0.pkl --class_drop 0 --load_path /scratch/gobi2/phil/JEM_experiment/resnet/res-d0/experiment/best_valid_ckpt_ood_0.pt --num_block 2 4 --score_fn pxy px py pxgrad pxygrad --act_func elu

# MLP example
python eval_wrn_ebm.py --dataset /scratch/gobi2/phil/data/pbmc_filtered.pkl --n_classes 10 --backbone mlp --arch 2000 1200 600 200 --rset test --fset ood --split_dict /scratch/gobi2/phil/JEM_experiment/mlp/mlp-d0/experiment/set_split_idx_ood_0.pkl --class_drop 0 --load_path /scratch/gobi2/phil/JEM_experiment/mlp/mlp-d0/experiment/best_valid_ckpt_ood_0.pt --score_fn pxy px py pxgrad pxygrad


## calibration example
## with JEM
## calibmodel:                  jem: default
python eval_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 10 --backbone resnet --arch 800 200 --num_block 2 4 --calibset train --load_path ~/Desktop/best_valid_ckpt_ood_0.pt --split_dict ~/Desktop/set_split_idx_ood_0.pkl --eval calib --class_drop 0 --calibmodel jem

## with other classifier
## calibmodel:                  pickled LinearSVC()
## clf_path:                    path to calibmodel
python eval_wrn_ebm.py --dataset ~/Desktop/pbmc_filtered.pkl --n_classes 10  --calibset train --split_dict ~/Desktop/set_split_idx_ood_0.pkl --eval calib --class_drop 0 --calibmodel clf --clf_path ~/Desktop/svm_0.pkl
