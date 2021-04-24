# CLAMS: Cell Type Classification Using Energy-Based Models

Official code for the paper [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263).

![clams](figs/clams.jpg)

Includes scripts for training JEM (Joint-Energy Model), evaluating models at various tasks, and running adversarial attacks.

A pretrained model on CIFAR10 can be found [here](http://www.cs.toronto.edu/~wgrathwohl/CIFAR10_MODEL.pt).


## Usage
### Training
Training CLAMS from scratch
```
# CLAMS: ResNet JEM
python train_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --n_classes 10 --backbone resnet --arch 800 200 --num_block 2 4 --act_func elu --class_drop 0 --decay_epochs 6 15 --n_epochs 20 --checkpoint

# CLAMS: MLP JEM
python train_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --n_classes 10 --backbone mlp --arch 2000 1200 600 200 --act_func elu --class_drop 0 --decay_epochs 6 15 --n_epochs 20 --checkpoint
```

### Evaluation
OOD ROC AUC
```
# For all models
for i in {0..10}
do
   python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --eval OOD --rset test --fset ood --n_classes 10 --backbone resnet --arch 800 200 --num_block 2 4 --act_func elu --split_dict PATH_TO_PICKLED_SET_SPLIT --class_drop ${i} --load_path PATH_TO_TRAINED_MODEL --score_fn pxy px py px_grad pxy_grad py_grad svm_cal --save_dir ./plots/resnet_${i}_img --svm_cal_path PATH_TO_TRAINED_SVM

   python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --eval OOD --rset test --fset ood --n_classes 10 --backbone mlp --arch 2000 1200 600 200 --act_func elu --split_dict PATH_TO_PICKLED_SET_SPLIT --class_drop ${i} --load_path PATH_TO_TRAINED_MODEL --score_fn pxy px py px_grad pxy_grad py_grad svm_cal --save_dir ./plots/mlp_${i}_img --svm_cal_path PATH_TO_TRAIN_SVM
done

# CLAMS: ResNet JEM
python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --eval OOD --rset test --fset ood --n_classes 10 --backbone resnet --arch 800 200 --num_block 2 4 --act_func elu --split_dict PATH_TO_PICKLED_SET_SPLIT --class_drop 0 --load_path PATH_TO_TRAINED_MODEL --score_fn pxy px py pxgrad pxygrad

# CLAMS: MLP JEM
python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --eval OOD --rset test --fset ood --n_classes 10 --backbone mlp --arch 2000 1200 600 200 --act_func elu --split_dict PATH_TO_PICKLED_SET_SPLIT --class_drop 0 --load_path PATH_TO_TRAINED_MODEL --score_fn pxy px py pxgrad pxygrad
```

Calibration
```
# CLAMS: ResNet JEM
python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --eval calib --calibmodel resnet --calibset test --n_classes 10 --backbone resnet --arch 800 200 --num_block 2 4 --act_func elu --split_dict PATH_TO_PICKLED_SET_SPLIT --class_drop 0 --load_path PATH_TO_TRAINED_MODEL

# CLAMS: MLP JEM
python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET --eval calib --calibmodel jem --calibset test --n_classes 10 --backbone mlp --arch 2000 1200 600 200 --act_func elu --split_dict PATH_TO_PICKLED_SET_SPLIT --class_drop 0 --load_path PATH_TO_TRAINED_MODEL

# SVM:
python eval_wrn_ebm.py --dataset PATH_TO_PICKLED_DATASET  --eval calib --calibmodel svm --calibset test --n_classes 10 --split_dictPATH_TO_PICKLED_SET_SPLIT --class_drop 0 --clf_path PATH_TO_TRAINED_SVM
```

SVM_reject baseline
```
python svm_calibrate.py --dataset PATH_TO_PICKLED_DATASET --split_dict PATH_TO_PICKLED_SET_SPLIT
```

Happy Energy-Based Modeling! 
