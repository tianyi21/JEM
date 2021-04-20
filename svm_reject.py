import utils
import torch 
from torch.utils.data import DataLoader, Dataset
import os 
import numpy as np
import pandas as pd
import time 
import argparse
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

# import functions from eval_wrn_ebm.py
from eval_wrn_ebm import return_set

def normalize_data(x, norm_type, eps=1e-4, rng=[-1., 1.]):
    """
    Function to normalize data:
        options:
        - using l2-norm: "norm"
        - using soft normalization: (x - mean) / (2*std + eps)
        - using hard  normalization: range [-1, 1] -> use rng to define norm range
    """
    if norm_type=="norm":
        for d in range(len(x[1, :])):
            x[:,d] = x[:,d] / np.linalg.norm(x[:,d])

    if norm_type=="soft":
        for d in range(len(x[1, :])):
            x[:, d] = (x[:, d]- np.mean(x[:, d])) / (2 * np.std(x[:, d]) + eps)

    if norm_type=="hard":
        a, b = rng[0], rng[1] # new range
        for d in range(len(x[1, :])):
            x[:,d] = 2 * (x[:,d] - np.min(x[:,d])) / (np.max(x[:,d]) - np.min(x[:,d]) + eps) + b

    return x


def main(args):

    # Load single-cell data
    # --
    if not os.path.isfile(args.split_dict):
        raise FileNotFoundError("set split not found.")
    set_split_dict = utils.pkl_io("r", args.split_dict)
    db = utils.pkl_io("r", args.dataset) # (N x 5000) - 5000 genes

    # load train datad
    dset_train = return_set(db, "train", set_split_dict) # splits datast into train/test/val/ood 
    dload_train = DataLoader(dset_train, batch_size=len(db), shuffle=True, drop_last=False) 
    x_train = next(iter(dload_train))[0].numpy()
    y_train = next(iter(dload_train))[1].numpy()
    x_train = normalize_data(x_train, norm_type=args.normalize)  # data for numerical stability
    print("Train data partition:"), print(x_train.shape), print(y_train.shape), print(y_train) # check if conversion to numpy array is correct

    # load val data
    dset_val = return_set(db, "valid", set_split_dict) # splits datast into train/test/val/ood 
    dload_val = DataLoader(dset_val, batch_size=len(db), shuffle=True, drop_last=False) 
    x_val = next(iter(dload_val))[0].numpy()
    y_val = next(iter(dload_val))[1].numpy()
    x_val = normalize_data(x_val, norm_type=args.normalize) # data for numerical stability
    print("Val data partition:"), print(x_val.shape), print(y_val.shape), print(y_val) # check if conversion to numpy array is correct

    # load test data
    dset_test = return_set(db, "test", set_split_dict) # splits datast into train/test/val/ood 
    dload_test = DataLoader(dset_test, batch_size=len(db), shuffle=True, drop_last=False) 
    x_test = next(iter(dload_test))[0].numpy()
    y_test = next(iter(dload_test))[1].numpy()
    x_test = normalize_data(x_test, norm_type=args.normalize) # data for numerical stability
    print("Test data partition:"), print(x_test.shape), print(y_test.shape), print(y_test) # check if conversion to numpy array is correct

    # load ood data
    dset_ood = return_set(db, "ood", set_split_dict) # splits datast into train/test/val/ood 
    dload_ood = DataLoader(dset_ood, batch_size=len(db), shuffle=True, drop_last=False) 
    x_ood = next(iter(dload_ood))[0].numpy()
    y_ood = next(iter(dload_ood))[1].numpy()
    x_ood = normalize_data(x_ood, norm_type=args.normalize) # data for numerical stability
    print("OOD data partition:"), print(x_ood.shape), print(y_ood.shape), print(y_ood) # check if conversion to numpy array is correct

    print(np.max(x_train)), print(np.min(x_train))

    # Train, Calibrate, and Test SVM-reject model
    # --
    # Train
    svm = LinearSVC(verbose=True, max_iter=args.max_iter)
    print("LinearSVC - model training")
    svm.fit(x_train, y_train) # train SVM classifier
    print("Model trained")

    # Callibrate
    svm_cal = CalibratedClassifierCV(svm, cv='prefit')
    print("Calibrating model")
    svm_cal.fit(x_val, y_val)
    print("Calibration finished")

    # Test
    predicted = svm_cal.predict(x_test)
    print("Test Accuracy:"), print(accuracy_score(y_test, predicted))
    print("Test F1-score:"), print(f1_score(y_test, predicted, average="macro"))
    print(predicted), print(y_test)

    # OOD (sort of?)
    Threshold = 0.7 # Rejection threshold
    prob = np.max(svm_cal.predict_proba(x_ood), axis = 1)
    unlabeled = np.where(prob < Threshold)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVM-reject for OOD-like comparison")

    parser.add_argument("--dataset", type=str) # path to anndata
    parser.add_argument("--split_dict", type=str, help="path to split dict")
    parser.add_argument("--rset", type=str, choices=["train", "test", "valid", "ood", "test+ood"], default="train", help="OODAUC real dateset")
    parser.add_argument("--max_iter", type=int, default=1000) # maximum number of iterations for training SVM
    parser.add_argument("--normalize", type=str, default="soft", choices=["norm", "hard", "soft"])

    args = parser.parse_args()

    print(time.ctime())
    for item in args.__dict__:
        print("{:24}".format(item), "->\t", args.__dict__[item])
    
    main(args)