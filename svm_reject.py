import numpy as np
import argparse
import pickle as pkl
import anndata
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score


def pkl_io(mode, path, *args):
    if mode == "w":
        with open(path, "wb") as f:
            print("O: written to {}".format(path))
            pkl.dump(args[0], f)
    elif mode == "r":
        with open(path, "rb") as f:
            print("I: read from {}".format(path))
            return pkl.load(f)


def svm_rej(clf, ood_set, Threshold=0.7):
    """
    Inputs: 
        - clf: trained and calibrated sklearn SVM classifier model
        - ood_set: dataset with left out cell class for OOD
        - Threshold: threshold value for rejection
    Outputs:
        - predicted: numpy array of predicted labels and repsective
                     rejected labels. The rejected entries in 'predicted'
                     are labeled as 'Unknown'. 
    """
    predicted = clf.predict(ood_set)
    prob = np.max(clf.predict_proba(ood_set), axis = 1)
    unlabeled = np.where(prob < Threshold)
    predicted = predicted.astype("object")
    predicted[unlabeled] = 'Unknown'
    return predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVM-reject for OOD-like comparison")

    parser.add_argument("--dataset", type=str) # path to anndata
    parser.add_argument("--split_dict", type=str, help="path to split dict")

    args = parser.parse_args()

    db = pkl_io("r", args.dataset)
    dic = pkl_io("r", args.split_dict)

    tr, va, te, oo = dic.values()
    #train_set = db.X[tr, :]
    #train_lab = np.array(db.obs["int_label"][tr]).astype("int")

    valid_set = db.X[va, :]
    valid_lab = np.array(db.obs["int_label"][va]).astype("int")

    test_set = db.X[te, :]
    test_lab = np.array(db.obs["int_label"][te]).astype("int")

    ood_set = db.X[oo, :]
    ood_lab = np.array(db.obs["int_label"][oo]).astype("int")

    import time
    tic = time.time()

    print("Start rejection")
    clf = pkl_io("r", "svm_cal/svm_cal-d8/svm_cal_" + args.split_dict.split("_")[-1])
    print("OOD score: {}".format(clf.score(ood_set, ood_lab)))

    # rejection procedure
    pred_rej = svm_rej(clf, ood_set)
    print(pred_rej)

    toc = time.time()
    print("Rejection time: {} s".format(toc - tic))
 