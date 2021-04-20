import numpy as np
import argparse
import pickle as pkl
import anndata
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def pkl_io(mode, path, *args):
    if mode == "w":
        with open(path, "wb") as f:
            print("O: written to {}".format(path))
            pkl.dump(args[0], f)
    elif mode == "r":
        with open(path, "rb") as f:
            print("I: read from {}".format(path))
            return pkl.load(f)


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
    print("Start calibrating")
    #clf = LinearSVC()
    #pkl_io("r", "svm/svm-d0/svm_" + args.split_dict.split("_")[-1], clf)
    #clf = pickle.load(open("tuple_model.pkl", 'rb'))
    clf = pkl_io("r", "svm/svm-d10/svm_" + args.split_dict.split("_")[-1])
    clf = CalibratedClassifierCV(clf, cv='prefit')
    clf.fit(valid_set, valid_lab)
    toc = time.time()
    print("Calibration time: {} s".format(toc - tic))
    
    pkl_io("w", "svm_cal/svm_cal-d10/svm_cal_" + args.split_dict.split("_")[-1], clf)

    # Print accuracy and f1-scores on validation and test sets
    print("Valid f1-score: {}".format(f1_score(valid_lab, clf.predict(valid_set), average="macro")))
    print("Valid accuracy: {}".format(accuracy_score(valid_lab, clf.predict(valid_set))))
    print("Test f1-score: {}".format(f1_score(test_lab, clf.predict(test_set), average="macro")))
    print("Test accuracy: {}".format(accuracy_score(test_lab, clf.predict(test_set))))