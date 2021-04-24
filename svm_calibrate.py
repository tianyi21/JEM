import time
import utils
import numpy as np
import argparse
import pickle as pkl
import anndata
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SVM-reject for OOD-like comparison")

    parser.add_argument("--dataset", type=str) # path to anndata
    parser.add_argument("--split_dict", type=str, help="path to split dict")
    parser.add_argument("--pretrain", action="store_true", help="set when uncalib available as pkl")
    parser.add_argument("--precalib", action="store_true", help="set when both uncalib and calib available as pkl")

    args = parser.parse_args()

    db = utils.pkl_io("r", args.dataset)
    dic = utils.pkl_io("r", args.split_dict)

    tr, va, te, oo = dic.values()
    train_set = db.X[tr, :]
    train_lab = np.array(db.obs["int_label"][tr]).astype("int")

    valid_set = db.X[va, :]
    valid_lab = np.array(db.obs["int_label"][va]).astype("int")

    test_set = db.X[te, :]
    test_lab = np.array(db.obs["int_label"][te]).astype("int")

    ood_set = db.X[oo, :]
    ood_lab = np.array(db.obs["int_label"][oo]).astype("int")
    
    if args.precalib:
        # both svm and svm_cal avail
        clf = utils.pkl_io("r", "./svm_" + args.split_dict.split("_")[-1])
        clf_calib = utils.pkl_io("r", "./svm_cal_" + args.split_dict.split("_")[-1])
    else:
        # svm_cal not avail
        if args.pretrain:
            # svm avail
            clf = utils.pkl_io("r", "./svm_" + args.split_dict.split("_")[-1])
        else:
            # svm not avail either, train from scratch
            tic = time.time()
            print("Start training")
            clf = LinearSVC()
            clf.fit(train_set, train_lab)
            toc = time.time()
            print("Train time: {} s".format(toc - tic))
            utils.pkl_io("w", "./svm_" + args.split_dict.split("_")[-1], clf)
        # calib
        tic = time.time()
        clf_calib = CalibratedClassifierCV(clf, cv='prefit')
        clf_calib.fit(valid_set, valid_lab)
        toc = time.time()
        print("Calibration time: {} s".format(toc - tic))
        utils.pkl_io("w", "./svm_cal_" + args.split_dict.split("_")[-1], clf_calib)

    # pre-calibration stats are deprecated: SVM_reject is a calibrated LinearSVC()
    # print("Pre-Calibration stats:")
    # print("Valid f1-score: {}".format(f1_score(valid_lab, clf.predict(valid_set), average="macro")))
    # print("Valid accuracy: {}".format(clf.score(valid_set, valid_lab)))
    # print("Test f1-score: {}".format(f1_score(test_lab, clf.predict(test_set), average="macro")))
    # print("Test accuracy: {}".format(clf.score(test_set, test_lab)))

    # Print accuracy and f1-scores on validation and test sets
    print("Calibrated stats:")
    print("Valid f1-score: {}".format(f1_score(valid_lab, clf_calib.predict(valid_set), average="macro")))
    print("Valid accuracy: {}".format(clf_calib.score(valid_set, valid_lab)))
    print("Test f1-score: {}".format(f1_score(test_lab, clf_calib.predict(test_set), average="macro")))
    print("Test accuracy: {}".format(clf_calib.score(test_set, test_lab)))
