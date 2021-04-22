#   -*- coding: utf-8 -*-
#
#   calib_group_plot.py
#
#   Created by Tianyi Liu on 2021-04-22 as tianyi
#   Copyright (c) 2021. All Rights Reserved.

"""

"""


import os
import pickle as pkl 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pkl_io(mode, path, *args):
    if mode == "w":
        with open(path, "wb") as f:
            print("O: written to {}".format(path))
            pkl.dump(args[0], f)
    elif mode == "r":
        if not os.path.isfile(path):
            raise FileNotFoundError("{} not found".format(path))
        with open(path, "rb") as f:
            print("I: read from {}".format(path))
            return pkl.load(f)


def to_percentage(dec, pos=2):
    return str(np.round(dec * 100, pos)) + " %"


svm = pkl_io("r", "./svm.pkl")
svm_cal = pkl_io("r", "./svm_calib.pkl")
resnet = pkl_io("r", "./resnet.pkl")
mlp = pkl_io("r", "./mlp.pkl")

sns.set()
sns.set_context('talk')

plt.figure(figsize=(25, 6))
plt.tight_layout()
plt.subplot(1,4,1)
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.bar(svm["xval"][:len(svm["xval"]) - 1], svm["acc_avg"], width=1/len(svm["acc_avg"]), align="edge")
plt.plot([0,1], [0,1], "r--")
plt.title("Uncalibrated SVM ECE={}".format(to_percentage(svm["ece"])))

plt.subplot(1,4,2)
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.bar(svm_cal["xval"][:len(svm_cal["xval"]) - 1], svm_cal["acc_avg"], width=1/len(svm_cal["acc_avg"]), align="edge")
plt.plot([0,1], [0,1], "r--")
plt.title("Calibrated SVM ECE={}".format(to_percentage(svm_cal["ece"])))

plt.subplot(1,4,3)
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.bar(resnet["xval"][:len(resnet["xval"]) - 1], resnet["acc_avg"], width=1/len(resnet["acc_avg"]), align="edge")
plt.plot([0,1], [0,1], "r--")
plt.title("ResNet JEM ECE={}".format(to_percentage(resnet["ece"])))

plt.subplot(1,4,4)
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.bar(mlp["xval"][:len(mlp["xval"]) - 1], mlp["acc_avg"], width=1/len(mlp["acc_avg"]), align="edge")
plt.plot([0,1], [0,1], "r--")
plt.title("MLP JEM ECE={}".format(to_percentage(mlp["ece"])))

# plt.show()
plt.savefig("ece.pdf")
