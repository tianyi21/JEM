# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import sys
import argparse
import numpy as np
import clamsnet
import pdb
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV

from tqdm import tqdm
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
n_gene = 5000


class SingleCellDataset(Dataset):
    """
    Create torch dataset from anndata
    """
    def __init__(self, adata):
        super(SingleCellDataset, self).__init__()
        self.data = adata.X
        self.dim = self.data.shape[1]
        self.gene = adata.var["n_counts"].index.values
        self.barcode = adata.obs["bulk_labels"].index.values
        self.label = np.array(adata.obs["int_label"].values)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, backbone, arch, num_block, req_bn, act_func, dropout_rate, n_classes):
        super(F, self).__init__()
        self.backbone = backbone
        self.arch = arch
        self.f = clamsnet.Clams_Net(backbone, arch, num_block, req_bn, act_func, dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        # forward is just a linear layer to single value
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        # This is just an output with n classes
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, backbone, arch, num_block, req_bn, act_func, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(
            backbone=backbone, arch=arch, num_block=num_block, req_bn=req_bn, act_func=act_func, dropout_rate=dropout_rate, n_classes=n_classes)
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(bs):
    return t.FloatTensor(bs, n_gene).uniform_(-1, 1)


def sample_p_0(device, replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not args.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs)
    choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(args, device, f, replay_buffer, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(device, replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(args.n_steps):
        f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def uncond_samples(f, args, device, save=True):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    # plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    replay_buffer = t.FloatTensor(args.buffer_size, n_gene).uniform_(-1, 1)
    tmp = []
    for i in range(args.n_sample_steps):
        samples = sample_q(args, device, f, replay_buffer)
        #if i % args.print_every == 0 and save:
        #    plot('{}/samples_{}.png'.format(args.save_dir, i), samples)
        tmp.append(samples)
        print(i)
    utils.makedirs("./samples")
    utils.pkl_io("w", "./samples/uncond.pkl", tmp)
    return replay_buffer


def cond_samples(f, replay_buffer, args, device, fresh=False):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    # plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    if fresh:
        replay_buffer = uncond_samples(f, args, device, save=False)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[i * 100: (i + 1) * 100].to(device)
        y = f.classify(x).max(1)[1]
        all_y.append(y)

    all_y = t.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    tmp = []
    for i in range(100):
        this_im = []
        for l in range(10):
            this_l = each_class[l][i * 10: (i + 1) * 10]
            this_im.append(this_l)
        this_im = t.cat(this_im, 0)
        tmp.append(this_im)
        # if this_im.size(0) > 0:
        #    plot('{}/samples_{}.png'.format(args.save_dir, i), this_im)
        print(i)
    utils.makedirs("./samples")
    utils.pkl_io("w", "./samples/cond.pkl", tmp)


def return_set(db, set, set_split_dict, clf=False):
    train_inds, valid_inds, test_inds, ood_inds = set_split_dict.values()
    print("Split Dict:\tTrain: {}\tValid: {}\tTest:{}\tOOD: {}".format(len(train_inds), len(valid_inds), len(test_inds), len(ood_inds)))
    if not clf:
        if set == "test+ood":
            # default, test in ttv + ood
            dataset = DataSubset(SingleCellDataset(db), inds=(test_inds + ood_inds))
        elif set == "ood":
            dataset = DataSubset(SingleCellDataset(db), inds=ood_inds)
        elif set == "test":
            dataset = DataSubset(SingleCellDataset(db), inds=test_inds)
        elif set == "train":
            dataset = DataSubset(SingleCellDataset(db), inds=train_inds)
        elif set == "valid":
            dataset = DataSubset(SingleCellDataset(db), inds=valid_inds)
        else:
            raise ValueError
        print("{} set retrived for OOD".format(set))
        return dataset
    else:
        if set == "test+ood":
            data = db.X[test_inds + ood_inds, :]
            label = np.array(db.obs["int_label"][test_inds + ood_inds]).astype("int")
        elif set == "ood":
            data = db.X[ood_inds, :]
            label = np.array(db.obs["int_label"][ood_inds]).astype("int")
        elif set == "test":
            data = db.X[test_inds, :]
            label = np.array(db.obs["int_label"][test_inds]).astype("int")
        elif set == "train":
            data = db.X[train_inds, :]
            label = np.array(db.obs["int_label"][train_inds]).astype("int")
        elif set == "valid":
            data = db.X[valid_inds, :]
            label = np.array(db.obs["int_label"][valid_inds]).astype("int")
        else:
            raise ValueError
        print("{} set retrived for OOD".format(set))
        return data, label


def logp_hist(f, args, device):
    sns.set()
    plt.switch_backend('agg')
    def sample(x, n_steps=args.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples
    def grad_norm(x):
        x_k = t.autograd.Variable(x, requires_grad=True)
        f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)
    def score_fn(x):
        if args.score_fn == "px":
            return f(x).detach().cpu()
        elif args.score_fn == "py":
            return nn.Softmax()(f.classify(x)).max(1)[0].detach().cpu()
        elif args.score_fn == "pxgrad":
            return -t.log(grad_norm(x).detach().cpu())
        elif args.score_fn == "refine":
            init_score = f(x)
            x_r = sample(x)
            final_score = f(x_r)
            delta = init_score - final_score
            return delta.detach().cpu()
        elif args.score_fn == "refinegrad":
            init_score = -grad_norm(x).detach()
            x_r = sample(x)
            final_score = -grad_norm(x_r).detach()
            delta = init_score - final_score
            return delta.detach().cpu()
        elif args.score_fn == "refinel2":
            x_r = sample(x)
            norm = (x - x_r).view(x.size(0), -1).norm(p=2, dim=1)
            return -norm.detach().cpu()
        else:
            return f.classify(x).max(1)[0].detach().cpu()

    if not os.path.isfile(args.split_dict):
        raise FileNotFoundError("set split not found.")
    set_split_dict = utils.pkl_io("r", args.split_dict)
    db = utils.pkl_io("r", args.dataset)
    dataset = return_set(db, args.logpset, set_split_dict)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    
    this_scores = []
    for x, _ in dataloader:
        x = x.to(device)
        scores = score_fn(x)
        # print(scores.mean())
        this_scores.extend(scores.numpy())

    # for name, scores in score_dict.items():
    plt.hist(scores.cpu().numpy(), label=args.logpset, bins=100, density=True, alpha=.5)
    plt.legend()
    plt.savefig("./img/logp_hist.pdf")


def OODAUC(f, args, device):
    sns.set()
    sns.set_context('talk')
    def score_fn(x, score_type='px'):
        permitted_score_types = ["px", "py", "pxgrad", "pxy", "pxygrad"]
        assert score_type in permitted_score_types, f"score function needs to be in {permitted_score_types}"
        # pdb.set_trace()
        if score_type == "px":
            return - f(x).detach().cpu()
        elif score_type == "py":
            return - nn.Softmax()(f.classify(x)).max(1)[0].detach().cpu()
        elif score_type == 'pxgrad':
            return -grad_norm(x).detach().cpu()
        elif score_type == 'pxy':
            return pxy(x).detach().cpu()
        elif score_type == 'pxygrad':
            return -grad_norm(x, fn=pxy).detach().cpu()
        else:
            raise ValueError

    def pxy(x):
        py = - nn.Softmax()(f.classify(x)).max(1)[0]
        px = f(x)
        return px * py

    # JEM grad norm function
    def grad_norm(x ,fn=f):
        x_k = t.autograd.Variable(x, requires_grad=True)
        f_prime = t.autograd.grad(fn(x_k).sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    print("OOD Evaluation")

    if not os.path.isfile(args.split_dict):
        raise FileNotFoundError("set split not found.")
    set_split_dict = utils.pkl_io("r", args.split_dict)
    db = utils.pkl_io("r", args.dataset)
    # load dataset specified by rset
    dset_real = return_set(db, args.rset, set_split_dict)
    dload_real = DataLoader(dset_real, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    print("Calculating real scores\n")
    all_scores = dict()
    for score_type in args.score_fn:
        real_scores = []
        for x, _ in dload_real:
            x = x.to(device)
            scores = score_fn(x, score_type=score_type)
            real_scores.extend(scores.numpy())

        # save the scores with the rset value as key
        all_scores[args.rset + "_" + score_type] = real_scores

    # we are differentiating vs these scores
    for ds in args.fset:
        # load fake dataset
        dset_fake = return_set(db, ds, set_split_dict)
        dload_fake = DataLoader(dset_fake, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        print("Calculating fake scores for {}\n".format(ds))
        for score_type in args.score_fn:
            fake_scores = []
            for x, _ in dload_fake:
                x = x.to(device)
                scores = score_fn(x, score_type=score_type)
                fake_scores.extend(scores.numpy())
                # print(scores.mean())
            all_scores[ds + '_' + score_type] = fake_scores
            # Create a histogram for fake scores

    # plot histogram
    # these are the real scores
    for score_type in args.score_fn:
        # Plot the rset dataset
        plt.figure(figsize=(10, 10))

        plt.hist(all_scores[args.rset + '_' + score_type],
                 density=True, label=args.rset, bins=100, alpha=.8, fill='black', lw=0)

        for dataset in args.fset:
            # plot the datasets in fset
            plt.hist(all_scores[dataset + '_' + score_type],
                     density=True, label=dataset, bins=100, alpha=.5, fill='black', lw=0)
        plt.legend()
        # if score_type == "px":
        #     plt.xlim(0, 20)
        # elif score_type == "pxgrad":
        #     plt.xlim(-5, 0)
        # elif score_type == "pxy":
        #     plt.xlim(0, 20)
        # elif score_type == "py":
        #     plt.xlim(0.7, 1)

        plt.title(f"Histogram of OOD detection {score_type}")
        plt.savefig(f"./img/OOD_hist_{score_type}.pdf")
        plt.close()
    # for every fake dataset make an ROC plot
    for dataset_name in args.fset:
        plt.figure(figsize=(10, 10))

        for score_type in args.score_fn:
            # make labels
            real_labels = np.ones_like(all_scores[args.rset + "_" + score_type])
            fake_labels = np.zeros_like(all_scores[dataset_name + "_" + score_type])

            labels = np.concatenate([real_labels, fake_labels])
            scores = np.concatenate([
                all_scores[args.rset + "_" + score_type],
                all_scores[dataset_name + '_' + score_type]
            ])
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
            auc_value = sklearn.metrics.auc(fpr, tpr)
            plt.step(fpr, tpr, where='post', label=f'{score_type} AUC: {round(auc_value, 3)}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.01])
        plt.xlim([0.0, 1.01])
        plt.title(f"Reciever Operating Characteristic (ROC)\n Plot OOD detection {dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./img/OOD_roc_{dataset_name}.pdf")
        plt.close()


def test_clf(f, args, device):
    def sample(x, n_steps=args.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if not os.path.isfile(args.split_dict):
        raise FileNotFoundError("set split not found.")
    set_split_dict = utils.pkl_io("r", args.split_dict)
    db = utils.pkl_io("r", args.dataset)
    dset = return_set(db, args.clfset, set_split_dict)
    dload = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if args.n_steps > 0:
            x_p_d = sample(x_p_d)
        logits = f.classify(x_p_d)
        py = nn.Softmax()(f.classify(x_p_d)).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits
                   .max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    t.save({"losses": losses, "corrects": corrects, "pys": pys}, os.path.join(args.save_dir, "vals.pt"))
    print(loss, correct)


def jem_calib(f, args, device):
    if not os.path.isfile(args.split_dict):
        raise FileNotFoundError("set split not found.")
    set_split_dict = utils.pkl_io("r", args.split_dict)
    db = utils.pkl_io("r", args.dataset)
    dset = return_set(db, args.calibset, set_split_dict)
    dload = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    conf = []
    acc = []
    for x, y in dload:
        x = x.to(device)
        if args.calibset in ["train", "valid", "test"]:
            y = utils.convert_label(args.class_drop, y, "r2t")
        logits = nn.Softmax(dim=1)(f.classify(x)).max(1)
        conf.extend(logits[0].detach().cpu().numpy())
        acc.extend((logits[1].detach().cpu().numpy() == y.cpu().numpy()).astype("int"))
    return np.array(conf), np.array(acc)


def clf_calib(args):
    if not os.path.isfile(args.split_dict):
        raise FileNotFoundError("set split not found.")
    set_split_dict = utils.pkl_io("r", args.split_dict)
    db = utils.pkl_io("r", args.dataset)
    data, label = return_set(db, args.calibset, set_split_dict, True)
    clf = utils.pkl_io("r", args.clf_path)
    acc = (clf.predict(data) == label).astype("int")
    conf = np.max(softmax(clf.decision_function(data), axis=1),axis=1)
    return np.array(conf), np.array(acc)


def calibration(f, args, device):
    def calib_bar(conf_sorted, acc_sorted, num_chunk):
        assert len(conf_sorted) == len(acc_sorted)
        count, xval = np.histogram(conf_sorted, range=(0, 1), bins=num_chunk)
        cummulate_count = 0
        conf_avg = []
        for i in count:
            if i == 0:
                conf_avg.append(0)
            else:
                conf_avg.append(np.average(acc_sorted[cummulate_count:cummulate_count+i]))
                cummulate_count += i
        return conf_avg, count, xval

    def cal_ece(conf_avg, count, acc_sorted):
        cummulate_count = 0
        ece = 0
        for step, i in enumerate(count):
            if i == 0:
                # sanity check
                assert conf_avg[step] == 0
                continue
            else:
                ece += i * np.abs(np.average(acc_sorted[cummulate_count:cummulate_count+i])- conf_avg[step])
        return ece / len(acc_sorted)

    def calib_plot(conf_avg, ece, xval, calibmodel, class_drop):
        xval = xval[:len(xval)-1]
        plt.bar(xval, conf_avg, width=1/len(conf_avg), align="edge")
        plt.plot([0,1], [0,1], "r--")
        plt.title("Calibration ECE={}".format(utils.to_percentage(ece)))
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.savefig("./img/calib_{}_ood_{}.pdf".format(calibmodel, class_drop))
        # plt.show()
    
    if args.calibmodel == "jem":
        conf, acc = jem_calib(f, args, device)
    else:
        conf, acc = clf_calib(args)

    idx = np.argsort(conf)
    conf_sorted = conf[idx]
    acc_sorted = acc[idx]
    conf_avg, count, xval = calib_bar(conf_sorted, acc_sorted, args.num_chunk)
    ece = cal_ece(conf_avg, count, acc_sorted)
    calib_plot(conf_avg, ece, xval, args.calibmodel, args.class_drop)


def main(args):
    utils.makedirs(args.save_dir)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    model_cls = F if args.uncond else CCF
    f = model_cls(args.backbone, args.arch, args.num_block, args.req_bn, args.act_func, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    print(f"loading model from {args.load_path}")

    # load em up
    ckpt_dict = t.load(args.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]
    class_drop = ckpt_dict["class_drop"]
    assert class_drop == args.class_drop

    f = f.to(device)

    if args.eval == "OOD":
        OODAUC(f, args, device)

    if args.eval == "test_clf":
        test_clf(f, args, device)

    if args.eval == "cond_samples":
        cond_samples(f, replay_buffer, args, device, args.fresh_samples)

    if args.eval == "uncond_samples":
        uncond_samples(f, args, device)

    if args.eval == "logp_hist":
        logp_hist(f, args, device)

    if args.eval == "calib":
        calibration(f, args, device)


if __name__ == "__main__":
    __spec__ = None
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--eval", default="OOD", type=str,
                        choices=["uncond_samples", "cond_samples", "logp_hist", "OOD", "test_clf", "calib"])
    parser.add_argument("--score_fn", default=["px", "py", "pxgrad", "pxy"], type=str, nargs="+",
                        help="For OODAUC, chooses what score function we use.")
    parser.add_argument("--dataset", type=str) # path to anndata

    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=3e-2)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./img')
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--load_path", type=str, default="./experiment/best_valid_ckpt.pt")
    parser.add_argument("--print_to_log", action="store_true")
    parser.add_argument("--fresh_samples", action="store_true",
                        help="If set, then we generate a new replay buffer from scratch for conditional sampling,"
                             "Will be much slower.")

    parser.add_argument("--n_classes", type=int, default=11)

    parser.add_argument("--backbone", choices=["resnet", "mlp"], required=True)
    parser.add_argument("--arch", nargs="+", type=int, help="dimension of each layer")
    parser.add_argument("--num_block", nargs="+", type=int, default=None, help="For resnet backbone only: number of block per layer")
    parser.add_argument("--req_bn", action="store_true", help="If set, uses BatchNorm in CLAMSNet")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--act_func", choices=["relu", "sigmoid", "tanh", "lrelu", "elu"], default="elu")

    parser.add_argument("--rset", type=str, choices=["train", "test", "valid", "ood", "test+ood"], default="train", help="OODAUC real dateset")
    parser.add_argument("--fset", nargs="+", type=str, default=["test+ood"], choices=["train", "test", "valid", "ood", "test+ood"], help="OODAUC fake dataset")
    parser.add_argument("--clfset", type=str, choices=["train", "test", "valid", "ood", "test+ood"], help="test_clf dataset")
    parser.add_argument("--logpset", type=str, choices=["train", "test", "valid", "ood", "test+ood"], default="ood+test", help="test_clf dataset")
    parser.add_argument("--calibset", type=str, choices=["train", "test", "valid", "ood", "test+ood"], default="train", help="calibration dataset")
    parser.add_argument("--num_chunk", type=int, default=20, help="number of chunks in calibration")
    parser.add_argument("--class_drop", type=int, default=-1, help="drop the class for ood detection")
    parser.add_argument("--split_dict", type=str, help="path to split dict")
    parser.add_argument("--calibmodel", choices=["jem", "clf"], help="use either JEM or clf to plot calibration")
    parser.add_argument("--clf_path", type=str, help="path to clf if calibmodel=clf") 

    args = parser.parse_args()

    print(time.ctime())
    for item in args.__dict__:
        print("{:24}".format(item), "->\t", args.__dict__[item])
    
    main(args)

