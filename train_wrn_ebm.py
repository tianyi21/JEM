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

from numpy.core.numeric import full
from numpy.lib.arraysetops import isin
import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
#import wideresnet
import clamsnet
import json
# Sampling
from tqdm import tqdm
import time

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
n_gene = 5000

# supress openmp error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

LABEL_RAW = np.array(['CD14+', 'CD19+', 'CD34+', 'CD4+',
            'CD4+/CD25', 'CD4+/CD45RA+/CD25-',
            'CD4+/CD45RO+', 'CD56+', 'CD8+',
            'CD8+/CD45RA+', 'Dendritic'])
CACHE_DATA = True


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
    def __init__(self, backbone, arch, req_bn, dropout_rate, n_classes,):
        super(F, self).__init__()
        self.backbone = backbone
        self.arch = arch
        self.f = clamsnet.Clams_Net(backbone, arch, req_bn, dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, backbone, arch, req_bn, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(backbone=backbone, arch=arch, req_bn=req_bn, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])
    
    def __str__():



def cycle(loader):
    while True:
        for data in loader:
            yield data


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs):
    return t.FloatTensor(bs, n_gene).uniform_(-1, 1)


def get_model_and_buffer(args, device, sample_q):
    model_cls = F if args.uncond else CCF
    f = model_cls(args.backbone, args.arch, args.req_bn, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer


def get_data(args):
    """
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
    """

    # get all training inds
    # full_train = dataset_fn(True, transform_train)
    db = utils.pkl_io("r", args.dataset)
    full_train = SingleCellDataset(db)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        test_inds, valid_inds, train_inds = all_inds[:args.n_test], all_inds[args.n_test:args.n_valid+args.n_test], all_inds[args.n_valid+args.n_test:]
    else:
        test_inds, valid_inds, train_inds = all_inds[:args.n_test], [], all_inds[args.n_test:]
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])

    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(SingleCellDataset(db), inds=train_inds)
    dset_valid = DataSubset(SingleCellDataset(db), inds=valid_inds)
    dset_test = DataSubset(SingleCellDataset(db), inds=test_inds)
    dset_train_labeled = DataSubset(SingleCellDataset(db), inds=train_labeled_inds)

    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_test = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_valid = DataLoader(dset_valid, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    if CACHE_DATA:
        utils.makedirs(args.save_dir)
        utils.pkl_io("w", os.path.join(args.save_dir, "ttv_idx.pkl", {"train": train_inds, "test": test_inds, "valid": valid_inds})
    return dload_train, dload_train_labeled, dload_valid, dload_test


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q


def eval_classification(f, dload, device, set, epoch=0, cm_normalize="pred"):
    corrects, losses = [], []
    ys, preds = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        ys.extend(y_p_d.cpu().numpy())
        preds.extend(logits.max(1)[1].cpu().numpy())
    loss = np.mean(losses)
    correct = np.mean(corrects)
    # plot confusion matrix
    plot_cm(ys, preds, cm_normalize, correct, epoch, set=set)
    return correct, loss

def plot_cm(y_p_d, logits, cm_normalize, correct, epoch=0, set="test"):
    cm = confusion_matrix(y_p_d, logits, normalize=cm_normalize)
    plt.imshow(cm)
    # sanity check
    batch_label_uniq = sorted(np.unique(y_p_d))
    pred_class_uniq = sorted(np.unique(logits))
    plot_label = sorted(np.unique(np.concatenate((batch_label_uniq, pred_class_uniq))))
    plt.xticks(range(len(plot_label)), LABEL_RAW[plot_label], rotation=45, horizontalalignment="right")
    plt.yticks(range(len(plot_label)), LABEL_RAW[plot_label])
    plt.title("Confusion Matrix set={}, epoch={}, acc={}".format(set, epoch + 1, correct))
    utils.makedirs("./img")
    plt.savefig("./img/cm_{}_{}.jpg".format(set, epoch + 1))


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            L = 0.
            if args.p_x_weight > 0:  # maximize log p(x) default 1
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp

                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()

                l_p_x = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch + 1, i + 1, fp, fq,
                                                                                                   fp - fq))
                L += args.p_x_weight * l_p_x

            if args.p_y_given_x_weight > 0:  # maximize log p(y | x) by xeloss
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch + 1,
                                                                                 cur_iter + 1,
                                                                                 l_p_y_given_x.item(),
                                                                                 acc.item()))
                L += args.p_y_given_x_weight * l_p_y_given_x

            if args.p_x_y_weight > 0:  # maximize log p(x, y), default 0
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch + 1, i + 1, fp, fq,
                                                                                                      fp - fq))

                L += args.p_x_y_weight * l_p_x_y

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                1/0

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1

            # no plot needed
            """
            if cur_iter % 100 == 0:
                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)
            """

        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # train
                correct, loss = eval_classification(f, dload_train, device, "train", epoch)
                print("Epoch {}: Train Loss {}, Train Acc {}".format(epoch + 1, loss, correct))
                # validation set
                correct, loss = eval_classification(f, dload_valid, device, "valid")
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch + 1, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device, "test", epoch)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch + 1, loss, correct))
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="./pbmc_filtered.pkl") # path to anndata
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=11000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--n_classes", type=int, default=11)

    # CLAMSNet arch
    parser.add_argument("--backbone", choice=["resnet, mlp"], required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    # mlp is defined based on arch
    # resnet is defeind based on either arch or pooling
    group.add_argument("--arch", type=list, help="dimension of each layer")
    group.add_argument("--pooling",type=list, help="For resnet backbone only: kernel size of pooling after each layer, last layer doesn't have max pooling")
   
    parser.add_argument("--num_block", type=list, required=True, help="For resnet backbone only: number of block per layer")
    parser.add_argument("--act_func", choice=["relu, sigmoid, tanh, lrelu"], required=True, default="lrelu")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--req_bn", action="store_true", help="If set, uses BatchNorm in CLAMSNet")

    args = parser.parse_args()

    print(time.ctime())
    for item in args.__dict__:
        print("{:24}".format(item), "->\t", args.__dict__[item])
    
    main(args)
