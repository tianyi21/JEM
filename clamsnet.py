from typing import Sequence
from numpy.lib.arraysetops import isin
import torch.nn as nn


def activation_function(act_func):
    if act_func.lower() == "relu":
        return nn.ReLU
    elif act_func.lower() == "tanh":
        return nn.Tanh
    elif act_func.lower() == "sigmoid":
        return nn.Sigmoid
    elif act_func.lower() == "lrelu":
        return nn.LeakyReLU
    else:
        raise ValueError("Invalid activation function.")


def format_model_str(model, intro):
    """
    format __str__ method of model
    """
    model_str = intro
    model_str += "\n"
    for layer in list(model.modules()):
        if isinstance(layer, nn.Linear):
            model_str += "\n"
            model_str += str(layer)
            model_str += "\n"
        elif isinstance(layer, (nn.BatchNorm1d, nn.Dropout, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
            model_str += str(layer)
            model_str += "\n"
    return model_str


class ResBlock(nn.Module):
    def __init__(self, dim, req_bn, dropout_rate, act_func):
        """
        Residual Block
        # act(side-path [fc + (bn) + act + drop + fc + (bn) + drop] + shortcut)
        """
        super(ResBlock, self).__init__()
        self.res_act = activation_function(act_func)
        layer = []
        layer.append(nn.Linear(dim, dim))
        if req_bn:
            layer.append(nn.BatchNorm1d(dim))
        layer.append(self.res_act())
        layer.append(nn.Dropout(p=dropout_rate))
        layer.append(nn.Linear(dim, dim))
        if req_bn:
            layer.append(nn.BatchNorm1d(dim))
        layer.append(nn.Dropout(p=dropout_rate))
        self.side = nn.Sequential(*layer)
        self.shortcut = nn.Sequential()
    
    def forward(self, x):
        return self.res_act(self.side(x) + self.shortcut(x))
    
    def __str__(self):
        return format_model_str(self.side, "ResBlock Arch")


class ClamsResNet(nn.Module):
    def __init__(self, in_dim, arch, num_block, pooling, req_bn, dropout_rate, act_func):
        """
        in_dim:     inital data input dim
        arch:       dimension of a residual layer           len = n
        num_block:  number of block per layer               len = n 
        pooling:    max pool kernel at the end of layer     len = n - 1
        """
        super(ClamsResNet, self).__init__()
        self.in_dim = in_dim
        self.arch = arch
        self.num_block = num_block
        self.pooling = pooling
        
        # sanity check
        assert len(arch) == len(num_block)
        assert len(num_block) == len(pooling) + 1
        tmp = arch[0]
        for i in pooling:
            tmp = tmp // i
        assert tmp > 0

        # last layer does not have pooling
        pooling += [1]

        layer = []
        layer.append(nn.Linear(in_dim, arch[0]))
        for dim, num_blk, pool in zip(arch, num_block, pooling):
            layer.append(self._make_layer(dim, num_blk, pool, req_bn, dropout_rate, act_func))
        self.model = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.model(x)
    
    def _make_layer(self, dim, num_block, pooling, req_bn, dropout_rate, act_func):
        layer = []
        for _ in num_block:
            layer.append(ResBlock(dim, req_bn, dropout_rate, act_func))
            layer.append(nn.MaxPool1d(pooling))
        return nn.Sequential(*layer)
    
    def __str__(self):
        model_str = "CLAMS ResNet Arch\n"
        model_str += "Input dim: {}\n".format(self.in_dim)
        for i, (dim, num_blk, pool) in enumerate(zip(self.arch, self.num_block, self.pooling)):
            model_str += "\nLayer: {}\t{} ResBlocks of dimension {} each. End-layer pooling: {}\n".format(i + 1, num_blk, dim, pool)
        return model_str
        

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, arch, req_bn, dropout_rate, act_func):
        """
        in_dim:     inital data input dim
        arch:       dimension of each layer                 len = n
        """
        super(MultiLayerPerceptron, self).__init__()
        mlp_act = activation_function(act_func)
        layer = []
        arch = [in_dim] + arch
        for i in range(len(arch) - 1):
            # weights -> bn -> act -> dropout
            layer.append(nn.Linear(arch[i], arch[i + 1]))
            if req_bn:
                layer.append(nn.BatchNorm1d(arch[i + 1]))
            layer.append(mlp_act())
            layer.append(nn.Dropout(p=dropout_rate))
        self.mlp = nn.Sequential(*layer)
        
    def forward(self, x):
        return self.mlp(x)
    
    def __str__(self):
        return format_model_str(self.mlp, "MLP Arch")


class Clams_Net(nn.Module):
    in_dim = 5000

    def __init__(self, backbone, arch, req_bn, dropout_rate, act_func, num_block, pooling):
        super(Clams_Net, self).__init__()
        self.last_dim = arch[-1]
        if backbone == "mlp":
            self.model = MultiLayerPerceptron(self.in_dim, arch, req_bn, dropout_rate, act_func)
        elif backbone == "resnet":
            self.model = ClamsResNet(self.in_dim, arch, num_block, pooling, req_bn, dropout_rate, act_func)
    
    def forward(self, x):
        return self.model(x)

    @classmethod
    def update_class_dim(cls, dim):
        cls.in_dim = dim

    def __str__(self):
        return self.model.__str__()

