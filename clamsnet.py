import torch
import torch.nn as nn

class Clams_Net(nn.Module):
    in_dim = 5000
    
    def __init__(self, last_dim=50):
        super(Clams_Net, self).__init__()
        self.toy_layer = nn.Linear(self.in_dim, last_dim)
        self.last_dim = last_dim
    
    def forward(self, x):
        return self.toy_layer(x)

    @classmethod
    def update_class_dim(cls, dim):
        cls.in_dim = dim