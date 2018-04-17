import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

class FeedForwardPolicy(nn.Module):
    def __init__(self, config):
        super(FeedForwardPolicy, self).__init__()
        pass

    def forward(self, state):
        pass

class RNNPolicy(nn.Module):
    def __init__(self, config):
        super(RNNPolicy, self).__init__()
        pass

    def forward(self, state):
        pass

class LSTMPolicy(nn.Module):
    def __init__(self, config):
        super(LSTMPolicy, self).__init__()
        pass

    def forward(self, state):
        pass

