
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, x):
        # import logging
        # l = logging.getLogger('autonet')
        # l.debug(x.shape)
        # l.debug((x.reshape(-1, self.size)).shape)
        return x.reshape(-1, self.size)