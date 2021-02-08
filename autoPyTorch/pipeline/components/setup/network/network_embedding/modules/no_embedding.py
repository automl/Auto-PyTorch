from torch import nn


class NoEmbedding(nn.Module):
    def __init__(self, config, in_features, num_numerical_features):
        super(NoEmbedding, self).__init__()
        self.config = config
        self.n_feats = in_features
        self.num_numerical = num_numerical_features

    def forward(self, x):
        return x