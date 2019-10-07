#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parent Class of all Networks based on features.
"""

import torch.nn as nn
from collections import OrderedDict
import ConfigSpace

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class BaseNet(nn.Module):
    """ Parent class for all Networks"""
    def __init__(self, config, in_features, out_features, final_activation):
        """
        Initialize the BaseNet.
        """

        super(BaseNet, self).__init__()
        self.layers = nn.Sequential()
        self.config = config
        self.n_feats = in_features
        self.n_classes = out_features
        self.epochs_trained = 0
        self.budget_trained = 0
        self.stopped_early = False
        self.last_compute_result = None
        self.logs = []
        self.num_epochs_no_progress = 0
        self.current_best_epoch_performance = None
        self.best_parameters = None
        self.final_activation = final_activation

    def forward(self, x):
        x = self.layers(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x
    
    def snapshot(self):
        self.best_parameters = OrderedDict({key: value.cpu().clone() for key, value in self.state_dict().items()})
    
    def load_snapshot(self):
        if self.best_parameters is not None:
            self.load_state_dict(self.best_parameters)

    @staticmethod
    def get_config_space():
        return ConfigSpace.ConfigurationSpace()
    

class BaseFeatureNet(BaseNet):
    """ Parent class for MlpNet, ResNet, ... Can use entity embedding for cagtegorical features"""
    def __init__(self, config, in_features, out_features, embedding, final_activation):
        """
        Initialize the BaseFeatureNet.
        """

        super(BaseFeatureNet, self).__init__(config, in_features, out_features, final_activation)
        self.embedding = embedding

    def forward(self, x):
        x = self.embedding(x)
        return super(BaseFeatureNet, self).forward(x)


class BaseImageNet(BaseNet):
    def __init__(self, config, in_features, out_features, final_activation):
        super(BaseImageNet, self).__init__(config, in_features, out_features, final_activation)
        
        if len(in_features) == 2:
            self.channels = 1
            self.iw = in_features[0]
            self.ih = in_features[1]
        if len(in_features) == 3:
            self.channels = in_features[0]
            self.iw = in_features[1]
            self.ih = in_features[2]
