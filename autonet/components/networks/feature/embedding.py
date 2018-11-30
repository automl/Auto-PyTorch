#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to learn an embedding for categorical hyperparameters.
"""

import ConfigSpace as CS
import ConfigSpace.conditions as CSC
import ConfigSpace.hyperparameters as CSH
import torch
import torch.nn as nn
import numpy as np

from autonet.components.preprocessing.preprocessor_base import PreprocessorBase

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class LearnedEntityEmbedding(nn.Module):
    """ Parent class for MlpNet, ResNet, ... Can use entity embedding for cagtegorical features"""
    def __init__(self, config, in_features, one_hot_encoder):
        """
        Initialize the BaseFeatureNet.
        
        Arguments:
            config: The configuration sampled by the hyperparameter optimizer
            in_features: the number of features of the dataset
            one_hot_encoder: OneHot encoder, that is used to encode X
        """
        super(LearnedEntityEmbedding, self).__init__()
        self.config = config
        self.n_feats = in_features
        self.one_hot_encoder = one_hot_encoder

        self.num_numerical = len([f for f in one_hot_encoder.categorical_features if not f])
        self.num_input_features = [len(c) for c in one_hot_encoder.categories_]
        self.embed_features = [num_in >= config["min_unique_values_for_embedding"] for num_in in self.num_input_features]
        self.num_output_dimensions = [config["dimension_reduction_" + str(i)] * num_in for i, num_in in enumerate(self.num_input_features)]
        self.num_output_dimensions = [int(np.clip(num_out, 1, num_in - 1)) for num_out, num_in in zip(self.num_output_dimensions, self.num_input_features)]
        self.num_output_dimensions = [num_out if embed else num_in for num_out, embed, num_in in zip(self.num_output_dimensions, self.embed_features, self.num_input_features)]
        self.num_out_feats = self.num_numerical + sum(self.num_output_dimensions)

        self.ee_layers = self._create_ee_layers(in_features)


    def forward(self, x):
        # pass the columns of each categorical feature through entity embedding layer
        # before passing it through the model
        concat_seq = []
        last_concat = 0
        x_pointer = 0
        layer_pointer = 0
        for num_in, embed in zip(self.num_input_features, self.embed_features):
            if not embed:
                x_pointer += 1
                continue
            if x_pointer > last_concat:
                concat_seq.append(x[:, last_concat : x_pointer])
            categorical_feature_slice = x[:, x_pointer : x_pointer + num_in]
            concat_seq.append(self.ee_layers[layer_pointer](categorical_feature_slice))
            layer_pointer += 1
            x_pointer += num_in
            last_concat = x_pointer
        
        concat_seq.append(x[:, last_concat:])
        return torch.cat(concat_seq, dim=1)
    
    def _create_ee_layers(self, in_features):
        # entity embeding layers are Linear Layers
         layers = nn.ModuleList()
         for i, (num_in, embed, num_out) in enumerate(zip(self.num_input_features, self.embed_features, self.num_output_dimensions)):
            if not embed:
                continue
            layers.append(nn.Linear(num_in, num_out))
         return layers

    @staticmethod
    def get_config_space(categorical_features=None):
        # dimension of entity embedding layer is a hyperparameter
        if categorical_features is None or not any(categorical_features):
            return CS.ConfigurationSpace()
        cs = CS.ConfigurationSpace()
        min_hp = CSH.UniformIntegerHyperparameter("min_unique_values_for_embedding", lower=3, upper=300, default_value=3, log=True)
        cs.add_hyperparameter(min_hp)
        for i in range(len([x for x in categorical_features if x])):
            ee_dimensions = CSH.UniformFloatHyperparameter("dimension_reduction_" + str(i), lower=0, upper=1, default_value=1, log=False)
            cs.add_hyperparameter(ee_dimensions)
        return cs

class NoEmbedding(nn.Module):
    def __init__(self, config, in_features, one_hot_encoder):
        super(NoEmbedding, self).__init__()
        self.config = config
        self.n_feats = in_features
        self.num_out_feats = self.n_feats
    
    def forward(self, x):
        return x
    
    @staticmethod
    def get_config_space(*args, **kwargs):
        return CS.ConfigurationSpace()