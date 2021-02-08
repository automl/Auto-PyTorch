"""
Class to learn an embedding for categorical hyperparameters.
"""

import torch
import torch.nn as nn
import numpy as np


class LearnedEntityEmbedding(nn.Module):
    """ Parent class for MlpNet, ResNet, ... Can use entity embedding for categorical features"""

    def __init__(self, config, in_features, num_numerical_features):
        """
        Initialize the BaseFeatureNet.
        Arguments:
            config: The configuration sampled by the hyperparameter optimizer
            in_features: the number of features of the dataset
            one_hot_encoder: OneHot encoder, that is used to encode X
        """
        super(LearnedEntityEmbedding, self).__init__()
        self.config = config

        # self.num_numerical = len([f for f in one_hot_encoder.categorical_features if not f])
        # self.num_input_features = [len(c) for c in one_hot_encoder.categories_]
        self.num_numerical = num_numerical_features
        self.embed_features = [num_in >= config["min_unique_values_for_embedding"] for num_in in
                               self.num_input_features]
        self.num_output_dimensions = [config["dimension_reduction_" + str(i)] * num_in for i, num_in in
                                      enumerate(self.num_input_features)]
        self.num_output_dimensions = [int(np.clip(num_out, 1, num_in - 1)) for num_out, num_in in
                                      zip(self.num_output_dimensions, self.num_input_features)]
        self.num_output_dimensions = [num_out if embed else num_in for num_out, embed, num_in in
                                      zip(self.num_output_dimensions, self.embed_features, self.num_input_features)]
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
                concat_seq.append(x[:, last_concat: x_pointer])
            categorical_feature_slice = x[:, x_pointer: x_pointer + num_in]
            concat_seq.append(self.ee_layers[layer_pointer](categorical_feature_slice))
            layer_pointer += 1
            x_pointer += num_in
            last_concat = x_pointer

        concat_seq.append(x[:, last_concat:])
        return torch.cat(concat_seq, dim=1)

    def _create_ee_layers(self, in_features):
        # entity embeding layers are Linear Layers
        layers = nn.ModuleList()
        for i, (num_in, embed, num_out) in enumerate(
            zip(self.num_input_features, self.embed_features, self.num_output_dimensions)):
            if not embed:
                continue
            layers.append(nn.Linear(num_in, num_out))
        return layers