import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

from autoPyTorch.components.networks.base_net import BaseImageNet
from autoPyTorch.components.networks.image.utils.utils import initialize_weights
from autoPyTorch.components.networks.image.utils.shakeshakeblock import shake_shake, generate_alpha_beta
from autoPyTorch.components.networks.image.utils.shakedrop import shake_drop, generate_alpha_beta_single
from .utils.utils import get_layer_params

class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SkipConnection, self).__init__()

        self.s1 = nn.Sequential()
        self.s1.add_module('Skip_1_AvgPool',
                           nn.AvgPool2d(1, stride=stride))
        self.s1.add_module('Skip_1_Conv',
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2),
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.s2 = nn.Sequential()
        self.s2.add_module('Skip_2_AvgPool',
                           nn.AvgPool2d(1, stride=stride))
        self.s2.add_module('Skip_2_Conv',
                           nn.Conv2d(in_channels,
                                     int(out_channels / 2) if out_channels % 2 == 0 else int(out_channels / 2) + 1,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False))

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(x, inplace=False)
        out1 = self.s1(out1)

        out2 = F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1))
        out2 = self.s2(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.batch_norm(out)

        return out


class ResidualBranch(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride, padding, branch_index):
        super(ResidualBranch, self).__init__()

        self.residual_branch = nn.Sequential()

        self.residual_branch.add_module('Branch_{}:ReLU_1'.format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_1'.format(branch_index),
                                        nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=filter_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_1'.format(branch_index),
                                        nn.BatchNorm2d(out_channels))
        self.residual_branch.add_module('Branch_{}:ReLU_2'.format(branch_index),
                                        nn.ReLU(inplace=False))
        self.residual_branch.add_module('Branch_{}:Conv_2'.format(branch_index),
                                        nn.Conv2d(out_channels,
                                                  out_channels,
                                                  kernel_size=filter_size,
                                                  stride=1,
                                                  padding=round(filter_size / 3),
                                                  bias=False))
        self.residual_branch.add_module('Branch_{}:BN_2'.format(branch_index),
                                        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.residual_branch(x)


class BasicBlock(nn.Module):
    def __init__(self, n_input_plane, n_output_plane, filter_size, res_branches, stride, padding, shake_config):
        super(BasicBlock, self).__init__()

        self.shake_config = shake_config
        self.branches = nn.ModuleList([ResidualBranch(n_input_plane, n_output_plane, filter_size, stride, padding, branch + 1) for branch in range(res_branches)])

        # Skip connection
        self.skip = nn.Sequential()
        if n_input_plane != n_output_plane or stride != 1:
            self.skip.add_module('Skip_connection',
                                 SkipConnection(n_input_plane, n_output_plane, stride))
                                 

    def forward(self, x):
        out = sum([self.branches[i](x) for i in range(len(self.branches))])
            
        if len(self.branches) == 1:
            if self.config.apply_shakeDrop:
                alpha, beta = generate_alpha_beta_single(out.size(), self.shake_config if self.training else (False, False, False), x.is_cuda)
                out = shake_drop(out, alpha, beta, self.config.death_rate, self.training)
            else:
                out = self.branches[0](x)
        else:
            if self.config.apply_shakeShake:
                alpha, beta = generate_alpha_beta(len(self.branches), x.size(0), self.shake_config if self.training else (False, False, False), x.is_cuda)
                branches = [self.branches[i](x) for i in range(len(self.branches))]
                out = shake_shake(alpha, beta, *branches)
            else:
                out = sum([self.branches[i](x) for i in range(len(self.branches))])

        return out + self.skip(x)


class ResidualGroup(nn.Module):
    def __init__(self, block, n_input_plane, n_output_plane, n_blocks, filter_size, res_branches, stride, padding, shake_config):
        super(ResidualGroup, self).__init__()
        self.group = nn.Sequential()
        self.n_blocks = n_blocks

        # The first residual block in each group is responsible for the input downsampling
        self.group.add_module('Block_1',
                              block(n_input_plane,
                                    n_output_plane,
                                    filter_size,
                                    res_branches,
                                    stride=stride,
                                    padding=padding,
                                    shake_config=shake_config))

        # The following residual block do not perform any downsampling (stride=1)
        for block_index in range(2, n_blocks + 1):
            block_name = 'Block_{}'.format(block_index)
            self.group.add_module(block_name,
                                  block(n_output_plane,
                                        n_output_plane,
                                        filter_size,
                                        res_branches,
                                        stride=1,
                                        padding=0
                                        shake_config=shake_config))

    def forward(self, x):
        return self.group(x)


class ResNet(BaseImageNet):
    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(ResNet, self).__init__(config, in_features, out_features, final_activation)

        nn.Module.config = config
        self.final_activation = final_activation
        self.nr_main_blocks = 3 #config['nr_main_blocks']
        config.initial_filters = config['initial_filters']
        config.nr_convs = config['nr_convs']
        config.death_rate = config['death_rate']

        config.forward_shake = True
        config.backward_shake = True
        config.shake_image = True
        config.apply_shakeDrop = True
        config.apply_shakeShake = True

        self.nr_residual_blocks = dict([
            ('Group_%d' % (i+1), config['nr_residual_blocks_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])
 
        self.widen_factors = dict([
            ('Group_%d' % (i+1), config['widen_factor_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.res_branches = dict([
            ('Group_%d' % (i+1), config['res_branches_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.filters_size = dict([
            ('Group_%d' % (i+1), 3) #config['filters_size_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])
        
        shake_config = (config.forward_shake, config.backward_shake,
                             config.shake_image)

        ##########
        self.model = nn.Sequential()

        depth = sum([config.nr_convs * self.nr_residual_blocks['Group_{}'.format(i)] + 2 for i in range(1, self.nr_main_blocks + 1)])
        print(' | Multi-branch ResNet-' + str(depth) + ' CIFAR-10')

        block = BasicBlock

        self.model.add_module('Conv_0',
                                nn.Conv2d(self.channels,
                                        config.initial_filters,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False))
        self.model.add_module('BN_0',
                                nn.BatchNorm2d(config.initial_filters))

        feature_maps_in = int(round(config.initial_filters * self.widen_factors['Group_1']))
        self.model.add_module('Group_1',
                                ResidualGroup(block, 
                                            config.initial_filters, 
                                            feature_maps_in, 
                                            self.nr_residual_blocks['Group_1'], 
                                            self.filters_size['Group_1'],
                                            self.res_branches['Group_1'],
                                            1, 
                                            shake_config))

        in_size = min(self.iw, self.ih)
        out_size = max(1, in_size * config['last_image_size'])
        size_reduction = math.pow(in_size / out_size, 1 / (self.nr_main_blocks))
        sizes = [max(1, round(in_size / math.pow(size_reduction, i+1))) for i in range(self.nr_main_blocks + 1)]

        for main_block_nr in range(2, self.nr_main_blocks + 1):
            feature_maps_out = int(round(feature_maps_in * self.widen_factors['Group_{}'.format(main_block_nr)]))

            in_size, kernel_size, stride, padding = get_layer_params(in_size, sizes[main_block_nr-2], self.filters_size['Group_{}'.format(main_block_nr)])
            self.model.add_module('Group_{}'.format(main_block_nr),
                                    ResidualGroup(block, 
                                                feature_maps_in, 
                                                feature_maps_out, 
                                                self.nr_residual_blocks['Group_{}'.format(main_block_nr)],
                                                kernel_size,
                                                self.res_branches['Group_{}'.format(main_block_nr)],
                                                2 if main_block_nr in (self.nr_main_blocks, self.nr_main_blocks - 1) else 1, 
                                                shake_config))
            feature_maps_in = feature_maps_out

        self.feature_maps_out = feature_maps_in
        self.model.add_module('ReLU_0',
                                nn.ReLU(inplace=True))
        self.model.add_module('AveragePool',
                                nn.AvgPool2d(8, stride=1))
        self.fc = nn.Linear(self.feature_maps_out, out_features)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.feature_maps_out)
        x = self.fc(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    @staticmethod
    def get_config_space(   nr_main_blocks=[3, 3], nr_residual_blocks=[1, 16], 
                            nr_convs=[2, 3], initial_filters=[8, 32], widen_factor=[0.5, 6], 
                            res_branches=[1, 5], filters_size=[3, 3], **kwargs):
                            
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter

        cs = CS.ConfigurationSpace()

        main_blocks = add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'nr_main_blocks', nr_main_blocks)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'initial_filters', initial_filters, log=True)
        add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'nr_convs', nr_convs, log=True)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'death_rate', [0, 1], log=False)
        add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'last_image_size', [0, 0.4])
	    
        for i in range(1, nr_main_blocks[1] + 1):
            blocks = add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'nr_residual_blocks_%d' % i, nr_residual_blocks, log=True)
            widen = add_hyperparameter(cs, CSH.UniformFloatHyperparameter, 'widen_factor_%d' % i, widen_factor, log=True)
            branches = add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'res_branches_%d' % i, res_branches, log=False)
            filters = add_hyperparameter(cs, CSH.UniformIntegerHyperparameter, 'filters_size_%d' % i, filters_size, log=False)

            if i > nr_main_blocks[0]:
                cs.add_condition(CS.GreaterThanCondition(blocks, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(widen, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(branches, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(filters, main_blocks, i-1))

        return cs