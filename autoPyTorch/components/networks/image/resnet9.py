import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

from autoPyTorch.components.networks.base_net import BaseImageNet
from autoPyTorch.components.networks.image.utils.utils import initialize_weights


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
  
class conv_bn_self(nn.Module):
    def __init__(self, c_in, c_out, batch_norm, kernel_size=3, activation=None):
        super(conv_bn_self, self).__init__()  
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn = batch_norm(c_out) #nn.BatchNorm2d(self.depth)
        self.activation=activation
        if self.activation:
            self.act_func = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.act_func(x)
        return x

class conv_bn_act_pool(nn.Module):
    def __init__(self, c_in, c_out, batch_norm, activation, pool, kernel_size=3):
        super(conv_bn_act_pool, self).__init__()
        self.conv_bn = conv_bn_self(c_in, c_out, batch_norm, kernel_size, activation)
        self.pool = pool
    
    def forward(self, x):
        x = self.conv_bn(x)
        out = self.pool(x)
        return out

class conv_bn_pool_act(nn.Module):
    def __init__(self, c_in, c_out, batch_norm, activation, pool, kernel_size=3):
        super(conv_bn_pool_act, self).__init__()
        self.conv_bn = conv_bn_self(c_in, c_out, batch_norm, kernel_size)
        self.pool = pool
        self.act_fun = activation()
    
    def forward(self, x):
        x = self.conv_bn(x)
        x = self.pool(x)
        out = self.act_func(x)
        return out

class conv_pool_bn_act(nn.Module):
    def __init__(self, c_in, c_out, batch_norm, activation, pool, kernel_size=3):
        super(conv_pool_bn_act, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.pool = pool
        self.bn = batch_norm(c_out)
        self.act_func = activation()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        out = self.act_func(x)
        return out

class Residual(nn.Module):
    def __init__(self, c, batch_norm, activation, downsample=None):
        super(Residual, self).__init__()
        self.conv = conv_bn_self(c, c, batch_norm=batch_norm, activation=activation)
        self.conv2 = conv_bn_self(c, c, batch_norm=batch_norm, activation=activation)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.conv2(out)
    
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity

        return out 


class ResNet9(BaseImageNet):
    def __init__(self, config, in_features, out_features, final_activation, batch_norm=nn.BatchNorm2d, weight=0.125, pool=nn.MaxPool2d(2), prep_block=conv_bn_self):
        super(ResNet9, self).__init__(config, in_features, out_features, final_activation)
        if config['conv_bn'] == 'conv_bn_pool_act':
            conv_bn = conv_bn_pool_act
        if config['conv_bn'] == 'conv_pool_bn_act':
            conv_bn = conv_pool_bn_act
        if config['conv_bn'] == 'conv_bn_act_pool':
            conv_bn = conv_bn_act_pool

        activation = nn.ReLU
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        self.model = nn.Sequential()
        self.model.add_module('prep_layer', prep_block(3, channels['prep'], batch_norm=batch_norm, activation=activation))
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv_bn', conv_bn(channels['prep'], channels['layer1'], pool=pool, batch_norm=batch_norm, activation=activation)),
            ('residual', Residual(channels['layer1'], batch_norm=batch_norm, activation=activation))
        ]))
        self.model.add_module('Layer1', self.layer1)

        self.layer2 = nn.Sequential(OrderedDict([
            ('conv_bn', conv_bn(channels['layer1'], channels['layer2'], pool=pool, batch_norm=batch_norm, activation=activation)),
        ]))
        self.model.add_module('Layer2', self.layer2)

        self.layer3 = nn.Sequential(OrderedDict([
            ('conv_bn', conv_bn(channels['layer2'], channels['layer3'], pool=pool, batch_norm=batch_norm, activation=activation)),
            ('residual', Residual(channels['layer3'], batch_norm=batch_norm, activation=activation))
        ]))
        self.model.add_module('Layer3', self.layer3)

        self.pool = nn.MaxPool2d(4)
        self.model.add_module('pool', self.pool)
        self.apply(initialize_weights)
        self.flatten = Flatten()
        self.linear =  nn.Linear(channels['layer3'], 10, bias=False)
        self.mul = Mul(weight)
    
        

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.mul(x)

        return x
    
    @staticmethod
    def get_config_space(user_updates=None):
        
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('conv_bn', ['conv_bn_pool_act', 'conv_pool_bn_act', 'conv_bn_act_pool']))

        return(cs)

