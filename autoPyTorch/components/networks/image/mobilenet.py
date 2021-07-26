import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from autoPyTorch.components.networks.base_net import BaseImageNet
from .utils.mobilenet_utils import GenEfficientNet, _decode_arch_def, _resolve_bn_args, _round_channels, swish, sigmoid, hard_swish, hard_sigmoid, SelectAdaptivePool2d
from .utils.utils import get_layers

# TODO
# EXPANSION RATIO HARDCODED
# ADD ACTIVATION? (already from utils imported)
# ADD BN ARGS? (atm same as timm 0.1, 1e-5)


class Arch_Encoder():
    """ Encode block definition string
    Encodes a list of config space (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type. For further usage with _decode_architecure and timm.
    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block hyperpar dict as coming from MobileNet class
    Returns:
        Architecture encoded as string for further usage with _decode_architecure and timm.
    """

    def __init__(self, block_types, nr_sub_blocks, kernel_sizes, strides, output_filters, se_ratios, skip_connections, expansion_rates=0):
        self.block_types = block_types
        self.nr_sub_blocks = nr_sub_blocks
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.expansion_rates = expansion_rates
        self.output_filters = output_filters
        self.se_ratios = se_ratios
        self.skip_connections = skip_connections

        self.arch_encoded = [[""] for ind in range(len(self.block_types))]
        self._encode_architecture()

    def _encode_architecture(self):
        encoding_functions = [self._get_encoded_blocks, self._get_encoded_nr_sub_bocks, self._get_encoded_kernel_sizes, self._get_encoded_strides,
                              self._get_encoded_expansion_rates , self._get_encoded_output_filters, self._get_encoded_se_ratios, self._get_encoded_skip_connections]

        for func in encoding_functions:
            return_val = func()
            self._add_specifications(return_val)

    def _add_specifications(self, arguments):
        for ind, arg in enumerate(arguments):
            if len(self.arch_encoded[ind][0])!=0 and arg!="" and not self.arch_encoded[ind][0].endswith("_") :
                self.arch_encoded[ind][0] = self.arch_encoded[ind][0] + "_"
            self.arch_encoded[ind][0] = self.arch_encoded[ind][0] + arg

    def _get_encoded_blocks(self):
        block_type_dict = {"inverted_residual":"ir", "dwise_sep_conv":"ds", "conv_bn_act":"cn"}
        block_type_list = self._dict_to_list(self.block_types)
        return [block_type_dict[item] for item in block_type_list]

    def _get_encoded_nr_sub_bocks(self):
        nr_sub_blocks_dict = dict([(i, "r"+str(i)) for i in range(10)])
        nr_sub_blocks_list = self._dict_to_list(self.nr_sub_blocks)
        return [nr_sub_blocks_dict[item] for item in nr_sub_blocks_list]

    def _get_encoded_kernel_sizes(self):
        kernel_sizes_dict = dict([(i, "k"+str(i)) for i in range(10)])
        kernel_sizes_list = self._dict_to_list(self.kernel_sizes)
        return [kernel_sizes_dict[item] for item in kernel_sizes_list]

    def _get_encoded_strides(self):
        strides_dict = dict([(i, "s"+str(i)) for i in range(10)])
        strides_list = self._dict_to_list(self.strides)
        return [strides_dict[item] for item in strides_list]

    def _get_encoded_expansion_rates(self):
        if self.expansion_rates == 0:
            exp_list = ["e1","e6","e6","e6","e6","e6","e6"]
            return exp_list[0:len(self.block_types)]
        else:
            expansion_rates_dict = dict([(i, "e"+str(i)) for i in range(10)])
            expansion_rates_list = self._dict_to_list(self.expansion_rates)
            return [expansion_rates_dict[item] for item in expansion_rates_list]

    def _get_encoded_output_filters(self):
        output_filters_dict = dict([(i, "c"+str(i)) for i in range(5000)])
        output_filters_list = self._dict_to_list(self.output_filters)
        return [output_filters_dict[item] for item in output_filters_list]

    def _get_encoded_se_ratios(self):
        se_ratios_dict = {0:"", 0.25:"se0.25"}
        se_ratios_list = self._dict_to_list(self.se_ratios)
        return [se_ratios_dict[item] for item in se_ratios_list]

    def _get_encoded_skip_connections(self):
        skip_connections_dict = {True : "", False: "no_skip"}
        skip_connections_list = self._dict_to_list(self.skip_connections)
        return [skip_connections_dict[item] for item in skip_connections_list]

    def _dict_to_list(self, input_dict):
        output_list = []
        dict_len = len(input_dict)
        for ind in range(dict_len):
            output_list.append(input_dict["Group_" + str(ind+1)])
        return output_list
        
    def get_encoded_architecture(self):
        return self.arch_encoded


class MobileNet(BaseImageNet):
    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(MobileNet, self).__init__(config, in_features, out_features, final_activation)

        # Initialize hyperpars for architecture
        nn.Module.config = config
        self.final_activation = final_activation
        self.nr_main_blocks = config['nr_main_blocks']
        self.initial_filters = config['initial_filters']
        self.drop_rate = config['dropout_rate']
        self.drop_connect_rate = config['drop_connect_rate']


        self.nr_sub_blocks = dict([
            ('Group_%d' % (i+1), config['nr_sub_blocks_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.op_types = dict([
            ('Group_%d' % (i+1), config['op_type_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.kernel_sizes = dict([
            ('Group_%d' % (i+1), config['kernel_size_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.strides = dict([
            ('Group_%d' % (i+1), config['stride_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.output_filters = dict([
            ('Group_%d' % (i+1), config['out_filters_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.skip_cons = dict([
            ('Group_%d' % (i+1), config['skip_con_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        self.se_ratios = dict([
            ('Group_%d' % (i+1), config['se_ratio_%i' % (i+1)]) 
            for i in range(self.nr_main_blocks)])

        
        ########## Create model
        encoder = Arch_Encoder(block_types=self.op_types,
                               nr_sub_blocks=self.nr_sub_blocks, 
                               kernel_sizes=self.kernel_sizes,
                               strides=self.strides,
                               expansion_rates=0,
                               output_filters=self.output_filters,
                               se_ratios=self.se_ratios,
                               skip_connections=self.skip_cons)
        arch_enc = encoder.get_encoded_architecture()

        kwargs["bn_momentum"] = 0.01

        self.model = GenEfficientNet(_decode_arch_def(arch_enc, depth_multiplier=1.0),
                                     num_classes=out_features,
                                     stem_size=self.initial_filters,
                                     channel_multiplier=1.0,
                                     num_features=_round_channels(1280, 1.0, 8, None),
                                     bn_args=_resolve_bn_args(kwargs),
                                     act_fn=swish,
                                     drop_connect_rate=self.drop_connect_rate,
                                     drop_rate=self.drop_rate,
                                     **kwargs)

        def _cfg(url='', **kwargs):
            return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
                    'crop_pct': 0.875, 'interpolation': 'bicubic',
                    'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
                    'first_conv': 'conv_stem', 'classifier': 'classifier', **kwargs}

        self.model.default_cfg = _cfg(url='', input_size=in_features, pool_size=(10, 10), crop_pct=0.904, num_classes=out_features)

        # CONTINUE HERE
        #im_size = max(self.ih, self.iw)


        #self.feature_maps_out = feature_maps_in
        #self.model.add_module('ReLU_0', nn.ReLU(inplace=True))
        #self.model.add_module('AveragePool', nn.AdaptiveAvgPool2d(1))
        #self.fc = nn.Linear(self.feature_maps_out, out_features)

        #self.apply(initialize_weights)

    def forward(self, x):
        # make sure channels first
        x = self.model(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def train(self, *args, **kwargs):
        super(MobileNet, self).train(*args,**kwargs)
        self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        super(MobileNet, self).eval(*args, **kwargs)
        self.model.eval(*args, **kwargs)

    @staticmethod
    def get_config_space(   nr_main_blocks=[3, 7], initial_filters=[8, 32], nr_sub_blocks=[1, 4],
                            op_types = ["inverted_residual", "dwise_sep_conv"], kernel_sizes=[3, 5],  strides=[1,2],
                            output_filters = [[12, 16, 20],
                                              [18, 24, 30],
                                              [24, 32, 40],
                                              [48, 64, 80],
                                              [72, 96, 120],
                                              [120, 160, 200], 
                                              [240, 320, 400]],   # the idea is to search for e.g. 0.75, 1, 1.25* output_filters(mainblock number)
                            skip_connection = [True, False], se_ratios = [0, 0.25], dropout_rate=[0.0, 0.6], drop_connect_rate=[0.0,0.6], **kwargs):
                            
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH

        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('dropout_rate', lower=dropout_rate[0], upper=dropout_rate[1]))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('drop_connect_rate', lower=drop_connect_rate[0], upper=drop_connect_rate[1]))

        main_blocks = cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('nr_main_blocks', lower=nr_main_blocks[0], upper=nr_main_blocks[1]))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('initial_filters', lower=initial_filters[0], upper=initial_filters[1], log=True))
	    
        for i in range(1, nr_main_blocks[1] + 1):
            sub_blocks = cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('nr_sub_blocks_%d' % i, lower=nr_sub_blocks[0], upper=nr_sub_blocks[1], log=True))
            op_type = cs.add_hyperparameter(CSH.CategoricalHyperparameter('op_type_%d' % i, op_types))
            kernel_size = cs.add_hyperparameter(CSH.CategoricalHyperparameter('kernel_size_%d' % i, kernel_sizes))
            stride = cs.add_hyperparameter(CSH.CategoricalHyperparameter('stride_%d' % i, strides))
            out_filters = cs.add_hyperparameter(CSH.CategoricalHyperparameter('out_filters_%d' % i, output_filters[i-1]))             # take output_filters list i-1 as options
            skip_con = cs.add_hyperparameter(CSH.CategoricalHyperparameter('skip_con_%d' % i, [True, False]))
            se_ratio = cs.add_hyperparameter(CSH.CategoricalHyperparameter('se_ratio_%d' % i, se_ratios))

            if i > nr_main_blocks[0]:
                cs.add_condition(CS.GreaterThanCondition(sub_blocks, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(op_type, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(kernel_size, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(stride, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(out_filters, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(skip_con, main_blocks, i-1))
                cs.add_condition(CS.GreaterThanCondition(se_ratio, main_blocks, i-1))

        return cs


class ParallelMobileNet(MobileNet):

    def __init__(self, parent_net, n_gpus):
        super(ParallelMobileNet, self).__init__(parent_net.config, parent_net.n_feats, parent_net.n_classes, parent_net.final_activation)

        self.model = parent_net.model
        self.n_gpus = n_gpus
        self.layer_list = []

        get_layers(self.model, self.layer_list)
        self.parallelize_model()

    def auto_forward_layer(self, x, layer_ind_start, layer_ind_end=None):
        if layer_ind_end is None:
            layer_ind_end = len(self.layer_list)
        if layer_ind_start > -1:
            increment = layer_ind_start
        else:
            increment = len(self.layer_list) + layer_ind_start
        for ind,layer in enumerate(self.layer_list[layer_ind_start:layer_ind_end]):
            x = layer(x.to("cuda:"+str(int((ind+increment) // self.chunk_size))))
        return x

    def forward_features(self, x, pool=True):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act_fn(x, inplace=True)
        x = self.auto_forward_layer(x, 2, -4)
        if self.model.efficient_head:
            # efficient head, currently only mobilenet-v3 performs pool before last 1x1 conv
            x = self.model.global_pool(x)  # always need to pool here regardless of flag
            x = self.model.conv_head(x)
            # no BN
            x = self.model.act_fn(x, inplace=True)
            if pool:
                # expect flattened output if pool is true, otherwise keep dim
                x = x.view(x.size(0), -1)
        else:
            if self.model.conv_head is not None:
                x = self.model.conv_head(x)
                x = self.model.bn2(x)
            x = self.model.act_fn(x, inplace=True)
            if pool:
                x = self.model.global_pool(x)
                x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.model.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = self.model.classifier(x)

        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x.to("cuda:0")

    def parallelize_model(self):
        # TODO
        # This should just slice the mobilenet blocks, not the conv head and so on. conv stem... should be placed on first gpu,  conv head ... on last gpu
        n_layers = len(self.layer_list)

        self.chunk_size = np.ceil(n_layers/self.n_gpus)

        for ind, layer in enumerate(self.layer_list):
            layer.to("cuda:" + str(int(ind // self.chunk_size)))
