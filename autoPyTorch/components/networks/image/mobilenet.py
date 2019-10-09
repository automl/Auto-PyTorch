import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import ConfigSpace
from autoPyTorch.components.networks.base_net import BaseImageNet
from autoPyTorch.utils.config_space_hyperparameter import add_hyperparameter, get_hyperparameter

from torch.autograd import Variable
from autoPyTorch.components.networks.base_net import BaseImageNet

from .utils.mobilenet_utils import GenEfficientNet, _decode_arch_def, _resolve_bn_args, _round_channels, swish, sigmoid, hard_swish, hard_sigmoid, SelectAdaptivePool2d

# TODO
# EXPANSION RATIO (currenty hardcoded)
# ACTIVATION? (currently swish)

class Arch_Encoder():
    """ Encode block definition string
    Encodes a list of config space (dicts) through a string notation of arguments for further usage with _decode_architecure and timm.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
    
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
    """
    Implements a search space as in MnasNet (https://arxiv.org/abs/1807.11626) using inverted residuals.
    """
    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(MobileNet, self).__init__(config, in_features, out_features, final_activation)

        # Initialize hyperpars for architecture
        nn.Module.config = config
        self.final_activation = final_activation
        self.nr_main_blocks = config['nr_main_blocks']
        self.initial_filters = config['initial_filters']


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
                                     drop_connect_rate=0.2,
                                     drop_rate=0.2,
                                     **kwargs)

        def _cfg(url='', **kwargs):
            return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
                    'crop_pct': 0.875, 'interpolation': 'bicubic',
                    'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
                    'first_conv': 'conv_stem', 'classifier': 'classifier', **kwargs}

        self.model.default_cfg = _cfg(url='', input_size=in_features, pool_size=(10, 10), crop_pct=0.904, num_classes=out_features)

    def forward(self, x):
        # make sure channels first
        x = self.model(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    @staticmethod
    def get_config_space(   nr_main_blocks=[3, 7], initial_filters=([8, 32], True), nr_sub_blocks=([1, 4], False),
                            op_types = ["inverted_residual", "dwise_sep_conv"], kernel_sizes=[3, 5],  strides=[1,2],
                            output_filters = [[12, 16, 20],
                                              [18, 24, 30],
                                              [24, 32, 40],
                                              [48, 64, 80],
                                              [72, 96, 120],
                                              [120, 160, 200], 
                                              [240, 320, 400]],   # the idea is to search for e.g. 0.75, 1, 1.25* output_filters(mainblock number)
                            skip_connection = [True, False], se_ratios = [0, 0.25], **kwargs):
                            
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH

        cs = CS.ConfigurationSpace()

        
        main_blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, "nr_main_blocks", nr_main_blocks)
        initial_filters_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, "initial_filters", initial_filters)
        cs.add_hyperparameter(main_blocks_hp)
        cs.add_hyperparameter(initial_filters_hp)

        if type(nr_main_blocks[0]) == int:
            min_blocks = nr_main_blocks[0]
            max_blocks = nr_main_blocks[1]
        else:
            min_blocks = nr_main_blocks[0][0]
            max_blocks = nr_main_blocks[0][1]
	    
        for i in range(1, max_blocks + 1):
            sub_blocks_hp = get_hyperparameter(ConfigSpace.UniformIntegerHyperparameter, 'nr_sub_blocks_%d' % i, nr_sub_blocks)
            op_type_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'op_type_%d' % i, op_types)
            kernel_size_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'kernel_size_%d' % i, kernel_sizes)
            stride_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'stride_%d' % i, strides)
            out_filters_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'out_filters_%d' % i, output_filters[i-1])             # take output_filters list i-1 as options
            se_ratio_hp = get_hyperparameter(ConfigSpace.CategoricalHyperparameter, 'se_ratio_%d' % i, se_ratios)
            cs.add_hyperparameter(sub_blocks_hp)
            cs.add_hyperparameter(op_type_hp)
            cs.add_hyperparameter(kernel_size_hp)
            cs.add_hyperparameter(stride_hp)
            cs.add_hyperparameter(out_filters_hp)
            cs.add_hyperparameter(se_ratio_hp)
            skip_con = cs.add_hyperparameter(CSH.CategoricalHyperparameter('skip_con_%d' % i, [True, False]))

            if i > min_blocks:
                cs.add_condition(CS.GreaterThanCondition(sub_blocks_hp, main_blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(op_type_hp, main_blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(kernel_size_hp, main_blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(stride_hp, main_blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(out_filters_hp, main_blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(skip_con, main_blocks_hp, i-1))
                cs.add_condition(CS.GreaterThanCondition(se_ratio_hp, main_blocks_hp, i-1))

        return cs
