import torch.nn as nn
import math

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        #nn.init.kaiming_normal(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

def get_layer_params(in_size, out_size, kernel_size):
    kernel_size = int(kernel_size)
    stride = int(max(1, math.ceil((in_size - kernel_size) / (out_size - 1)) if out_size > 1 else 1))
    cur_out_size = _get_out_size(in_size, kernel_size, stride, 0)
    required_padding = (stride / 2) * (in_size - cur_out_size)

    cur_padding = int(math.ceil(required_padding))
    cur_out_size = _get_out_size(in_size, kernel_size, stride, cur_padding)
    if cur_padding < kernel_size and cur_out_size <= in_size and cur_out_size >= 1:
        return cur_out_size, kernel_size, stride, cur_padding
    
    cur_padding = int(math.floor(required_padding))
    cur_out_size = _get_out_size(in_size, kernel_size, stride, cur_padding)
    if cur_padding < kernel_size and cur_out_size <= in_size and cur_out_size >= 1:
        return cur_out_size, kernel_size, stride, cur_padding

    if stride > 1:
        stride = int(stride - 1)
        cur_padding = 0
        cur_out_size = int(_get_out_size(in_size, kernel_size, stride, cur_padding))
        if cur_padding < kernel_size and cur_out_size <= in_size and cur_out_size >= 1:
            return cur_out_size, kernel_size, stride, cur_padding

    if (kernel_size % 2) == 0 and out_size == in_size:
        return get_layer_params(in_size, out_size, kernel_size + 1) # an odd kernel can always keep the dimension (with stride 1)

    raise Exception('Could not find padding and stride to reduce ' + str(in_size) + ' to ' + str(out_size) + ' using kernel ' + str(kernel_size))

def _get_out_size(in_size, kernel_size, stride, padding):
    return int(math.floor((in_size - kernel_size + 2 * padding) / stride + 1))