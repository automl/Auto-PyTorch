import torch.nn as nn
import inspect


all_activations = {
    'relu' :        nn.ReLU,
    'sigmoid' :     nn.Sigmoid,
    'tanh' :        nn.Tanh,
    'leakyrelu' :   nn.LeakyReLU,
    'selu' :        nn.SELU,
    'rrelu' :       nn.RReLU,
    'tanhshrink' :  nn.Tanhshrink,
    'hardtanh' :    nn.Hardtanh,
    'elu' :         nn.ELU,
    'prelu' :       nn.PReLU,
}

def get_activation(name, inplace=False):
    if name not in all_activations:
        raise ValueError('Activation ' + str(name) + ' not defined')
    activation = all_activations[name]
    activation_kwargs = { 'inplace': True } if 'inplace' in inspect.getfullargspec(activation)[0] else dict()
    return activation(**activation_kwargs)
