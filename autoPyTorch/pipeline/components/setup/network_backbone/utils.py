import typing
import warnings

import torch
from torch.autograd import Function

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


_activations = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid
}


def get_output_shape(network: torch.nn.Module, input_shape: typing.Tuple[int, ...]
                     ) -> typing.Tuple[int, ...]:
    """
    Run a dummy forward pass to get the output shape of the backbone.
    Can and should be overridden by subclasses that know the output shape
    without running a dummy forward pass.
    :param input_shape: shape of the input
    :return: output_shape
    """
    placeholder = torch.randn((2, *input_shape), dtype=torch.float)
    with torch.no_grad():
        output = network(placeholder)
    return tuple(output.shape[1:])


class ShakeShakeFunction(Function):
    """
    References:
        Title: Shake-Shake regularization
        Authors: Xavier Gastaldi
        URL: https://arxiv.org/pdf/1705.07485.pdf
        Github URL: https://github.com/hysts/pytorch_shake_shake/blob/master/functions/shake_shake_function.py
    """
    @staticmethod
    def forward(
        ctx: typing.Any,  # No typing for AutogradContext
        x1: torch.Tensor,
        x2: torch.Tensor,
        alpha: torch.tensor,
        beta: torch.tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx: typing.Any,
                 grad_output: torch.Tensor
                 ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, x2, alpha, beta = ctx.saved_tensors
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_shake = ShakeShakeFunction.apply


class ShakeDropFunction(Function):
    """
    References:
        Title: ShakeDrop Regularization for Deep Residual Learning
        Authors: Yoshihiro Yamada et. al.
        URL: https://arxiv.org/pdf/1802.02375.pdf

        Title: ShakeDrop Regularization
        Authors: Yoshihiro Yamada et. al.
        URL: https://openreview.net/pdf?id=S1NHaMW0b

        Github URL: https://github.com/owruby/shake-drop_pytorch/blob/master/models/shakedrop.py
    """
    @staticmethod
    def forward(ctx: typing.Any,
                x: torch.tensor,
                alpha: torch.tensor,
                beta: torch.tensor,
                bl: torch.tensor,
                ) -> torch.Tensor:
        ctx.save_for_backward(x, alpha, beta, bl)

        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx: typing.Any,
                 grad_output: torch.Tensor
                 ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, alpha, beta, bl = ctx.saved_tensors
        grad_x = grad_alpha = grad_beta = grad_bl = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)

        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_get_alpha_beta(is_training: bool, is_cuda: bool
                         ) -> typing.Tuple[torch.tensor, torch.tensor]:
    if is_training:
        result = (torch.FloatTensor([0.5]), torch.FloatTensor([0.5]))
        return result if not is_cuda else (result[0].cuda(), result[1].cuda())

    # TODO implement other update methods
    alpha = torch.rand(1)
    beta = torch.rand(1)

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return alpha, beta


def shake_drop_get_bl(
        block_index: int,
        min_prob_no_shake: float,
        num_blocks: int,
        is_training: bool,
        is_cuda: bool
) -> torch.tensor:
    pl = 1 - ((block_index + 1) / num_blocks) * (1 - min_prob_no_shake)

    if not is_training:
        # Move to torch.randn(1) for reproducibility
        bl = torch.tensor(1.0) if torch.randn(1) <= pl else torch.tensor(0.0)
    if is_training:
        bl = torch.tensor(pl)

    if is_cuda:
        bl = bl.cuda()

    return bl


def get_shaped_neuron_counts(
    shape: str,
    in_feat: int,
    out_feat: int,
    max_neurons: int,
    layer_count: int
) -> typing.List[int]:
    counts = []  # type: typing.List[int]

    if (layer_count <= 0):
        return counts

    if (layer_count == 1):
        counts.append(out_feat)
        return counts

    max_neurons = max(in_feat, max_neurons)
    # https://mikkokotila.github.io/slate/#shapes

    if shape == 'brick':
        #
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |        |
        #   |___  ___|
        #
        for _ in range(layer_count - 1):
            counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'triangle':
        #
        #        /  \
        #       /    \
        #      /      \
        #     /        \
        #    /          \
        #   /_____  _____\
        #
        previous = in_feat
        step_size = int((max_neurons - previous) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous + step_size
            counts.append(previous)
        counts.append(max_neurons)
        counts.append(out_feat)

    if shape == 'funnel':
        #
        #   \            /
        #    \          /
        #     \        /
        #      \      /
        #       \    /
        #        \  /
        #
        previous = max_neurons
        counts.append(previous)

        step_size = int((previous - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(layer_count - 2):
            previous = previous - step_size
            counts.append(previous)

        counts.append(out_feat)

    if shape == 'long_funnel':
        #
        #   |        |
        #   |        |
        #   |        |
        #    \      /
        #     \    /
        #      \  /
        #
        brick_layer = int(layer_count / 2)
        funnel_layer = layer_count - brick_layer
        counts.extend(get_shaped_neuron_counts(
                      'brick', in_feat, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts(
                      'funnel', in_feat, out_feat, max_neurons, funnel_layer))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: long funnel layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'diamond':
        #
        #     /  \
        #    /    \
        #   /      \
        #   \      /
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 2) + 1
        funnel_layer = layer_count - triangle_layer
        counts.extend(get_shaped_neuron_counts(
                      'triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        remove_triangle_layer = len(counts) > 1
        if (remove_triangle_layer):
            # remove the last two layers since max_neurons == out_features
            # (-> two layers with the same size)
            counts = counts[0:-2]
        counts.extend(get_shaped_neuron_counts(
                      'funnel',
                      max_neurons,
                      out_feat,
                      max_neurons,
                      funnel_layer + (2 if remove_triangle_layer else 0)))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: diamond layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'hexagon':
        #
        #     /  \
        #    /    \
        #   |      |
        #   |      |
        #    \    /
        #     \  /
        #
        triangle_layer = int(layer_count / 3) + 1
        funnel_layer = triangle_layer
        brick_layer = layer_count - triangle_layer - funnel_layer
        counts.extend(get_shaped_neuron_counts(
                      'triangle', in_feat, max_neurons, max_neurons, triangle_layer))
        counts.extend(get_shaped_neuron_counts(
                      'brick', max_neurons, max_neurons, max_neurons, brick_layer))
        counts.extend(get_shaped_neuron_counts(
                      'funnel', max_neurons, out_feat, max_neurons, funnel_layer))

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: hexagon layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    if shape == 'stairs':
        #
        #   |          |
        #   |_        _|
        #     |      |
        #     |_    _|
        #       |  |
        #       |  |
        #
        previous = max_neurons
        counts.append(previous)

        if layer_count % 2 == 1:
            counts.append(previous)

        step_size = 2 * int((max_neurons - out_feat) / (layer_count - 1))
        step_size = max(0, step_size)
        for _ in range(int(layer_count / 2 - 1)):
            previous = previous - step_size
            counts.append(previous)
            counts.append(previous)

        counts.append(out_feat)

        if (len(counts) != layer_count):
            warnings.warn("\nWarning: stairs layer count does not match "
                          "" + str(layer_count) + " != " + str(len(counts)) + "\n")

    return counts
