import warnings
from typing import Any, List, Tuple

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


def get_output_shape(network: torch.nn.Module, input_shape: Tuple[int, ...], has_hidden_states: bool = False
                     ) -> Tuple[int, ...]:
    """
    Run a dummy forward pass to get the output shape of the backbone.
    Can and should be overridden by subclasses that know the output shape
    without running a dummy forward pass.
    :param input_shape: shape of the input
    :param has_hidden_states: bool, if the network backbone contains a hidden_states. if yes,
        the network will return a Tuple, we will then only consider the first item
    :return: output_shape
    """
    placeholder = torch.randn((2, *input_shape), dtype=torch.float)
    with torch.no_grad():
        if has_hidden_states:
            output = network(placeholder)[0]
        else:
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
        ctx: Any,  # No typing for AutogradContext
        x1: torch.Tensor,
        x2: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx: Any,
                 grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def forward(ctx: Any,
                x: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor,
                bl: torch.Tensor,
                ) -> torch.Tensor:
        ctx.save_for_backward(x, alpha, beta, bl)

        y = (bl + alpha - bl * alpha) * x
        return y

    @staticmethod
    def backward(ctx: Any,
                 grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, alpha, beta, bl = ctx.saved_tensors
        grad_x = grad_alpha = grad_beta = grad_bl = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)

        return grad_x, grad_alpha, grad_beta, grad_bl


shake_drop = ShakeDropFunction.apply


def shake_get_alpha_beta(is_training: bool, is_cuda: bool
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The methods used in this function have been introduced in 'ShakeShake Regularisation'
    Currently, this function supports `shake-shake`.

    Args:
        is_training (bool): Whether the computation for the training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        alpha, beta (Tuple[float, float]):
            alpha (in [0, 1]) is the weight coefficient  for the forward pass
            beta (in [0, 1]) is the weight coefficient for the backward pass

    Reference:
        Title: Shake-shake regularization
        Author: Xavier Gastaldi
        URL: https://arxiv.org/abs/1705.07485

    Note:
        The names have been taken from the paper as well.
        Currently, this function supports `shake-shake`.
    """
    if not is_training:
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
) -> torch.Tensor:
    """
    The sampling of Bernoulli random variable
    based on Eq. (4) in the paper

    Args:
        block_index (int): The index of the block from the input layer
        min_prob_no_shake (float): The initial shake probability
        num_blocks (int): The total number of building blocks
        is_training (bool): Whether it is training
        is_cuda (bool): Whether the tensor is on CUDA

    Returns:
        bl (torch.Tensor): a Bernoulli random variable in {0, 1}

    Reference:
        ShakeDrop Regularization for Deep Residual Learning
        Yoshihiro Yamada et. al. (2020)
        paper: https://arxiv.org/pdf/1802.02375.pdf
        implementation: https://github.com/imenurok/ShakeDrop
    """

    pl = 1 - ((block_index + 1) / num_blocks) * (1 - min_prob_no_shake)

    if is_training:
        # Move to torch.rand(1) for reproducibility
        bl = torch.as_tensor(1.0) if torch.rand(1) <= pl else torch.as_tensor(0.0)
    else:
        bl = torch.as_tensor(pl)

    if is_cuda:
        bl = bl.cuda()

    return bl


def get_shaped_neuron_counts(
    shape: str,
    in_feat: int,
    out_feat: int,
    max_neurons: int,
    layer_count: int
) -> List[int]:

    counts: List[int] = []

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
