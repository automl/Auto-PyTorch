#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for shake-shake and shake-drop regularization.
"""

import torch
import random
from torch.autograd import Function

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class ShakeShakeFunction(Function):
    @staticmethod
    def forward(ctx, x1, x2, alpha, beta):
        ctx.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, alpha, beta = ctx.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if ctx.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta
shake_shake = ShakeShakeFunction.apply


class ShakeDropFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, bl):
        ctx.save_for_backward(x, alpha, beta, bl)

        y = (bl + alpha - bl * alpha ) * x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, bl = ctx.saved_variables
        grad_x = grad_alpha = grad_beta = grad_bl = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (bl + beta - bl * beta)

        return grad_x, grad_alpha, grad_beta, grad_bl
shake_drop = ShakeDropFunction.apply

def shake_get_alpha_beta(is_training, is_cuda):
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

def shake_drop_get_bl(block_index, min_prob_no_shake, num_blocks, is_training, is_cuda):
    pl = 1 - ((block_index + 1)/ num_blocks) * (1 - min_prob_no_shake)

    if not is_training:
        bl = torch.tensor(1.0) if random.random() <= pl else torch.tensor(0.0)
    if is_training:
        bl = torch.tensor(pl)

    if is_cuda:
        bl = bl.cuda()

    return bl
