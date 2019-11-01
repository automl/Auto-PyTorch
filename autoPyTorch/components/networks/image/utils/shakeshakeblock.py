# coding: utf-8

import torch
from torch.autograd import Variable, Function


class ShakeShakeBlock(Function):
    @staticmethod
    def forward(ctx, alpha, beta, *args):
        ctx.save_for_backward(beta)

        y = sum(alpha[i] * args[i] for i in range(len(args)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        beta = ctx.saved_variables
        grad_x = [beta[0][i] * grad_output for i in range(beta[0].shape[0])]

        return (None, None, *grad_x)

shake_shake = ShakeShakeBlock.apply


def generate_alpha_beta(num_branches, batch_size, shake_config, is_cuda):
    forward_shake, backward_shake, shake_image = shake_config

    if forward_shake and not shake_image:
        alpha = torch.rand(num_branches)
    elif forward_shake and shake_image:
        alpha = torch.rand(num_branches, batch_size).view(num_branches, batch_size, 1, 1, 1)
    else:
        alpha = torch.ones(num_branches)

    if backward_shake and not shake_image:
        beta = torch.rand(num_branches) 
    elif backward_shake and shake_image:
        beta = torch.rand(num_branches, batch_size).view(num_branches, batch_size, 1, 1, 1)
    else:
        beta = torch.ones(num_branches)

    alpha = torch.nn.Softmax(0)(Variable(alpha))    
    beta = torch.nn.Softmax(0)(Variable(beta))

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
    
    return alpha, beta