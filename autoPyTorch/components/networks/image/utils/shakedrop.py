import torch
from torch.autograd import Variable, Function


class ShakeDrop(Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, death_rate, is_train):
        gate = (torch.rand(1) > death_rate).numpy()
        ctx.gate = gate
        ctx.save_for_backward(x, alpha, beta)

        if is_train:
            if not gate:
                y = alpha * x
            else:
                y = x
        else:
            y = x.mul(1 - (death_rate * 1.0))

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta = ctx.saved_variables
        grad_x1 = grad_alpha = grad_beta = None

        if ctx.needs_input_grad[0]:
            if not ctx.gate:
                grad_x = grad_output * beta
            else:
                grad_x = grad_output

        return grad_x, grad_alpha, grad_beta, None, None

shake_drop = ShakeDrop.apply


def generate_alpha_beta_single(tensor_size, shake_config, is_cuda):
    forward_shake, backward_shake, shake_image = shake_config

    if forward_shake and not shake_image:
        alpha = torch.rand(tensor_size).mul(2).add(-1)
    elif forward_shake and shake_image:
        alpha = torch.rand(tensor_size[0]).view(tensor_size[0], 1, 1, 1)
        alpha.mul_(2).add_(-1) # alpha from -1 to 1
    else:
        alpha = torch.FloatTensor([0.5])

    if backward_shake and not shake_image:
        beta = torch.rand(tensor_size)
    elif backward_shake and shake_image:
        beta = torch.rand(tensor_size[0]).view(tensor_size[0], 1, 1, 1)
    else:
        beta = torch.FloatTensor([0.5])

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return Variable(alpha), Variable(beta)