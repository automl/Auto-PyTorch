from autoPyTorch.pipeline.components.setup.network_backbone.MLPBackbone import MLPBackbone
import torch


def seq2tab(x: torch.Tensor):
    # https://discuss.pytorch.org/t/how-could-i-flatten-two-dimensions-of-a-tensor/44570/4
    return x.view(-1, *x.shape[2:])


