import re
from typing import Dict

import torch


def update_model_state_dict_from_swa(model: torch.nn.Module, swa_state_dict: Dict) -> None:
    """
    swa model adds a module keyword to each parameter,
    this function updates the state dict of the model
    using the state dict of the swa model
    Args:
        model:
        swa_state_dict:

    Returns:

    """
    model_state = model.state_dict()
    for name, param in swa_state_dict.items():
        name = re.sub('module.', '', name)
        if name not in model_state.keys():
            continue
        model_state[name].copy_(param)
