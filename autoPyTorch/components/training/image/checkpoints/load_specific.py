
import os
import math
import torch
import torch.nn as nn


import logging


def load_model(model, checkpoint):

    if checkpoint is None:
        return model

    pretrained_state = checkpoint['state']
    model_state = model.state_dict()

    pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
    logging.getLogger('autonet').debug('=> Resuming model using ' + str(len(pretrained_state.keys())) + '/' + str(len(model_state.keys())) + ' parameters')
    model_state.update(pretrained_state)
    model.load_state_dict(model_state)
    
    return model

# def load_optimizer(optimizer, checkpoint, device):
    
#     if checkpoint is None:
#         return optimizer

#     opti_state = optimizer.state_dict()
#     pretrained_state = checkpoint['optimizer']

#     logging.getLogger('autonet').debug(str(len(pretrained_state['state'])))
#     logging.getLogger('autonet').debug(str(len(opti_state['param_groups'][0]['params'])))
#     logging.getLogger('autonet').debug(str(len(pretrained_state['param_groups'][0]['params'])))
#     logging.getLogger('autonet').debug(str(set(pretrained_state['param_groups'][0]['params']).intersection(set(opti_state['param_groups'][0]['params']))))


#     pretrained_state = {k: pretrained_state[k] for state in opti_state.items() for k, v in enumerate(state) if state in pretrained_state and k in pretrained_state[state] and v.size() == opti_state[state][k].size()}
#     logging.getLogger('autonet').debug('=> Resuming optimizer using ' + str(len(pretrained_state.keys())) + '/' + str(len(opti_state.keys())))
#     opti_state.update(pretrained_state)
#     optimizer.load_state_dict(opti_state)

#     for state in optimizer.state.values():
#         for k, v in state.items():
#             if isinstance(v, torch.Tensor):
#                 state[k] = v.to(device)
#     return optimizer

# def load_scheduler(scheduler, checkpoint):

#     if checkpoint is None:
#         return scheduler

#     loaded_scheduler = checkpoint['scheduler']
#     loaded_scheduler.optimizer = scheduler.optimizer
#     return loaded_scheduler