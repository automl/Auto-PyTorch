import torch
import os

import logging


def get_checkpoint_name(config_id, budget):
    return 'checkpoint_' + str(config_id) + '_Budget_' + str(int(budget)) + '.pt'

def get_checkpoint_dir(working_directory):
    return os.path.join(working_directory, 'checkpoints')

def save_checkpoint(path, config_id, budget, model, optimizer, scheduler):

    name = get_checkpoint_name(config_id, budget)
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, name)

    torch.save({
       'state': model.state_dict(),
    }, open(path, 'wb'))

    logging.getLogger('autonet').debug('=> Model {} saved to {}'.format(str(type(model)), path))
    return path


def load_checkpoint(path, config_id, budget):
    name = get_checkpoint_name(config_id, budget)

    path = os.path.join(path, name)
    if not os.path.exists(path):
        return None

    logging.getLogger('autonet').debug('=> Loading checkpoint ' + path)
    checkpoint = torch.load(path)
    return checkpoint


