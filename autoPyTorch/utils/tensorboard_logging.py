import logging
import os

from torch.utils.tensorboard import SummaryWriter

def get_tb_logger():
    global apt_tensorboard_writer
    if 'apt_tensorboard_writer' not in globals() or apt_tensorboard_writer is None:
        raise RuntimeError("Please initialize apt_tensorboard_writer before logging.")
    return apt_tensorboard_writer

def configure_tb_log_dir(logdir):
    logger = logging.getLogger('autopytorch.tb_logging')
    global apt_tensorboard_writer

    if 'apt_tensorboard_writer' not in globals():
        logger.debug("Initializing global variable apt_tensorboard_writer")
        apt_tensorboard_writer = None

    # Initialize summary writer if log_dir changed
    if apt_tensorboard_writer is not None and logdir == apt_tensorboard_writer.log_dir:
        return

    if  apt_tensorboard_writer is None:
        logger.debug("No tensorboard summary writer detected.")
    elif logdir != apt_tensorboard_writer.log_dir:
        logger.debug("Logdir changed from \"{}\" to \"{}\".".format(apt_tensorboard_writer.log_dir, logdir))
    logger.debug("Initializing tensorboard logger for directory \"{}\".".format(logdir))
    os.makedirs(logdir, exist_ok=True)
    apt_tensorboard_writer = SummaryWriter(log_dir=logdir)
