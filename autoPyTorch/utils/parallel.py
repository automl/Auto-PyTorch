import multiprocessing
import sys


def preload_modules(context: multiprocessing.context.BaseContext) -> None:
    """
    This function is meant to be used with the forkserver multiprocessing context.
    More details about it can be found here:
    https://docs.python.org/3/library/multiprocessing.html

    Forkserver is known to be slower than other contexts. We use it, because it helps
    reduce the probability of a deadlock. To make it fast, we pre-load modules so that
    forked children have the desired modules available.

    We do not inherit dead-lock problematic modules like logging.

    Args:
        context (multiprocessing.context.BaseContext): One of the three supported multiprocessing
            contexts being fork, forkserver or spawn.
    """
    all_loaded_modules = sys.modules.keys()
    preload = [
        loaded_module for loaded_module in all_loaded_modules
        if loaded_module.split('.')[0] in (
            'smac',
            'autoPyTorch',
            'numpy',
            'scipy',
            'pandas',
            'pynisher',
            'sklearn',
            'ConfigSpace',
            'torch',
            'torchvision',
            'tensorboard',
            'imgaug',
            'catboost',
            'lightgbm',
        ) and 'logging' not in loaded_module
    ]
    context.set_forkserver_preload(preload)
