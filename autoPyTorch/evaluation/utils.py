import queue
from multiprocessing.queues import Queue
from typing import List, Optional, Union

import numpy as np

from smac.runhistory.runhistory import RunValue

__all__ = [
    'read_queue',
    'convert_multioutput_multiclass_to_multilabel',
    'extract_learning_curve',
    'empty_queue'
]


def read_queue(queue_: Queue) -> List[RunValue]:
    stack: List[RunValue] = []
    while True:
        try:
            rval: RunValue = queue_.get(timeout=1)
        except queue.Empty:
            break

        # Check if there is a special placeholder value which tells us that
        # we don't have to wait until the queue times out in order to
        # retrieve the final value!
        if 'final_queue_element' in rval:
            del rval['final_queue_element']
            do_break = True
        else:
            do_break = False
        stack.append(rval)
        if do_break:
            break

    if len(stack) == 0:
        raise queue.Empty
    else:
        return stack


def empty_queue(queue_: Queue) -> None:
    while True:
        try:
            queue_.get(block=False)
        except queue.Empty:
            break

    queue_.close()


def extract_learning_curve(stack: List[RunValue], key: Optional[str] = None) -> List[List]:
    learning_curve = []
    for entry in stack:
        if key is not None:
            learning_curve.append(entry['additional_run_info'][key])
        else:
            learning_curve.append(entry['loss'])
    return list(learning_curve)


def convert_multioutput_multiclass_to_multilabel(probas: Union[List, np.ndarray]) -> np.ndarray:
    if isinstance(probas, np.ndarray) and len(probas.shape) > 2:
        raise ValueError('New unsupported sklearn output!')
    if isinstance(probas, list):
        multioutput_probas = np.ndarray((probas[0].shape[0], len(probas)))
        for i, output in enumerate(probas):
            if output.shape[1] > 2:
                raise ValueError('Multioutput-Multiclass supported by '
                                 'scikit-learn, but not by auto-pytorch!')
            # Only copy the probability of something having class 1
            elif output.shape[1] == 2:
                multioutput_probas[:, i] = output[:, 1]
            # This label was never observed positive in the training data,
            # therefore it is only the probability for the label being False
            else:
                multioutput_probas[:, i] = 0
        probas = multioutput_probas
    return probas
