import queue
from multiprocessing.queues import Queue
from typing import List, Optional, Union

import numpy as np

from sklearn.ensemble import VotingRegressor

from smac.runhistory.runhistory import RunValue

from autoPyTorch.utils.common import autoPyTorchEnum


__all__ = [
    'read_queue',
    'convert_multioutput_multiclass_to_multilabel',
    'extract_learning_curve',
    'empty_queue',
    'VotingRegressorWrapper'
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


class VotingRegressorWrapper(VotingRegressor):
    """
    Wrapper around the sklearn voting regressor that properly handles
    predictions with shape (B, 1)
    """

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # overriding the _predict function should be enough
        predictions = []
        for est in self.estimators_:
            pred = est.predict(X)

            if pred.ndim > 1 and pred.shape[1] > 1:
                raise ValueError("Multi-output regression not yet supported with VotingRegressor. "
                                 "Issue is addressed here: https://github.com/scikit-learn/scikit-learn/issues/18289")

            predictions.append(pred.ravel())

        return np.asarray(predictions).T


class DisableFileOutputParameters(autoPyTorchEnum):
    """
    Contains literals that can be passed in to `disable_file_output` list.
    These include:

    + `y_optimization`:
        do not save the predictions for the optimization set,
        which would later on be used to build an ensemble. Note that SMAC
        optimizes a metric evaluated on the optimization set.
    + `pipeline`:
        do not save any individual pipeline files
    + `pipelines`:
        In case of cross validation, disables saving the joint model of the
        pipelines fit on each fold.
    + `y_test`:
        do not save the predictions for the test set.
    + `all`:
        do not save any of the above.
    """
    pipeline = 'pipeline'
    pipelines = 'pipelines'
    y_optimization = 'y_optimization'
    y_test = 'y_test'
    all = 'all'

    @classmethod
    def check_compatibility(
        cls,
        disable_file_output: List[Union[str, 'DisableFileOutputParameters']]
    ) -> None:
        for item in disable_file_output:
            if item not in cls.__members__ and not isinstance(item, cls):
                raise ValueError(f"Expected {item} to be in the members ("
                                 f"{list(cls.__members__.keys())}) of {cls.__name__}"
                                 f" or as string value of a member.")
