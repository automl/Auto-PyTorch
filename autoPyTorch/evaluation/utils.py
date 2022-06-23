import queue
from multiprocessing.queues import Queue
from typing import List, Optional, Union

import numpy as np

from sklearn.ensemble import VotingRegressor

from smac.runhistory.runhistory import RunValue

from autoPyTorch.constants import (
    MULTICLASS,
    STRING_TO_OUTPUT_TYPES
)
from autoPyTorch.utils.common import autoPyTorchEnum


__all__ = [
    'read_queue',
    'convert_multioutput_multiclass_to_multilabel',
    'ensure_prediction_array_sizes',
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


def ensure_prediction_array_sizes(
    prediction: np.ndarray,
    output_type: str,
    num_classes: Optional[int],
    unique_train_labels: Optional[List[int]]
) -> np.ndarray:
    """
    This function formats a prediction to match the dimensionality of the provided
    labels `unique_train_labels`. This should be used exclusively for classification tasks.
    This function is typically important when using cross validation, which might cause
    some splits not having some class in the training split.

    Args:
        prediction (np.ndarray):
            The un-formatted predictions of a pipeline
        output_type (str):
            Output type specified in constants. (TODO: Fix it to enum)
        unique_train_labels (Optional[List[int]]):
            The labels from the dataset to give an intuition of the expected
            predictions dimensionality

    Returns:
        (np.ndarray):
            The formatted prediction
    """
    if num_classes is None:
        raise RuntimeError("ensure_prediction_array_sizes is only for classification tasks")
    if unique_train_labels is None:
        raise ValueError('unique_train_labels must be provided, but got None')

    if STRING_TO_OUTPUT_TYPES[output_type] != MULTICLASS or prediction.shape[1] == num_classes:
        return prediction

    mapping = {
        unique_train_labels.index(class_idx): class_idx
        for class_idx in range(num_classes) if class_idx in unique_train_labels
    }
    # augment the array size when the output shape is different
    modified_pred = np.zeros((prediction.shape[0], num_classes), dtype=np.float32)

    for index, class_index in mapping.items():
        modified_pred[:, class_index] = prediction[:, index]

    return modified_pred


def extract_learning_curve(stack: List[RunValue], key: Optional[str] = None) -> List[float]:
    """
    Extract learning curve from the additional info.

    Args:
        stack (List[RunValue]):
            The stack of the additional information.
        key (Optional[str]):
            The key to extract.

    Returns:
        learning_curve (List[float]):
            The list of the extracted information

    Note:
        This function is experimental.
        The source of information in RunValue might require modifications.
    """
    learning_curve = []
    key = 'loss' if key is None else key

    for entry in stack:
        try:
            info = entry.additional_info
            learning_curve.append(getattr(entry, key, info[key]))
        except AttributeError:  # additional info is not RunValue
            pass
        except KeyError:  # Key does not exist
            pass

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

    + `y_opt`:
        do not save the predictions for the optimization set,
        which would later on be used to build an ensemble. Note that SMAC
        optimizes a metric evaluated on the optimization set.
    + `model`:
        do not save any individual pipeline files
    + `cv_model`:
        In case of cross validation, disables saving the joint model of the
        pipelines fit on each fold.
    + `y_test`:
        do not save the predictions for the test set.
    + `all`:
        do not save any of the above.
    """
    model = 'model'
    cv_model = 'cv_model'
    y_opt = 'y_opt'
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
