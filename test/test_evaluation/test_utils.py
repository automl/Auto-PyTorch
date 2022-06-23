"""
Tests the functionality in autoPyTorch.evaluation.utils
"""
import numpy as np

import pytest

from autoPyTorch.constants import STRING_TO_OUTPUT_TYPES
from autoPyTorch.evaluation.utils import (
    DisableFileOutputParameters,
    ensure_prediction_array_sizes,
)


def test_ensure_prediction_array_sizes_errors():
    dummy = np.random.random(20)
    with pytest.raises(RuntimeError):
        ensure_prediction_array_sizes(dummy, 'binary', None, dummy)
    with pytest.raises(ValueError):
        ensure_prediction_array_sizes(dummy, 'binary', 1, None)


def test_ensure_prediction_array_sizes():
    output_types = list(STRING_TO_OUTPUT_TYPES.keys())
    dummy = np.random.random((20, 3))
    for output_type in output_types:
        if output_type == 'multiclass':
            num_classes = dummy.shape[-1]
            label_examples = np.array([0, 2, 0, 2])
            unique_train_labels = list(np.unique(label_examples))
            pred = np.array([
                [0.1, 0.9],
                [0.2, 0.8],
            ])
            ans = np.array([
                [0.1, 0.0, 0.9],
                [0.2, 0.0, 0.8]
            ])
            ret = ensure_prediction_array_sizes(
                prediction=pred,
                output_type=output_type,
                num_classes=num_classes,
                unique_train_labels=unique_train_labels
            )
            assert np.allclose(ans, ret)
        else:
            num_classes = 1

        ret = ensure_prediction_array_sizes(dummy, output_type, num_classes, dummy)
        assert np.allclose(ret, dummy)


@pytest.mark.parametrize('disable_file_output',
                         [['model', 'cv_model'],
                          [DisableFileOutputParameters.model, DisableFileOutputParameters.cv_model]])
def test_disable_file_output_no_error(disable_file_output):
    """
    Checks that `DisableFileOutputParameters.check_compatibility`
    does not raise an error for the parameterized values of `disable_file_output`.

    Args:
        disable_file_output ([List[Union[str, DisableFileOutputParameters]]]):
            Options that should be compatible with the `DisableFileOutputParameters`
            defined in `autoPyTorch`.
    """
    DisableFileOutputParameters.check_compatibility(disable_file_output=disable_file_output)


def test_disable_file_output_error():
    """
    Checks that `DisableFileOutputParameters.check_compatibility` raises an error
    for a value not present in `DisableFileOutputParameters` and ensures that the
    expected error is raised.
    """
    disable_file_output = ['dummy']
    with pytest.raises(ValueError, match=r"Expected .*? to be in the members (.*?) of"
                                         r" DisableFileOutputParameters or as string value"
                                         r" of a member."):
        DisableFileOutputParameters.check_compatibility(disable_file_output=disable_file_output)
