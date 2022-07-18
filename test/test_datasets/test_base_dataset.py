import numpy as np

import pytest

from autoPyTorch.datasets.base_dataset import _get_output_properties


@pytest.mark.parametrize(
    "target_labels,dim,task_type", (
        (np.arange(5), 5, "multiclass"),
        (np.linspace(0, 1, 3), 1, "continuous"),
        (np.linspace(0, 1, 3)[:, np.newaxis], 1, "continuous")
    )
)
def test_get_output_properties(target_labels, dim, task_type):
    train_tensors = np.array([np.empty_like(target_labels), target_labels])
    output_dim, output_type = _get_output_properties(train_tensors)
    assert output_dim == dim
    assert output_type == task_type
