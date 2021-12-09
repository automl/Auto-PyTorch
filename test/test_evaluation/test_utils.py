"""
Tests the functionality in autoPyTorch.evaluation.utils
"""
import pytest

from autoPyTorch.evaluation.utils import DisableFileOutputParameters


@pytest.mark.parametrize('disable_file_output',
                         [['pipeline', 'pipelines'],
                          [DisableFileOutputParameters.pipelines, DisableFileOutputParameters.pipeline]])
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
    disable_file_output = ['model']
    with pytest.raises(ValueError, match=r"Expected .*? to be in the members (.*?) of"
                                         r" DisableFileOutputParameters or as string value"
                                         r" of a member."):
        DisableFileOutputParameters.check_compatibility(disable_file_output=disable_file_output)
