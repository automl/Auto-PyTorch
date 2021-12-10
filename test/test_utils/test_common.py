"""
This tests the functionality in autoPyTorch/utils/common.
"""
import pytest

from autoPyTorch.utils.common import autoPyTorchEnum


class SubEnum(autoPyTorchEnum):
    x = "x"
    y = "y"


@pytest.mark.parametrize('iter',
                         [[SubEnum.x],
                          ["x"],
                          {SubEnum.x: "hello"},
                          {'x': 'hello'}])
def test_autopytorch_enum(iter):
    """
    This test ensures that a subclass of `autoPyTorchEnum`
    can be used with strings.

    Args:
        iter (Iterable):
            iterable to check for compaitbility
    """

    e = SubEnum.x

    assert e in iter
