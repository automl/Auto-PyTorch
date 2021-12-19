"""
This tests the functionality in autoPyTorch/utils/common.
"""
from enum import Enum

import pytest

from autoPyTorch.utils.common import autoPyTorchEnum


class SubEnum(autoPyTorchEnum):
    x = "x"
    y = "y"


class DummyEnum(Enum):  # You need to move it on top
    x = "x"


@pytest.mark.parametrize('iter',
                         ([SubEnum.x],
                          ["x"],
                          {SubEnum.x: "hello"},
                          {'x': 'hello'},
                          SubEnum,
                          ["x", "y"]))
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


@pytest.mark.parametrize('iter',
                         [[SubEnum.y],
                          ["y"],
                          {SubEnum.y: "hello"},
                          {'y': 'hello'}])
def test_autopytorch_enum_false(iter):
    """
    This test ensures that a subclass of `autoPyTorchEnum`
    can be used with strings.
    Args:
        iter (Iterable):
            iterable to check for compaitbility
    """

    e = SubEnum.x

    assert e not in iter


@pytest.mark.parametrize('others', (1, 2.0, SubEnum, DummyEnum.x))
def test_raise_errors_autopytorch_enum(others):
    """
    This test ensures that a subclass of `autoPyTorchEnum`
    raises error properly.
    Args:
        others (Any):
            Variable to compare with SubEnum.
    """

    with pytest.raises(RuntimeError):
        SubEnum.x == others
