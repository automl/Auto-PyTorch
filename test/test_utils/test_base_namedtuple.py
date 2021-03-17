from autoPyTorch.utils.common import create_dictlike_namedtuple
from typing import NamedTuple
import numpy as np


class DummyNamedTuple(NamedTuple):
    a: int = 1
    b: int = 2
    arr: np.ndarray = None


def test_base_namedtuple():
    dummy_namedtuple = create_dictlike_namedtuple(DummyNamedTuple)
    assert dummy_namedtuple.a == 1 and dummy_namedtuple['a'] == 1
    assert dummy_namedtuple.b == 2 and dummy_namedtuple['b'] == 2
    assert dummy_namedtuple.arr is None and dummy_namedtuple.arr is None

    for key, value in dummy_namedtuple.items():
        if key == 'a':
            assert value == 1
        elif key == 'b':
            assert value == 2
        elif key == 'arr':
            assert value is None
        else:
            raise KeyError(f'Key {key} does not exist.')

    try:
        dummy_namedtuple['c'] = 3
        raise TypeError('BaseNamedTuple must not support item assignment')
    except TypeError:
        pass

    try:
        dummy_namedtuple['a'] = 4
        raise TypeError('BaseNamedTuple must not support item assignment')
    except TypeError:
        pass

    dummy_namedtuple = create_dictlike_namedtuple(DummyNamedTuple, a=2, b=4, arr=np.arange(5))
    assert id(dummy_namedtuple.arr) == id(dummy_namedtuple['arr'])

    for key, value in dummy_namedtuple.items():
        if key == 'a':
            assert value == 2
        elif key == 'b':
            assert value == 4
        elif key == 'arr':
            assert all(v1 == v2 for v1, v2 in zip(value, np.arange(5)))
        else:
            raise KeyError(f'Key {key} does not exist.')


if __name__ == '__main__':
    test_base_namedtuple()
    print("Test complete.")
