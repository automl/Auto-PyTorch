from autoPyTorch.utils.common import BaseDict
import numpy as np


class DummyDict(BaseDict):
    a: int = 1
    b: int = 2
    arr: np.ndarray


def test_base_dict():
    dummy_dict = DummyDict()
    assert dummy_dict.a == 1 and dummy_dict['a'] == 1
    assert dummy_dict.b == 2 and dummy_dict['b'] == 2

    dummy_dict = DummyDict(a=2, b=4)
    assert dummy_dict.a == 2 and dummy_dict['a'] == 2
    assert dummy_dict.b == 4 and dummy_dict['b'] == 4

    dummy_dict.b = 6
    assert dummy_dict.b == 6 and dummy_dict['b'] == 6

    dummy_dict['c'] = 8
    assert dummy_dict.c == 8 and dummy_dict['c'] == 8

    dummy_dict.d = 10
    assert dummy_dict.d == 10 and dummy_dict['d'] == 10

    assert dummy_dict.arr is None and dummy_dict['arr'] is None

    dummy_dict.arr = np.arange(5)
    assert id(dummy_dict['arr']) == id(dummy_dict.arr)


if __name__ == '__main__':
    test_base_dict()
    print("Test complete.")
