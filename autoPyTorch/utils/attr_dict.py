from typing import Any


_prohibited = ['clear', 'copy', 'fromkeys', 'get', 'items', 'keys',
               'pop', 'popitem', 'setdefault', 'update', 'values']


class AttrDict(dict):
    """non-static version of NamedTuple
    When we would like to define variables explicitly,
    we use this class.
    If neither a default value or an input
    are not given, we will use None.
    Examples:
    >>> class NewDict(AttrDict):
    >>>     a: int = 3
    >>>     b: float = 2.0
    >>>     c: str
    >>> new_dict = NewDict(a=1, d=5)
    >>> print(new_dict)
        AttrDict('NewDict', {'a': 1, 'd': 5, 'b': 2.0, 'c': None})
    >>> print(new_dict.a, new_dict.b, new_dict.c, new_dict.d)
        1 2.0 None 5
    >>> new_dict.a = 100
    >>> print(new_dict.a, new_dict['a'])
        100 100
    """
    def __init__(self, **kwargs: Any):
        if not hasattr(self, "__annotations__"):
            self.__annotations__ = {}

        var_dict = {var_name: getattr(self, var_name)
                    for var_name in self.__annotations__.keys()
                    if hasattr(self, var_name)}

        for var_name, default_value in var_dict.items():
            self._prohibited_overwrite(var_name)
            if var_name not in kwargs.keys():
                kwargs[var_name] = default_value

        for var_name in kwargs.keys():
            self._prohibited_overwrite(var_name)

        dict.__init__(self, **kwargs)
        self.__dict__ = self

    def __repr__(self) -> str:
        super_cls = set([obj.__name__ for obj in self.__class__.__mro__])
        super_cls -= set(['AttrDict', 'dict', 'object'])
        dict_name = list(super_cls)[0]

        seg = [f"AttrDict('{dict_name}', ", "{"]
        seg += [f"'{key}': {value}, " for key, value in self.items()]

        seg[-1] = seg[-1][:-2] + "})" if len(seg) > 2 else seg[-1] + "})"
        return "".join(seg)

    def _prohibited_overwrite(self, name: str) -> None:
        if name in _prohibited:
            raise AttributeError(f"Cannot overwrite dict attribute '{name}'. "
                                 "Use another variable name.")

    def __setattr__(self, name: str, value: Any) -> None:
        self._prohibited_overwrite(name)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        self._prohibited_overwrite(key)
        super().__setitem__(key, value)
