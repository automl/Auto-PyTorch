from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import dask.distributed


class DummyFuture(dask.distributed.Future):
    """
    A class that mimics a distributed Future, the outcome of
    performing submit on a distributed client.
    """
    def __init__(self, result: Any):
        self._result: Any = result

    def result(self, timeout: Optional[int] = None) -> Any:
        return self._result

    def cancel(self) -> None:
        pass

    def done(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "DummyFuture: {}".format(self._result)

    def __del__(self) -> None:
        pass


class SingleThreadedClient(dask.distributed.Client):
    """
    A class to Mock the Distributed Client class.

    Using dask requires a scheduler which submits jobs on a different process. Also,
    pynisher submits jobs in a further additional process.

    When using a single core, we would prefer using the same main process without any
    multiprocessing overhead (that is, without the need of a LocalCluster in
    dask.distributed.Client). In other words, this class enriches the Client() class
    with the capability to run a future in the same thread (without any deadlock).
    """
    def __init__(self) -> None:

        # Raise a not implemented error if using a method from Client
        implemented_methods = ['submit', 'close', 'shutdown', 'write_scheduler_file',
                               '_get_scheduler_info', 'nthreads']
        method_list = [func for func in dir(dask.distributed.Client) if callable(
            getattr(dask.distributed.Client, func)) and not func.startswith('__')]
        for method in method_list:
            if method in implemented_methods:
                continue
            setattr(self, method, self._unsupported_method)
        pass

    def _unsupported_method(self) -> None:
        raise NotImplementedError()

    def submit(
        self,
        func: Callable,
        *args: List,
        priority: int = 0,
        key: Any = None,
        workers: Any = None,
        resources: Any = None,
        retries: Any = None,
        fifo_timeout: Any = "100 ms",
        allow_other_workers: Any = False,
        actor: Any = False,
        actors: Any = False,
        pure: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Note
        ----
        The keyword arguments caught in `dask.distributed.Client` need to
        be specified here so they don't get passed in as ``**kwargs`` to the
        ``func``.
        """
        return DummyFuture(func(*args, **kwargs))

    def close(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def write_scheduler_file(self, scheduler_file: str) -> None:
        Path(scheduler_file).touch()
        return

    def _get_scheduler_info(self) -> Dict:
        return {
            'workers': ['127.0.0.1'],
            'type': 'Scheduler',
        }

    def nthreads(self) -> Dict:
        return {
            '127.0.0.1': 1,
        }

    def __repr__(self) -> str:
        return 'SingleThreadedClient()'

    def __del__(self) -> None:
        pass
