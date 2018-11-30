
from hpbandster.optimizers.bohb import BOHB
from autoPyTorch.core.hpbandster_extensions.run_with_time import run_with_time

class BOHBExt(BOHB):
    def run_until(self, runtime=1, n_iterations=float("inf"), min_n_workers=1, iteration_kwargs = {},):
        """
            Parameters:
            -----------
            runtime: int
                time for this run in seconds
            n_iterations:
                the number of hyperband iterations to run
            min_n_workers: int
                minimum number of workers before starting the run
        """
        return run_with_time(self, runtime, n_iterations, min_n_workers, iteration_kwargs)