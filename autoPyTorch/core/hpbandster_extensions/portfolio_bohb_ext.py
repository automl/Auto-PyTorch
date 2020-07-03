import os
import time
import math
import copy
import json
import logging
import numpy as np

import ConfigSpace as CS
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB_CG

from autoPyTorch.core.hpbandster_extensions.run_with_time import run_with_time

def get_portfolio(portfolio_type):
    dirname = os.path.dirname(os.path.abspath(__file__))
    portfolio_file = os.path.join(dirname, portfolio_type+"_portfolio.json")

    with open(portfolio_file, "r") as f:
        portfolio_configs = json.load(f)
    return portfolio_configs


class PortfolioBOHB_CG(BOHB_CG):
    def __init__(self, initial_configs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initial_configs = initial_configs

    def get_config(self, budget):

        # return a portfolio member first
        if len(self.initial_configs) > 0 and True:
            c = self.initial_configs.pop(0)
            return (c, {'portfolio_member': True})

        return (super().get_config(budget))

    def new_result(self, job):
        # notify ensemble script or something
        super().new_result(job)


class PortfolioBOHB(Master):
    def __init__(self, configspace = None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model = None, top_n_percent=15,
                 num_samples = 64, random_fraction=1/3, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 portfolio_type="greedy",
                 **kwargs ):

        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        portfolio_configs = get_portfolio(portfolio_type=portfolio_type)

        cg = PortfolioBOHB_CG(initial_configs=portfolio_configs,
                              configspace = configspace,
                              min_points_in_model = min_points_in_model,
                              top_n_percent=top_n_percent,
                              num_samples = num_samples,
                              random_fraction=random_fraction,
                              bandwidth_factor=bandwidth_factor,
                              min_bandwidth = min_bandwidth)

        super().__init__(config_generator=cg, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

        self.config.update({
            'eta'        : eta,
            'min_budget' : min_budget,
            'max_budget' : max_budget,
            'budgets'    : self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model' : min_points_in_model,
            'top_n_percent' : top_n_percent,
            'num_samples' : num_samples,
            'random_fraction' : random_fraction,
            'bandwidth_factor' : bandwidth_factor,
            'min_bandwidth': min_bandwidth})

    def get_next_iteration(self, iteration, iteration_kwargs={}):
		
        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration%self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]

        return(SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s-1):], config_sampler=self.config_generator.get_config, **iteration_kwargs))

    def load_portfolio_configs(self):
        with open(self.portfolio_dir, "r") as f:
            configs = json.load(f)
        return configs


class PortfolioBOHBExt(PortfolioBOHB):
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
