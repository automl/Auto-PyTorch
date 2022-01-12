import json
import os
import warnings
from typing import Any, Dict, List, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from smac.optimizer.smbo import SMBO
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.initial_design.initial_design import InitialDesign
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.intensification.abstract_racer import AbstractRacer
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.ei_optimization import AbstractAcquisitionFunction, AcquisitionFunctionMaximizer
from smac.tae import FirstRunCrashedException, StatusType, TAEAbortException
from smac.tae.base import BaseRunner
from smac.optimizer.random_configuration_chooser import RandomConfigurationChooser, ChooserNoCoolDown


def read_return_initial_configurations(
    config_space: ConfigurationSpace,
    portfolio_selection: str
) -> List[Configuration]:

    # read and validate initial configurations
    portfolio_path = portfolio_selection if portfolio_selection != "greedy" else \
        os.path.join(os.path.dirname(__file__), '../configs/greedy_portfolio.json')
    try:
        initial_configurations_dict: List[Dict[str, Any]] = json.load(open(portfolio_path))
    except FileNotFoundError:
        raise FileNotFoundError("The path: {} provided for 'portfolio_selection' for "
                                "the file containing the portfolio configurations "
                                "does not exist. Please provide a valid path".format(portfolio_path))
    initial_configurations: List[Configuration] = list()
    for configuration_dict in initial_configurations_dict:
        try:
            configuration = Configuration(config_space, configuration_dict)
            initial_configurations.append(configuration)
        except Exception as e:
            warnings.warn(f"Failed to convert {configuration_dict} into"
                          f" a Configuration with error {e}. "
                          f"Therefore, it can't be used as an initial "
                          f"configuration as it does not match the current config space. ")
    return initial_configurations


class AdjustRunHistoryCallback:
    """
    Allows manipulating run history for custom needs
    """
    def __call__(self, smbo: 'SMBO') -> RunHistory:
        pass

class autoPyTorchSMBO(SMBO):
    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: AbstractRacer,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 tae_runner: BaseRunner,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: Union[RandomConfigurationChooser] = ChooserNoCoolDown(2.0),
                 predict_x_best: bool = True,
                 min_samples_model: int = 1):
        super().__init__(
            scenario, 
            stats, 
            initial_design, 
            runhistory, 
            runhistory2epm, 
            intensifier, 
            num_run, 
            model, 
            acq_optimizer, 
            acquisition_func, 
            rng, 
            tae_runner, 
            restore_incumbent, 
            random_configuration_chooser, 
            predict_x_best, 
            min_samples_model, 
        )
        self._callbacks.update({'_adjust_run_history': list()})
        self._callback_to_key.update({AdjustRunHistoryCallback: '_adjust_run_history'})

    def _incorporate_run_results(self, run_info: RunInfo, result: RunValue, time_left: float) -> None:
        # update SMAC stats
        self.stats.ta_time_used += float(result.time)
        self.stats.finished_ta_runs += 1

        self.logger.debug(
            "Return: Status: %r, cost: %f, time: %f, additional: %s" % (
                result.status, result.cost, result.time, str(result.additional_info)
            )
        )

        self.runhistory.add(
            config=run_info.config,
            cost=result.cost,
            time=result.time,
            status=result.status,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
            starttime=result.starttime,
            endtime=result.endtime,
            force_update=True,
            additional_info=result.additional_info,
        )
        self.stats.n_configs = len(self.runhistory.config_ids)

        if result.status == StatusType.ABORT:
            raise TAEAbortException("Target algorithm status ABORT - SMAC will "
                                    "exit. The last incumbent can be found "
                                    "in the trajectory-file.")
        elif result.status == StatusType.STOP:
            self._stop = True
            return

        if self.scenario.abort_on_first_run_crash:  # type: ignore[attr-defined] # noqa F821
            if self.stats.finished_ta_runs == 1 and result.status == StatusType.CRASHED:
                raise FirstRunCrashedException(
                    "First run crashed, abort. Please check your setup -- we assume that your default "
                    "configuration does not crashes. (To deactivate this exception, use the SMAC scenario option "
                    "'abort_on_first_run_crash'). Additional run info: %s" % result.additional_info
                )
        for callback in self._callbacks['_incorporate_run_results']:
            response = callback(smbo=self, run_info=run_info, result=result, time_left=time_left)
            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called
            if response is False:
                self.logger.debug("An IncorporateRunResultCallback returned False, requesting abort.")
                self._stop = True

        for callback in self._callbacks['_adjust_run_history']:
            result = callback(smbo=self)
        # Update the intensifier with the result of the runs
        self.incumbent, inc_perf = self.intensifier.process_results(
            run_info=run_info,
            incumbent=self.incumbent,
            run_history=self.runhistory,
            time_bound=max(self._min_time, time_left),
            result=result,
        )

        if self.scenario.save_results_instantly:  # type: ignore[attr-defined] # noqa F821
            self.save()

        return