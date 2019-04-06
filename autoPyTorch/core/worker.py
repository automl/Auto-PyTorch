import logging
import torch
import time
import numpy as np
import Pyro4
from hpbandster.core.worker import Worker

from autoPyTorch.components.training.budget_types import BudgetTypeTime

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class ModuleWorker(Worker):
    def __init__(self, pipeline, pipeline_config,
            X_train, Y_train, X_valid, Y_valid, dataset_info, budget_type, max_budget,
            shutdownables, use_pynisher, *args, **kwargs):
        self.X_train = X_train #torch.from_numpy(X_train).float()
        self.Y_train = Y_train #torch.from_numpy(Y_train).long()
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.dataset_info = dataset_info
        self.shutdownables = shutdownables

        self.max_budget = max_budget
        self.budget_type = budget_type

        self.pipeline = pipeline
        self.pipeline_config = pipeline_config

        self.autonet_logger = logging.getLogger('autonet')

        # We can only use user defined limits (memory) if we have the required module 'resource' - not available on windows!
        self.guarantee_limits = use_pynisher and module_exists("resource") and module_exists("pynisher")
        if (not self.guarantee_limits):
            self.autonet_logger.info("Can not guarantee memory and time limit because module 'resource' is not available")


        super().__init__(*args, **kwargs)
    
    def compute(self, config, budget, working_directory, config_id, **kwargs):

        self.autonet_logger.debug("Budget " + str(budget) + " config: " + str(config))

        start_time = time.time()
        self.autonet_logger.debug("Starting optimization!")

        if self.guarantee_limits:
            import pynisher
            time_limit=None

            if self.budget_type == BudgetTypeTime:
                grace_time = 10
                time_limit = int(budget + 240)

            limit_train = pynisher.enforce_limits(mem_in_mb=self.pipeline_config['memory_limit_mb'], wall_time_in_s=time_limit)(self.optimize_pipeline)
            result = limit_train(config, config_id, budget, start_time)

            if (limit_train.exit_status == pynisher.TimeoutException):
                raise Exception("Time limit reached. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
            elif (limit_train.exit_status == pynisher.MemorylimitException):
                raise Exception("Memory limit reached. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
            elif (limit_train.exit_status != 0):
                self.autonet_logger.info('Exception occurred using config:\n' + str(config))
                raise Exception("Exception in train pipeline. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
        else:
            result = self.optimize_pipeline(config, config_id, budget, start_time)

        loss = result['loss']
        info = result['info']
        self.autonet_logger.debug("Result: " + str(loss) + " info: " + str(info))

        # that is not really elegant but we can want to achieve some kind of feedback
        network_name = [v for k, v in config.items() if k.endswith('network')] or "None"

        self.autonet_logger.info("Training " + str(network_name) + " with budget " + str(budget) + " resulted in score: " + str(loss) + " took " + str((time.time()-start_time)) + " seconds")

        return  result
    
    def optimize_pipeline(self, config, config_id, budget, optimize_start_time):
        try:
            self.autonet_logger.info("Fit optimization pipeline")
            return self.pipeline.fit_pipeline(hyperparameter_config=config, pipeline_config=self.pipeline_config, 
                                            X_train=self.X_train, Y_train=self.Y_train, X_valid=self.X_valid, Y_valid=self.Y_valid, 
                                            budget=budget, budget_type=self.budget_type, max_budget=self.max_budget, optimize_start_time=optimize_start_time,
                                            refit=False, rescore=False, hyperparameter_config_id=config_id, dataset_info=self.dataset_info)
        except Exception as e:
            if 'use_tensorboard_logger' in self.pipeline_config and self.pipeline_config['use_tensorboard_logger']:            
                import tensorboard_logger as tl
                tl.log_value('Exceptions/' + str(e), budget, int(time.time()))
            self.autonet_logger.info(str(e))
            raise e
    
    @Pyro4.expose
    @Pyro4.oneway
    def shutdown(self):
        for s in self.shutdownables:
            s.shutdown()
        super().shutdown()

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True