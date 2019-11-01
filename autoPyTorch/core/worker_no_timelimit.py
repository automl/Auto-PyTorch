import logging
import torch
import time
import random
from hpbandster.core.worker import Worker

from autoPyTorch.components.training.image.budget_types import BudgetTypeTime

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


class ModuleWorkerNoTimeLimit(Worker):
    def __init__(self, pipeline, pipeline_config, constant_hyperparameter,
            X_train, Y_train, X_valid, Y_valid, budget_type, max_budget, working_directory, permutations=None, *args, **kwargs):
        self.X_train = X_train #torch.from_numpy(X_train).float()
        self.Y_train = Y_train #torch.from_numpy(Y_train).long()
        self.X_valid = X_valid
        self.Y_valid = Y_valid

        if permutations is None:
            self.permutations = [idx for idx in range(len(X_train))]
        else:
            self.permutations = permutations

        self.max_budget = max_budget
        self.budget_type = budget_type

        self.pipeline = pipeline
        self.pipeline_config = pipeline_config
        self.constant_hyperparameter = constant_hyperparameter

        self.working_directory = working_directory

        self.autonet_logger = logging.getLogger('autonet')
        # self.end_time = None

        # We can only use user defined limits (memory) if we have the required module 'resource' - not available on windows!
        self.guarantee_limits = module_exists("resource") and module_exists("pynisher")
        if (not self.guarantee_limits):
            self.autonet_logger.info("Can not guarantee memory and time limit because module 'resource' is not available")

        super().__init__(*args, **kwargs)
    
    def compute(self, config, budget, working_directory, config_id, **kwargs):

        start_time = time.time()

        self.autonet_logger.debug("Starting optimization!")

        config.update(self.constant_hyperparameter)
        
        self.autonet_logger.debug("Budget " + str(budget) + " config: " + str(config))

        if self.guarantee_limits and self.budget_type == 'time':
            import pynisher

            limit_train = pynisher.enforce_limits(wall_time_in_s=int(budget * 4))(self.optimize_pipeline)
            result, randomstate = limit_train(config, budget, config_id, random.getstate())

            if (limit_train.exit_status == pynisher.MemorylimitException):
                raise Exception("Memory limit reached. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))
            elif (limit_train.exit_status != 0):
                self.autonet_logger.info('Exception occurred using config:\n' + str(config))
                raise Exception("Exception in train pipeline. Took " + str((time.time()-start_time)) + " seconds with budget " + str(budget))

        else:
            result, randomstate = self.optimize_pipeline(config, budget, config_id, random.getstate())

        random.setstate(randomstate)

        loss = result['loss']
        if 'losses' in result.keys():
            losses = result['losses']
        else:
            losses = loss
        info = result['info']

        self.autonet_logger.debug("Result: " + str(loss) + " info: " + str(info))

        # that is not really elegant but we can want to achieve some kind of feedback
        network_name = [v for k, v in config.items() if k.endswith('network')] or "None"

        self.autonet_logger.info("Training " + str(network_name) + " with budget " + str(budget) + " resulted in score: " + str(loss) + " took " + str((time.time()-start_time)) + " seconds")

        if 'use_tensorboard_logger' in self.pipeline_config and self.pipeline_config['use_tensorboard_logger']:
            import os
            log_file = os.path.join(self.working_directory, "worker_logs_" + str(self.pipeline_config['task_id']), 'results.log')
            sep = '\t'
            with open(log_file, 'a+') as f:
                f.write('Result: ' + str(round(loss, 2)) + sep + \
                        'Budget: ' + str(round(budget)) + '/' + str(round(self.pipeline_config['max_budget'])) + sep + \
                        'Used time: ' + str(round((time.time()-start_time))) + 'sec (' + str(round((time.time()-start_time)/budget, 2)) + 'x)' + sep + \
                        'ID: ' + str(config_id) + '\n')

        return  {
                    'loss': loss,
                    'info': info,
                    'losses': losses
                }
    
    def optimize_pipeline(self, config, budget, config_id, random_state):
        
        random.setstate(random_state)

        if self.permutations is not None:
            current_sh_run = config_id[0]
            self.pipeline_config["dataset_order"] = self.permutations[current_sh_run%len(self.permutations)].tolist()

        try:
            self.autonet_logger.info("Fit optimization pipeline")
            return self.pipeline.fit_pipeline(hyperparameter_config=config, pipeline_config=self.pipeline_config, 
                                            X_train=self.X_train, Y_train=self.Y_train, X_valid=self.X_valid, Y_valid=self.Y_valid, 
                                            budget=budget, budget_type=self.budget_type, max_budget=self.max_budget, 
                                            config_id=config_id, working_directory=self.working_directory), random.getstate()
        except Exception as e:
            if 'use_tensorboard_logger' in self.pipeline_config and self.pipeline_config['use_tensorboard_logger']:            
                import tensorboard_logger as tl
                tl.log_value('Exceptions/' + str(e), budget, int(time.time()))
            #self.autonet_logger.exception('Exception occurred')
            raise e

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True
