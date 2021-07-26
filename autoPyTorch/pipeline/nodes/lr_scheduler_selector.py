__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.components.lr_scheduler.lr_schedulers import AutoNetLearningRateSchedulerBase

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.training.lr_scheduling import LrScheduling

class LearningrateSchedulerSelector(PipelineNode):
    def __init__(self):
        super(LearningrateSchedulerSelector, self).__init__()

        self.lr_scheduler = dict()

    def fit(self, hyperparameter_config, pipeline_config, optimizer, training_techniques):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        scheduler_name = config['lr_scheduler'] if 'lr_scheduler' in config else pipeline_config['lr_scheduler'][0]
        lr_scheduler_type = self.lr_scheduler[scheduler_name]
        lr_scheduler_config = ConfigWrapper(scheduler_name, config)

        return {'training_techniques': [LrScheduling({"lr_scheduler": lr_scheduler_type(optimizer, lr_scheduler_config)})] + training_techniques}

    def add_lr_scheduler(self, name, lr_scheduler_type):
        if (not issubclass(lr_scheduler_type, AutoNetLearningRateSchedulerBase)):
            raise ValueError("learningrate scheduler type has to inherit from AutoNetLearningRateSchedulerBase")
        self.lr_scheduler[name] = lr_scheduler_type

    def remove_lr_scheduler(self, name):
        del self.lr_scheduler[name]

    def get_hyperparameter_search_space(self, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        selector = None
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("lr_scheduler", list(self.lr_scheduler.keys())))
        
        for lr_scheduler_name, lr_scheduler_type in self.lr_scheduler.items():
            lr_scheduler_cs = lr_scheduler_type.get_config_space()

            parent = {'parent': selector, 'value': lr_scheduler_name}
            cs.add_configuration_space( prefix=lr_scheduler_name, configuration_space=lr_scheduler_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter=parent)

        possible_lr_scheduler = sorted(set(pipeline_config["lr_scheduler"]).intersection(self.lr_scheduler.keys()))
        self._update_hyperparameter_range('lr_scheduler', possible_lr_scheduler, check_validity=False, override_if_already_modified=False)

        return self._apply_user_updates(cs)

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="lr_scheduler", default=list(self.lr_scheduler.keys()), type=str, list=True, choices=list(self.lr_scheduler.keys())),
        ]
        return options