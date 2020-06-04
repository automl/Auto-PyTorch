__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.components.lr_scheduler.lr_schedulers import AutoNetLearningRateSchedulerBase

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption
from autoPyTorch.components.training.lr_scheduling import LrScheduling

class LearningrateSchedulerSelector(PipelineNode):
    def __init__(self):
        super(LearningrateSchedulerSelector, self).__init__()

        self.lr_scheduler = dict()
        self.lr_scheduler_settings = dict()

    def fit(self, hyperparameter_config, optimizer, training_techniques):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        lr_scheduler_type = self.lr_scheduler[config["lr_scheduler"]]
        lr_scheduler_config = ConfigWrapper(config["lr_scheduler"], config)
        lr_scheduler_settings = self.lr_scheduler_settings[config["lr_scheduler"]]
        lr_scheduling = LrScheduling(training_components={"lr_scheduler": lr_scheduler_type(optimizer, lr_scheduler_config)},
                                     **lr_scheduler_settings)
        return {'training_techniques': [lr_scheduling] + training_techniques}

    def add_lr_scheduler(self, name, lr_scheduler_type, lr_step_after_batch=False, lr_step_with_time=False, allow_snapshot=True):
        if (not issubclass(lr_scheduler_type, AutoNetLearningRateSchedulerBase)):
            raise ValueError("learningrate scheduler type has to inherit from AutoNetLearningRateSchedulerBase")
        self.lr_scheduler[name] = lr_scheduler_type
        self.lr_scheduler_settings[name] = {
            "lr_step_after_batch": lr_step_after_batch,
            "lr_step_with_time": lr_step_with_time,
            "allow_snapshot": allow_snapshot
        }

    def remove_lr_scheduler(self, name):
        del self.lr_scheduler[name]
        del self.lr_scheduler_settings[name]

    def get_hyperparameter_search_space(self, dataset_info=None, **pipeline_config):
        pipeline_config = self.pipeline.get_pipeline_config(**pipeline_config)
        cs = ConfigSpace.ConfigurationSpace()

        possible_lr_scheduler = set(pipeline_config["lr_scheduler"]).intersection(self.lr_scheduler.keys())
        selector = cs.add_hyperparameter(CSH.CategoricalHyperparameter("lr_scheduler", sorted(possible_lr_scheduler)))
        
        for lr_scheduler_name, lr_scheduler_type in self.lr_scheduler.items():
            if (lr_scheduler_name not in possible_lr_scheduler):
                continue
            lr_scheduler_cs = lr_scheduler_type.get_config_space(
                **self._get_search_space_updates(prefix=lr_scheduler_name))
            cs.add_configuration_space( prefix=lr_scheduler_name, configuration_space=lr_scheduler_cs, delimiter=ConfigWrapper.delimiter, 
                                        parent_hyperparameter={'parent': selector, 'value': lr_scheduler_name})

        self._check_search_space_updates((possible_lr_scheduler, "*"))
        return cs

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name="lr_scheduler", default=list(self.lr_scheduler.keys()), type=str, list=True, choices=list(self.lr_scheduler.keys())),
        ]
        return options
