__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.base.pipeline_node import PipelineNode

from autoPyTorch.pipeline.nodes.lr_scheduler_selector import LearningrateSchedulerSelector

import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption

class SimpleLearningrateSchedulerSelector(LearningrateSchedulerSelector):

    def fit(self, hyperparameter_config, pipeline_config, optimizer):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)

        scheduler_name = config['lr_scheduler']

        lr_scheduler_type = self.lr_scheduler[scheduler_name]
        lr_scheduler_config = ConfigWrapper(scheduler_name, config)
        lr_scheduler = lr_scheduler_type(optimizer, lr_scheduler_config)

        return {'lr_scheduler': lr_scheduler}