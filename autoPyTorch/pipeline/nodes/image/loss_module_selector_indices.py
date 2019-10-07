__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
import numpy as np

import ConfigSpace
import ConfigSpace.hyperparameters as CSH

from autoPyTorch.pipeline.nodes.loss_module_selector import LossModuleSelector
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption


class LossModuleSelectorIndices(LossModuleSelector):
    def fit(self, hyperparameter_config, pipeline_config, X, Y, train_indices, dataset_info):

        if Y.shape[0] == dataset_info.y_shape[0]:
            return super(LossModuleSelectorIndices, self).fit(hyperparameter_config, pipeline_config, X=np.zeros((Y.shape[0], 1)), Y=Y, train_indices=train_indices)

        print(Y.shape[0], dataset_info.y_shape[0])

        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)
        
        loss_module_name = hyperparameter_config["loss_module"]
        loss_module = self.loss_modules[loss_module_name]
        loss = loss_module.module
        if inspect.isclass(loss):
            loss = loss()
        loss_module.set_loss_function(loss)
        
        return {'loss_function': loss_module}

        
