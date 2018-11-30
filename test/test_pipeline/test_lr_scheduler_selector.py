__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

import unittest
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from autonet.pipeline.base.pipeline import Pipeline
from autonet.pipeline.nodes.network_selector import NetworkSelector
from autonet.pipeline.nodes.optimizer_selector import OptimizerSelector
from autonet.pipeline.nodes.lr_scheduler_selector import LearningrateSchedulerSelector

from autonet.components.networks.feature.mlpnet import MlpNet
from autonet.components.networks.feature.shapedmlpnet import ShapedMlpNet
from autonet.components.optimizer.optimizer import AdamOptimizer, SgdOptimizer
from autonet.components.lr_scheduler.lr_schedulers import SchedulerStepLR, SchedulerExponentialLR

class TestLearningrateSchedulerSelectorMethods(unittest.TestCase):

    def test_selector(self):
        pipeline = Pipeline([
            NetworkSelector(),
            OptimizerSelector(),
            LearningrateSchedulerSelector(),
        ])

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_network("mlpnet", MlpNet)
        net_selector.add_network("shapedmlpnet", ShapedMlpNet)
        net_selector.add_final_activation('none', nn.Sequential())

        opt_selector = pipeline[OptimizerSelector.get_name()]
        opt_selector.add_optimizer("adam", AdamOptimizer)
        opt_selector.add_optimizer("sgd", SgdOptimizer)

        lr_scheduler_selector = pipeline[LearningrateSchedulerSelector.get_name()]
        lr_scheduler_selector.add_lr_scheduler("step", SchedulerStepLR)
        lr_scheduler_selector.add_lr_scheduler("exp", SchedulerExponentialLR)


        pipeline_config = pipeline.get_pipeline_config()
        hyper_config = pipeline.get_hyperparameter_search_space().sample_configuration()

        pipeline.fit_pipeline(hyperparameter_config=hyper_config, pipeline_config=pipeline_config,
                                X_train=torch.rand(3,3), Y_train=torch.rand(3, 2), embedding=nn.Sequential(), training_techniques=[])

        sampled_lr_scheduler = pipeline[lr_scheduler_selector.get_name()].fit_output['training_techniques'][0].training_components['lr_scheduler']

        self.assertIn(type(sampled_lr_scheduler), [lr_scheduler.ExponentialLR, lr_scheduler.StepLR])



