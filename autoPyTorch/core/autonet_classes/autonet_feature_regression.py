
import numpy as np
import torch
import torch.nn as nn
import copy
from autoPyTorch.core.autonet_classes.autonet_feature_data import AutoNetFeatureData

class AutoNetRegression(AutoNetFeatureData):
    preset_folder_name = "feature_regression"

    # OVERRIDE
    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
        from autoPyTorch.pipeline.nodes.loss_module_selector import LossModuleSelector
        from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
        from autoPyTorch.pipeline.nodes.train_node import TrainNode
        from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation

        import torch.nn as nn
        from autoPyTorch.components.metrics.standard_metrics import mean_distance

        AutoNetFeatureData._apply_default_pipeline_settings(pipeline)

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_final_activation('none', nn.Sequential())

        loss_selector = pipeline[LossModuleSelector.get_name()]
        loss_selector.add_loss_module('l1_loss', nn.L1Loss)

        metric_selector = pipeline[MetricSelector.get_name()]
        metric_selector.add_metric('mean_distance', mean_distance, loss_transform=False, requires_target_class_labels=False)

        train_node = pipeline[TrainNode.get_name()]
        train_node.default_minimize_value = True

        cv = pipeline[CrossValidation.get_name()]
        cv.use_stratified_cv_split_default = False
