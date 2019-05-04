from autoPyTorch.core.autonet_classes.autonet_feature_data import AutoNetFeatureData

class AutoNetMultilabel(AutoNetFeatureData):
    preset_folder_name = "feature_multilabel"

    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
        from autoPyTorch.pipeline.nodes.loss_module_selector import LossModuleSelector
        from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
        from autoPyTorch.pipeline.nodes.train_node import TrainNode
        from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation

        import torch.nn as nn
        from autoPyTorch.components.metrics import multilabel_accuracy, auc_metric, pac_metric
        from autoPyTorch.components.preprocessing.loss_weight_strategies import LossWeightStrategyWeightedBinary

        AutoNetFeatureData._apply_default_pipeline_settings(pipeline)

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_final_activation('sigmoid', nn.Sigmoid())

        loss_selector = pipeline[LossModuleSelector.get_name()]
        loss_selector.add_loss_module('bce_with_logits', nn.BCEWithLogitsLoss, None, False)
        loss_selector.add_loss_module('bce_with_logits_weighted', nn.BCEWithLogitsLoss, LossWeightStrategyWeightedBinary(), False)

        metric_selector = pipeline[MetricSelector.get_name()]
        metric_selector.add_metric('multilabel_accuracy', multilabel_accuracy,
                                   loss_transform=True, requires_target_class_labels=True)
        metric_selector.add_metric('auc_metric', auc_metric, loss_transform=True,
                                   requires_target_class_labels=False)
        metric_selector.add_metric('pac_metric', pac_metric, loss_transform=True,
                                   requires_target_class_labels=False)

        train_node = pipeline[TrainNode.get_name()]
        train_node.default_minimize_value = False

        cv = pipeline[CrossValidation.get_name()]
        cv.use_stratified_cv_split_default = False
