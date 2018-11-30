from autonet.core.autonet_classes.autonet_feature_data import AutoNetFeatureData

class AutoNetMultilabel(AutoNetFeatureData):

    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autonet.pipeline.nodes.network_selector import NetworkSelector
        from autonet.pipeline.nodes.loss_module_selector import LossModuleSelector
        from autonet.pipeline.nodes.metric_selector import MetricSelector
        from autonet.pipeline.nodes.train_node import TrainNode
        from autonet.pipeline.nodes.cross_validation import CrossValidation

        import torch.nn as nn
        from autonet.components.metrics.standard_metrics import multilabel_accuracy
        from autonet.components.preprocessing.loss_weight_strategies import LossWeightStrategyWeightedBinary

        AutoNetFeatureData._apply_default_pipeline_settings(pipeline)

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_final_activation('sigmoid', nn.Sigmoid())

        loss_selector = pipeline[LossModuleSelector.get_name()]
        loss_selector.add_loss_module('bce_with_logits', nn.BCEWithLogitsLoss, None, False)
        loss_selector.add_loss_module('bce_with_logits_weighted', nn.BCEWithLogitsLoss, LossWeightStrategyWeightedBinary(), False)

        metric_selector = pipeline[MetricSelector.get_name()]
        metric_selector.add_metric('multilabel_accuracy', multilabel_accuracy)

        train_node = pipeline[TrainNode.get_name()]
        train_node.default_minimize_value = False

        cv = pipeline[CrossValidation.get_name()]
        cv.use_stratified_cv_split_default = False