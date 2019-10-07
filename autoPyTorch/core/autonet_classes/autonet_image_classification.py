from autoPyTorch.core.autonet_classes.autonet_image_data import AutoNetImageData


class AutoNetImageClassification(AutoNetImageData):
    preset_folder_name = "image_classification"

    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        import torch.nn as nn
        from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
        from autoPyTorch.pipeline.nodes.image.simple_train_node import SimpleTrainNode
        from autoPyTorch.pipeline.nodes.image.cross_validation_indices import CrossValidationIndices
        from autoPyTorch.pipeline.nodes.image.loss_module_selector_indices import LossModuleSelectorIndices
        from autoPyTorch.pipeline.nodes.image.network_selector_datasetinfo import NetworkSelectorDatasetInfo
        from autoPyTorch.components.metrics.standard_metrics import accuracy
        from autoPyTorch.components.preprocessing.loss_weight_strategies import LossWeightStrategyWeighted

        AutoNetImageData._apply_default_pipeline_settings(pipeline)

        net_selector = pipeline[NetworkSelectorDatasetInfo.get_name()]
        net_selector.add_final_activation('softmax', nn.Softmax(1))

        loss_selector = pipeline[LossModuleSelectorIndices.get_name()]
        loss_selector.add_loss_module('cross_entropy', nn.CrossEntropyLoss, None, True)
        loss_selector.add_loss_module('cross_entropy_weighted', nn.CrossEntropyLoss, LossWeightStrategyWeighted(), True)

        metric_selector = pipeline[MetricSelector.get_name()]
        metric_selector.add_metric('accuracy', accuracy)

        train_node = pipeline[SimpleTrainNode.get_name()]
        train_node.default_minimize_value = False
        
        cv = pipeline[CrossValidationIndices.get_name()]
        cv.use_stratified_cv_split_default = True
