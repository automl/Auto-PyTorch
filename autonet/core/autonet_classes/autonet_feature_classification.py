from autonet.core.autonet_classes.autonet_feature_data import AutoNetFeatureData

class AutoNetClassification(AutoNetFeatureData):

    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autonet.pipeline.nodes.network_selector import NetworkSelector
        from autonet.pipeline.nodes.loss_module_selector import LossModuleSelector
        from autonet.pipeline.nodes.metric_selector import MetricSelector
        from autonet.pipeline.nodes.train_node import TrainNode
        from autonet.pipeline.nodes.resampling_strategy_selector import ResamplingStrategySelector
        from autonet.pipeline.nodes.cross_validation import CrossValidation
        from autonet.pipeline.nodes.one_hot_encoding import OneHotEncoding
        from autonet.pipeline.nodes.resampling_strategy_selector import ResamplingStrategySelector
        from autonet.components.preprocessing.resampling import RandomOverSamplingWithReplacement, RandomUnderSamplingWithReplacement, SMOTE, \
            TargetSizeStrategyAverageSample, TargetSizeStrategyDownsample, TargetSizeStrategyMedianSample, TargetSizeStrategyUpsample

        import torch.nn as nn
        from autonet.components.metrics.standard_metrics import accuracy
        from autonet.components.preprocessing.loss_weight_strategies import LossWeightStrategyWeighted

        AutoNetFeatureData._apply_default_pipeline_settings(pipeline)


        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_final_activation('softmax', nn.Softmax(1))

        loss_selector = pipeline[LossModuleSelector.get_name()]
        loss_selector.add_loss_module('cross_entropy', nn.CrossEntropyLoss, None, True)
        loss_selector.add_loss_module('cross_entropy_weighted', nn.CrossEntropyLoss, LossWeightStrategyWeighted(), True)

        metric_selector = pipeline[MetricSelector.get_name()]
        metric_selector.add_metric('accuracy', accuracy)

        resample_selector = pipeline[ResamplingStrategySelector.get_name()]
        resample_selector.add_over_sampling_method('random', RandomOverSamplingWithReplacement)
        resample_selector.add_over_sampling_method('smote', SMOTE)
        resample_selector.add_under_sampling_method('random', RandomUnderSamplingWithReplacement)
        resample_selector.add_target_size_strategy('upsample', TargetSizeStrategyUpsample)
        resample_selector.add_target_size_strategy('downsample', TargetSizeStrategyDownsample)
        resample_selector.add_target_size_strategy('average', TargetSizeStrategyAverageSample)
        resample_selector.add_target_size_strategy('median', TargetSizeStrategyMedianSample)

        train_node = pipeline[TrainNode.get_name()]
        train_node.default_minimize_value = False
        
        cv = pipeline[CrossValidation.get_name()]
        cv.use_stratified_cv_split_default = True

        one_hot_encoding_node = pipeline[OneHotEncoding.get_name()]
        one_hot_encoding_node.encode_Y = True

        return pipeline