from autoPyTorch.core.autonet_classes.autonet_image_classification import AutoNetImageClassification


class AutoNetImageClassificationMultipleDatasets(AutoNetImageClassification):
    @classmethod
    def get_default_pipeline(cls):
        from autoPyTorch.pipeline.base.pipeline import Pipeline
        from autoPyTorch.pipeline.nodes.image.optimization_algorithm_no_timelimit import OptimizationAlgorithmNoTimeLimit
        from autoPyTorch.pipeline.nodes.optimizer_selector import OptimizerSelector
        from autoPyTorch.pipeline.nodes.log_functions_selector import LogFunctionsSelector
        from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector

        from autoPyTorch.pipeline.nodes.image.simple_scheduler_selector import SimpleLearningrateSchedulerSelector
        from autoPyTorch.pipeline.nodes.image.cross_validation_indices import CrossValidationIndices
        from autoPyTorch.pipeline.nodes.image.autonet_settings_no_shuffle import AutoNetSettingsNoShuffle
        from autoPyTorch.pipeline.nodes.image.network_selector_datasetinfo import NetworkSelectorDatasetInfo
        from autoPyTorch.pipeline.nodes.image.loss_module_selector_indices import LossModuleSelectorIndices
        from autoPyTorch.pipeline.nodes.image.image_augmentation import ImageAugmentation
        from autoPyTorch.pipeline.nodes.image.create_image_dataloader import CreateImageDataLoader
        from autoPyTorch.pipeline.nodes.image.create_dataset_info import CreateDatasetInfo
        from autoPyTorch.pipeline.nodes.image.simple_train_node import SimpleTrainNode

        from autoPyTorch.pipeline.nodes.image.multiple_datasets import MultipleDatasets
        from autoPyTorch.pipeline.nodes.image.image_dataset_reader import ImageDatasetReader
        
        
        # build the pipeline
        pipeline = Pipeline([
            AutoNetSettingsNoShuffle(),
            OptimizationAlgorithmNoTimeLimit([
                
                MultipleDatasets([

                    ImageDatasetReader(),
                    CreateDatasetInfo(),
                    CrossValidationIndices([
                        
                        NetworkSelectorDatasetInfo(),
                        OptimizerSelector(),
                        SimpleLearningrateSchedulerSelector(),
                        
                        LogFunctionsSelector(),
                        MetricSelector(),
                        
                        LossModuleSelectorIndices(),

                        ImageAugmentation(),
                        CreateImageDataLoader(),
                        SimpleTrainNode()
                    ])
                ])
            ])
        ])

        cls._apply_default_pipeline_settings(pipeline)
        return pipeline