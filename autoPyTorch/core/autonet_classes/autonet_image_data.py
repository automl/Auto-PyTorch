import numpy as np
import torch
from autoPyTorch.core.api import AutoNet


__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


class AutoNetImageData(AutoNet):

    @classmethod
    def get_default_pipeline(cls):
        from autoPyTorch.pipeline.base.pipeline import Pipeline
        from autoPyTorch.pipeline.nodes.image.optimization_algorithm_no_timelimit import OptimizationAlgorithmNoTimeLimit
        from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
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
        from autoPyTorch.pipeline.nodes.image.image_dataset_reader import ImageDatasetReader
        from autoPyTorch.pipeline.nodes.image.single_dataset import SingleDataset

        # build the pipeline
        pipeline = Pipeline([
            AutoNetSettingsNoShuffle(),
            OptimizationAlgorithmNoTimeLimit([

                SingleDataset([

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

    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autoPyTorch.pipeline.nodes.optimizer_selector import OptimizerSelector
        from autoPyTorch.pipeline.nodes.image.simple_scheduler_selector import SimpleLearningrateSchedulerSelector

        from autoPyTorch.pipeline.nodes.image.network_selector_datasetinfo import NetworkSelectorDatasetInfo
        from autoPyTorch.pipeline.nodes.image.simple_train_node import SimpleTrainNode
        from autoPyTorch.pipeline.nodes.image.create_image_dataloader import CreateImageDataLoader
        from autoPyTorch.pipeline.nodes.image.image_augmentation import ImageAugmentation

        from autoPyTorch.components.networks.image import DenseNet, ResNet, MobileNet
        from autoPyTorch.components.networks.image.densenet_flexible import DenseNetFlexible
        from autoPyTorch.components.networks.image.resnet152 import ResNet152
        from autoPyTorch.components.networks.image.darts.model import DARTSImageNet

        from autoPyTorch.components.optimizer.optimizer import AdamOptimizer, AdamWOptimizer, SgdOptimizer, RMSpropOptimizer
        from autoPyTorch.components.lr_scheduler.lr_schedulers import SchedulerCosineAnnealingWithRestartsLR, SchedulerNone, \
            SchedulerCyclicLR, SchedulerExponentialLR, SchedulerReduceLROnPlateau, SchedulerReduceLROnPlateau, SchedulerStepLR, \
            SchedulerAlternatingCosineLR, SchedulerAdaptiveLR, SchedulerExponentialLR, SchedulerCosineAnnealingLR

        from autoPyTorch.components.training.image.early_stopping import EarlyStopping
        from autoPyTorch.components.training.image.mixup import Mixup

        net_selector = pipeline[NetworkSelectorDatasetInfo.get_name()]
        net_selector.add_network('densenet', DenseNet)
        net_selector.add_network('densenet_flexible', DenseNetFlexible)
        net_selector.add_network('resnet', ResNet)
        net_selector.add_network('resnet152', ResNet152)
        net_selector.add_network('darts', DARTSImageNet)
        net_selector.add_network('mobilenet', MobileNet)
        net_selector._apply_search_space_update('resnet:nr_main_blocks', [2, 4], log=False)
        net_selector._apply_search_space_update('resnet:widen_factor_1', [0.5, 8], log=True)

        opt_selector = pipeline[OptimizerSelector.get_name()]
        opt_selector.add_optimizer('adam', AdamOptimizer)
        opt_selector.add_optimizer('adamw', AdamWOptimizer)
        opt_selector.add_optimizer('sgd',  SgdOptimizer)
        opt_selector.add_optimizer('rmsprop',  RMSpropOptimizer)

        lr_selector = pipeline[SimpleLearningrateSchedulerSelector.get_name()]
        lr_selector.add_lr_scheduler('cosine_annealing', SchedulerCosineAnnealingLR)
        lr_selector.add_lr_scheduler('cosine_annealing_with_restarts', SchedulerCosineAnnealingWithRestartsLR)
        lr_selector.add_lr_scheduler('cyclic', SchedulerCyclicLR)
        lr_selector.add_lr_scheduler('step', SchedulerStepLR)
        lr_selector.add_lr_scheduler('adapt', SchedulerAdaptiveLR)
        lr_selector.add_lr_scheduler('plateau', SchedulerReduceLROnPlateau)
        lr_selector.add_lr_scheduler('alternating_cosine',SchedulerAlternatingCosineLR)
        lr_selector.add_lr_scheduler('exponential',      SchedulerExponentialLR)
        lr_selector.add_lr_scheduler('none', SchedulerNone)
        
        train_node = pipeline[SimpleTrainNode.get_name()]
        #train_node.add_training_technique("early_stopping", EarlyStopping)
        train_node.add_batch_loss_computation_technique("mixup", Mixup)

        data_node = pipeline[CreateImageDataLoader.get_name()]

        data_node._apply_search_space_update('batch_size', [32, 160], log=True)

        augment_node = pipeline[ImageAugmentation.get_name()]
        augment_node._apply_search_space_update('augment', [False, True])
        augment_node._apply_search_space_update('autoaugment', [False, True])
        augment_node._apply_search_space_update('fastautoaugment', [False, True])
        augment_node._apply_search_space_update('length', [2,6])
        augment_node._apply_search_space_update('cutout', [False, True])
        augment_node._apply_search_space_update('cutout_holes', [1, 50])
