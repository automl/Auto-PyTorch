import numpy as np
import torch

from autoPyTorch.core.api import AutoNet


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
                
                #SingleDataset([

                    #ImageDatasetReader(),
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
                    #])
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
        from autoPyTorch.components.networks.image.darts.model import DARTSImageNet # uses cifar10 base as we train ImageNet mostly with 64x64 images
        from autoPyTorch.components.networks.image.efficientnets import (EfficientNetB0,
                                                                         EfficientNetB1,
                                                                         EfficientNetB2,
                                                                         EfficientNetB3,
                                                                         EfficientNetB4,
                                                                         EfficientNetB5,
                                                                         EfficientNetB6,
                                                                         EfficientNetB7)

        from autoPyTorch.components.optimizer.optimizer import AdamOptimizer, AdamWOptimizer, SgdOptimizer, RMSpropOptimizer
        from autoPyTorch.components.lr_scheduler.lr_schedulers import SchedulerCosineAnnealingWithRestartsLR, SchedulerNone, \
            SchedulerCyclicLR, SchedulerExponentialLR, SchedulerReduceLROnPlateau, SchedulerReduceLROnPlateau, SchedulerStepLR, SchedulerAlternatingCosineLR, \
            SchedulerAdaptiveLR, SchedulerCosineAnnealingLR

        from autoPyTorch.training.early_stopping import EarlyStopping
        from autoPyTorch.training.mixup import Mixup

        net_selector = pipeline[NetworkSelectorDatasetInfo.get_name()]
        net_selector.add_network('densenet', DenseNet)
        net_selector.add_network('densenet_flexible', DenseNetFlexible)
        net_selector.add_network('resnet', ResNet)
        net_selector.add_network('resnet152', ResNet152)
        net_selector.add_network('darts', DARTSImageNet)
        net_selector.add_network('efficientnetb0', EfficientNetB0)
        net_selector.add_network('efficientnetb1', EfficientNetB1)
        net_selector.add_network('efficientnetb2', EfficientNetB2)
        net_selector.add_network('efficientnetb3', EfficientNetB3)
        net_selector.add_network('efficientnetb4', EfficientNetB4)
        net_selector.add_network('efficientnetb5', EfficientNetB5)
        net_selector.add_network('efficientnetb6', EfficientNetB6)
        net_selector.add_network('efficientnetb7', EfficientNetB7)
        net_selector.add_network('mobilenet', MobileNet)
        net_selector._update_hyperparameter_range('resnet:nr_main_blocks', [2, 4], log=False, check_validity=False)
        net_selector._update_hyperparameter_range('resnet:widen_factor_1', [0.5, 8], log=True, check_validity=False)
        net_selector._update_hyperparameter_range('resnet:auxiliary', [False], check_validity=False)
        net_selector._update_hyperparameter_range('mobilenet:nr_main_blocks', [3, 7], log=False, check_validity=False)
        net_selector._update_hyperparameter_range('mobilenet:initial_filters', [8, 32], log=True, check_validity=False)
        net_selector._update_hyperparameter_range('mobilenet:nr_sub_blocks', [1, 4], log=False, check_validity=False)
        net_selector._update_hyperparameter_range('mobilenet:op_types', ["inverted_residual", "dwise_sep_conv"], log=True, check_validity=False)
        net_selector._update_hyperparameter_range('mobilenet:kernel_sizes', [3, 5], log=False, check_validity=False)
        net_selector._update_hyperparameter_range('mobilenet:strides', [1,2], log=True, check_validity=False)

        opt_selector = pipeline[OptimizerSelector.get_name()]
        opt_selector.add_optimizer('adam', AdamOptimizer)
        opt_selector.add_optimizer('adamw', AdamWOptimizer)
        opt_selector.add_optimizer('sgd',  SgdOptimizer)
        opt_selector.add_optimizer('rmsprop',  RMSpropOptimizer)

        # ENGINEERED
        opt_selector._update_hyperparameter_range('adamw:learning_rate', [1e-4, 0.3], log=True, check_validity=False)
        opt_selector._update_hyperparameter_range('adamw:weight_decay', [1e-5, 1e-3], log=True, check_validity=False)
        opt_selector._update_hyperparameter_range('sgd:learning_rate', [1e-4, 0.3], log=True, check_validity=False)
        opt_selector._update_hyperparameter_range('sgd:momentum', [0.1, 0.99], log=False, check_validity=False)
        opt_selector._update_hyperparameter_range('sgd:weight_decay', [1e-5, 1e-3], log=True, check_validity=False)
        opt_selector._update_hyperparameter_range('rmsprop:learning_rate', [1e-4, 0.3], log=True, check_validity=False)
        opt_selector._update_hyperparameter_range('rmsprop:momentum', [0.1, 0.99], log=False, check_validity=False)
        opt_selector._update_hyperparameter_range('rmsprop:weight_decay', [1e-5, 1e-3], log=True, check_validity=False)
        opt_selector._update_hyperparameter_range('rmsprop:alpha', [0.8, 0.99], log=False, check_validity=False)

        lr_selector = pipeline[SimpleLearningrateSchedulerSelector.get_name()]
        lr_selector.add_lr_scheduler('cosine_annealing_with_restarts', SchedulerCosineAnnealingWithRestartsLR)
        lr_selector.add_lr_scheduler('cosine_annealing',           SchedulerCosineAnnealingLR)
        lr_selector.add_lr_scheduler('cyclic',           SchedulerCyclicLR)
        lr_selector.add_lr_scheduler('step',           SchedulerStepLR)
        lr_selector.add_lr_scheduler('adapt',           SchedulerAdaptiveLR)
        lr_selector.add_lr_scheduler('plateau',             SchedulerReduceLROnPlateau)
        lr_selector.add_lr_scheduler('alternating_cosine',SchedulerAlternatingCosineLR)
        lr_selector.add_lr_scheduler('none',             SchedulerNone)
        
        #lr_selector._update_hyperparameter_range('step:step_size', [10, 50], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('cyclic:cycle_length', [10, 50], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('cosine_annealing:T_max', [10, 50], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('alternating_cosine:T_max', [10, 50], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('plateau:patience', [3, 8], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('adapt:T_max', [10, 50], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('adapt:patience', [3, 3], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('adapt:threshold', [0.01, 0.01], log=False, check_validity=False)

        # ENGINEERED
        lr_selector._update_hyperparameter_range('step:step_size', [5, 100], log=False, check_validity=False)
        lr_selector._update_hyperparameter_range('step:gamma', [0.001, 0.99], log=True, check_validity=False)
        #lr_selector._update_hyperparameter_range('cosine_annealing_with_restarts:T_max', [10, 100], log=False, check_validity=False)
        #lr_selector._update_hyperparameter_range('cosine_annealing_with_restarts:T_mult', [1., 2.], log=False, check_validity=False)
        
        train_node = pipeline[SimpleTrainNode.get_name()]
        # train_node.add_training_technique("early_stopping", EarlyStopping)
        train_node.add_batch_loss_computation_technique("mixup", Mixup)

        # ENGINEERED
        train_node._update_hyperparameter_range("mixup:alpha", [0.01,0.7], log=True, check_validity=False)

        data_node = pipeline[CreateImageDataLoader.get_name()]
        #data_node._update_hyperparameter_range('batch_size', [32, 128], log=True, check_validity=False)

        # ENGINEERED
        data_node._update_hyperparameter_range('batch_size', [64, 128], log=True, check_validity=False)

        #augment_node = pipeline[ImageAugmentation.get_name()]
        #augment_node._update_hyperparameter_range('augment', [True], check_validity=False)
        #augment_node._update_hyperparameter_range('autoaugment', [False, True], check_validity=False)
        #augment_node._update_hyperparameter_range('fastautoaugment', [False, True], check_validity=False)
        #augment_node._update_hyperparameter_range('cutout', [False,True], check_validity=False)
        #augment_node._update_hyperparameter_range('cutout_holes', [1, 2], check_validity=False)

        # ENGINEERED
        augment_node = pipeline[ImageAugmentation.get_name()]
        augment_node._update_hyperparameter_range('augment', [True], check_validity=False)
        augment_node._update_hyperparameter_range('autoaugment', [True], check_validity=False)
        augment_node._update_hyperparameter_range('fastautoaugment', [False], check_validity=False)
        augment_node._update_hyperparameter_range('cutout', [True], check_validity=False)
        augment_node._update_hyperparameter_range('cutout_holes', [1, 1], check_validity=False)
        augment_node._update_hyperparameter_range('cutout_length', [1,30],  check_validity=False)
