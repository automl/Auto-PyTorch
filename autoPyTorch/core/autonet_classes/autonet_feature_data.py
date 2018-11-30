
__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from autoPyTorch.core.api import AutoNet

class AutoNetFeatureData(AutoNet):

    @classmethod
    def get_default_pipeline(cls):
        from autoPyTorch.pipeline.base.pipeline import Pipeline
        from autoPyTorch.pipeline.nodes.autonet_settings import AutoNetSettings
        from autoPyTorch.pipeline.nodes.optimization_algorithm import OptimizationAlgorithm
        from autoPyTorch.pipeline.nodes.cross_validation import CrossValidation
        from autoPyTorch.pipeline.nodes.imputation import Imputation
        from autoPyTorch.pipeline.nodes.normalization_strategy_selector import NormalizationStrategySelector
        from autoPyTorch.pipeline.nodes.one_hot_encoding import OneHotEncoding
        from autoPyTorch.pipeline.nodes.preprocessor_selector import PreprocessorSelector
        from autoPyTorch.pipeline.nodes.resampling_strategy_selector import ResamplingStrategySelector
        from autoPyTorch.pipeline.nodes.embedding_selector import EmbeddingSelector
        from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
        from autoPyTorch.pipeline.nodes.optimizer_selector import OptimizerSelector
        from autoPyTorch.pipeline.nodes.lr_scheduler_selector import LearningrateSchedulerSelector
        from autoPyTorch.pipeline.nodes.log_functions_selector import LogFunctionsSelector
        from autoPyTorch.pipeline.nodes.metric_selector import MetricSelector
        from autoPyTorch.pipeline.nodes.loss_module_selector import LossModuleSelector
        from autoPyTorch.pipeline.nodes.train_node import TrainNode
        
        # build the pipeline
        pipeline = Pipeline([
            AutoNetSettings(),
            OptimizationAlgorithm([
                CrossValidation([
                    Imputation(),
                    NormalizationStrategySelector(),
                    OneHotEncoding(),
                    PreprocessorSelector(),
                    ResamplingStrategySelector(),
                    EmbeddingSelector(),
                    NetworkSelector(),
                    OptimizerSelector(),
                    LearningrateSchedulerSelector(),
                    LogFunctionsSelector(),
                    MetricSelector(),
                    LossModuleSelector(),
                    TrainNode()
                ])
            ])
        ])

        cls._apply_default_pipeline_settings(pipeline)
        return pipeline

    
    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autoPyTorch.pipeline.nodes.normalization_strategy_selector import NormalizationStrategySelector
        from autoPyTorch.pipeline.nodes.preprocessor_selector import PreprocessorSelector
        from autoPyTorch.pipeline.nodes.embedding_selector import EmbeddingSelector
        from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
        from autoPyTorch.pipeline.nodes.optimizer_selector import OptimizerSelector
        from autoPyTorch.pipeline.nodes.lr_scheduler_selector import LearningrateSchedulerSelector
        from autoPyTorch.pipeline.nodes.train_node import TrainNode

        from autoPyTorch.components.networks.feature import MlpNet, ResNet, ShapedMlpNet, ShapedResNet

        from autoPyTorch.components.optimizer.optimizer import AdamOptimizer, SgdOptimizer
        from autoPyTorch.components.lr_scheduler.lr_schedulers import SchedulerCosineAnnealingWithRestartsLR, SchedulerNone, \
            SchedulerCyclicLR, SchedulerExponentialLR, SchedulerReduceLROnPlateau, SchedulerReduceLROnPlateau, SchedulerStepLR
        from autoPyTorch.components.networks.feature import LearnedEntityEmbedding

        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

        from autoPyTorch.components.preprocessing.feature_preprocessing import \
                TruncatedSVD, FastICA, RandomKitchenSinks, KernelPCA, Nystroem

        from autoPyTorch.training.early_stopping import EarlyStopping
        from autoPyTorch.training.mixup import Mixup

        pre_selector = pipeline[PreprocessorSelector.get_name()]
        pre_selector.add_preprocessor('truncated_svd', TruncatedSVD)
        pre_selector.add_preprocessor('fast_ica', FastICA)
        pre_selector.add_preprocessor('kitchen_sinks', RandomKitchenSinks)
        pre_selector.add_preprocessor('kernel_pca', KernelPCA)
        pre_selector.add_preprocessor('nystroem', Nystroem)

        norm_selector = pipeline[NormalizationStrategySelector.get_name()]
        norm_selector.add_normalization_strategy('minmax',   MinMaxScaler)
        norm_selector.add_normalization_strategy('standardize', StandardScaler)
        norm_selector.add_normalization_strategy('maxabs', MaxAbsScaler)

        emb_selector = pipeline[EmbeddingSelector.get_name()]
        emb_selector.add_embedding_module('learned', LearnedEntityEmbedding)

        net_selector = pipeline[NetworkSelector.get_name()]
        net_selector.add_network('mlpnet',       MlpNet)
        net_selector.add_network('shapedmlpnet', ShapedMlpNet)
        net_selector.add_network('resnet',       ResNet)
        net_selector.add_network('shapedresnet', ShapedResNet)

        opt_selector = pipeline[OptimizerSelector.get_name()]
        opt_selector.add_optimizer('adam', AdamOptimizer)
        opt_selector.add_optimizer('sgd',  SgdOptimizer)

        lr_selector = pipeline[LearningrateSchedulerSelector.get_name()]
        lr_selector.add_lr_scheduler('cosine_annealing', SchedulerCosineAnnealingWithRestartsLR)
        lr_selector.add_lr_scheduler('cyclic',           SchedulerCyclicLR)
        lr_selector.add_lr_scheduler('exponential',      SchedulerExponentialLR)
        lr_selector.add_lr_scheduler('step',             SchedulerStepLR)
        lr_selector.add_lr_scheduler('plateau',          SchedulerReduceLROnPlateau)
        lr_selector.add_lr_scheduler('none',             SchedulerNone)

        train_node = pipeline[TrainNode.get_name()]
        train_node.add_training_technique("early_stopping", EarlyStopping)
        train_node.add_batch_loss_computation_technique("mixup", Mixup)
