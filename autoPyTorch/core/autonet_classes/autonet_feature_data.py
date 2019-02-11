
__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from autoPyTorch.core.api import AutoNet

class AutoNetFeatureData(AutoNet):

    @classmethod
    def get_default_ensemble_pipeline(cls):
        from autoPyTorch.pipeline.base.pipeline import Pipeline
        from autoPyTorch.pipeline.nodes import AutoNetSettings, OptimizationAlgorithm, \
            CrossValidation, Imputation, NormalizationStrategySelector, OneHotEncoding, PreprocessorSelector, ResamplingStrategySelector, \
            EmbeddingSelector, NetworkSelector, OptimizerSelector, LearningrateSchedulerSelector, LogFunctionsSelector, MetricSelector, \
            LossModuleSelector, TrainNode, CreateDataLoader, CreateDatasetInfo, EnableComputePredictionsForEnsemble, SavePredictionsForEnsemble, \
            BuildEnsemble, AddEnsembleLogger, InitializationSelector
        
        # build the pipeline
        pipeline = Pipeline([
            AutoNetSettings(),
            CreateDatasetInfo(),
            AddEnsembleLogger(),
            OptimizationAlgorithm([
                CrossValidation([
                    Imputation(),
                    NormalizationStrategySelector(),
                    OneHotEncoding(),
                    PreprocessorSelector(),
                    ResamplingStrategySelector(),
                    EmbeddingSelector(),
                    NetworkSelector(),
                    InitializationSelector(),
                    OptimizerSelector(),
                    LearningrateSchedulerSelector(),
                    LogFunctionsSelector(),
                    MetricSelector(),
                    EnableComputePredictionsForEnsemble(),
                    LossModuleSelector(),
                    CreateDataLoader(),
                    TrainNode(),
                    SavePredictionsForEnsemble()
                ])
            ]),
            BuildEnsemble()
        ])

        cls._apply_default_pipeline_settings(pipeline)
        return pipeline
    
    @classmethod
    def get_default_pipeline(cls):
        from autoPyTorch.pipeline.base.pipeline import Pipeline
        from autoPyTorch.pipeline.nodes import AutoNetSettings, OptimizationAlgorithm, \
            CrossValidation, Imputation, NormalizationStrategySelector, OneHotEncoding, PreprocessorSelector, ResamplingStrategySelector, \
            EmbeddingSelector, NetworkSelector, OptimizerSelector, LearningrateSchedulerSelector, LogFunctionsSelector, MetricSelector, \
            LossModuleSelector, TrainNode, CreateDataLoader, CreateDatasetInfo, InitializationSelector
        
        # build the pipeline
        pipeline = Pipeline([
            AutoNetSettings(),
            CreateDatasetInfo(),
            OptimizationAlgorithm([
                CrossValidation([
                    Imputation(),
                    NormalizationStrategySelector(),
                    OneHotEncoding(),
                    PreprocessorSelector(),
                    ResamplingStrategySelector(),
                    EmbeddingSelector(),
                    NetworkSelector(),
                    InitializationSelector(),
                    OptimizerSelector(),
                    LearningrateSchedulerSelector(),
                    LogFunctionsSelector(),
                    MetricSelector(),
                    LossModuleSelector(),
                    CreateDataLoader(),
                    TrainNode()
                ])
            ])
        ])

        cls._apply_default_pipeline_settings(pipeline)
        return pipeline

    
    @staticmethod
    def _apply_default_pipeline_settings(pipeline):
        from autoPyTorch.pipeline.nodes import NormalizationStrategySelector, PreprocessorSelector, EmbeddingSelector, NetworkSelector, \
            OptimizerSelector, LearningrateSchedulerSelector, TrainNode, CrossValidation, InitializationSelector

        from autoPyTorch.components.networks.feature import MlpNet, ResNet, ShapedMlpNet, ShapedResNet
        from autoPyTorch.components.networks.initialization import SimpleInitializer, SparseInitialization

        from autoPyTorch.components.optimizer.optimizer import AdamOptimizer, SgdOptimizer
        from autoPyTorch.components.lr_scheduler.lr_schedulers import SchedulerCosineAnnealingWithRestartsLR, SchedulerNone, \
            SchedulerCyclicLR, SchedulerExponentialLR, SchedulerReduceLROnPlateau, SchedulerReduceLROnPlateau, SchedulerStepLR
        from autoPyTorch.components.networks.feature import LearnedEntityEmbedding

        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
        from sklearn.model_selection import KFold

        from autoPyTorch.components.preprocessing.feature_preprocessing import \
                TruncatedSVD, FastICA, RandomKitchenSinks, KernelPCA, Nystroem, PowerTransformer

        from autoPyTorch.training.early_stopping import EarlyStopping
        from autoPyTorch.training.mixup import Mixup

        pre_selector = pipeline[PreprocessorSelector.get_name()]
        pre_selector.add_preprocessor('truncated_svd', TruncatedSVD)
        pre_selector.add_preprocessor('power_transformer', PowerTransformer)
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

        init_selector = pipeline[InitializationSelector.get_name()]
        init_selector.add_initialization_method("sparse", SparseInitialization)
        init_selector.add_initializer("simple_initializer", SimpleInitializer)

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

        cv = pipeline[CrossValidation.get_name()]
        cv.add_cross_validator("k_fold", KFold)