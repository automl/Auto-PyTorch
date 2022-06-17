import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.forbidden import (
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause
)

import numpy as np

import pandas as pd

from sklearn.base import RegressorMixin

import torch

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import (
    TimeSeriesFeatureTransformer
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding import TimeSeriesEncoderChoice
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.imputation.TimeSeriesImputer import (
    TimeSeriesFeatureImputer,
    TimeSeriesTargetImputer
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler import (
    BaseScaler
)
from autoPyTorch.pipeline.components.setup.early_preprocessor.TimeSeriesEarlyPreProcessing import (
    TimeSeriesEarlyPreprocessing,
    TimeSeriesTargetEarlyPreprocessing
)
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.forecasting_training_loss import ForecastingLossChoices
from autoPyTorch.pipeline.components.setup.lr_scheduler import SchedulerChoice
from autoPyTorch.pipeline.components.setup.network.forecasting_network import ForecastingNetworkComponent
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone import ForecastingNetworkChoice
from autoPyTorch.pipeline.components.setup.network_embedding import NetworkEmbeddingChoice
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead
from autoPyTorch.pipeline.components.setup.network_initializer import NetworkInitializerChoice
from autoPyTorch.pipeline.components.setup.optimizer import OptimizerChoice
from autoPyTorch.pipeline.components.training.data_loader.time_series_forecasting_data_loader import (
    TimeSeriesForecastingDataLoader
)
from autoPyTorch.pipeline.components.training.trainer.forecasting_trainer import ForecastingTrainerChoice
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TimeSeriesForecastingPipeline(RegressorMixin, BasePipeline):
    """This class is a proof of concept to integrate AutoPyTorch Components

    It implements a pipeline, which includes as steps:

        ->One preprocessing step
        ->One neural network

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available regressors at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    ConfigSpace.configuration_space.Configuration.


    Args:
        config (Configuration):
            The configuration to evaluate.
        random_state (Optional[RandomState):
            random_state is the random number generator

    Attributes:
    """

    def __init__(self,
                 config: Optional[Configuration] = None,
                 steps: Optional[List[Tuple[str, Union[autoPyTorchComponent, autoPyTorchChoice]]]] = None,
                 dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 random_state: Optional[np.random.RandomState] = None,
                 init_params: Optional[Dict[str, Any]] = None,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None,
                 ):
        BasePipeline.__init__(self,
                              config, steps, dataset_properties, include, exclude,
                              random_state, init_params, search_space_updates)

        # Because a pipeline is passed to a worker, we need to honor the random seed
        # in this context. A tabular regression pipeline will implement a torch
        # model, so we comply with https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(self.random_state.get_state()[1][0])

    def score(self, X: List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]],
              y: np.ndarray, batch_size: Optional[int] = None, **score_kwargs: Any) -> float:
        """Scores the fitted estimator on (X, y)

        Args:
            X (List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]]):
                input to the pipeline, from which to guess targets
            batch_size (Optional[int]):
                batch_size controls whether the pipeline will be called on small chunks of the data.
                 Useful when calling the predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray:
                coefficient of determination R^2 of the prediction
        """
        from autoPyTorch.pipeline.components.training.metrics.utils import (
            calculate_score, get_metrics)
        metrics = get_metrics(self.dataset_properties, ['mean_MAPE_forecasting'])
        y_pred = self.predict(X, batch_size=batch_size)  # type: ignore[arg-types]
        r2 = calculate_score(y, y_pred, task_type=STRING_TO_TASK_TYPES[str(self.dataset_properties['task_type'])],
                             metrics=metrics, **score_kwargs)['mean_MAPE_forecasting']
        return r2

    def _get_hyperparameter_search_space(self,
                                         dataset_properties: Dict[str, Any],
                                         include: Optional[Dict[str, Any]] = None,
                                         exclude: Optional[Dict[str, Any]] = None,
                                         ) -> ConfigurationSpace:
        """Create the hyperparameter configuration space.

        For the given steps, and the Choices within that steps,
        this procedure returns a configuration space object to
        explore.

        Args:
            include (Optional[Dict[str, Any]]):
                what hyper-parameter configurations to honor when creating the configuration space
            exclude (Optional[Dict[str, Any]]):
                what hyper-parameter configurations to remove from the configuration space
            dataset_properties (Optional[Dict[str, Union[str, int]]]):
                Characteristics of the dataset to guide the pipeline choices of components

        Returns:
            cs (Configuration):
                The configuration space describing the TimeSeriesRegressionPipeline.
        """
        cs = ConfigurationSpace()

        if not isinstance(dataset_properties, dict):
            warnings.warn('The given dataset_properties argument contains an illegal value.'
                          'Proceeding with the default value')
            dataset_properties = dict()

        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'time_series_forecasting'
        if dataset_properties['target_type'] != 'time_series_forecasting':
            warnings.warn('Time series forecasting is being used, however the target_type'
                          'is not given as "time_series_forecasting". Overriding it.')
            dataset_properties['target_type'] = 'time_series_forecasting'
        # get the base search space given this
        # dataset properties. Then overwrite with custom
        # regression requirements
        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        # Here we add custom code, like this with this
        # is not a valid configuration
        # Learned Entity Embedding is only valid when encoder is one hot encoder
        if 'network_embedding' in self.named_steps.keys():
            embeddings = cs.get_hyperparameter('network_embedding:__choice__').choices
            if 'LearnedEntityEmbedding' in embeddings:
                if 'feature_encoding' in self.named_steps.keys():
                    feature_encodings = cs.get_hyperparameter('feature_encoding:__choice__').choices
                    default = cs.get_hyperparameter('network_embedding:__choice__').default_value
                    possible_default_embeddings = copy.copy(list(embeddings))
                    del possible_default_embeddings[possible_default_embeddings.index(default)]

                    for encoding in feature_encodings:
                        if encoding == 'OneHotEncoder':
                            continue
                        while True:
                            try:
                                cs.add_forbidden_clause(ForbiddenAndConjunction(
                                    ForbiddenEqualsClause(cs.get_hyperparameter(
                                        'network_embedding:__choice__'), 'LearnedEntityEmbedding'),
                                    ForbiddenEqualsClause(
                                        cs.get_hyperparameter('feature_encoding:__choice__'), encoding
                                    )))
                                break
                            except ValueError:
                                # change the default and try again
                                try:
                                    default = possible_default_embeddings.pop()
                                except IndexError:
                                    raise ValueError("Cannot find a legal default configuration")
                                cs.get_hyperparameter('network_embedding:__choice__').default_value = default

                if 'network_backbone:flat_encoder:__choice__' in cs:
                    hp_flat_encoder = cs.get_hyperparameter('network_backbone:flat_encoder:__choice__')
                    if 'NBEATSEncoder' in hp_flat_encoder.choices:
                        cs.add_forbidden_clause(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(hp_flat_encoder, 'NBEATSEncoder'),
                            ForbiddenEqualsClause(cs.get_hyperparameter(
                                'network_embedding:__choice__'), 'LearnedEntityEmbedding'))
                        )

        # dist_cls and auto_regressive are only activate if the network outputs distribution
        if 'loss' in self.named_steps.keys() and 'network_backbone' in self.named_steps.keys():
            hp_loss = cs.get_hyperparameter('loss:__choice__')

            ar_forbidden = True

            hp_deepAR = []
            for hp_name in cs.get_hyperparameter_names():
                if hp_name.startswith('network_backbone:'):
                    if hp_name.endswith(':auto_regressive'):
                        hp_deepAR.append(cs.get_hyperparameter(hp_name))

            # DeepAR
            forbidden_losses_all = []
            losses_non_ar = []
            for loss in hp_loss.choices:
                if loss != "DistributionLoss":
                    losses_non_ar.append(loss)
            if losses_non_ar:
                forbidden_hp_regression_loss = ForbiddenInClause(hp_loss, losses_non_ar)
                for hp_ar in hp_deepAR:
                    if True in hp_ar.choices:
                        forbidden_hp_dist = ForbiddenEqualsClause(hp_ar, ar_forbidden)
                        forbidden_hp_dist = ForbiddenAndConjunction(forbidden_hp_dist, forbidden_hp_regression_loss)
                        forbidden_losses_all.append(forbidden_hp_dist)

            if "network_backbone:seq_encoder:decoder_auto_regressive" in cs:
                decoder_auto_regressive = cs.get_hyperparameter("network_backbone:seq_encoder:decoder_auto_regressive")
                forecast_strategy = cs.get_hyperparameter("loss:DistributionLoss:forecast_strategy")
                use_tf = cs.get_hyperparameter("network_backbone:seq_encoder:use_temporal_fusion")

                if True in decoder_auto_regressive.choices and\
                        'sample' in forecast_strategy.choices and True in use_tf.choices:
                    cs.add_forbidden_clause(
                        ForbiddenAndConjunction(
                            ForbiddenEqualsClause(decoder_auto_regressive, True),
                            ForbiddenEqualsClause(forecast_strategy, 'sample'),
                            ForbiddenEqualsClause(use_tf, True)
                        )
                    )

            if 'network_backbone:flat_encoder:__choice__' in cs:
                network_flat_encoder_hp = cs.get_hyperparameter('network_backbone:flat_encoder:__choice__')

                if 'MLPEncoder' in network_flat_encoder_hp.choices:
                    forbidden = ['MLPEncoder']
                    forbidden_deepAREncoder = [
                        forbid for forbid in forbidden if forbid in network_flat_encoder_hp.choices
                    ]
                    for hp_ar in hp_deepAR:
                        if True in hp_ar.choices:
                            forbidden_hp_ar = ForbiddenEqualsClause(hp_ar, ar_forbidden)
                            forbidden_hp_mlpencoder = ForbiddenInClause(network_flat_encoder_hp,
                                                                        forbidden_deepAREncoder)
                            forbidden_hp_ar_mlp = ForbiddenAndConjunction(forbidden_hp_ar, forbidden_hp_mlpencoder)
                            forbidden_losses_all.append(forbidden_hp_ar_mlp)

            if 'loss:DistributionLoss:forecast_strategy' in cs:
                forecast_strategy = cs.get_hyperparameter('loss:DistributionLoss:forecast_strategy')
                if 'mean' in forecast_strategy.choices:
                    for hp_ar in hp_deepAR:
                        if True in hp_ar.choices:

                            forbidden_hp_ar = ForbiddenEqualsClause(hp_ar, ar_forbidden)
                            forbidden_hp_forecast_strategy = ForbiddenEqualsClause(forecast_strategy, 'mean')
                            forbidden_hp_ar_forecast_strategy = ForbiddenAndConjunction(forbidden_hp_ar,
                                                                                        forbidden_hp_forecast_strategy)
                            forbidden_losses_all.append(forbidden_hp_ar_forecast_strategy)
            if forbidden_losses_all:
                cs.add_forbidden_clauses(forbidden_losses_all)

            # NBEATS
            network_encoder_hp = cs.get_hyperparameter("network_backbone:__choice__")
            forbidden_NBEATS = []
            encoder_non_flat = [choice for choice in network_encoder_hp.choices if choice != 'flat_encoder']
            loss_non_regression = [choice for choice in hp_loss.choices if choice != 'RegressionLoss']
            data_loader_backcast = cs.get_hyperparameter('data_loader:backcast')

            forbidden_encoder_non_flat = ForbiddenInClause(network_encoder_hp, encoder_non_flat)
            forbidden_loss_non_regression = ForbiddenInClause(hp_loss, loss_non_regression)
            forbidden_backcast = ForbiddenEqualsClause(data_loader_backcast, True)

            if 'network_backbone:flat_encoder:__choice__' in cs:
                hp_flat_encoder = cs.get_hyperparameter("network_backbone:flat_encoder:__choice__")

                # Ensure that NBEATS encoder only works with NBEATS decoder
                if 'NBEATSEncoder' in hp_flat_encoder.choices:
                    forbidden_NBEATS.append(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(hp_flat_encoder, 'NBEATSEncoder'),
                        forbidden_loss_non_regression)
                    )
                    transform_time_features = "data_loader:transform_time_features"
                    if transform_time_features in cs:
                        hp_ttf = cs.get_hyperparameter(transform_time_features)
                        forbidden_NBEATS.append(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(hp_flat_encoder, 'NBEATSEncoder'),
                            ForbiddenEqualsClause(hp_ttf, True))
                        )

            forbidden_NBEATS.append(ForbiddenAndConjunction(
                forbidden_backcast,
                forbidden_encoder_non_flat
            ))

            cs.add_forbidden_clauses(forbidden_NBEATS)

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]]) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.
        One key difference between Forecasting pipeline and others is that we put "data_loader"
        before "network_backbone" such that

        Returns:
            List[Tuple[str, autoPyTorchChoice]]:
                list of steps sequentially exercised by the pipeline.
        """
        steps = []  # type: List[Tuple[str, autoPyTorchChoice]]

        default_dataset_properties: Dict[str, BaseDatasetPropertiesType] = {'target_type': 'time_series_prediction'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        if not default_dataset_properties.get("uni_variant", False):
            steps.extend([("impute", TimeSeriesFeatureImputer(random_state=self.random_state)),
                          ("scaler", BaseScaler(random_state=self.random_state)),
                          ('feature_encoding', TimeSeriesEncoderChoice(default_dataset_properties,
                                                                       random_state=self.random_state)),
                          ("time_series_transformer", TimeSeriesFeatureTransformer(random_state=self.random_state)),
                          ("preprocessing", TimeSeriesEarlyPreprocessing(random_state=self.random_state)),
                          ])

        steps.extend([
            ("target_imputer", TimeSeriesTargetImputer(random_state=self.random_state)),
            ("target_preprocessing", TimeSeriesTargetEarlyPreprocessing(random_state=self.random_state)),
            ('loss', ForecastingLossChoices(default_dataset_properties, random_state=self.random_state)),
            ("target_scaler", BaseTargetScaler(random_state=self.random_state)),
            ("data_loader", TimeSeriesForecastingDataLoader(random_state=self.random_state)),
            ("network_embedding", NetworkEmbeddingChoice(default_dataset_properties,
                                                         random_state=self.random_state)),
            ("network_backbone", ForecastingNetworkChoice(dataset_properties=default_dataset_properties,
                                                          random_state=self.random_state)),
            ("network_head", ForecastingHead(random_state=self.random_state)),
            ("network", ForecastingNetworkComponent(random_state=self.random_state)),
            ("network_init", NetworkInitializerChoice(default_dataset_properties,
                                                      random_state=self.random_state)),
            ("optimizer", OptimizerChoice(default_dataset_properties,
                                          random_state=self.random_state)),
            ("lr_scheduler", SchedulerChoice(default_dataset_properties,
                                             random_state=self.random_state)),
            ("trainer", ForecastingTrainerChoice(default_dataset_properties, random_state=self.random_state)),
        ])
        return steps

    def get_pipeline_representation(self) -> Dict[str, str]:
        """
        Returns a representation of the pipeline, so that it can be
        consumed and formatted by the API.

        It should be a representation that follows:
        [{'PreProcessing': <>, 'Estimator': <>}]

        Returns:
            Dict: contains the pipeline representation in a short format
        """
        preprocessing: List[str] = []
        estimator: List[str] = []
        skip_steps = ['data_loader', 'trainer', 'lr_scheduler', 'optimizer', 'network_init',
                      'preprocessing', 'time_series_transformer']
        for step_name, step_component in self.steps:
            if step_name in skip_steps:
                continue
            properties: Dict[str, Any] = {}
            if isinstance(step_component, autoPyTorchChoice) and step_component.choice is not None:
                properties = step_component.choice.get_properties()
            elif isinstance(step_component, autoPyTorchComponent):
                properties = step_component.get_properties()
            if 'shortname' in properties:
                if 'network' in step_name:
                    estimator.append(properties['shortname'])
                else:
                    preprocessing.append(properties['shortname'])
        return {
            'Preprocessing': ','.join(preprocessing),
            'Estimator': ','.join(estimator),
        }

    def _get_estimator_hyperparameter_name(self) -> str:
        """
        Returns the name of the current estimator.

        Returns:
            str: name of the pipeline type
        """
        return "time_series_forecasting"

    def predict(self,
                X: List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]],  # type: ignore[override]
                batch_size: Optional[int] = None) -> np.ndarray:
        """Predict the output using the selected model.

        Args:
            X (List[Union[np.ndarray, pd.DataFrame, TimeSeriesSequence]]):
                input data to predict
            batch_size (Optional[int]):
                batch_size controls whether the pipeline will be called on small chunks of the data.
                Useful when calling the predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray:
                the predicted values given input X
        """

        # Pre-process X
        if batch_size is None:
            warnings.warn("Batch size not provided. "
                          "Will use 1000 instead")
            batch_size = 1000

        loader = self.named_steps['data_loader'].get_loader(X=X, batch_size=batch_size)
        try:
            return self.named_steps['network'].predict(loader).flatten()
        except Exception as e:
            # https://github.com/pytorch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L283
            if 'out of memory' in str(e):
                if batch_size <= 1:
                    raise e
                warnings.warn('| WARNING: ran out of memory, retrying batch')
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                return self.predict(X, batch_size=batch_size // 2).flatten()
            else:
                raise e
