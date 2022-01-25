import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause

import numpy as np

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

import torch

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import (
    TimeSeriesTransformer
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import EarlyPreprocessing
from autoPyTorch.pipeline.components.setup.lr_scheduler import SchedulerChoice
from autoPyTorch.pipeline.components.setup.network.forecasting_network import ForecastingNetworkComponent
from autoPyTorch.pipeline.components.setup.network_embedding import NetworkEmbeddingChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone import ForecastingBackboneChoice
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder import \
    ForecastingDecoderChoice
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.forecasting_head import ForecastingHead
from autoPyTorch.pipeline.components.setup.network_initializer import (
    NetworkInitializerChoice
)
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.forecasting_target_scaling import \
    TargetScalerChoice
from autoPyTorch.pipeline.components.setup.optimizer import OptimizerChoice
from autoPyTorch.pipeline.components.setup.forecasting_training_loss import ForecastingLossChoices
from autoPyTorch.pipeline.components.training.data_loader.time_series_forecasting_data_loader import \
    TimeSeriesForecastingDataLoader
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
        config (Configuration)
            The configuration to evaluate.
        random_state (Optional[RandomState): random_state is the random number generator

    Attributes:
    Examples
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
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params, search_space_updates)

        self.target_scaler = None

        # Because a pipeline is passed to a worker, we need to honor the random seed
        # in this context. A tabular regression pipeline will implement a torch
        # model, so we comply with https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(self.random_state.get_state()[1][0])

    def score(self, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Scores the fitted estimator on (X, y)

        Args:
            X (np.ndarray): input to the pipeline, from which to guess targets
            batch_size (Optional[int]): batch_size controls whether the pipeline
                will be called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray: coefficient of determination R^2 of the prediction
        """
        # TODO adjust to sktime's losses
        from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics, calculate_score
        metrics = get_metrics(self.dataset_properties, ['r2'])
        y_pred = self.predict(X, batch_size=batch_size)
        r2 = calculate_score(y, y_pred, task_type=STRING_TO_TASK_TYPES[self.dataset_properties['task_type']],
                             metrics=metrics)['r2']
        return r2

    def fit(self, X: Dict[str, Any], y: Optional[np.ndarray] = None,
            **fit_params: Any) -> Pipeline:
        super().fit(X, y, **fit_params)
        self.target_scaler = X['target_scaler']

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
            include (Optional[Dict[str, Any]]): what hyper-parameter configurations
                to honor when creating the configuration space
            exclude (Optional[Dict[str, Any]]): what hyper-parameter configurations
                to remove from the configuration space
            dataset_properties (Optional[Dict[str, Union[str, int]]]): Characteristics
                of the dataset to guide the pipeline choices of components

        Returns:
            cs (Configuration): The configuration space describing the TimeSeriesRegressionPipeline.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
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
        if 'network_embedding' in self.named_steps.keys() and 'encoder' in self.named_steps.keys():
            embeddings = cs.get_hyperparameter('network_embedding:__choice__').choices
            if 'LearnedEntityEmbedding' in embeddings:
                encoders = cs.get_hyperparameter('encoder:__choice__').choices
                default = cs.get_hyperparameter('network_embedding:__choice__').default_value
                possible_default_embeddings = copy.copy(list(embeddings))
                del possible_default_embeddings[possible_default_embeddings.index(default)]

                for encoder in encoders:
                    if encoder == 'OneHotEncoder':
                        continue
                    while True:
                        try:
                            cs.add_forbidden_clause(ForbiddenAndConjunction(
                                ForbiddenEqualsClause(cs.get_hyperparameter(
                                    'network_embedding:__choice__'), 'LearnedEntityEmbedding'),
                                ForbiddenEqualsClause(cs.get_hyperparameter('encoder:__choice__'), encoder)
                            ))
                            break
                        except ValueError:
                            # change the default and try again
                            try:
                                default = possible_default_embeddings.pop()
                            except IndexError:
                                raise ValueError("Cannot find a legal default configuration")
                            cs.get_hyperparameter('network_embedding:__choice__').default_value = default
                            """
                            # in this case we cannot deactivate the hps, we might need to think about this
                            if 'RegressionLoss' in hp_loss.choices:
                                forbidden_hp_regression_loss = ForbiddenEqualsClause(hp_loss, 'RegressionLoss')
                                for hp_dist in hp_distribution_children:
                                    forbidden_hp_dist = ForbiddenEqualsClause(hp_dist, True)
                                    forbidden_hp_dist = AndConjunction(forbidden_hp_dist, forbidden_hp_regression_loss)
                                    forbidden_regression_losses_all.append(forbidden_hp_dist)
                            else:
                                for hp_dist in hp_distribution_children:
                                    forbidden_hp_dist = ForbiddenEqualsClause(hp_dist, True)
                                    forbidden_regression_losses_all.append(forbidden_hp_dist)
                            """

        # dist_cls and auto_regressive are only activate if the network outputs distribution
        if 'loss' in self.named_steps.keys() and 'network_backbone' in self.named_steps.keys():
            hp_loss = cs.get_hyperparameter('loss:__choice__')

            hp_auto_regressive = []
            for hp_name in cs.get_hyperparameter_names():
                if hp_name.startswith('network_backbone:'):
                    if hp_name.endswith(':auto_regressive'):
                        hp_auto_regressive.append(cs.get_hyperparameter(hp_name))

            # Auto-Regressive is incompatible with regression losses
            forbidden_losses_all = []
            if 'RegressionLoss' in hp_loss.choices:
                forbidden_hp_regression_loss = ForbiddenEqualsClause(hp_loss, 'RegressionLoss')
                for hp_ar in hp_auto_regressive:
                    forbidden_hp_dist = ForbiddenEqualsClause(hp_ar, True)
                    forbidden_hp_dist = ForbiddenAndConjunction(forbidden_hp_dist, forbidden_hp_regression_loss)
                    forbidden_losses_all.append(forbidden_hp_dist)

            hp_net_output_type = []
            if 'network' in self.named_steps.keys():
                hp_net_output_type.append(cs.get_hyperparameter('network:net_out_type'))

            if 'RegressionLoss' in hp_loss.choices:
                # TODO Quantile loses need to be added here
                forbidden_hp_loss = ForbiddenInClause(hp_loss, ['RegressionLoss'])
                # RegressionLos only allow regression hp_net_out
                for hp_net_out in hp_net_output_type:
                    forbidden_hp_dist = ForbiddenInClause(hp_net_out, ['distribution'])
                    forbidden_hp_dist = ForbiddenAndConjunction(forbidden_hp_dist, forbidden_hp_loss)
                    forbidden_losses_all.append(forbidden_hp_dist)

            if 'DistributionLoss' in hp_loss.choices:
                # TODO Quantile loses need to be added here
                forbidden_hp_loss = ForbiddenInClause(hp_loss, ['DistributionLoss'])
                # DistributionLoss only allow distribution hp_net_out
                for hp_net_out in hp_net_output_type:
                    forbidden_hp_dist = ForbiddenInClause(hp_net_out, ['regression'])
                    forbidden_hp_dist = ForbiddenAndConjunction(forbidden_hp_dist, forbidden_hp_loss)
                    forbidden_losses_all.append(forbidden_hp_dist)

            network_encoder_hp = cs.get_hyperparameter('network_backbone:__choice__')

            if 'MLPEncoder' in network_encoder_hp.choices:
                forbidden = ['MLPEncoder']
                forbidden_deepAREncoder = [forbid for forbid in forbidden if forbid in network_encoder_hp.choices]
                for hp_ar in hp_auto_regressive:
                    forbidden_hp_ar = ForbiddenEqualsClause(hp_ar, True)
                    forbidden_hp_mlpencoder = ForbiddenInClause(network_encoder_hp, forbidden_deepAREncoder)
                    forbidden_hp_ar_mlp = ForbiddenAndConjunction(forbidden_hp_ar, forbidden_hp_mlpencoder)
                    forbidden_losses_all.append(forbidden_hp_ar_mlp)

            if 'MLPEncoder' in network_encoder_hp.choices:
                forbidden = ['MLPEncoder']
                forbidden_deepAREncoder = [forbid for forbid in forbidden if forbid in network_encoder_hp.choices]
                for hp_ar in hp_auto_regressive:
                    forbidden_hp_ar = ForbiddenEqualsClause(hp_ar, True)
                    forbidden_hp_mlpencoder = ForbiddenInClause(network_encoder_hp, forbidden_deepAREncoder)
                    forbidden_hp_ar_mlp = ForbiddenAndConjunction(forbidden_hp_ar, forbidden_hp_mlpencoder)
                    forbidden_losses_all.append(forbidden_hp_ar_mlp)

            forecast_strategy = cs.get_hyperparameter('network:forecast_strategy')
            if 'mean' in forecast_strategy.choices:
                for hp_ar in hp_auto_regressive:
                    forbidden_hp_ar = ForbiddenEqualsClause(hp_ar, True)
                    forbidden_hp_forecast_strategy = ForbiddenEqualsClause(forecast_strategy, 'mean')
                    forbidden_hp_ar_forecast_strategy = ForbiddenAndConjunction(forbidden_hp_ar,
                                                                                forbidden_hp_forecast_strategy)
                    forbidden_losses_all.append(forbidden_hp_ar_forecast_strategy)

            cs.add_forbidden_clauses(forbidden_losses_all)

            # NBEATS
            forbidden_NBEATS = []
            encoder_non_BEATS = [choice for choice in network_encoder_hp.choices if choice != 'NBEATSEncoder']
            loss_non_regression = [choice for choice in hp_loss.choices if choice != 'RegressionLoss']
            data_loader_backcast = cs.get_hyperparameter('data_loader:backcast')

            forbidden_encoder_NBEATS = ForbiddenInClause(network_encoder_hp, encoder_non_BEATS)
            forbidden_loss_non_regression = ForbiddenInClause(hp_loss, loss_non_regression)
            forbidden_backcast = ForbiddenEqualsClause(data_loader_backcast, True)
            forbidden_backcast_false = ForbiddenEqualsClause(data_loader_backcast, False)


            # Ensure that NBEATS encoder only works with NBEATS decoder
            if 'NBEATSEncoder' in network_encoder_hp.choices:
                forbidden_NBEATS.append(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(network_encoder_hp, 'NBEATSEncoder'),
                    forbidden_loss_non_regression)
                )
                """
                forbidden_NBEATS.append(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(network_encoder_hp, 'NBEATSEncoder'),
                    forbidden_backcast_false)
                )
                """
            """
            if 'NBEATSDecoder' in network_decoder_hp.choices:
                forbidden_NBEATS.append(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(network_decoder_hp, 'NBEATSDecoder'),
                    forbidden_encoder_NBEATS)
                )
                forbidden_NBEATS.append(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(network_decoder_hp, 'NBEATSDecoder'),
                    forbidden_loss_non_regression)
                )
                forbidden_NBEATS.append(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(network_decoder_hp, 'NBEATSDecoder'),
                    forbidden_backcast_false)
                )
            forbidden_NBEATS.append(ForbiddenAndConjunction(
                forbidden_backcast,
                forbidden_decoder_NBEATS
            ))
            """
            forbidden_NBEATS.append(ForbiddenAndConjunction(
                forbidden_backcast,
                forbidden_encoder_NBEATS
            ))

            cs.add_forbidden_clauses(forbidden_NBEATS)

        """
        # rnn head only allow rnn backbone
        if 'network_encoder' in self.named_steps.keys() and 'network_decoder' in self.named_steps.keys():
            hp_encoder_choice = cs.get_hyperparameter('network_encoder:__choice__')
            hp_decoder_choice = cs.get_hyperparameter('network_decoder:__choice__')

            if 'RNNDecoder' in hp_decoder_choice.choices:
                if len(hp_decoder_choice.choices) == 1 and 'RNNEncoder' not in hp_encoder_choice.choices:
                    raise ValueError("RNN Header is only compatible with RNNBackbone, RNNHead is not allowed to be "
                                     "the only network head choice if the backbone choices do not contain RNN!")
                encoder_choices = [choice for choice in hp_encoder_choice.choices if choice != 'RNNEncoder']
                forbidden_clause_encoder = ForbiddenInClause(hp_encoder_choice, encoder_choices)
                forbidden_clause_decoder = ForbiddenEqualsClause(hp_decoder_choice, 'RNNDecoder')

                cs.add_forbidden_clause(ForbiddenAndConjunction(forbidden_clause_encoder, forbidden_clause_decoder))
            cs.get_hyperparameter_names()
        """


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
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps = []  # type: List[Tuple[str, autoPyTorchChoice]]

        default_dataset_properties = {'target_type': 'time_series_prediction'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)
        # TODO consider the correct way of doing imputer for time series forecasting tasks.
        steps.extend([
            ('loss', ForecastingLossChoices(default_dataset_properties, random_state=self.random_state)),
            ("imputer", SimpleImputer(random_state=self.random_state)),
            # ("scaler", ScalerChoice(default_dataset_properties, random_state=self.random_state)),
            ("time_series_transformer", TimeSeriesTransformer(random_state=self.random_state)),
            ("preprocessing", EarlyPreprocessing(random_state=self.random_state)),
            ("target_scaler", TargetScalerChoice(default_dataset_properties,
                                                 random_state=self.random_state)),
            ("data_loader", TimeSeriesForecastingDataLoader(random_state=self.random_state)),
            ("network_embedding", NetworkEmbeddingChoice(default_dataset_properties,
                                                         random_state=self.random_state)),
            ("network_backbone", ForecastingBackboneChoice(dataset_properties=default_dataset_properties,
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
        preprocessing = []
        estimator = []
        skip_steps = ['data_loader', 'trainer', 'lr_scheduler', 'optimizer', 'network_init',
                      'preprocessing', 'time_series_transformer']
        for step_name, step_component in self.steps:
            if step_name in skip_steps:
                continue
            properties = {}
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


    def predict(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Predict the output using the selected model.

        Args:
            X (np.ndarray): input data to the array
            batch_size (Optional[int]): batch_size controls whether the pipeline will be
                called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.

        Returns:
            np.ndarray: the predicted values given input X
        """

        # Pre-process X
        if batch_size is None:
            warnings.warn("Batch size not provided. "
                          "Will use 1000 instead")
            batch_size = 1000

        loader = self.named_steps['data_loader'].get_loader(X=X, batch_size=batch_size)
        try:
            return self.named_steps['network'].predict(loader, self.target_scaler).flatten()
        except Exception as e:
            # https://github.com/pytorch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L283
            if 'out of memory' in str(e):
                if batch_size == 1:
                    raise e
                warnings.warn('| WARNING: ran out of memory, retrying batch')
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                return self.predict(X, batch_size=batch_size // 2)