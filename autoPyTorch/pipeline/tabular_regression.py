import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause

import numpy as np

from sklearn.base import RegressorMixin

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import (
    TabularColumnTransformer
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder_choice import (
    EncoderChoice
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor_choice import FeatureProprocessorChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import EarlyPreprocessing
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import SchedulerChoice
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone_choice import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding_choice import NetworkEmbeddingChoice
from autoPyTorch.pipeline.components.setup.network_head.base_network_head_choice import NetworkHeadChoice
from autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice import (
    NetworkInitializerChoice
)
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice import OptimizerChoice
from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import FeatureDataLoader
from autoPyTorch.pipeline.components.training.trainer.base_trainer_choice import (
    TrainerChoice
)
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TabularRegressionPipeline(RegressorMixin, BasePipeline):
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
                 steps: Optional[List[Tuple[str, autoPyTorchChoice]]] = None,
                 dataset_properties: Optional[Dict[str, Any]] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 random_state: Optional[np.random.RandomState] = None,
                 init_params: Optional[Dict[str, Any]] = None,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                 ):
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params, search_space_updates)

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
        from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics, calculate_score
        metrics = get_metrics(self.dataset_properties, ['r2'])
        y_pred = self.predict(X, batch_size=batch_size)
        r2 = calculate_score(y, y_pred, task_type=STRING_TO_TASK_TYPES[self.dataset_properties['task_type']],
                             metrics=metrics)['r2']
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
            include (Optional[Dict[str, Any]]): what hyper-parameter configurations
                to honor when creating the configuration space
            exclude (Optional[Dict[str, Any]]): what hyper-parameter configurations
                to remove from the configuration space
            dataset_properties (Optional[Dict[str, Union[str, int]]]): Characteristics
                of the dataset to guide the pipeline choices of components

        Returns:
            cs (Configuration): The configuration space describing the TabularRegressionPipeline.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            if not isinstance(dataset_properties, dict):
                warnings.warn('The given dataset_properties argument contains an illegal value.'
                              'Proceeding with the default value')
            dataset_properties = dict()

        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'tabular_regression'
        if dataset_properties['target_type'] != 'tabular_regression':
            warnings.warn('Tabular regression is being used, however the target_type'
                          'is not given as "tabular_regression". Overriding it.')
            dataset_properties['target_type'] = 'tabular_regression'
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

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]]) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps: List[Tuple[str, autoPyTorchChoice]] = []

        default_dataset_properties = {'target_type': 'tabular_regression'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("imputer", SimpleImputer(random_state=self.random_state)),
            ("encoder", EncoderChoice(default_dataset_properties, random_state=self.random_state)),
            ("scaler", ScalerChoice(default_dataset_properties, random_state=self.random_state)),
            ("feature_preprocessor", FeatureProprocessorChoice(default_dataset_properties,
                                                               random_state=self.random_state)),
            ("tabular_transformer", TabularColumnTransformer(random_state=self.random_state)),
            ("preprocessing", EarlyPreprocessing(random_state=self.random_state)),
            ("network_embedding", NetworkEmbeddingChoice(default_dataset_properties,
                                                         random_state=self.random_state)),
            ("network_backbone", NetworkBackboneChoice(default_dataset_properties,
                                                       random_state=self.random_state)),
            ("network_head", NetworkHeadChoice(default_dataset_properties,
                                               random_state=self.random_state)),
            ("network", NetworkComponent(random_state=self.random_state)),
            ("network_init", NetworkInitializerChoice(default_dataset_properties,
                                                      random_state=self.random_state)),
            ("optimizer", OptimizerChoice(default_dataset_properties,
                                          random_state=self.random_state)),
            ("lr_scheduler", SchedulerChoice(default_dataset_properties,
                                             random_state=self.random_state)),
            ("data_loader", FeatureDataLoader(random_state=self.random_state)),
            ("trainer", TrainerChoice(default_dataset_properties, random_state=self.random_state)),
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
                      'preprocessing', 'tabular_transformer']
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
        return "tabular_regressor"
