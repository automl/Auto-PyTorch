import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.base import ClassifierMixin

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.base_pipeline import BasePipeline, PipelineStepType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.traditional_ml import ModelChoice
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TraditionalTabularClassificationPipeline(ClassifierMixin, BasePipeline):
    """
    A pipeline to fit traditional ML methods for tabular classification.

    Args:
        config (Configuration)
            The configuration to evaluate.
        steps (Optional[List[Tuple[str, Union[autoPyTorchComponent, autoPyTorchChoice]]]]):
            the list of `autoPyTorchComponent` or `autoPyTorchChoice`
            that build the pipeline. If provided, they won't be
            dynamically produced.
        include (Optional[Dict[str, Any]]):
            Allows the caller to specify which configurations
            to honor during the creation of the configuration space.
        exclude (Optional[Dict[str, Any]]):
            Allows the caller to specify which configurations
            to avoid during the creation of the configuration space.
        random_state (np.random.RandomState):
            Allows to produce reproducible results by
            setting a seed for randomized settings
        init_params (Optional[Dict[str, Any]]):
            Optional initial settings for the config
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            Search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline

    Attributes:
        steps (List[Tuple[str, PipelineStepType]]):
            The steps of the current pipeline. Each step in an AutoPyTorch
            pipeline is either a autoPyTorchChoice or autoPyTorchComponent.
            Both of these are child classes of sklearn 'BaseEstimator' and
            they perform operations on and transform the fit dictionary.
            For more info, check documentation of 'autoPyTorchChoice' or
            'autoPyTorchComponent'.
        config (Configuration):
            A configuration to delimit the current component choice
        random_state (Optional[np.random.RandomState]):
            Allows to produce reproducible results by setting a
            seed for randomized settings
    """

    def __init__(
        self,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, Union[autoPyTorchComponent, autoPyTorchChoice]]]] = None,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        include: Optional[Dict[str, Any]] = None,
        exclude: Optional[Dict[str, Any]] = None,
        random_state: Optional[np.random.RandomState] = None,
        init_params: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ):
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params, search_space_updates)

    def predict(self, X: np.ndarray, batch_size: Optional[int] = None
                ) -> np.ndarray:
        """Predict the output using the selected model.

        Args:
            X (np.ndarray):
                Input data to the array
            batch_size (Optional[int]):
                Controls whether the pipeline will be
                called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.

        Returns:
            np.ndarray: the predicted values given input X
        """

        if batch_size is None:
            return self.named_steps['model_trainer'].predict(X)

        else:
            if not isinstance(batch_size, int):
                raise ValueError("Argument 'batch_size' must be of type int, "
                                 "but is '%s'" % type(batch_size))
            if batch_size <= 0:
                raise ValueError("Argument 'batch_size' must be positive, "
                                 "but is %d" % batch_size)

            else:
                # Probe for the target array dimensions
                target = self.predict(X[0:2].copy())
                if (target.shape) == 1:
                    target = target.reshape((-1, 1))
                y = np.zeros((X.shape[0], target.shape[1]),
                             dtype=np.float32)

                for k in range(max(1, int(np.ceil(float(X.shape[0]) / batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    pred_prob = self.predict(X[batch_from:batch_to], batch_size=None)
                    y[batch_from:batch_to] = pred_prob.astype(np.float32)

                return y

    def predict_proba(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """predict_proba.

        Args:
            X (np.ndarray):
                Input to the pipeline, from which to guess targets
            batch_size (Optional[int]):
                Controls whether the pipeline will be called on
                small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray:
                Probabilities of the target being certain class
        """
        if batch_size is None:
            return self.named_steps['model_trainer'].predict_proba(X)

        else:
            if not isinstance(batch_size, int):
                raise ValueError("Argument 'batch_size' must be of type int, "
                                 "but is '%s'" % type(batch_size))
            if batch_size <= 0:
                raise ValueError("Argument 'batch_size' must be positive, "
                                 "but is %d" % batch_size)

            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0:2].copy())

                y = np.zeros((X.shape[0], target.shape[1]),
                             dtype=np.float32)

                for k in range(max(1, int(np.ceil(float(X.shape[0]) / batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    pred_prob = self.predict_proba(X[batch_from:batch_to], batch_size=None)
                    y[batch_from:batch_to] = pred_prob.astype(np.float32)

                return y

    def _get_hyperparameter_search_space(
            self,
            dataset_properties: Dict[str, BaseDatasetPropertiesType],
            include: Optional[Dict[str, Any]] = None,
            exclude: Optional[Dict[str, Any]] = None,
    ) -> ConfigurationSpace:
        """Create the hyperparameter configuration space.

        For the given steps, and the Choices within that steps,
        this procedure returns a configuration space object to
        explore.

        Args:
            include (Optional[Dict[str, Any]]):
                What hyper-parameter configurations
                to honor when creating the configuration space
            exclude (Optional[Dict[str, Any]]):
                What hyper-parameter configurations
                to remove from the configuration space
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]):
                Characteristics of the dataset to guide the pipeline choices
                of components

        Returns:
            cs (Configuration):
                The configuration space describing
                the TraditionalTabularClassificationPipeline.
        """
        cs = ConfigurationSpace()

        if not isinstance(dataset_properties, dict):
            warnings.warn('The given dataset_properties argument contains an illegal value.'
                          'Proceeding with the default value')
            dataset_properties = dict()

        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'tabular_classification'
        if dataset_properties['target_type'] != 'tabular_classification':
            warnings.warn('Tabular classification is being used, however the target_type'
                          'is not given as "tabular_classification". Overriding it.')
            dataset_properties['target_type'] = 'tabular_classification'
        # get the base search space given this
        # dataset properties. Then overwrite with custom
        # classification requirements
        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        # Here we add custom code, like this with this
        # is not a valid configuration

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _get_pipeline_steps(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]],
    ) -> List[Tuple[str, PipelineStepType]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, PipelineStepType]]:
                List of steps sequentially exercised
                by the pipeline.
        """
        steps: List[Tuple[str, PipelineStepType]] = []

        default_dataset_properties: Dict[str, BaseDatasetPropertiesType] = {'target_type': 'tabular_classification'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("model_trainer", ModelChoice(default_dataset_properties,
                                          random_state=self.random_state)),
        ])
        return steps

    def _get_estimator_hyperparameter_name(self) -> str:
        """
        Returns the name of the current estimator.

        Returns:
            str:
                Name of the pipeline type
        """
        return "traditional_tabular_learner"

    def get_pipeline_representation(self) -> Dict[str, str]:
        """
        Returns a representation of the pipeline, so that it can be
        consumed and formatted by the API.

        It should be a representation that follows:
        [{'PreProcessing': <>, 'Estimator': <>}]

        Returns:
            Dict:
                Contains the pipeline representation in a short format
        """
        estimator_name = 'TraditionalTabularClassification'
        if self.steps[0][1].choice is not None:
            if self.steps[0][1].choice.model is None:
                estimator_name = self.steps[0][1].choice.__class__.__name__
            else:
                estimator_name = cast(
                    str,
                    self.steps[0][1].choice.model.get_properties()['shortname']
                )
        return {
            'Preprocessing': 'None',
            'Estimator': estimator_name,
        }
