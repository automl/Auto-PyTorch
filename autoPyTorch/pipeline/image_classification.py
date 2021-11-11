import warnings
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.base import ClassifierMixin

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.base_pipeline import BasePipeline, PipelineStepType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise import (
    NormalizerChoice
)
from autoPyTorch.pipeline.components.setup.augmentation.image.ImageAugmenter import ImageAugmenter
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import EarlyPreprocessing
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
# from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import SchedulerChoice
# from autoPyTorch.pipeline.components.setup.network.base_network_choice import NetworkChoice
# from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice import OptimizerChoice
# from autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice import (
#     NetworkInitializerChoice
# )


class ImageClassificationPipeline(ClassifierMixin, BasePipeline):
    """This class is a proof of concept to integrate AutoSklearn Components

    It implements a pipeline, which includes as steps:

        ->One preprocessing step
        ->One neural network

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    ConfigSpace.configuration_space.Configuration.


    Args:
        config (Configuration)
            The configuration to evaluate.
        steps (Optional[List[Tuple[str, autoPyTorchChoice]]]): the list of steps that
            build the pipeline. If provided, they won't be dynamically produced.
        include (Optional[Dict[str, Any]]): Allows the caller to specify which configurations
            to honor during the creation of the configuration space.
        exclude (Optional[Dict[str, Any]]): Allows the caller to specify which configurations
            to avoid during the creation of the configuration space.
        random_state (np.random.RandomState): allows to produce reproducible results by
            setting a seed for randomized settings
        init_params (Optional[Dict[str, Any]])
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline

    Attributes:
    Examples
    """

    def __init__(
        self,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, autoPyTorchChoice]]] = None,
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

    def predict_proba(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """predict_proba.

        Args:
            X (np.ndarray):
                input to the pipeline, from which to guess targets
            batch_size (Optional[int]):
                batch_size controls whether the pipeline
                will be called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray:
                Probabilities of the target being certain class
        """
        if batch_size is None:
            return super().predict_proba(X)

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

    def _get_hyperparameter_search_space(self,
                                         dataset_properties: Dict[str, BaseDatasetPropertiesType],
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
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]):
                Characteristics of the dataset to guide the pipeline choices of components

        Returns:
            cs (Configuration): The configuration space describing
                the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if not isinstance(dataset_properties, dict):
            warnings.warn('Expected dataset_properties to be of type dict, got {}'
                          'Proceeding with the default value'.format(type(dataset_properties)))
            dataset_properties = dict()
        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'image_classification'
        if dataset_properties['target_type'] != 'image_classification':
            dataset_properties['target_type'] = 'image_classification'
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
                list of steps sequentially exercised by the pipeline.
        """
        steps: List[Tuple[str, PipelineStepType]] = []

        default_dataset_properties: Dict[str, BaseDatasetPropertiesType] = {'target_type': 'image_classification'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("normalizer", NormalizerChoice(default_dataset_properties)),
            ("preprocessing", EarlyPreprocessing()),
            ("image_augmenter", ImageAugmenter())
            # ("network", NetworkChoice(default_dataset_properties)),
            # ("network_init", NetworkInitializerChoice(default_dataset_properties)),
            # ("optimizer", OptimizerChoice(default_dataset_properties)),
            # ("lr_scheduler", SchedulerChoice(default_dataset_properties)),
        ])
        return steps

    def _get_estimator_hyperparameter_name(self) -> str:
        """
        Returns the name of the current estimator.

        Returns:
            str: name of the pipeline type
        """
        return "image_classifier"
