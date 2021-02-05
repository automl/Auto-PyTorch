import warnings
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

import sklearn.preprocessing
from sklearn.base import ClassifierMixin

from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
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


class TabularClassificationPipeline(ClassifierMixin, BasePipeline):
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
        random_state (Optional[RandomState): random_state is the random number generator

    Attributes:
    Examples
    """

    def __init__(
            self,
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

    def fit_transformer(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Fits the pipeline given a training (X,y) pair

        Args:
            X (np.ndarray): features from which to guess targets
            y (np.ndarray): classification targets for this task
            fit_params (Optional[Dict[str, Any]]]): handy communication dictionary,
                so that inter-stages of the pipeline can share information

        Returns:
            np.ndarray: the transformed features
            Optional[Dict[str, Any]]]: A dictionary to share fit informations
                within the pipeline stages
        """

        if fit_params is None:
            fit_params = {}

        X, fit_params = super().fit_transformer(
            X, y, fit_params=fit_params)

        return X, fit_params

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Pre-process X
        loader = self.named_steps['data_loader'].get_loader(X=X)
        pred = self.named_steps['network'].predict(loader)
        if self.dataset_properties['output_shape'] == 1:
            proba = pred[:, :self.dataset_properties['num_classes']]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.dataset_properties['output_shape']):
                proba_k = pred[:, k, :self.dataset_properties['num_classes'][k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def predict_proba(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """predict_proba.

        Args:
            X (np.ndarray): input to the pipeline, from which to guess targets
            batch_size (Optional[int]): batch_size controls whether the pipeline
                will be called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray: Probabilities of the target being certain class
        """
        if batch_size is None:
            y = self._predict_proba(X)

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

        # Neural networks might not be fit to produce a [0-1] output
        # For instance, after small number of epochs.
        y = np.clip(y, 0, 1)
        y = sklearn.preprocessing.normalize(y, axis=1, norm='l1')

        return y

    def _get_hyperparameter_search_space(
            self,
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
            cs (Configuration): The configuration space describing
                the SimpleRegressionClassifier.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
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

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]],
                            ) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps = []  # type: List[Tuple[str, autoPyTorchChoice]]

        default_dataset_properties = {'target_type': 'tabular_classification'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("imputer", SimpleImputer()),
            ("encoder", EncoderChoice(default_dataset_properties)),
            ("scaler", ScalerChoice(default_dataset_properties)),
            ("feature_preprocessor", FeatureProprocessorChoice(default_dataset_properties)),
            ("tabular_transformer", TabularColumnTransformer()),
            ("preprocessing", EarlyPreprocessing()),
            ("network_backbone", NetworkBackboneChoice(default_dataset_properties)),
            ("network_head", NetworkHeadChoice(default_dataset_properties)),
            ("network", NetworkComponent(default_dataset_properties)),
            ("network_init", NetworkInitializerChoice(default_dataset_properties)),
            ("optimizer", OptimizerChoice(default_dataset_properties)),
            ("lr_scheduler", SchedulerChoice(default_dataset_properties)),
            ("data_loader", FeatureDataLoader()),
            ("trainer", TrainerChoice(default_dataset_properties)),
        ])
        return steps

    def _get_estimator_hyperparameter_name(self) -> str:
        """
        Returns the name of the current estimator.

        Returns:
            str: name of the pipeline type
        """
        return "tabular_classifier"
