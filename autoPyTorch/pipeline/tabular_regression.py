import warnings
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.base import RegressorMixin

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.TabularColumnTransformer import (
    TabularColumnTransformer
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding.base_encoder_choice import (
    EncoderChoice
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.imputation.SimpleImputer import SimpleImputer
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler_choice import ScalerChoice
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import EarlyPreprocessing
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import SchedulerChoice
from autoPyTorch.pipeline.components.setup.network.base_network_choice import NetworkChoice
from autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice import (
    NetworkInitializerChoice
)
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice import OptimizerChoice
from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import FeatureDataLoader
from autoPyTorch.pipeline.components.training.trainer.base_trainer_choice import (
    TrainerChoice
)


class TabularRegressionPipeline(RegressorMixin, BasePipeline):
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
        init_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params)

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

    def score(self, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """score.

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
            dataset_properties['target_type'] = 'tabular_regression'
        if dataset_properties['target_type'] != 'tabular_regression':
            warnings.warn('Tabular classification is being used, however the target_type'
                          'is not given as "tabular_regression". Overriding it.')
            dataset_properties['target_type'] = 'tabular_regression'
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

        default_dataset_properties = {'target_type': 'tabular_regression'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("imputer", SimpleImputer()),
            ("encoder", EncoderChoice(default_dataset_properties)),
            ("scaler", ScalerChoice(default_dataset_properties)),
            ("tabular_transformer", TabularColumnTransformer()),
            ("preprocessing", EarlyPreprocessing()),
            ("network", NetworkChoice(default_dataset_properties)),
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
        return "tabular_regresser"
