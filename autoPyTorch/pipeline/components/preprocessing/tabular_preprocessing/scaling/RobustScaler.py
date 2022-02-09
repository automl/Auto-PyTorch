from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

from sklearn.preprocessing import RobustScaler as SklearnRobustScaler

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace, add_hyperparameter


class RobustScaler(BaseScaler):
    """
    Remove the median and scale features according to the quantile_range to make
    the features robust to outliers.

    For more details of the preprocessor, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    """
    def __init__(
        self,
        q_min: float = 0.25,
        q_max: float = 0.75,
        random_state: Optional[np.random.RandomState] = None
    ):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('issparse', (bool,), user_defined=True, dataset_property=True)])
        self.random_state = random_state
        self.q_min = q_min
        self.q_max = q_max

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)
        with_centering = bool(not X['dataset_properties']['issparse'])

        self.preprocessor['numerical'] = SklearnRobustScaler(quantile_range=(self.q_min, self.q_max),
                                                             with_centering=with_centering,
                                                             copy=False)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        q_min: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="q_min",
                                                                     value_range=(0.001, 0.3),
                                                                     default_value=0.25),
        q_max: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="q_max",
                                                                     value_range=(0.7, 0.999),
                                                                     default_value=0.75)
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, q_min, UniformFloatHyperparameter)
        add_hyperparameter(cs, q_max, UniformFloatHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RobustScaler',
            'name': 'RobustScaler',
            'handles_sparse': True
        }
