from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter
)

import numpy as np

from sklearn.preprocessing import Normalizer as SklearnNormalizer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class Normalizer(BaseScaler):
    """
    Normalises samples individually according to norm {mean_abs, mean_squared, max}
    """

    def __init__(self, random_state: Optional[Union[np.random.RandomState, int]] = None, norm: str = 'mean_squared'):
        """
        Args:
            random_state (Optional[Union[np.random.RandomState, int]]): Determines random number generation for
            subsampling and smoothing noise.
            norm (str): {mean_abs, mean_squared, max} default: mean_squared
        """
        super().__init__()
        self.random_state = random_state
        self.norm = norm

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:
        self.check_requirements(X, y)

        map_norm = dict({"mean_abs": "l1", "mean_squared": "l2", "max": "max"})
        self.preprocessor['numerical'] = SklearnNormalizer(norm=map_norm[self.norm], copy=False)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        norm: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="norm",
                                                                    value_range=("mean_abs", "mean_squared", "max"),
                                                                    default_value="mean_squared",
                                                                    )
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        add_hyperparameter(cs, norm, CategoricalHyperparameter)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'Normalizer',
            'name': 'Normalizer',
            'handles_sparse': True
        }
