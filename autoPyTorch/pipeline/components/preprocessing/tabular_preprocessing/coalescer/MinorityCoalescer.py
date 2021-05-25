from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer.base_coalescer import BaseCoalescer
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter
from autoPyTorch.utils.implementations import MinorityCoalescing


class MinorityCoalescer(BaseCoalescer):
    """
    Groups together classes in a categorical feature if the frequency
    of occurrence is less than minimum_fraction
    """
    def __init__(self, minimum_fraction: float, random_state: Optional[Union[np.random.RandomState, int]] = None):
        super().__init__()
        self.minimum_fraction = minimum_fraction
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseCoalescer:

        self.check_requirements(X, y)

        self.preprocessor['categorical'] = MinorityCoalescing(minimum_fraction=self.minimum_fraction)
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MinorityCoalescer',
            'name': 'Minority Feature-class coalescer',
            'handles_sparse': False
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict] = None,
        minimum_fraction: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="minimum_fraction",
                                                                                value_range=(0.0001, 0.5),
                                                                                default_value=0.01,
                                                                                log=True),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, minimum_fraction, UniformFloatHyperparameter)

        return cs
