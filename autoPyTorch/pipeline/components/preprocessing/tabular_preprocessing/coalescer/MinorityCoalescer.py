from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import numpy as np

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer.base_coalescer import BaseCoalescer
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter
from autoPyTorch.utils.implementations import MinorityCoalesceTransformer


class MinorityCoalescer(BaseCoalescer):
    """Group together categories whose occurence is less than a specified min_frac """
    def __init__(self, min_frac: float, random_state: np.random.RandomState):
        super().__init__()
        self.min_frac = min_frac
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseCoalescer:
        self.check_requirements(X, y)
        self.preprocessor['categorical'] = MinorityCoalesceTransformer(min_frac=self.min_frac)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, Any]] = None,
        min_frac: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='min_frac',
                                                                        value_range=(1e-4, 0.5),
                                                                        default_value=1e-2,
                                                                        ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, min_frac, UniformFloatHyperparameter)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MinorityCoalescer',
            'name': 'MinorityCoalescer',
            'handles_sparse': False
        }
