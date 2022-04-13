from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling.base_scaler import BaseScaler
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class QuantileTransformer(BaseScaler):
    """
    Transform the features to follow a uniform or a normal distribution
    using quantiles information.

    For more details of each attribute, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    """
    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: str = "normal",  # Literal["normal", "uniform"]
        random_state: Optional[np.random.RandomState] = None
    ):
        super().__init__()
        self.random_state = random_state
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseScaler:

        self.check_requirements(X, y)

        self.preprocessor['numerical'] = SklearnQuantileTransformer(n_quantiles=self.n_quantiles,
                                                                    output_distribution=self.output_distribution,
                                                                    copy=False)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        n_quantiles: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="n_quantiles",
                                                                           value_range=(10, 2000),
                                                                           default_value=1000,
                                                                           ),
        output_distribution: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="output_distribution",
                                                                                   value_range=("uniform", "normal"),
                                                                                   default_value="normal",
                                                                                   )
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # TODO parametrize like the Random Forest as n_quantiles = n_features^param
        add_hyperparameter(cs, n_quantiles, UniformIntegerHyperparameter)
        add_hyperparameter(cs, output_distribution, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'QuantileTransformer',
            'name': 'QuantileTransformer',
            'handles_sparse': False
        }
