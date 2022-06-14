from typing import Any, Dict, Iterable, List, Optional

from ConfigSpace import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.cells import \
    TemporalFusionLayer
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    NetworkStructure
from autoPyTorch.utils.common import (FitRequirement,
                                      HyperparameterSearchSpace,
                                      add_hyperparameter, get_hyperparameter)


class TemporalFusion(autoPyTorchComponent):
    """
    Temporal Fusion layer. For details we refer to
    Lim et al. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    https://arxiv.org/abs/1912.09363
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 random_state: Optional[np.random.RandomState] = None,
                 attention_n_head_log: int = 2,
                 attention_d_model_log: int = 4,
                 use_dropout: bool = False,
                 dropout_rate: Optional[float] = None, ):
        autoPyTorchComponent.__init__(self, random_state=random_state)
        self.add_fit_requirements(
            self._required_fit_requirements
        )
        self.attention_n_head_log = attention_n_head_log
        self.attention_d_model_log = attention_d_model_log
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.temporal_fusion: Optional[torch.nn.Module] = None
        self.n_decoder_output_features = 0

    @property
    def _required_fit_requirements(self) -> List[FitRequirement]:
        return [
            FitRequirement('window_size', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('n_decoder_output_features', (int,), user_defined=False, dataset_property=False),
            FitRequirement('network_encoder', (Dict,), user_defined=False, dataset_property=False),
            FitRequirement('network_structure', (NetworkStructure,), user_defined=False, dataset_property=False),
        ]

    def fit(self, X: Dict[str, Any], y: Any = None) -> autoPyTorchComponent:
        network_structure = X['network_structure']  # type: NetworkStructure

        self.temporal_fusion = TemporalFusionLayer(window_size=X['window_size'],
                                                   network_structure=network_structure,
                                                   network_encoder=X['network_encoder'],
                                                   n_decoder_output_features=X['n_decoder_output_features'],
                                                   d_model=2 ** self.attention_d_model_log,
                                                   n_head=2 ** self.attention_n_head_log,
                                                   dropout=self.dropout_rate
                                                   )
        self.n_decoder_output_features = 2 ** self.attention_d_model_log
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({"n_decoder_output_features": self.n_decoder_output_features,
                  "temporal_fusion": self.temporal_fusion})
        return X

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {
            'shortname': 'TemporalFusion',
            'name': 'TemporalFusion',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            attention_n_head_log: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='attention_n_head_log',
                value_range=(1, 3),
                default_value=2,
            ),
            attention_d_model_log: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='attention_d_model_log',
                value_range=(4, 8),
                default_value=4,
            ),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='use_dropout',
                value_range=(True, False),
                default_value=True,
            ),
            dropout_rate: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='dropout_rate',
                value_range=(0.0, 0.8),
                default_value=0.1,
            )
    ) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Args:
            dataset_properties (Optional[Dict[str, Union[str, int]]):
                Describes the dataset to work on
            attention_n_head_log (HyperparameterSearchSpace):
                log value of number of heads for interpretable
            attention_d_model_log (HyperparameterSearchSpace):
                log value of input of attention model
            use_dropout (HyperparameterSearchSpace):
                if dropout is applied to temporal fusion layer
            dropout_rate (HyperparameterSearchSpace):
                dropout rate of the temporal fusion  layer
        Returns:
            ConfigurationSpace:
                The configuration space of this algorithm.
        """
        cs = ConfigurationSpace()
        add_hyperparameter(cs, attention_n_head_log, UniformIntegerHyperparameter)
        add_hyperparameter(cs, attention_d_model_log, UniformIntegerHyperparameter)
        use_dropout = get_hyperparameter(use_dropout, CategoricalHyperparameter)
        dropout_rate = get_hyperparameter(dropout_rate, UniformFloatHyperparameter)

        cs.add_hyperparameters([use_dropout, dropout_rate])
        cond_dropout = EqualsCondition(dropout_rate, use_dropout, True)
        cs.add_condition(cond_dropout)
        return cs
