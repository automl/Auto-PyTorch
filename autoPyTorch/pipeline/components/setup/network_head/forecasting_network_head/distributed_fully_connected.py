from typing import Dict, Optional, Tuple, Union, List

from torch import nn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter

from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distribution import ALL_DISTRIBUTIONS
from autoPyTorch.pipeline.components.setup.network_head.forecasting_network_head.distributed_network_head import \
    DistributionNetworkHeadComponents
from autoPyTorch.pipeline.components.setup.network_head.fully_connected import FullyConnectedHead


class DistributionFullyConnectedHead(DistributionNetworkHeadComponents, FullyConnectedHead):
    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'DistributionFullyConnectedHead',
            'name': 'DistributionFullyConnectedHead',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    def _build_head(self, input_shape: Tuple[int, ...]) -> Tuple[List[nn.Module], int]:
        layers = []
        in_features = input_shape[-1]
        for i in range(1, self.config["num_layers"]):
            layers.append(nn.Linear(in_features=in_features,
                                    out_features=self.config[f"units_layer_{i}"]))
            layers.append(_activations[self.config["activation"]]())
            in_features = self.config[f"units_layer_{i}"]
        head_base_output_features = in_features

        return layers, head_base_output_features

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_layers",
                                                                          value_range=(1, 4),
                                                                          default_value=2),
        units_layer: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="units_layer",
                                                                           value_range=(64, 512),
                                                                           default_value=128),
        activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                          value_range=tuple(_activations.keys()),
                                                                          default_value=list(_activations.keys())[0]),
        dist_cls: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="dist_cls",
                                                                        value_range=tuple(ALL_DISTRIBUTIONS.keys()),
                                                                        default_value=list(ALL_DISTRIBUTIONS.keys())[0]),
        #auto_regressive: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="auto_regressive",
        #                                                                       value_range=(True, False),
        #                                                                       default_value=False)
    ) -> ConfigurationSpace:
        cs = FullyConnectedHead.get_hyperparameter_search_space(dataset_properties=dataset_properties,
                                                                num_layers=num_layers,
                                                                units_layer=units_layer,
                                                                activation=activation)

        add_hyperparameter(cs, dist_cls, CategoricalHyperparameter)
        # TODO let dataset_properties decide if autoregressive models is applied
        #add_hyperparameter(cs, auto_regressive, CategoricalHyperparameter)
        return cs
