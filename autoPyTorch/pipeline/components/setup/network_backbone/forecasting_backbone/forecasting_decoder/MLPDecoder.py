from typing import Dict, Optional, Tuple, Union, Any

from torch import nn

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import GreaterThanCondition, EqualsCondition

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, get_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.\
    forecasting_backbone.forecasting_decoder.base_forecasting_decoder import BaseForecastingDecoder


class ForecastingMLPDecoder(BaseForecastingDecoder):
    def _build_decoder(self,
                       input_shape: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> Tuple[nn.Module, int]:
        layers = []
        in_features = input_shape[-1]
        num_decoder_output_features = in_features
        if 'num_layers' in self.config and self.config["num_layers"] > 0:
            for i in range(1, self.config["num_layers"]):
                layers.append(nn.Linear(in_features=in_features,
                                        out_features=self.config[f"units_layer_{i}"]))
                layers.append(_activations[self.config["activation"]]())
                in_features = self.config[f"units_layer_{i}"]
                num_decoder_output_features = in_features
        if 'units_local_layer' in self.config:
            layers.append(nn.Linear(in_features=in_features,
                                    out_features=self.config['units_local_layer'] * n_prediction_heads))
            if 'activation' in self.config:
                layers.append(_activations[self.config["activation"]]())
            num_decoder_output_features = self.config['units_local_layer']

        return nn.Sequential(*layers), num_decoder_output_features

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MLPDecoder',
            'name': 'MLPDecoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'mlp_has_local_layer': self.config.get('has_local_layer', True)})
        return super().transform(X)

    @property
    def fitted_encoder(self):
        return ['RNNEncoder', 'TCNEncoder', 'MLEncoder', 'NBEATSEncoder']

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_layers: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_layers",
                                                                              value_range=(0, 3),
                                                                              default_value=1),
            units_layer: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="units_layer",
                                                                               value_range=(16, 512),
                                                                               default_value=64,
                                                                               log=True),
            activation: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="activation",
                                                                              value_range=tuple(_activations.keys()),
                                                                              default_value=list(_activations.keys())[
                                                                                  0]),
            auto_regressive: bool = False,
            has_local_layer: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='has_local_layer',
                                                                                   value_range=(True, False),
                                                                                   default_value=True),
            units_local_layer: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="units_local_layer",
                                                                                     value_range=(16, 128),
                                                                                     default_value=32,
                                                                                     log=True),
    ) -> ConfigurationSpace:
        """
        Builds the mlp head layer. The decoder implementation follows the idea from:

        Wen et al, A Multi-Horizon Quantile Recurrent Forecaster, NeurIPS 2017, Time Series Workshop
        https://arxiv.org/abs/1711.11053

        This model acts as the global MLP, local MLP is implemented under forecasting_head, that maps the output
        features to the final output

        Additionally, this model also determines if DeepAR is applied to do prediction

        Salinas et al. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
        https://arxiv.org/abs/1704.04110

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]): Dataset Properties
            num_layers (HyperparameterSearchSpace): number of decoder layers (the last layer is not included, thus it
            could start from 0)
            units_layer (HyperparameterSearchSpace): number of units of each layer (except for the last layer)
            activation (HyperparameterSearchSpace): activation function
            auto_regressive (bool): if the model acts as a DeepAR model, the corresponding hyperparaemter is
            controlled by seq_encoder
            has_local_layer (HyperparameterSearchSpace): if local MLP layer is applied, if not, the output of the
                network will be directly attached with different heads
            units_local_layer (HyperparameterSearchSpace): number of units of local layer. The size of this layer is
                smaller as it needs to be expanded to adapt to the number of predictions
        Returns:
            cs (ConfigurationSpace): ConfigurationSpace
        """
        cs = ConfigurationSpace()

        min_num_layers: int = num_layers.value_range[0]  # type: ignore
        max_num_layers: int = num_layers.value_range[-1]  # type: ignore
        num_layers_is_constant = (min_num_layers == max_num_layers)

        num_layers_hp = get_hyperparameter(num_layers, UniformIntegerHyperparameter)
        activation_hp = get_hyperparameter(activation, CategoricalHyperparameter)
        cs.add_hyperparameter(num_layers_hp)

        if not num_layers_is_constant:
            cs.add_hyperparameter(activation_hp)
            # HERE WE replace 1 with 0 to be compatible with our modification
            cs.add_condition(GreaterThanCondition(activation_hp, num_layers_hp, 0))
        elif max_num_layers > 1:
            # only add activation if we have more than 1 layer
            cs.add_hyperparameter(activation_hp)

        for i in range(1, max_num_layers + 1):
            num_units_search_space = HyperparameterSearchSpace(
                hyperparameter=f"units_layer_{i}",
                value_range=units_layer.value_range,
                default_value=units_layer.default_value,
                log=units_layer.log,
            )
            num_units_hp = get_hyperparameter(num_units_search_space, UniformIntegerHyperparameter)
            cs.add_hyperparameter(num_units_hp)

            if i >= min_num_layers and not num_layers_is_constant:
                # In the case of a constant, the max and min number of layers are the same.
                # So no condition is needed. If it is not a constant but a hyperparameter,
                # then a condition has to be made so that it accounts for the value of the
                # hyperparameter.
                cs.add_condition(GreaterThanCondition(num_units_hp, num_layers_hp, i - 1))

        # add_hyperparameter(cs, units_final_layer, UniformIntegerHyperparameter)

        if not auto_regressive:
            has_local_layer = get_hyperparameter(has_local_layer, CategoricalHyperparameter)
            units_local_layer = get_hyperparameter(units_local_layer, UniformIntegerHyperparameter)

            cond_units_local_layer = EqualsCondition(units_local_layer, has_local_layer, True)
            cs.add_hyperparameters([has_local_layer, units_local_layer])
            cs.add_conditions([cond_units_local_layer])

        return cs
