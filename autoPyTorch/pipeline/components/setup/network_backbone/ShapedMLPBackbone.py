from typing import Any, Dict, List, Optional, Tuple, Union

import ConfigSpace as CS
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import (
    NetworkBackboneComponent,
)
from autoPyTorch.pipeline.components.setup.network_backbone.utils import (
    _activations,
    get_shaped_neuron_counts,
)


class ShapedMLPBackbone(NetworkBackboneComponent):
    """
        Implementation of a Shaped MLP -- an MLP with the number of units
        arranged so that a given shape is honored
    """
    supported_tasks = {"tabular_classification", "tabular_regression"}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        layers = list()  # type: List[nn.Module]
        in_features = input_shape[0]
        out_features = self.config["output_dim"]
        neuron_counts = get_shaped_neuron_counts(self.config['mlp_shape'],
                                                 in_features,
                                                 out_features,
                                                 self.config['max_units'],
                                                 self.config['num_groups'])
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            dropout_shape = get_shaped_neuron_counts(
                self.config['mlp_shape'], 0, 0, 1000, self.config['num_groups']
            )

        previous = in_features
        for i in range(self.config['num_groups'] - 1):
            if i >= len(neuron_counts):
                break
            if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
                dropout = dropout_shape[i] / 1000 * self.config["max_dropout"]
            else:
                dropout = 0.0
            self._add_layer(layers, previous, neuron_counts[i], dropout)
            previous = neuron_counts[i]
        layers.append(nn.Linear(previous, out_features))

        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.config["output_dim"],)

    def _add_layer(self, layers: List[nn.Module],
                   in_features: int, out_features: int, dropout: float
                   ) -> None:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(_activations[self.config["activation"]]())
        if self.config["use_dropout"] and self.config["max_dropout"] > 0.05:
            layers.append(nn.Dropout(dropout))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ShapedMLPBackbone',
            'name': 'ShapedMLPBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        min_num_gropus: int = 1,
                                        max_num_groups: int = 15,
                                        min_num_units: int = 10,
                                        max_num_units: int = 1024,
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The number of groups that will compose the resnet. That is,
        # a group can have N Resblock. The M number of this N resblock
        # repetitions is num_groups
        num_groups = UniformIntegerHyperparameter(
            "num_groups", lower=min_num_gropus, upper=max_num_groups, default_value=5)

        mlp_shape = CategoricalHyperparameter('mlp_shape', choices=[
            'funnel', 'long_funnel', 'diamond', 'hexagon', 'brick', 'triangle', 'stairs'
        ])

        activation = CategoricalHyperparameter(
            "activation", choices=list(_activations.keys())
        )

        max_units = UniformIntegerHyperparameter(
            "max_units",
            lower=min_num_units,
            upper=max_num_units,
            default_value=200,
        )

        output_dim = UniformIntegerHyperparameter(
            "output_dim",
            lower=min_num_units,
            upper=max_num_units
        )

        cs.add_hyperparameters([num_groups, activation, mlp_shape, max_units, output_dim])

        # We can have dropout in the network for
        # better generalization
        use_dropout = CategoricalHyperparameter(
            "use_dropout", choices=[True, False])
        max_dropout = UniformFloatHyperparameter("max_dropout", lower=0.0, upper=1.0)
        cs.add_hyperparameters([use_dropout, max_dropout])
        cs.add_condition(CS.EqualsCondition(max_dropout, use_dropout, True))

        return cs
