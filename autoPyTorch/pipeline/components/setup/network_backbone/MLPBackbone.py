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
)


class MLPBackbone(NetworkBackboneComponent):
    """
    This component automatically creates a Multi Layer Perceptron based on a given config.

    This MLP allows for:
        - Different number of layers
        - Specifying the activation. But this activation is shared among layers
        - Using or not dropout
        - Specifying the number of units per layers
    """
    supported_tasks = {"tabular_classification", "tabular_regression"}

    def build_backbone(self, input_shape: Tuple[int, ...]) -> nn.Module:
        layers = list()  # type: List[nn.Module]
        in_features = input_shape[0]

        self._add_layer(layers, in_features, self.config['num_units_1'], 1)

        for i in range(2, self.config['num_groups'] + 1):
            self._add_layer(layers, self.config["num_units_%d" % (i - 1)],
                            self.config["num_units_%d" % i], i)
        backbone = nn.Sequential(*layers)
        self.backbone = backbone
        return backbone

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (self.config["num_units_%d" % self.config["num_groups"]],)

    def _add_layer(self, layers: List[nn.Module], in_features: int, out_features: int,
                   layer_id: int) -> None:
        """
        Dynamically add a layer given the in->out specification

        Args:
            layers (List[nn.Module]): The list where all modules are added
            in_features (int): input dimensionality of the new layer
            out_features (int): output dimensionality of the new layer

        """
        layers.append(nn.Linear(in_features, out_features))
        layers.append(_activations[self.config["activation"]]())
        if self.config['use_dropout']:
            layers.append(nn.Dropout(self.config["dropout_%d" % layer_id]))

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'MLPBackbone',
            'name': 'MLPBackbone',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: Optional[Dict] = None,
                                        num_groups=((1, 15), 5),
                                        dropout: bool = True,
                                        use_dropout=((True, False), False),
                                        num_units=((10, 1024), 200),
                                        dropout_=((0, 0.8), 0.5)
                                        ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        # The number of hidden layers the network will have.
        # Layer blocks are meant to have the same architecture, differing only
        # by the number of units
        min_mlp_layers, max_mlp_layers = num_groups[0]
        num_groups = UniformIntegerHyperparameter(
            "num_groups", min_mlp_layers, max_mlp_layers, default_value=num_groups[1])

        activation = CategoricalHyperparameter(
            "activation", choices=list(_activations.keys())
        )
        cs.add_hyperparameters([num_groups, activation])

        # We can have dropout in the network for
        # better generalization
        if dropout:
            use_dropout = CategoricalHyperparameter(
                "use_dropout", choices=use_dropout[0], default_value=use_dropout[1])
            cs.add_hyperparameters([use_dropout])

        for i in range(1, max_mlp_layers + 1):
            n_units_hp = UniformIntegerHyperparameter("num_units_%d" % i,
                                                      lower=num_units[0][0],
                                                      upper=num_units[0][1],
                                                      default_value=num_units[1])
            cs.add_hyperparameter(n_units_hp)

            if i > min_mlp_layers:
                # The units of layer i should only exist
                # if there are at least i layers
                cs.add_condition(
                    CS.GreaterThanCondition(
                        n_units_hp, num_groups, i - 1
                    )
                )

            if dropout:
                dropout_hp = UniformFloatHyperparameter(
                    "dropout_%d" % i,
                    lower=dropout_[0][0],
                    upper=dropout_[0][1],
                    default_value=dropout_[1]
                )
                cs.add_hyperparameter(dropout_hp)
                dropout_condition_1 = CS.EqualsCondition(dropout_hp, use_dropout, True)

                if i > min_mlp_layers:
                    dropout_condition_2 = CS.GreaterThanCondition(dropout_hp, num_groups, i - 1)
                    cs.add_condition(CS.AndConjunction(dropout_condition_1, dropout_condition_2))
                else:
                    cs.add_condition(dropout_condition_1)

        return cs
