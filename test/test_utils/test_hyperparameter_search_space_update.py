from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class DummyComponent(autoPyTorchComponent):
    def __init__(self):
        self._cs_updates = {}

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties=None,
        X=HyperparameterSearchSpace("X",
                                    value_range=[-5, 5],
                                    default_value=0),
        Y=HyperparameterSearchSpace("Y",
                                    value_range=[0, 1],
                                    default_value=0),
        Z=HyperparameterSearchSpace("Z",
                                    value_range=['a', 'b', 1],
                                    default_value='a'),
    ):
        cs = ConfigurationSpace()
        add_hyperparameter(cs, X, UniformIntegerHyperparameter)
        add_hyperparameter(cs, Y, UniformFloatHyperparameter)
        add_hyperparameter(cs, Z, CategoricalHyperparameter)
        return cs


def test_hyperparameter_search_space_update():
    updates = HyperparameterSearchSpaceUpdates()
    updates.append(node_name="dummy_node",
                   hyperparameter="X",
                   value_range=[1, 3],
                   default_value=2)
    updates.append(node_name="dummy_node",
                   hyperparameter="Y",
                   value_range=[0.1, 0.5],
                   default_value=0.1,
                   log=True)
    updates.append(node_name="dummy_node",
                   hyperparameter="Z",
                   value_range=['a', 3],
                   default_value=3)
    dummy_component = DummyComponent()
    updates.apply([("dummy_node", dummy_component)])
    new_updates = dummy_component._get_search_space_updates()
    config_space = dummy_component.get_hyperparameter_search_space(**new_updates)

    for i, (update, hp_type) in enumerate(zip(['X', 'Y', 'Z'],
                                              [UniformIntegerHyperparameter,
                                               UniformFloatHyperparameter,
                                               CategoricalHyperparameter])):

        search_space_update = updates.updates[i]
        search_space = search_space_update.get_search_space()

        assert search_space.hyperparameter == search_space_update.hyperparameter
        assert search_space.value_range == search_space_update.value_range
        assert search_space.default_value == search_space_update.default_value
        assert search_space.log == search_space_update.log

        assert update in dummy_component._cs_updates
        assert update in new_updates
        assert update in config_space

        hp = config_space.get_hyperparameter(update)
        assert isinstance(hp, hp_type)

        if update == 'Z':
            assert all(a == b for a, b in zip(hp.choices, search_space.value_range))
        else:
            assert hp.lower == search_space.value_range[0]
            assert hp.upper == search_space.value_range[1]
            assert hp.log == search_space.log

        assert hp.default_value == search_space.default_value
