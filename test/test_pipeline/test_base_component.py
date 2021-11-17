from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

import pytest

from autoPyTorch.pipeline.components.base_component import ThirdPartyComponents, autoPyTorchComponent


class DummyComponentRequiredFailuire(autoPyTorchComponent):
    _required_properties = {'required'}

    def __init__(self, random_state=None):
        self.fitted = False
        self._cs_updates = {}

    def fit(self, X, y):
        self.fitted = True
        return self

    def get_properties(dataset_properties=None):
        return {"name": 'DummyComponentRequiredFailuire',
                "shortname": "Dummy"}


class DummyComponentExtraPropFailuire(autoPyTorchComponent):

    def __init__(self, random_state=None):
        self.fitted = False
        self._cs_updates = {}

    def fit(self, X, y):
        self.fitted = True
        return self

    def get_properties(dataset_properties=None):
        return {"name": 'DummyComponentExtraPropFailuire',
                "shortname": 'Dummy',
                "must_not_be_there": True}


class DummyComponent(autoPyTorchComponent):
    def __init__(self, a=0, b='orange', random_state=None):
        self.a = a
        self.b = b
        self.fitted = False
        self.random_state = random_state
        self._cs_updates = {}

    def get_hyperparameter_search_space(self, dataset_properties=None):
        cs = ConfigurationSpace()
        a = UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
        b = CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])
        cs.add_hyperparameters([a, b])
        return cs

    def fit(self, X, y):
        self.fitted = True
        return self

    def get_properties(dataset_properties=None):
        return {"name": 'DummyComponent',
                "shortname": 'Dummy'}


def test_third_party_component_failure():
    _addons = ThirdPartyComponents(autoPyTorchComponent)

    with pytest.raises(ValueError, match=r"Property required not specified for .*"):
        _addons.add_component(DummyComponentRequiredFailuire)

    with pytest.raises(ValueError, match=r"Property must_not_be_there must not be specified for algorithm .*"):
        _addons.add_component(DummyComponentExtraPropFailuire)

    with pytest.raises(TypeError, match=r"add_component works only with a subclass of .*"):
        _addons.add_component(1)


def test_set_hyperparameters_not_found_failure():
    dummy_component = DummyComponent()
    dummy_config_space = dummy_component.get_hyperparameter_search_space()
    success_configuration = dummy_config_space.sample_configuration()
    dummy_config_space.add_hyperparameter(CategoricalHyperparameter('c', choices=[1, 2]))
    failure_configuration = dummy_config_space.sample_configuration()
    with pytest.raises(ValueError, match=r"Cannot set hyperparameter c for autoPyTorch.pipeline "
                                         r"DummyComponent because the hyperparameter does not exist."):
        dummy_component.set_hyperparameters(failure_configuration)
    with pytest.raises(ValueError, match=r"Cannot set init param r for autoPyTorch.pipeline "
                                         r"DummyComponent because the init param does not exist."):
        dummy_component.set_hyperparameters(success_configuration, init_params={'r': 1})
