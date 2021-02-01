import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import pytest

from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


class DummyComponent(autoPyTorchComponent):
    def __init__(self, a=0, b='orange', random_state=None):
        self.a = a
        self.b = b
        self.fitted = False
        self._cs_updates = {}

    def get_hyperparameter_search_space(self, dataset_properties=None):
        cs = CS.ConfigurationSpace()
        a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False)
        b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'])
        cs.add_hyperparameters([a, b])
        return cs

    def fit(self, X, y):
        self.fitted = True
        return self


class DummyChoice(autoPyTorchChoice):
    def get_components(self):
        return {
            'DummyComponent2': DummyComponent,
            'DummyComponent3': DummyComponent,
        }

    def get_hyperparameter_search_space(self, dataset_properties=None, default=None,
                                        include=None, exclude=None):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(
            CSH.CategoricalHyperparameter(
                '__choice__',
                list(self.get_components().keys()),
            )
        )
        return cs


class BasePipelineMock(BasePipeline):
    def __init__(self):
        pass

    def _get_pipeline_steps(self, dataset_properties):
        return [
            ('DummyComponent1', DummyComponent(a=10, b='red')),
            ('DummyChoice', DummyChoice(self.dataset_properties))
        ]


@pytest.fixture
def base_pipeline():
    """Create a pipeline and test the different properties of it"""
    base_pipeline = BasePipelineMock()
    base_pipeline.dataset_properties = {}
    base_pipeline.steps = [
        ('DummyComponent1', DummyComponent(a=10, b='red')),
        ('DummyChoice', DummyChoice(base_pipeline.dataset_properties))
    ]
    base_pipeline.search_space_updates = None
    return base_pipeline


def test_pipeline_base_config_space(base_pipeline):
    """Makes sure that the pipeline can build a proper
    configuration space via its base config methods"""
    cs = base_pipeline._get_base_search_space(
        cs=CS.ConfigurationSpace(),
        include={}, exclude={}, dataset_properties={},
        pipeline=base_pipeline.steps
    )

    # The hyperparameters a and b of the dummy component
    # must be in the hyperparameter search space
    # If parsing the configuration correctly, hyper param a
    # lower bound should be properly defined
    assert 'DummyComponent1:a' in cs
    assert 10 == cs.get_hyperparameter('DummyComponent1:a').lower
    assert 'DummyComponent1:b' in cs

    # For the choice, we make sure the choice
    # is among components 2 and 4
    assert 'DummyChoice:__choice__' in cs
    assert ('DummyComponent2', 'DummyComponent3') == cs.get_hyperparameter(
        'DummyChoice:__choice__').choices


def test_pipeline_set_config(base_pipeline):
    config = base_pipeline._get_base_search_space(
        cs=CS.ConfigurationSpace(),
        include={}, exclude={}, dataset_properties={},
        pipeline=base_pipeline.steps
    ).sample_configuration()

    base_pipeline.set_hyperparameters(config)

    # Check that the proper hyperparameters where set
    config_dict = config.get_dictionary()
    assert config_dict['DummyComponent1:a'] == base_pipeline.named_steps['DummyComponent1'].a
    assert config_dict['DummyComponent1:b'] == base_pipeline.named_steps['DummyComponent1'].b

    # Make sure that the proper component choice was made
    # according to the config
    # The orange check makes sure that the pipeline is setting the
    # hyperparameters individually, as orange should only happen on the
    # choice, as it is not a hyperparameter from the cs
    assert isinstance(base_pipeline.named_steps['DummyChoice'].choice, DummyComponent)
    assert 'orange' == base_pipeline.named_steps['DummyChoice'].choice.b


def test_get_default_options(base_pipeline):
    default_options = base_pipeline.get_default_pipeline_options()
    # test if dict is returned
    assert isinstance(default_options, dict)
    for option, default in default_options.items():
        # check whether any defaults is none
        assert default is not None
