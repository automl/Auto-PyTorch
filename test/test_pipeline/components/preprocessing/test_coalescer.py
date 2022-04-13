import copy
import unittest

import numpy as np

import pytest

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer import (
    CoalescerChoice
)
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer.MinorityCoalescer import (
    MinorityCoalescer
)


def test_transform_before_fit():
    with pytest.raises(RuntimeError):
        mc = MinorityCoalescer(min_frac=None, random_state=np.random.RandomState())
        mc.transform(np.random.random((4, 4)))


class TestCoalescerChoice(unittest.TestCase):
    def test_raise_error_in_check_update_compatiblity(self):
        dataset_properties = {'numerical_columns': [], 'categorical_columns': []}
        cc = CoalescerChoice(dataset_properties)
        choices = ["NoCoescer"]  # component name with typo
        with pytest.raises(ValueError):
            # raise error because no categorical columns, but choices do not have no coalescer
            cc._check_update_compatiblity(choices_in_update=choices, dataset_properties=dataset_properties)

    def test_raise_error_in_get_component_without_updates(self):
        dataset_properties = {'numerical_columns': [], 'categorical_columns': []}
        cc = CoalescerChoice(dataset_properties)
        with pytest.raises(ValueError):
            # raise error because no categorical columns, but choices do not have no coalescer
            cc._get_component_without_updates(
                avail_components={},
                dataset_properties=dataset_properties,
                default="",
                include=[]
            )

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the Coalescer
        choice"""
        dataset_properties = {'numerical_columns': list(range(4)), 'categorical_columns': [5]}
        coalescer_choice = CoalescerChoice(dataset_properties)
        cs = coalescer_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(coalescer_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for _ in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            coalescer_choice.set_hyperparameters(config)

            self.assertEqual(coalescer_choice.choice.__class__,
                             coalescer_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(coalescer_choice.choice))
                self.assertEqual(value, coalescer_choice.choice.__dict__[key])

    def test_only_numerical(self):
        dataset_properties = {'numerical_columns': list(range(4)), 'categorical_columns': []}

        chooser = CoalescerChoice(dataset_properties)
        configspace = chooser.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        self.assertEqual(configspace['__choice__'], 'NoCoalescer')


if __name__ == '__main__':
    unittest.main()
