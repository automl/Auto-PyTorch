import copy
import unittest

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.coalescer import (
    CoalescerChoice
)


class TestCoalescerChoice(unittest.TestCase):
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
        for i in range(5):
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
