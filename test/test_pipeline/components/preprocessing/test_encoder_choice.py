import copy
import unittest

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.encoding import (
    EncoderChoice
)


class TestEncoderChoice(unittest.TestCase):
    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        dataset_properties = {'numerical_columns': list(range(4)), 'categorical_columns': [5]}
        encoder_choice = EncoderChoice(dataset_properties)
        cs = encoder_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(encoder_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            encoder_choice.set_hyperparameters(config)

            self.assertEqual(encoder_choice.choice.__class__,
                             encoder_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(encoder_choice.choice))
                self.assertEqual(value, encoder_choice.choice.__dict__[key])

    def test_only_numerical(self):
        dataset_properties = {'numerical_columns': list(range(4)), 'categorical_columns': []}

        chooser = EncoderChoice(dataset_properties)
        configspace = chooser.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        self.assertEqual(configspace['__choice__'], 'NoEncoder')


if __name__ == '__main__':
    unittest.main()
