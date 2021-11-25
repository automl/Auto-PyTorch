import copy
import unittest

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.scaling import ScalerChoice


class TestRescalerChoice(unittest.TestCase):

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        dataset_properties = {'categorical_columns': list(range(4)),
                              'numerical_columns': [5],
                              'issparse': False}
        rescaler_choice = ScalerChoice(dataset_properties)
        cs = rescaler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(rescaler_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            rescaler_choice.set_hyperparameters(config)

            self.assertEqual(rescaler_choice.choice.__class__,
                             rescaler_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(rescaler_choice.choice))
                self.assertEqual(value, rescaler_choice.choice.__dict__[key])

    def test_only_categorical(self):
        dataset_properties = {'categorical_columns': list(range(4)), 'numerical_columns': []}
        chooser = ScalerChoice(dataset_properties)
        configspace = chooser.get_hyperparameter_search_space(dataset_properties).sample_configuration().\
            get_dictionary()
        self.assertEqual(configspace['__choice__'], 'NoScaler')


if __name__ == '__main__':
    unittest.main()
