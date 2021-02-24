import copy
import unittest

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler_choice import \
    ScalerChoice


class TestTimeSeriesScalerChoice(unittest.TestCase):

    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice for the time series scaler"""
        dataset_properties = {'categorical_features': [],
                              'numerical_features': list(range(4)),
                              'issparse': False}
        scaler_choice = ScalerChoice(dataset_properties)
        cs = scaler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(scaler_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            scaler_choice.set_hyperparameters(config)

            self.assertEqual(scaler_choice.choice.__class__,
                             scaler_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(scaler_choice.choice))
                self.assertEqual(value, scaler_choice.choice.__dict__[key])


if __name__ == '__main__':
    unittest.main()
