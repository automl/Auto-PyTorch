import copy
import unittest

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing import (
    FeatureProprocessorChoice
)


class TestFeaturePreprocessorChoice(unittest.TestCase):
    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the feature preprocessor
        choice"""
        dataset_properties = {'numerical_columns': list(range(4)),
                              'categorical_columns': [5],
                              'task_type': 'tabular_classification'}
        feature_preprocessor_choice = FeatureProprocessorChoice(dataset_properties)
        cs = feature_preprocessor_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(feature_preprocessor_choice.get_available_components(
                dataset_properties=dataset_properties).keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            feature_preprocessor_choice.set_hyperparameters(config)

            self.assertEqual(feature_preprocessor_choice.choice.__class__,
                             feature_preprocessor_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(feature_preprocessor_choice.choice))
                # for score function in some feature preprocessors
                # this will fail
                if 'score_func' or 'pooling_func' in key:
                    continue
                self.assertEqual(value, feature_preprocessor_choice.choice.__dict__[key])

    def test_only_categorical(self):
        dataset_properties = {'numerical_columns': [],
                              'categorical_columns': [5],
                              'task_type': 'tabular_classification'}

        chooser = FeatureProprocessorChoice(dataset_properties)
        configspace = chooser.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        self.assertEqual(configspace['__choice__'], 'NoFeaturePreprocessor')


if __name__ == '__main__':
    unittest.main()
