import unittest

from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.encoding import TimeSeriesEncoderChoice


class TestEncoderChoice(unittest.TestCase):
    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        dataset_properties = {'numerical_columns': list(range(4)), 'categorical_columns': [5]}
        encoder_choice = TimeSeriesEncoderChoice(dataset_properties)
        cs = encoder_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(encoder_choice.get_components().keys()))
        )


if __name__ == '__main__':
    unittest.main()
