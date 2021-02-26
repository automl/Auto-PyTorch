import unittest
import unittest.mock

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.time_series_data_loader import (
    TimeSeriesDataLoader
)


class TestTimeSeriesDataLoader(unittest.TestCase):
    def test_build_transform_small_preprocess_true(self):
        """
        Makes sure a proper composition is created
        """
        loader = TimeSeriesDataLoader()

        fit_dictionary = {'dataset_properties': {'is_small_preprocess': True}}
        for thing in ['scaler']:
            fit_dictionary[thing] = [unittest.mock.Mock()]

        compose = loader.build_transform(fit_dictionary, mode='train')

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # No preprocessing needed here as it was done before, only from_numpy
        self.assertEqual(len(compose.transforms), 1)

    def test_build_transform_small_preprocess_false(self):
        """
        Makes sure a proper composition is created
        """
        loader = TimeSeriesDataLoader()

        fit_dictionary = {'dataset_properties': {'is_small_preprocess': False},
                          'preprocess_transforms': [unittest.mock.Mock()]}

        compose = loader.build_transform(fit_dictionary, mode='train')

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        print(compose)

        # We expect the preprocess_transforms and from_numpy
        self.assertEqual(len(compose.transforms), 2)
