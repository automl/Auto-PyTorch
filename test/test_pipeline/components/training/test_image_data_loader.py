import unittest
import unittest.mock

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.image_data_loader import (
    ImageDataLoader
)


class TestFeatureDataLoader(unittest.TestCase):
    def test_build_transform(self):
        """
        Makes sure a proper composition is created
        """
        loader = ImageDataLoader()

        fit_dictionary = dict()
        fit_dictionary['dataset_properties'] = dict()
        fit_dictionary['dataset_properties']['is_small_preprocess'] = unittest.mock.Mock(())
        fit_dictionary['image_augmenter'] = unittest.mock.Mock()

        compose = loader.build_transform(fit_dictionary, mode='train')

        self.assertIsInstance(compose, torchvision.transforms.Compose)

        # We expect to tensor and image augmenter
        self.assertEqual(len(compose.transforms), 2)
