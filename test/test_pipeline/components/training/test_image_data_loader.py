import unittest
import unittest.mock

import torchvision

from autoPyTorch.pipeline.components.training.data_loader.image_data_loader import (
    ImageDataLoader
)


def test_imageloader_build_transform():
    """
    Makes sure a proper composition is created
    """
    loader = ImageDataLoader()

    fit_dictionary = dict()
    fit_dictionary['dataset_properties'] = dict()
    fit_dictionary['dataset_properties']['is_small_preprocess'] = unittest.mock.Mock(())
    fit_dictionary['image_augmenter'] = unittest.mock.Mock()
    fit_dictionary['preprocess_transforms'] = unittest.mock.Mock()

    compose = loader.build_transform(fit_dictionary, mode='train')

    assert isinstance(compose, torchvision.transforms.Compose)

    # We expect to tensor and image augmenter
    assert len(compose.transforms) == 2

    compose = loader.build_transform(fit_dictionary, mode='test')
    assert isinstance(compose, torchvision.transforms.Compose)
    assert len(compose.transforms) == 2

    # Check the expected error msgs
    loader._check_transform_requirements(fit_dictionary)
