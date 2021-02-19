import unittest

from imgaug.augmenters.meta import Augmenter, Sequential

import numpy as np

from autoPyTorch.pipeline.components.setup.augmentation.image.ImageAugmenter import ImageAugmenter


class TestImageAugmenter(unittest.TestCase):
    def test_every_augmenter(self):
        image_augmenter = ImageAugmenter()
        #  To test every augmenter, we set the configuration as default where each augmenter
        #  has use_augmenter set to True
        configuration = image_augmenter.get_hyperparameter_search_space().get_default_configuration()
        image_augmenter = image_augmenter.set_hyperparameters(configuration=configuration)
        X = dict(X_train=np.random.randint(0, 255, (8, 3, 16, 16), dtype=np.uint8),
                 dataset_properties=dict(image_height=16, image_width=16))
        for name, augmenter in image_augmenter.available_augmenters.items():
            augmenter = augmenter.fit(X)
            # check if augmenter in the component has correct name
            self.assertEqual(augmenter.get_image_augmenter().name, name)
            # test if augmenter has an Augmenter attribute
            self.assertIsInstance(augmenter.get_image_augmenter(), Augmenter)

            # test if augmenter works on a random image
            train_aug = augmenter(X['X_train'])
            self.assertIsInstance(train_aug, np.ndarray)
            # check if data was changed
            self.assertIsNot(train_aug, X['X_train'])

    def test_get_set_config_space(self):
        X = dict(X_train=np.random.randint(0, 255, (8, 3, 16, 16), dtype=np.uint8),
                 dataset_properties=dict(image_height=16, image_width=16))
        image_augmenter = ImageAugmenter()
        configuration = image_augmenter.get_hyperparameter_search_space().sample_configuration()
        image_augmenter = image_augmenter.set_hyperparameters(configuration=configuration)
        image_augmenter = image_augmenter.fit(X)
        X = image_augmenter.transform(X)

        image_augmenter = X['image_augmenter']
        # test if a sequential augmenter was formed
        self.assertIsInstance(image_augmenter.augmenter, Sequential)

        # test if augmenter works on a random image
        train_aug = image_augmenter(X['X_train'])
        self.assertIsInstance(train_aug, np.ndarray)


if __name__ == '__main__':
    unittest.main()
