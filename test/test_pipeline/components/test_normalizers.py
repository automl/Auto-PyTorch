import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.ImageNormalizer import ImageNormalizer
from autoPyTorch.pipeline.components.preprocessing.image_preprocessing.normalise.NoNormalizer import NoNormalizer


class TestNormalizers(unittest.TestCase):
    def initialise(self):
        self.train = np.random.randint(0, 255, (3, 2, 2, 3))
        self.mean = np.array([np.mean(self.train[:, :, :, i]) for i in range(3)])
        self.std = np.array([np.std(self.train[:, :, :, i]) for i in range(3)])

    def test_image_normalizer(self):
        self.initialise()
        dataset_properties = {'mean': self.mean, 'std': self.std, }
        X = {'dataset_properties': dataset_properties, 'X_train': self.train}

        normalizer = ImageNormalizer()
        normalizer = normalizer.fit(X)
        X = normalizer.transform(X)

        # check if normalizer added to X is instance of self
        self.assertEqual(X['normalise'], normalizer)
        epsilon = 1e-8
        train = self.train - self.mean
        train *= 1.0 / (epsilon + self.std)

        assert_allclose(train, normalizer(self.train), rtol=1e-5)

    def test_no_normalizer(self):
        self.initialise()

        dataset_properties = {'mean': self.mean, 'std': self.std, }
        X = {'dataset_properties': dataset_properties, 'X_train': self.train}

        normalizer = NoNormalizer()
        normalizer = normalizer.fit(X)
        X = normalizer.transform(X)

        # check if normalizer added to X is instance of self
        self.assertEqual(X['normalise'], normalizer)

        assert_array_equal(self.train, normalizer(self.train))
