import unittest

import numpy as np

import torch

import torchvision

from autoPyTorch.datasets.image_dataset import ImageDataset


@unittest.skip(reason="Image Dataset issue")
class DatasetTest(unittest.TestCase):
    def runTest(self):
        dataset = torchvision.datasets.FashionMNIST(root='../../datasets/',
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)
        ds = ImageDataset(dataset)
        self.assertIsInstance(ds.mean, torch.Tensor)
        self.assertIsInstance(ds.std, torch.Tensor)
        for img, _ in ds.train_tensors:
            self.assertIsInstance(img, torch.Tensor)


@unittest.skip(reason="Image Dataset issue")
class NumpyArrayTest(unittest.TestCase):
    def runTest(self):
        matrix = np.random.randint(0, 255, (15, 3, 10, 10)).astype(np.float)
        target_df = np.random.randint(0, 5, (15, )).astype(np.float)
        ds = ImageDataset((matrix, target_df))
        self.assertIsInstance(ds.mean, torch.Tensor)
        self.assertIsInstance(ds.std, torch.Tensor)
        for img, _ in ds.train_tensors:
            self.assertIsInstance(img, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
