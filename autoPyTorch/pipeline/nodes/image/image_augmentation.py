__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"

import inspect
import logging
import numpy as np

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper

import torch
from torchvision import datasets, models, transforms
from autoPyTorch.components.preprocessing.image_preprocessing.transforms import Cutout, AutoAugment, FastAutoAugment


import time
from autoPyTorch.data_management.image_loader import ThreadCounter
class TimeCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.counters = [ThreadCounter() for _ in transforms]

    def __call__(self, img):
        for i, t in enumerate(self.transforms):
            start_time = time.time()
            img = t(img)
            self.counters[i].add(time.time() - start_time)
        return img

    def get_times(self):
        return {str(t): self.counters[i].value() for i, t in enumerate(self.transforms) }

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ImageAugmentation(PipelineNode):
    def __init__(self):
        super(ImageAugmentation, self).__init__()
        self.mean_std_cache = dict()

    def fit(self, pipeline_config, hyperparameter_config, dataset_info, X, Y, train_indices, valid_indices):
        mean, std = self.compute_mean_std(pipeline_config, hyperparameter_config, X, Y, train_indices, dataset_info) #dataset_info.mean, dataset_info.std

        hyperparameter_config = ConfigWrapper(self.get_name(), hyperparameter_config)

        transform_list = []
        image_size = min(dataset_info.x_shape[-2], dataset_info.x_shape[-1])

        if len(X.shape) > 1:
            transform_list.append(transforms.ToPILImage())
        
        if hyperparameter_config['augment']:
            if hyperparameter_config['fastautoaugment'] and hyperparameter_config['autoaugment']:
                # fast autoaugment and autoaugment
                transform_list.extend([
                    FastAutoAugment(),
                    AutoAugment(),
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip()
                ])
            elif hyperparameter_config['fastautoaugment']:
                # fast autoaugment
                transform_list.extend([
                    FastAutoAugment(),
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip()
                ])
            elif hyperparameter_config['autoaugment']:
                # autoaugment
                transform_list.extend([
                    AutoAugment(),
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip()
                ])
            else:
                # default augment color, rotation, size
                transform_list.extend([
                    transforms.ColorJitter(brightness=0.196, saturation=0.196, hue=0.141),
                    transforms.RandomAffine(degrees=10, shear=0.1, fillcolor=127),
                    transforms.RandomResizedCrop(image_size, scale=(0.533, 1), ratio=(0.75, 1.25)),
                    transforms.RandomHorizontalFlip()
                ])
        else:
            transform_list.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])

            
        # grayscale if only one channel
        if dataset_info.x_shape[1] == 1:
            transform_list.append(transforms.Grayscale(1))
            
        # normalize
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))

        # cutout
        if hyperparameter_config['cutout']:
            n_holes = hyperparameter_config['cutout_holes']
            transform_list.append(Cutout(n_holes=1, length=hyperparameter_config['length'], probability=0.5))


        train_transform = transforms.Compose(transform_list)

        transform_list = []
        if len(X.shape) > 1:
            transform_list.append(transforms.ToPILImage())

        transform_list.extend([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        valid_transform = transforms.Compose([transforms.Grayscale(1)] + transform_list if dataset_info.x_shape[1] == 1 else transform_list)

        return { 'train_transform': train_transform, 'valid_transform': valid_transform, 'mean': mean, 'std': std }

    def predict(self, pipeline_config, mean, std):
    
        predict_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        return {'predict_transform': predict_transform}

    def get_hyperparameter_search_space(self, **pipeline_config):
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()

        augment = cs.add_hyperparameter(CSH.CategoricalHyperparameter('augment', [True, False]))
        autoaugment = cs.add_hyperparameter(CSH.CategoricalHyperparameter('autoaugment', [True, False]))
        fastautoaugment = cs.add_hyperparameter(CSH.CategoricalHyperparameter('fastautoaugment', [True, False]))

        cutout = cs.add_hyperparameter(CSH.CategoricalHyperparameter('cutout', [True, False]))
        cutout_length = cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('length', lower=0, upper=20, log=False))
        cutout_holes = cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('cutout_holes', lower=1, upper=3, log=False))

        cs.add_condition(CS.EqualsCondition(cutout_length, cutout, True))
        cs.add_condition(CS.EqualsCondition(cutout_holes, cutout, True))
        
        cs.add_condition(CS.EqualsCondition(autoaugment, augment, True))
        cs.add_condition(CS.EqualsCondition(fastautoaugment, augment, True))

        return cs

    def compute_mean_std(self, pipeline_config, hyperparameter_config, X, Y, train_indices, dataset_info):
        log = logging.getLogger('autonet')

        if dataset_info.path in self.mean_std_cache:
            mean, std = self.mean_std_cache[dataset_info.path]
            log.debug('CACHED: MEAN: ' + str(mean) + ' -- STD: ' + str(std))
            return mean, std

        from autoPyTorch.pipeline.nodes.image.create_image_dataloader import CreateImageDataLoader
        loader = CreateImageDataLoader()

        image_size = min(dataset_info.x_shape[-2], dataset_info.x_shape[-1])
        transform_list = []
        if len(X.shape) > 1:
            transform_list.append(transforms.ToPILImage())
        transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.CenterCrop(image_size))
        if dataset_info.x_shape[1] == 1:
            transform_list.append(transforms.Grayscale(1))
        transform_list.append(transforms.ToTensor())
        train_transform = transforms.Compose(transform_list)

        cache_size = pipeline_config['dataloader_cache_size_mb']
        pipeline_config['dataloader_cache_size_mb'] = 0
        train_loader = loader.fit(pipeline_config, hyperparameter_config, X, Y, train_indices, None, train_transform, None, dataset_info)['train_loader']
        pipeline_config['dataloader_cache_size_mb'] = cache_size

        mean = 0.
        std = 0.
        nb_samples = 0.

        with torch.no_grad():
            for data, _ in train_loader:
                
                # import matplotlib.pyplot as plt
                # img = plt.imshow(data.numpy()[0,1,:])
                # plt.show()

                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean = mean + data.mean(2).sum(0)
                std = std + data.std(2).sum(0)
                nb_samples += batch_samples

        if nb_samples > 0.:
            mean /= nb_samples
            std /= nb_samples
            mean, std = mean.numpy().tolist(), std.numpy().tolist()
        else:
            mean, std = [mean], [std]

        log.debug('MEAN: ' + str(mean) + ' -- STD: ' + str(std))
        
        self.mean_std_cache[dataset_info.path] = [mean, std]
        return mean, std
