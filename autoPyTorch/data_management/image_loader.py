import torch.utils.data as data

import os
import os.path

import logging
logging.getLogger('PIL').setLevel(logging.CRITICAL)
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

from multiprocessing import Process, RawValue, Lock
import time

class ThreadCounter(object):
    def __init__(self):
        # RawValue because we don't need it to create a Lock:
        self.val = RawValue('d', 0)
        self.num = RawValue('i', 0)
        self.lock = Lock()

    def add(self, value):
        with self.lock:
            self.val.value += value
            self.num.value += 1

    def value(self):
        with self.lock:
            return self.val.value

    def avg(self):
        with self.lock:
            return self.val.value / self.num.value

    def reset(self):
        with self.lock:
            self.val.value = 0
            self.num.value = 0

class ImageFilelist(data.Dataset):
    def __init__(self, image_file_list, label_list, transform=None, target_transform=None, loader=default_loader, cache_size=0, image_size=None):
        self.image_file_list = image_file_list
        self.label_list = label_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # self.readTime = ThreadCounter()
        # self.augmentTime = ThreadCounter()
        # self.loadTime = ThreadCounter()
        self.fill_cache(cache_size, image_size)

    def get_times(self, prefix):
        times = dict()
        # times.update({prefix + k: v for k, v in self.transform.get_times().items()})
        # times[prefix + 'read_time'] = self.readTime.value()
        # times[prefix + 'read_time_avg'] = self.readTime.avg()
        # times[prefix + 'augment_time'] = self.augmentTime.value()
        # times[prefix + 'augment_time_avg'] = self.augmentTime.avg()
        # times[prefix + 'load_time'] = self.loadTime.value()
        return times

    def fill_cache(self, cache_size, image_size_pixels):
        self.cache = dict()
        if cache_size == 0:
            return
        import sys
        max_image_size = 0
        cur_size = 0
        for i, impath in enumerate(self.image_file_list):
            img = self.loader(impath)
            image_size = sys.getsizeof(img)
            max_image_size = max(max_image_size, image_size)
            cur_size += image_size
            if image_size_pixels is not None:
                img = img.resize(image_size_pixels)
            self.cache[impath] = img
            # logging.getLogger('autonet').info('Load image: ' + str(sys.getsizeof(self.cache[impath])) + ' bytes - Cache: ' + str(cur_size))
            if cur_size + max_image_size > cache_size:
                break
        logging.getLogger('autonet').info('Could load ' + str(i+1) + '/' + str(len(self.image_file_list)) + ' images into cache, used ' + str(cur_size) + '/' + str(cache_size) + ' bytes')

    def __getitem__(self, index):
        impath = self.image_file_list[index]
        target = self.label_list[index]
        # start_time = time.time()
        img = self.cache[impath] if impath in self.cache else self.loader(impath)
        # self.readTime.add(time.time() - start_time)
        # start_time = time.time()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # self.augmentTime.add(time.time() - start_time)
        # self.loadTime.add(time.time() - start_time)
        return img, target

    def __len__(self):
        return len(self.image_file_list)

class XYDataset(data.Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = self.X[index]
        target = self.Y[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.image_file_list)
