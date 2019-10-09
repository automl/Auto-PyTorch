import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path, genotype):
  pretrained_dict = torch.load(model_path)
  model_dict = model.state_dict()

  # keep only the weights for the specified genotype, 
  # and prune all the other weights from the MixedOps
  #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

  edge_dict = {(0,2): 0, (0,3): 2, (0,4): 5, (0,5): 9, (1,2): 1, (1,3): 3, (1,4): 6, (1,5): 10, (2,3): 4, (2,4): 7, (3,4): 8, (2,5): 11, (3,5): 12, (4,5): 13}

  for layer in range(8):
    first_number = layer
    
    for p in range(2):
      if layer in [3, 6] and p == 0:
        key = 'cells.{}.preprocess{}.conv_1.weight'.format(layer, p)
        key = 'cells.{}.preprocess{}.conv_2.weight'.format(layer, p)
      else:
        key = 'cells.{}.preprocess{}.op.1.weight'.format(layer, p)
      model_dict[key] = pretrained_dict[key]
      
    if layer in [2, 5]:
      gene = genotype.reduce
    else:
      gene = genotype.normal
      
    for i in range(4):
      for k in [2*i, 2*i + 1]:
        op, j = gene[k]
        second_number = edge_dict[(j, i + 2)]
        if op == 'sep_conv_3x3':
          third_number = 4
          for h in [1, 2, 5, 6]:
            key_model = 'cells.{}._ops.{}.op.{}.weight'.format(layer, k, h)
            key_pretrained = 'cells.{}._ops.{}._ops.{}.op.{}.weight'.format(first_number, second_number, third_number, h)
            model_dict[key_model] = pretrained_dict[key_pretrained] 
        elif op == 'max_pool_3x3':
          third_number = 1
        elif op == 'avg_pool_3x3':
          third_number = 2

  model.load_state_dict(model_dict)


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    try:
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    except:
        mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  import time, random
  time.sleep(random.uniform(1, 2))
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

