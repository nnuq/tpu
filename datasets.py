# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(object):
    def __init__(self, args, cur_img_size=None):
        img_size = cur_img_size if cur_img_size else args.img_size
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        self.train = torch.utils.data.DataLoader(
            Dt(root=args.data_path, train=True, transform=transform, download=True),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        self.valid = torch.utils.data.DataLoader(
            Dt(root=args.data_path, train=False, transform=transform),
            batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)