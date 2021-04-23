import os
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import warnings


class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """

    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
             else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
             else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """

    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class LoadDataset(Dataset):
    def __init__(self, data_path, train, resize_size=32, jitter=False,
                 cutout=False, norm=True):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.train = train
        self.resize_size = resize_size
        self.norm_mean = (0.3337, 0.3064, 0.3171)
        self.norm_std = (0.2672, 0.2564, 0.2629)

        if train:
            self.transforms = [
                RandomCropLongEdge(),
                transforms.Resize((self.resize_size, self.resize_size))]
            if jitter:
                self.transforms += [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0)
                ]
            self.transforms += [transforms.ToTensor()]
            if cutout:
                self.transforms += [transforms.RandomErasing(
                    p=1, scale=(0.1, 0.2), ratio=(0.5, 1.0)
                )]
        else:
            if jitter == True or cutout == True:
                warnings.warn(
                    "This is for testing, jitter, cutout will take no effect")
            self.transforms = [
                CenterCropLongEdge(),
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.ToTensor()
            ]
        if norm:
            self.transforms += [
                transforms.Normalize(self.norm_mean, self.norm_std)
            ]
        self.transforms = transforms.Compose(self.transforms)
        print(self.transforms)

        self.load_dataset()

    def load_dataset(self):
        self.root_dir = os.path.join(self.data_path, 'GTSRB')
        self.sub_directory = 'trainingset' if self.train else 'testset'
        self.csv_file_name = 'training.csv' if self.train else 'test.csv'

        csv_file_path = os.path.join(
            self.root_dir, self.sub_directory, self.csv_file_name
        )

        self.csv_data = pd.read_csv(csv_file_path)

    def __len__(self):
        num_dataset = len(self.csv_data)
        return num_dataset

    def __getitem__(self, index):
        img_path = os.path.join(
            self.root_dir, self.sub_directory, self.csv_data.iloc[index, 0]
        )
        img = Image.open(img_path)
        label = int(self.csv_data.iloc[index, 1])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
