'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset
from timm.data.transforms_factory import create_transform


# mean and std fr https://github.com/pytorch/examples/blob/master/imagenet/main.py
imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
# fr https://github.com/kakaobrain/fast-autoaugment
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}

imagenet_size=224
imagenet_train_transform = transforms.Compose([
                transforms.RandomResizedCrop(imagenet_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                transforms.ToTensor(),
                imagenet_normalize,
            ])

imagenet_test_transform = transforms.Compose([
            transforms.Resize(imagenet_size+32),
            transforms.CenterCrop(imagenet_size),
            transforms.ToTensor(),
            imagenet_normalize,
        ])

imagenet_train_transform_timm = create_transform(224, is_training=True,)
imagenet_test_transform_timm = create_transform(224, is_training=False,)

class ClassifierLoader:
    def __init__(self,
                 root='/data/imagenet',
                 batch_size=128, 
                 dataset=datasets.ImageNet, 
                 transform={'train':imagenet_train_transform_timm, 'test':imagenet_test_transform_timm},
                 device=None,
                 dataset_name="imagenet",
                 shuffle_test=False,
                 corruption=None):
        super(ClassifierLoader, self).__init__()
        self.test = None
        self.train = None
        self._build(root,
                    batch_size, 
                    dataset, 
                    transform, 
                    device, 
                    dataset_name,
                    shuffle_test,
                    corruption)

    
    def _build(self,
               root,
               batch_size, 
               dataset, 
               transform, 
               device,
               dataset_name,
               shuffle_test,
               corruption):
        DataLoader = torch.utils.data.DataLoader
        workers = torch.cuda.device_count() * 4
        if "cuda" in str(device):
            print("num_workers: ", workers)
            kwargs = {'num_workers': workers, 'pin_memory': True}
        else:
            kwargs = {}

        if dataset_name == "svhn" or dataset_name == "svhn-core":
            x_train = dataset(root=root,
                              split='train',
                              download=True,
                              transform=transform['train'])

            if dataset_name == "svhn":
                x_extra = dataset(root=root,
                                  split='extra',
                                  download=True, 
                                  transform=transform['train'])
                x_train = ConcatDataset([x_train, x_extra])

            x_test = dataset(root=root,
                             split='test',
                             download=True,
                             transform=transform['test'])
        elif dataset_name == "imagenet":
            x_train = dataset(root=root,
                              split='train', 
                              transform=transform['train'])
            if corruption is None:
                x_test = dataset(root=root,
                                 split='val', 
                                 transform=transform['test'])
            else:
                root = os.path.join(root, corruption)
                corrupt_test = []
                for i in range(1, 6):
                    folder = os.path.join(root, str(i))
                    x_test = datasets.ImageFolder(root=folder,
                                                  transform=transform['test'])
                    corrupt_test.append(x_test)
                x_test = ConcatDataset(corrupt_test)

        elif dataset_name == "speech_commands":
            x_train = dataset(root=root,
                              split='train', 
                              transform=transform['train'])
            x_val = dataset(root=root,
                            split='valid', 
                            transform=transform['test'])
            x_test = dataset(root=root,
                             split='test', 
                             transform=transform['test'])

            self.val = DataLoader(x_val,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  **kwargs)

            #self.train = DataLoader(x_train,
            #                        shuffle=True,
            #                        batch_size=batch_size,
            #                        **kwargs)

            #self.test = DataLoader(x_test,
            #                       shuffle=False,
            #                       batch_size=batch_size,
            #                       **kwargs)
            #return
        else:
            x_train = dataset(root=root,
                              train=True,
                              download=True,
                              transform=transform['train'])

            x_test = dataset(root=root,
                             train=False,
                             download=True,
                             transform=transform['test'])

        self.train = DataLoader(x_train,
                                shuffle=True,
                                batch_size=batch_size,
                                **kwargs)

        self.test = DataLoader(x_test,
                               shuffle=shuffle_test,
                               batch_size=batch_size,
                               **kwargs)

