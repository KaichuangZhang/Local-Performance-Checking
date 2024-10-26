from torchvision.datasets import MNIST, utils
from PIL import Image
import os
import torch
import shutil

from torchvision import transforms

class FEMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.download = download
        self.download_link = 'https://media.githubusercontent.com/media/GwenLegate/femnist-dataset-PyTorch/main/femnist.tar.gz'
        self.file_md5 = 'a8a28afae0e007f1acb87e37919a21db'
        self.train = train
        self.root = root
        self.training_file = f'{self.root}/FEMNIST/processed/femnist_train.pt'
        self.test_file = f'{self.root}/FEMNIST/processed/femnist_test.pt'
        self.user_list = f'{self.root}/FEMNIST/processed/femnist_user_keys.pt'

        if not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_test.pt') \
                or not os.path.exists(f'{self.root}/FEMNIST/processed/femnist_train.pt'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data_targets_users = torch.load(data_file)
        self.data, self.targets, self.users = torch.Tensor(data_targets_users[0]), torch.Tensor(data_targets_users[1]), data_targets_users[2]
        data_size = self.data.shape[0]
        self.data = self.data.view(data_size, 28, 28)
        self.user_ids = torch.load(self.user_list)

    def __getitem__(self, index):
        #print (f'get item index = {index}')
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        #print (f'img = {img}')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #print (f'data = {img}, target = {target}')
        return img, target

    def dataset_download(self):
        paths = [f'{self.root}/FEMNIST/raw/', f'{self.root}/FEMNIST/processed/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # download files
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/FEMNIST/raw/', filename=filename, md5=self.file_md5)

        files = ['femnist_train.pt', 'femnist_test.pt', 'femnist_user_keys.pt']
        for file in files:
            # move to processed dir
            shutil.move(os.path.join(f'{self.root}/FEMNIST/raw/', file), f'{self.root}/FEMNIST/processed/')

import torch

from datasets.dataset import torchDataset
from utils.tools import FEATURE_TYPE

class femnist(torchDataset):
    def __init__(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        ])
        root = 'datasets'
        torch_train_dataset = FEMNIST(root=root, train=True,
                                    transform=transform, download=False)
        print (f'train dataset size = {torch_train_dataset.data.shape}')
        torch_val_dataset = FEMNIST(root=root, train=False,
                                  transform=transform, download=False)
        print (f'test dataset size = {torch_val_dataset.data.shape}')
        super().__init__('femnist', torch_train_dataset, torch_val_dataset)
