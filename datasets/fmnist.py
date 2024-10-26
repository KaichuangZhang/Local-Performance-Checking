'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-22 11:19:20
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-07-04 10:08:53
FilePath: /Research/mycode/my_research/DFLandCFL_difference/datasets/mnist.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

from datasets.dataset import torchDataset
from utils.tools import FEATURE_TYPE

class fmnist(torchDataset):
    def __init__(self):
        from torchvision import transforms
        from torchvision.datasets import FashionMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        ])
        root = 'datasets'
        torch_train_dataset = FashionMNIST(root=root, train=True,
                                    transform=transform, download=True)
        print (f'->{torch_train_dataset}')
        torch_val_dataset = FashionMNIST(root=root, train=False,
                                  transform=transform, download=True)
        super().__init__('fmnist', torch_train_dataset, torch_val_dataset)