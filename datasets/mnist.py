'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-22 11:19:20
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-01 11:13:54
FilePath: /Research/mycode/my_research/DFLandCFL_difference/datasets/mnist.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

from datasets.dataset import torchDataset
from utils.tools import FEATURE_TYPE

class mnist(torchDataset):
    def __init__(self):
        from torchvision import transforms
        from torchvision.datasets import MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
            # transforms.Normalize(mean=[0.1307],std=[0.3081])
            transforms.Normalize(mean=[0.5],std=[0.5]),
            #transforms.Lambda(lambda x: x / 255)
        ])
        root = 'datasets'
        torch_train_dataset = MNIST(root=root, train=True,
                                    transform=transform, download=True)
        torch_val_dataset = MNIST(root=root, train=False,
                                  transform=transform, download=True)
        super().__init__('mnist', torch_train_dataset, torch_val_dataset)