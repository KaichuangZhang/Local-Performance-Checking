'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-22 11:19:20
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-07-04 09:48:41
FilePath: /Research/mycode/my_research/DFLandCFL_difference/datasets/mnist.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

from datasets.dataset import torchDataset
from utils.tools import FEATURE_TYPE

class emnist(torchDataset):
    def __init__(self):
        from torchvision import transforms
        from torchvision.datasets import EMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        ])
        root = 'datasets'
        torch_train_dataset = EMNIST(root=root, train=True,
                                    transform=transform, download=True, split="byclass")
        torch_val_dataset = EMNIST(root=root, train=False,
                                  transform=transform, download=True, split="byclass")
        super().__init__('emnist', torch_train_dataset, torch_val_dataset)