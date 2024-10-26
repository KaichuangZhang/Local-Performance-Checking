'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-22 11:19:25
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-01 11:37:11
FilePath: /Research/mycode/my_research/DFLandCFL_difference/datasets/cifar10.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from datasets.dataset import torchDataset
class cifar10(torchDataset):
    def __init__(self):
        from torchvision import transforms
        from torchvision.datasets import CIFAR10
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        root = 'dataset'
        torch_train_dataset = CIFAR10(root=root, train=True,
                                    transform=train_transform, download=True)
        torch_val_dataset = CIFAR10(root=root, train=False,
                                  transform=test_transform, download=True)
        super().__init__('cifar10', torch_train_dataset, torch_val_dataset)
