'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-22 11:00:41
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-07-04 18:52:36
FilePath: /Research/mycode/my_research/DFLandCFL_difference/datasets/dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
from datasets.cache_io import isfile_in_cache, load_file_in_cache, dump_file_in_cache
import torch

FEATURE_TYPE = torch.float64
CLASS_TYPE = torch.uint8
VALUE_TYPE = torch.float64

class Dataset():
    def __init__(self, name, features, targets,
                 val_features=None, val_targets=None):
        assert len(features) == len(targets)
        set_size = len(features)
        self.features = features
        self.targets = targets
        self.val_features = val_features
        self.val_targets = val_targets
        self.num_classes = torch.unique(self.targets).shape[0]

        # random shuffling
        self.__RR = False
        self.__order = list(range(set_size))

        self.name = name
        self.set_size = set_size
        self.__COUNT = None
        if len(features) != 0:
            self.feature_dimension = features[0].nelement()
            self.feature_size = features[0].size()
        else:
            self.feature_dimension = 0
            self.feature_size = 0
    def get_count(self):
        if self.__COUNT is None:
            count = {}
            for target_tensor in self.targets:
                target = target_tensor.item()
                if target in count.keys():
                    count[target] += 1
                else:
                    count[target] = 1
            self.__COUNT = count
        return self.__COUNT
        
    def randomReshuffle(self, on):
        self.__RR = on
        if on:
            random.shuffle(self.__order)
    
    def __getitem__(self, index):
        if self.__RR:
            i = self.__order[index]
            return self.features[i], self.targets[i]
        else:
            return self.features[index], self.targets[index]
    
    def __len__(self):
        return self.set_size

    def subset(self, indexes, name=''):
        if name == '':
            name = self.name + '_subset'
        return Dataset(self.name, self.features[indexes], self.targets[indexes])
    def get_val_set(self):
        if self.val_features is None or self.val_targets is None:
            self.val_features = self.features
            self.val_targets = self.targets
        print (f'val dataset {len(self.val_features)}')
        return Dataset(self.name+'_val', self.val_features, self.val_targets)

    def __str__(self) -> str:
        if self.val_targets is not None:
            return f'name[{self.name}] - train[{len(self.targets)}] - val[{len(self.val_targets)}]'
        else:
            return f'name[{self.name}] - train[{len(self.targets)}] - {self._get_dist()}'
    def _get_dist(self) ->str:
        #self.targets = targets
        data_dist = {}
        for target in self.targets:
            target = int(target.item())
            if target not in data_dist:
                data_dist[target] = 0
            data_dist[target] += 1
        data_dist_list = []
        for k, v in sorted(data_dist.items()):
            data_dist_list.append(f'{str(k)}:{str(v/len(self.targets))}')
        return ','.join(data_dist_list)

class torchDataset(Dataset):
    def __init__(self, name, torch_train_set, torch_val_set):
        self.torch_train_set = torch_train_set
        self.torch_val_set = torch_val_set
        dtype_str = str(FEATURE_TYPE).replace('torch.', '')
        cache_file_name = f'data_cache_{name}_{dtype_str}'
        if isfile_in_cache(cache_file_name):
            cache = load_file_in_cache(cache_file_name)
            train_features = cache['train_features']
            train_targets = cache['train_targets']
            val_features = cache['val_features']
            val_targets = cache['val_targets']
        else:
            train_features = \
                torch.stack([feature for feature, _ in torch_train_set], axis=0)
            val_features = \
                torch.stack([feature for feature, _ in torch_val_set], axis=0)
            print (f'train shape (in torch dataset) = {train_features.shape}')
            print (f'test shape (in torch dataset) = {val_features.shape}')
            # feature_size = torch_train_set[0][0].size()
            # dataset_size = len(torch_train_set)
            # train_features = torch.empty(dataset_size, *feature_size)
            train_targets = torch_train_set.targets
            val_targets = torch_val_set.targets
            # for i, (feature, _) in enumerate(torch_train_set):
            #     train_features[i].copy_(feature)
            cache = {
                'train_features': train_features,
                'train_targets': train_targets,
                'val_features': val_features,
                'val_targets': val_targets,
            }
            dump_file_in_cache(cache_file_name, cache)
        if type(torch_train_set.targets) != torch.Tensor:
            torch_train_set.targets = torch.Tensor(torch_train_set.targets)
        if type(torch_val_set.targets) != torch.Tensor:
            torch_val_set.targets = torch.Tensor(torch_val_set.targets)
        if train_features.dtype != FEATURE_TYPE:
            train_features = train_features.to(FEATURE_TYPE)
        if val_features.dtype != FEATURE_TYPE:
            val_features = val_features.to(FEATURE_TYPE)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device != 'cpu':
            train_features = train_features.to(device)
            torch_train_set.targets = torch_train_set.targets.to(device)
            val_features = val_features.to(device)
            torch_val_set.targets = torch_val_set.targets.to(device)
        self.num_classes = len(torch_train_set.classes)
        super().__init__(name, train_features, torch_train_set.targets,
                         val_features, torch_val_set.targets)
    def subset(self, indexes, name=''):
        s = super().subset(indexes, name)
        s.num_classes = self.num_classes
        return s