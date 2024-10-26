'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-03-30 22:10:09
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-27 22:38:11
FilePath: /Research/mycode/my_research/DFLandFL/datasets/pat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from datasets.dataset import Dataset

class Partition():
    def __init__(self, name, partition):
        self.name = name
        self.partition = partition
    def get_subsets(self, dataset):
        '''
        return all subsets of dataset
        '''
        raise NotImplementedError
    def __getitem__(self, i):
        return self.partition[i]
    def __len__(self):
        return len(self.partition)
    
    
class horizotalPartition(Partition):
    def get_subsets(self, dataset):
        return [
            Dataset(f'{dataset.name}_{i}',
                    features=dataset[p][0], targets=dataset[p][1])
            for i, p in enumerate(self.partition)
        ]
        
    
class EmptyPartition(horizotalPartition):
    def __init__(self, dataset, node_cnt):
        partition = [[] for _ in range(node_cnt)]
        super(EmptyPartition, self).__init__('EmptyPartition', partition)
    
    
class TrivalPartition(horizotalPartition):
    def __init__(self, dataset, node_cnt) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(i*len(dataset)) // node_cnt for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [list(range(seperation[i], seperation[i+1]))
                                for i in range(node_cnt)]
        super(TrivalPartition, self).__init__('TrivalDist', partition)

class iidPartition(horizotalPartition):
    def __init__(self, dataset, node_cnt) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        indexes = list(range(len(dataset)))
        print (f'indexes = {len(indexes)}')
        sep = [(i*len(dataset)) // node_cnt for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [[indexes[i] for i in range(sep[node], sep[node+1])]
                                for node in range(node_cnt)]
        super(iidPartition, self).__init__('iidPartition', partition)

class SharedData(horizotalPartition):
    def __init__(self, dataset, node_cnt) -> None:
        # all nodes have whole dataset
        partition = [list(range(len(dataset)))] * node_cnt
        super(SharedData, self).__init__('SharedData', partition)
        
class LabelSeperation(horizotalPartition):
    def __init__(self, dataset, node_cnt):
        partition = [[] for _ in range(node_cnt)]
        class_num = dataset.num_classes
        sample_num = len(dataset) // node_cnt
        print (f'class num = {class_num}, sample num = {sample_num}')
        for i, (_, label) in enumerate(dataset):
            sample_index = int(label)
            while sample_index < node_cnt and len(partition[sample_index]) >= sample_num:
                sample_index += class_num
            if sample_index >= node_cnt: sample_index -= class_num
            partition[sample_index].append(i)
        super(LabelSeperation, self).__init__('LabelSeperation', partition)
        
class verticalPartition(Partition):
    def __init__(self, dataset, node_cnt) -> None:
        # data seperation, with the form of [d(0), d(1), d(2), ..., d(n)]
        # Node i have the dataset indexed by [d(i), d(i+1))
        seperation = [(i*dataset.feature_dimension) // node_cnt
                      for i in range(node_cnt+1)]
        # data partition, with the form of 
        # [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        # Node i have the dataset indexed by [l(n), r(n))
        partition = [list(range(seperation[i], seperation[i+1]))
                                for i in range(node_cnt)]
        super().__init__('TrivalDist', partition)
    def get_subsets(self, dataset):
        return [
            Dataset(f'{dataset.name}_{i}',
                    features=dataset.features[:, p],
                    targets=dataset.targets)
            for i, p in enumerate(self.partition)
        ]


class QuantitySkew(horizotalPartition):
    def __init__(self, dataset, node_cnt, sub=0.02):
        start = (1. / node_cnt - (node_cnt - 1) * sub / 2.)
        end = start + (node_cnt - 1) * sub
        print (f'start = {start} end = {end}')
        partition_r = []
        for i in range(node_cnt):
            partition_r.append(start + i * sub)
        partition_r = [int(i * len(dataset)) for i in partition_r]
        partition = []
        part_partition = []
        index = 0
        for i in range(len(dataset)):
            part_partition.append(i)
            if len(part_partition) == partition_r[index]:
                partition.append(part_partition)
                part_partition = []
                index += 1
        super().__init__(f'QuantitySkew_{node_cnt}_sub={sub}', partition)

class LabelSkew(horizotalPartition):
    def __init__(self, dataset, node_cnt, c=4):
        sample_num = len(dataset) // node_cnt
        class_num = c
        sample_class_num = sample_num // class_num
        partition = [[] for _ in range(node_cnt)]
        # for this dict
        # partition_dict = {1: {1 : 0, 2: 0, 3: 0, 4: 0}}
        partition_dict = dict()
        for i in range(node_cnt):
            partition_dict[i] = dict()
        for i, (_, label) in enumerate(dataset):
            label = int(label)
            for j in range(node_cnt):
                if len(partition_dict[j].keys()) < c and label not in partition_dict[j]:
                    partition_dict[j][label] = 0
                if label in partition_dict[j]:
                    if partition_dict[j][label] < sample_class_num:
                        partition[j].append(i)
                        partition_dict[j][label] += 1
                        break
                    else: # len(partition_dict[j][label]) >= sample_class_num:
                        pass
            else:
                print (f'label = {label} is not all')
        super().__init__(f'LabelSkew_{node_cnt}_c={c}', partition)


class LabelQuantitySkew(horizotalPartition):
    def __init__(self, dataset, node_cnt, sub=0.02, c=4):
        start = (1. / node_cnt - (node_cnt - 1) * sub / 2.)
        end = start + (node_cnt - 1) * sub
        print (f'start = {start} end = {end}')
        partition_r = []
        for i in range(node_cnt):
            partition_r.append(start + i * sub)
        partition_r = [int(i * len(dataset)) for i in partition_r]
        print (partition_r)
        partition = [[] for _ in range(node_cnt)]
        partition_dict = dict()
        for i in range(node_cnt):
            partition_dict[i] = dict()
        for i, (_, label) in enumerate(dataset):
            label = int(label)
            for j in range(node_cnt):
                if len(partition_dict[j].keys()) < c and label not in partition_dict[j]:
                    partition_dict[j][label] = 0
                if label in partition_dict[j]:
                    if partition_dict[j][label] < partition_r[j] / c:
                        partition[j].append(i)
                        partition_dict[j][label] += 1
                        break
                    else: # len(partition_dict[j][label]) >= sample_class_num:
                        pass
            else:
                pass
                #print (f'label = {label} is not all')
        print (partition_dict)

        super().__init__(f'LabelQuantitySkew_{node_cnt}_sub={sub}_c={c}', partition)