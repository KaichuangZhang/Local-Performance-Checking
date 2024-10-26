'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 16:28:42
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-01 11:19:17
FilePath: /mycode/my_research/DFLandFL/datasets/distributedDatasets.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import logging as log
class DistributedDataSets():
    # TODO: just for train dataset or for train and test dataset
    def __init__(self, dataset, partition_cls, nodes, honest_nodes):
        
        self.dataset = dataset
        self.nodes = nodes
        self.honest_nodes = honest_nodes
        self.partition = partition_cls(dataset, len(honest_nodes))
        honest_subsets = self.partition.get_subsets(dataset)
        
        # allocate the partitions to all nodes
        next_pointer = 0
        self.subsets = [[] for _ in nodes]
        for node in nodes:
            if node in honest_nodes:
                self.subsets[node] = honest_subsets[next_pointer]
                next_pointer += 1
    def __getitem__(self, index):
        return self.subsets[index]
    def entire_set(self):
        return self.dataset

    def __str__(self) -> str:
        """
        [node:1-{1:10,2:10,3:100..},node:2-{1:20,2:30,3:10....},....]
        """
        node_data_dist = {}
        for node in self.nodes:
            node_data_dist[node] = {}
            if node in self.honest_nodes:
                # honest node
                log.info(self.subsets[node])
            else:
                # byzantine node
                node_data_dist[node] = "no data"
        return "ds"
