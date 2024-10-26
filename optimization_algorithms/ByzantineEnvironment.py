'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 16:24:18
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-24 16:49:26
FilePath: /mycode/my_research/DFLandFL/optimization_algorithms/ByzantineEnvironment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from optimization_algorithms.IterativeEnvironment import IterativeEnvironment
from datasets.data_partition import TrivalPartition

class ByzantineEnvironment(IterativeEnvironment):
    def __init__(self, name, lr, model, weight_decay, 
                 dataset, loss_fn, initialize_fn=None, lr_ctrl=None,
                 get_train_iter=None, get_val_iter=None, 
                 partition_cls=TrivalPartition, 
                 honest_size=-1, byzantine_size=-1, 
                 honest_nodes=None, byzantine_nodes=None, attack=None,
                 rounds=10, display_interval=1000, total_iterations=None,
                 seed=None, fix_seed=False,
                 communication_step=None, local_round=1, noise_level=1,
                 momentum=0.,
                 *args, **kw):
        super().__init__(name, lr, lr_ctrl, rounds, display_interval,
                         total_iterations, seed, fix_seed)

        # momentum
        self.momentum = momentum
        # noise
        self.noise_level = noise_level
        # local round
        self.local_round = local_round

        # ====== check validity ======
        assert (honest_nodes is not None and honest_size < 0) \
            or (honest_nodes is not None and len(honest_nodes) == honest_size > 0) \
            or (honest_nodes is None and honest_size > 0)
        assert (byzantine_nodes is not None and len(byzantine_nodes) == byzantine_size >= 0) \
            or (byzantine_nodes is not None and byzantine_size < 0) \
            or (byzantine_nodes is None and byzantine_size >= 0) \
        
        # ====== define node set ======
        if honest_nodes is None:
            self.honest_nodes = list(range(honest_size))
            self.honest_size = honest_size
        else:
            self.honest_nodes = sorted(honest_nodes)
            self.honest_size = len(self.honest_nodes)
        if byzantine_nodes is None:
            self.byzantine_nodes = list(range(honest_size, honest_size+byzantine_size))
            self.byzantine_size = byzantine_size
        else:
            self.byzantine_nodes = sorted(byzantine_nodes)
            self.byzantine_size = len(byzantine_nodes)
        self.nodes = sorted(self.honest_nodes + self.byzantine_nodes)
        self.node_size = self.honest_size + self.byzantine_size
        
        # ====== define properties ======
        assert self.byzantine_size == 0 or attack != None
        self.attack = attack
        self.model = model
        self.initialize_fn = initialize_fn
        
        # ====== task information ======
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.get_train_iter = get_train_iter
        self.get_val_iter = get_val_iter

        # distribute dataset
        self.dataset = dataset
        dist_dataset = DistributedDataSets(dataset=dataset, 
                                           partition_cls=partition_cls,
                                           nodes=self.nodes,
                                           honest_nodes=self.honest_nodes)
        self.partition_name = dist_dataset.partition.name
        self.dist_dataset = dist_dataset
        # communication step
        self.communication_step = communication_step

    def run(self, *args, **kw):
        raise NotImplementedError