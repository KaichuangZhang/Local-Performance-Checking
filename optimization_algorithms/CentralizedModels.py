'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-03-30 17:37:01
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-11 10:04:52
FilePath: /Research/mycode/my_research/DFLandFL/optimization_algorithms/DecentralizedMoldes.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import copy
from torch.nn.parameter import Parameter
from utils.tools import FEATURE_TYPE

class CentralizedModels():
    def __init__(self, base_model, nodesize, device="cuda:0") -> None:
        self.device = device
        self.base_model = base_model
        self.modelsize = self.get_modelsize()
        if self.device != 'cpu':
            self.base_model = self.base_model.to(self.device)
        self.global_model = base_model
        self.workmodel = base_model
        self.nodesize = nodesize
        if nodesize < 2:
            raise Exception(f'node size should >= 2. ')
        self.decentralized_mdoels = []
        for i in range(nodesize):
            self.decentralized_mdoels.append(copy.deepcopy(self.base_model))

    def get_modelsize(self):
        return sum([x.nelement() for x in self.base_model.parameters()])
    
    def active_model(self, index):
        if index < 0 or index >= self.nodesize:
            raise Exception("index error.")
        self.workmodel = self.decentralized_mdoels[index]

    def print_parameter(self):
        for i in range(self.nodesize):
            for name, p in self.decentralized_mdoels[i].named_parameters():
                print (name, p)

    def initilize_parameters(self, same=True, generator=None):
        for i in range(self.nodesize):
            for param in self.decentralized_mdoels[i].parameters():
                init_param = torch.randn(
                    param.size(), dtype=FEATURE_TYPE,
                    device=self.device
                )
                param.data.copy_(init_param)

    def to_vectors(self):
        result = torch.zeros([self.nodesize, self.modelsize], dtype=FEATURE_TYPE)
        with torch.no_grad():
            for i in range(self.nodesize):
                result[i].copy_(
                    torch.cat([p.flatten() for p in self.decentralized_mdoels[i].parameters()])
                )
        return result

    def update_all_models(self, param_vector):
        """ update the client node use parame_vector
        param_vector[torch]
        """
        beg = 0
        end = 0
        for p in self.global_model.parameters():
            end = beg + p.nelement()
            new_param = param_vector[beg:end]
            new_param = Parameter(new_param.view_as(p), requires_grad=True)
            p.data.copy_(new_param)
            beg = end
        if end != self.modelsize:
            raise Exception(f"update model end[{end}] != self.modelsize[{self.modelsize}]")
        for i in range(self.nodesize):
            self.decentralized_mdoels[i] = copy.deepcopy(self.base_model)
        