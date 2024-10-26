'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-04-08 18:51:36
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-08 18:51:59
FilePath: /Research/mycode/my_research/DFLandFL/aggregations/tools.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from utils.tools import FEATURE_TYPE

def MH_rule(graph):
    # Metropolis-Hastings rule
    node_size = graph.number_of_nodes()
    W = torch.eye(node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(node_size):
            if i == j or not graph.has_edge(j, i):
                continue
            i_n = graph.neighbor_sizes[i] + 1
            j_n = graph.neighbor_sizes[j] + 1
            W[i][j] = 1 / max(i_n, j_n)
            W[i][i] -= W[i][j]
    return W
    