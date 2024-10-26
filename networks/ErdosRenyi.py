'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-10 13:49:34
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-21 09:56:07
FilePath: /Research/mycode/my_research/DFLandCFL_difference/networks/networks.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from networks.network import Network
import networkx as nx
import random

class ErdosRenyi(Network):
    def __init__(self, node_size, byzantine_size, sparsity=0.7, seed=None):
        rng =  random if seed is None else random.Random(seed)
        while True:
            # get the graph 
            graph = nx.fast_gnp_random_graph(node_size, sparsity, seed=rng)
            byzantine_nodes = rng.sample(list(graph.nodes()), byzantine_size)
            honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
            valid = nx.connected.is_connected(graph.subgraph(honest_nodes)) and \
                nx.connected.is_connected(graph) 
            if valid:
                break
        name = f'ErdosRenyi_n={node_size}_b={byzantine_size}_p={sparsity}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super(ErdosRenyi, self).__init__(name = name, nx_graph = graph,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)