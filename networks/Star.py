'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-10 13:49:34
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-22 00:06:54
FilePath: /Research/mycode/my_research/DFLandCFL_difference/networks/networks.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from networks.network import Network
import networkx as nx
import random

class Star(Network):
    def __init__(self, node_size, byzantine_size, honest_size, seed=None):
        rng =  random if seed is None else random.Random(seed)
        center_node = 0
        nodes = list(range(node_size))
        center_node = nodes[0]
        byzantine_nodes = rng.sample(nodes[1:], byzantine_size)
        honest_nodes = [i for i in nodes if i not in byzantine_nodes and i != center_node]
        print (f'star: center {center_node}, byzantine nodes = {byzantine_nodes}, honest nodes = {honest_nodes}')
        graph= nx.Graph()
        graph.add_nodes_from(nodes)
        for node in nodes[1:]:
            graph.add_edge(0, node)
        name = f'Star_n={node_size}_b={byzantine_size}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super(Star, self).__init__(name = name, nx_graph = graph,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes, center_node=center_node)