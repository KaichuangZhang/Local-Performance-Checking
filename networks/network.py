'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-10 13:14:30
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-15 17:43:22
FilePath: /Research/mycode/my_research/DFLandCFL_difference/networks/network.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Network:
    def __init__(self, name, nx_graph, honest_nodes, byzantine_nodes, center_node=None):
        if center_node is None:
            self.init(name, nx_graph, honest_nodes, byzantine_nodes)
        else:
            self.init_star(name, nx_graph, honest_nodes, byzantine_nodes, center_node)
        
    def init(self, name, nx_graph, honest_nodes, byzantine_nodes):
        self.name = name
        self.nx_graph = nx_graph
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
        # node counting
        self.node_size = nx_graph.number_of_nodes()
        self.honest_size = len(honest_nodes)
        self.byzantine_size = len(byzantine_nodes)
        # neighbor list
        self.neighbors = [
            list(nx_graph.neighbors(node)) for node in nx_graph.nodes()
        ]
        self.honest_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in honest_nodes]
            for i in nx_graph.nodes()
        ]
        self.byzantine_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in byzantine_nodes]
            for i in nx_graph.nodes()
        ]
        self.honest_neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.honest_neighbors)
        ]
        self.neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.neighbors)
        ]
        # neighbor size list
        self.honest_sizes = [
            len(node_list) for node_list in self.honest_neighbors
        ]
        self.byzantine_sizes = [
            len(node_list) for node_list in self.byzantine_neighbors
        ]
        self.neighbor_sizes = [
            self.honest_sizes[node] + self.byzantine_sizes[node] 
            for node in nx_graph.nodes()
        ]
        
        # lost node refers to the the node has more than 1/2 byzantine neighbors
        self.lost_nodes = [
            node for node in self.honest_nodes
            if self.honest_sizes[node] <= 2 * self.byzantine_sizes[node]
        ]
        
    def honest_subgraph(self, name='', relabel=True):
        nx_subgraph = self.subgraph(self.honest_nodes)
        if name == '':
            name = self.name
        if relabel:
            nx_subgraph = nx.convert_node_labels_to_integers(nx_subgraph)
        return Network(name=name, nx_graph=nx_subgraph, 
                     honest_nodes=list(nx_subgraph.nodes()),
                     byzantine_nodes=[])
    
    def init_star(self, name, nx_graph, honest_nodes, byzantine_nodes, center_node):
        self.name = name
        self.nx_graph = nx_graph
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
        self.center_node = center_node
        # node counting
        self.node_size = nx_graph.number_of_nodes()
        self.honest_size = len(honest_nodes)
        self.byzantine_size = len(byzantine_nodes)

        # neighbor list
        self.neighbors = [
            list(nx_graph.neighbors(node)) for node in nx_graph.nodes()
        ]
        self.honest_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in honest_nodes]
            for i in nx_graph.nodes()
        ]
        self.byzantine_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
                and j in byzantine_nodes]
            for i in nx_graph.nodes()
        ]

         # neighbor size list
        self.honest_sizes = [
            len(node_list) for node_list in self.honest_neighbors
        ]
        self.byzantine_sizes = [
            len(node_list) for node_list in self.byzantine_neighbors
        ]
        self.neighbor_sizes = [
            self.honest_sizes[node] + self.byzantine_sizes[node] 
            for node in nx_graph.nodes()
        ]


    def save_as_figure(self, path, center_node=False):
        NODE_COLOR_CENTER = 'white'
        NODE_COLOR_HONEST = '#FFFF88'
        NODE_COLOR_BYZANTINE = '#CDEB8B'
        EDGE_WIDTH = 3
        NODE_SIZE = 1000
        FONT_SIZE = 28
        # loyout
        pos = nx.kamada_kawai_layout(self.nx_graph)

        # honest nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
            node_size = NODE_SIZE, 
            nodelist = self.honest_nodes,
            node_color = NODE_COLOR_HONEST,
        )
        # Byzantine nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, 
            node_size = NODE_SIZE,
            nodelist = self.byzantine_nodes,
            node_color = NODE_COLOR_BYZANTINE,
        )
        # center node
        if center_node:
            nx.draw_networkx_nodes(self.nx_graph, pos, 
                node_size = NODE_SIZE,
                nodelist = [self.center_node],
                node_color = NODE_COLOR_CENTER,
            )

        # edges
        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.5, width=EDGE_WIDTH)
        # label
        label_dict = {
            i: str(i) for i in range(self.nx_graph.number_of_nodes())
        }
        nx.draw_networkx_labels(self.nx_graph, pos, label_dict, font_size=FONT_SIZE)
        self.figure_path= path + self.name + ".png"
        plt.savefig(self.figure_path, format="PNG", dpi=600)

        
    def __getattr__(self, attr):
        '''
        inherit the properties of 'nx_graph'
        '''
        return getattr(self.nx_graph, attr)
    
    def __str__(self) -> str:
        return f"name[{self.name}] - honest nodes[{self.honest_nodes}] - Byzantine nodes[{self.byzantine_nodes}] - figure_path[{self.figure_path}]"