'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 16:23:36
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-24 16:46:46
FilePath: /mycode/my_research/DFLandFL/optimization_algorithms/DecentralizedByzantineEnvironment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from optimization_algorithms.ByzantineEnvironment import ByzantineEnvironment

class DecentralizedByzantineEnvironment(ByzantineEnvironment):
    def __init__(self, graph, *args, **kw):
        super(DecentralizedByzantineEnvironment, self).__init__(
            honest_nodes=graph.honest_nodes,
            byzantine_nodes=graph.byzantine_nodes,
            *args, **kw
        )
        self.graph = graph
        