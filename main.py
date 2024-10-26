'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 13:58:41
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-10-25 16:12:22
FilePath: /Research/mycode/my_research/DFLandFL/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import random
import logging as log
from utils.tools import TASK_TYPE, RANDOM_SEED
# algorithm
#from ByrdLab.decentralizedAlgorithm import DSGD
#from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr, decrease_lr
# data distribution
#from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
#                                   iidPartition)
import networkx as nx
from datasets import mnist, cifar10
from utils.parse_agument_functions import get_graph, get_task, get_dataset, \
    get_attack, get_aggregation, get_data_partition, get_LR_scheduler
from optimization_algorithms.DSGD import DecentralizedSGD
from optimization_algorithms.CSGD import CentralizedSGD
#import matplotlib.pyplot as plt
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Decentralized Federated Learning')
    
# Arguments
# DFL or FL
parser.add_argument('--task-type', choices=TASK_TYPE, default='DFL')
# network
parser.add_argument('--network', type=str, default='ER')
parser.add_argument('--network-sparsity', type=float, default=0.7)
parser.add_argument('--honest-size', type=int, default=10)
parser.add_argument('--byzantine-size', type=int, default=2)
parser.add_argument('--network-figure-path', type=str, default='./networks/network_figures/')

# data
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--dataset-partition', type=str, default='iid')

# task
parser.add_argument('--task', type=str, default='softmax_regression')
parser.add_argument('--local-rounds', type=int, default=1)
parser.add_argument('--learning-rate-scheduler', type=str, default='decrease')

# attack
parser.add_argument('--attack', type=str, default='none')

# aggregation
parser.add_argument('--aggregation', type=str, default='ios')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
#log
parser.add_argument('--log-file-path', type=str, default=None)
args = parser.parse_args()
log.basicConfig(
    filename=args.log_file_path,
    filemode='w+',
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


## network
if args.task_type == "DFL":
    # def get_graph(network, honest_size, byzantine_size, network_sparsity, random_seed):
    graph = get_graph(args.network, args.honest_size, \
                      args.byzantine_size, args.network_sparsity, 
                      random_seed=RANDOM_SEED)
    graph.save_as_figure(path=args.network_figure_path)
else:
    if args.network != 'Star':
        raise TypeError("FL is just only has the star network.")
    graph = get_graph(args.network, args.honest_size, \
                      args.byzantine_size,
                      random_seed=RANDOM_SEED)
    graph.save_as_figure(path=args.network_figure_path, center_node=True)

## dataset
dataset = get_dataset(args.dataset)
## task
task = get_task(args.task, dataset)
data_partition = get_data_partition(args.dataset_partition)
## attack
attack = get_attack(args.attack, graph=graph)
## aggregation
aggregation = get_aggregation(args.aggregation, graph=graph, data_partition=data_partition)

# before traing, print the task information
log.info(f'Graph {graph}')
log.info(f'Dataset {dataset}')
log.info(f'Data Partition {data_partition}')
log.info(f'Task {task}')
log.info(f'Attack {attack}')
log.info(f'Aggregation {aggregation}')
env = None
if args.task_type == "DFL":
    env = DecentralizedSGD(
        graph=graph,
        task=task,
        data_partition=data_partition,
        local_rounds=args.local_rounds,
        LR_scheduler=get_LR_scheduler(args.learning_rate_scheduler),
        aggregation=aggregation,
        agg_alpha=args.alpha,
        attack=attack,
        seed=args.seed
    )

else: # FL
    env = CentralizedSGD(
        graph=graph,
        task=task,
        data_partition=data_partition,
        local_rounds=args.local_rounds,
        LR_scheduler=get_LR_scheduler(args.learning_rate_scheduler),
        aggregation=aggregation,
        attack=attack,
        seed=args.seed
    )
env.run()