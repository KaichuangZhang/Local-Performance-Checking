'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 13:58:41
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-07-04 18:32:31
FilePath: /Research/mycode/my_research/DFLandFL/utils/parse_agument_functions.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import logging as log

from networks.ErdosRenyi import ErdosRenyi
from networks.Star import Star
from datasets.mnist import mnist
from datasets.cifar10 import cifar10
from datasets.fmnist import fmnist
from datasets.femnist import femnist
from datasets.emnist import emnist
from tasks.softmaxRegression import softmaxRegressionTask
from tasks.resnet import ResnetTask
from tasks.femnist_cnn import FEMNISTCNNTask
from attacks.attack import D_sign_flipping, D_gaussian, D_isolation, D_alie, D_zero_value
from aggregations.aggregation import D_mean, D_ios, D_trimmed_mean, D_median, D_geometric_median, D_self_centered_clipping, D_performance_checking, D_faba, D_Krum
from datasets.partitions import iidPartition, LabelSeperation, TrivalPartition, QuantitySkew, LabelSkew, LabelQuantitySkew
from optimization_algorithms.LearningrateScheduler import one_over_sqrt_k_lr, ladder_lr, decrease_lr

def get_graph(network, honest_size, byzantine_size, network_sparsity=None, random_seed=None):
    """ pass """
    if network == 'ER':
        honest_size = honest_size
        byzantine_size = byzantine_size
        node_size = honest_size + byzantine_size
        graph = ErdosRenyi(
            node_size, 
            byzantine_size, 
            sparsity=network_sparsity, 
            seed=random_seed)
    elif network == "Star":
        honest_size = honest_size
        byzantine_size = byzantine_size
        node_size = honest_size + byzantine_size + 1
        graph = Star(
            node_size,
            honest_size=honest_size,
            byzantine_size=byzantine_size, 
            seed=random_seed)
    else:
        raise NotImplemented
    return graph


def get_dataset(dataset):
    if dataset == "MNIST":
        return mnist()
    elif dataset == "CIFAR10":
        return cifar10()
    elif dataset == "FMNIST":
        return fmnist()
    elif dataset == "FEMNIST":
        return femnist()
    elif dataset == "EMNIST":
        return emnist()
    else:
        pass

def get_task(task, dataset):
    if task == "softmax_regression":
        return softmaxRegressionTask(dataset=dataset)
    elif task == 'resnet18_cifar10':
        return ResnetTask(dataset=dataset)
    elif task == 'femnist_alex':
        return FEMNISTCNNTask(dataset=dataset)
    else:
        raise NotImplemented



def get_aggregation(aggregation, graph, data_partition):
    # -------------------------------------------
    # define aggregation
    # -------------------------------------------
    if aggregation == 'mean':
        aggregation = D_mean(graph)
    elif aggregation == 'ios':
        aggregation = D_ios(graph)
    elif aggregation == 'krum':
        aggregation = D_Krum(graph)
    elif aggregation == 'trimmed-mean':
        aggregation = D_trimmed_mean(graph)
    elif aggregation == 'median':
        aggregation = D_median(graph)
    elif aggregation == 'D_faba':
        aggregation = D_faba(graph)
    elif aggregation == 'geometric-median':
        aggregation = D_geometric_median(graph)
    elif aggregation == 'scc':
        if data_partition == 'iid':
            threshold = 5
        elif data_partition == 'noniid':
            threshold = 0.3
        else:
            threshold = 0.3
        aggregation = D_self_centered_clipping(
            graph, threshold_selection='parameter', threshold=threshold)
    # D_self_centered_clipping(graph, threshold_selection='true'),
    elif aggregation == 'D_performance_checking':
        aggregation = D_performance_checking(graph)
    else:
        raise NotImplemented
    return aggregation

def get_attack(attack, graph):
    if attack == 'none':
        attack = None
    elif attack == 'sign_flipping':
        attack = D_sign_flipping(graph)
    elif attack == 'gaussian':
        attack = D_gaussian(graph)
    elif attack == 'isolation':
        attack = D_isolation(graph)
    elif attack == "alie":
        attack = D_alie(graph)
    elif attack == "samevalue":
        attack = D_zero_value(graph)
    else:
        raise ValueError(f'{attack} is not exist.')
    return attack

def get_data_partition(data_partition):
    if data_partition == 'trival':
        partition_cls = TrivalPartition
    elif data_partition == 'iid':
        partition_cls = iidPartition
    elif data_partition == 'noniid':
        partition_cls = LabelSeperation
    elif data_partition == 'QuantitySkew':
        partition_cls = QuantitySkew
    elif data_partition == 'LabelSkew':
        partition_cls = LabelSkew
    elif data_partition == 'LabelQuantitySkew':
        partition_cls = LabelQuantitySkew
    else:
        assert False, 'unknown data-partition'
    return partition_cls


def get_LR_scheduler(LR_scheduler):
    if LR_scheduler== 'constant':
        lr_ctrl = None
    elif LR_scheduler== '1_sqrt_k':
        lr_ctrl = one_over_sqrt_k_lr(a=0.9, b=1)
    elif LR_scheduler== 'ladder':
        decreasing_iter_ls = [30000, 60000]
        proportion_ls = [0.5, 0.2]
        lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
    elif LR_scheduler== 'decrease':
        lr_ctrl = decrease_lr()
    else:
        raise NotADirectoryError
    return lr_ctrl