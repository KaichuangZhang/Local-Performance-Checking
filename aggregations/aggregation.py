import copy

import torch
from utils.tools import FEATURE_TYPE
from scipy import stats
from aggregations.tools import MH_rule
import logging as log

#from ByrdLab.library.tool import MH_rule

# class aggregation():
#     def __init__(self, name, honest_size, byzantine_size):
#         self.name = name
#         self.honest_size = honest_size
#         self.byzantine_size = byzantine_size
#     def run(self, messages):
#         raise NotImplementedError


def noisymean(wList):
    x = []
    for i in range(len(wList)):
        r = (variance ** 0.5) * torch.randn_like(wList[i])
        x.append(r)

    # adding noise to gradient
    s = []
    for i in range(len(wList)):
        s.append(wList[i] + x[i])

    return torch.mean(s, dim=0)

def mean(wList):
    return torch.mean(wList, dim=0)

def geometric_median(wList, max_iter = 80, err=1e-5):
    guess = torch.mean(wList, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack([w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= err:
            break
    return guess

def medoid_index(wList):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    dist_sum = dist.sum(dim=1)
    return dist_sum.argmax()
    
def medoid(wList):
    return wList[medoid_index(wList)]

def Krum_index(wList, byzantine_size):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()**2
            # We need minimized distance so we add a minus sign here
            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    # The distance from any node to itself must be 0.00, so we add 1 here
    k = node_size - byzantine_size - 2 + 1 
    topv, _ = dist.topk(k=k, dim=1)
    scores = topv.sum(dim=1)
    return scores.argmax()
    
def Krum(wList, byzantine_size):
    index = Krum_index(wList, byzantine_size)
    return wList[index]

def mKrum(wList, byzantine_size, m=1):
    remain = wList
    result = torch.zeros_like(wList[0], dtype=FEATURE_TYPE)
    for _ in range(m):
        res_index = Krum_index(remain, byzantine_size)
        result += remain[res_index]
        remain = remain[torch.arange(remain.size(0)) != res_index]
    return result / m

def median(wList):
    return wList.median(dim=0)[0]


def noisytrimmed_mean(wList, byzantine_size):
    x = []
    for i in range(len(wList)):
        r = (variance ** 5) * torch.randn_like(wList[i])
        x.append(r)

    # adding noise to gradient
    s = []
    for i in range(len(wList)):
        s.append(wList[i] + x[i])
    node_size = s.size(0)
    proportion_to_cut = byzantine_size / node_size
    tm_np = stats.trim_mean(s, proportion_to_cut, axis=0)
    return torch.from_numpy(tm_np)

def trimmed_mean(wList, byzantine_size):
    #print (f'{type(wList)}')
    node_size = wList.size(0)
    proportion_to_cut = byzantine_size / node_size
    tm_np = stats.trim_mean(wList.detach().numpy(), proportion_to_cut, axis=0)
    return torch.from_numpy(tm_np)

def remove_outliers(wList, byzantine_size):
    mean = torch.mean(wList, dim=0)
    # remove the largest 'byzantine_size' model
    distances = torch.tensor([
        -torch.norm(model - mean) for model in wList
    ])
    node_size = wList.size(0)
    remain_cnt = node_size - byzantine_size
    (_, remove_index) = torch.topk(distances, k=remain_cnt)
    return wList[remove_index].mean(dim=0)
    
def faba(wList, byzantine_size):
    remain = wList
    for _ in range(byzantine_size):
        mean = remain.mean(dim=0)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
            torch.norm(model - mean) for model in remain
        ])
        remove_index = distances.argmax()
        remain = remain[torch.arange(remain.size(0)) != remove_index]
    return remain.mean(dim=0)

def bulyan(wList, byzantine_size):
    remain = wList
    selected_ls = []
    node_size = wList.size(0)
    selection_size = node_size-2*byzantine_size
    for _ in range(selection_size):
        res_index = Krum_index(remain, byzantine_size)
        selected_ls.append(remain[res_index])
        remain = remain[torch.arange(remain.size(0)) != res_index]
    selection = torch.stack(selected_ls)
    m = median(selection)
    dist = -(selection - m).abs()
    indices = dist.topk(k=selection_size-2*byzantine_size, dim=0)[1]
    if len(wList.size()) == 1:
        result = selection[indices].mean()
    else:
        result = torch.stack([
            selection[indices[:, d], d].mean() for d in range(wList.size(1))])
    return result
    
class DecentralizedAggregation():
    def __init__(self, name, graph, superparameter={}):
        self.name = name
        self.graph = graph
        # some aggregation may need global state of the system
        # this global state can be store in the global state dictionary
        self.global_state = {}
        self.superparam = superparameter
    def run(self, local_models, node):
        raise NotImplementedError
    def all_neighbor_models(self, local_models, node):
        return local_models[self.graph.neighbors[node]]
    def neighbor_models_and_itself(self, local_model, node):
        li = list(self.graph.neighbors[node]) + [node]
        return local_model[li]
    def __str__(self) -> str:
        return f'name[{self.name}]'
    
class D_mean(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_mean, self).__init__(name='mean', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return neighbor_models.mean(axis=0)
    
class D_no_communication(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_no_communication, self).__init__(name='no_communication',
                                                 graph=graph)
    def run(self, local_models, node):
        return local_models[node]

class D_meanWN(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_meanWN, self).__init__(name='meanWN', graph=graph)
        self.W = MH_rule(graph)
    def run(self, local_models, node):
        return torch.tensordot(self.W[node], local_models, dims=1)

class D_meanW(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_meanW, self).__init__(name='meanW', graph=graph)
        self.W = MH_rule(graph)
        print (self.W)
    def run(self, local_models, node):
        return torch.tensordot(self.W[node], local_models, dims=1)
    
class D_median(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_median, self).__init__(name='median', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return median(neighbor_models)
    
class D_geometric_median(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_geometric_median, self).__init__(name='geometric_median',
                                                 graph=graph)
    def run(self,  local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return geometric_median(neighbor_models)
    
class D_Krum(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_Krum, self).__init__(name='Krum', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return Krum(neighbor_models, byzantine_size=self.graph.byzantine_sizes[node])
    
class D_mKrum(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_mKrum, self).__init__(name='mKrum', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        m = self.graph.neighbor_sizes[node]-2*self.graph.byzantine_sizes[node]-3
        return mKrum(neighbor_models, 
                     byzantine_size=self.graph.byzantine_sizes[node],
                     m=m)
    
class D_trimmed_mean(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_trimmed_mean, self).__init__(name='trimmed_mean', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        q = self.graph.byzantine_sizes[node]
        trimmed_neighbor_size = len(neighbor_models) - 2 * q
        local_model = local_models[node]
        if trimmed_neighbor_size <= 0:
            return local_model
        tm = trimmed_mean(neighbor_models, byzantine_size=q)
        return (tm * trimmed_neighbor_size + local_model) / (trimmed_neighbor_size + 1)

class D_trimmed_mean1(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_trimmed_mean1, self).__init__(name='noisytrimmed_mean', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        q = self.graph.byzantine_sizes[node]
        tm = trimmed_mean(neighbor_models, byzantine_size=q)
        trimmed_neighbor_size = len(neighbor_models) - 2 * q
        local_model = local_models[node]
        return (tm * trimmed_neighbor_size + local_model) / (trimmed_neighbor_size + 1)
    
class D_remove_outliers(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_remove_outliers, self).__init__(name='remove_outliers',
                                                graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        local_model = local_models[node]
        rm = remove_outliers(neighbor_models, 
                             byzantine_size=self.graph.byzantine_sizes[node])
        neighbor_size = len(neighbor_models)
        res = (rm * neighbor_size + local_model) / (neighbor_size + 1)
        return res
    
class D_faba(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_faba, self).__init__(name='FABA', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        local_model = local_models[node]
        agg = faba(neighbor_models, byzantine_size=self.graph.byzantine_sizes[node])
        hoenst_neighbor_size = self.graph.honest_sizes[node]
        res = (agg * hoenst_neighbor_size + local_model) / (hoenst_neighbor_size + 1)
        return res
    
class D_ios(DecentralizedAggregation):
    def __init__(self, graph, exact_byz_cnt=True, byz_cnt=-1):
        assert exact_byz_cnt or byz_cnt > 0
        if exact_byz_cnt:
            name = 'IOS'
        else:
            if byz_cnt < 0:
                name = 'IOS_max'
            else:
                name = f'IOS_{byz_cnt}'
        super().__init__(name=name, graph=graph)
        node_size = graph.number_of_nodes()
        self.W = torch.eye(node_size, dtype=FEATURE_TYPE)
        self.exact_byz_cnt = exact_byz_cnt
        self.Byz_cnt = byz_cnt
        for i in range(node_size):
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i):
                    continue
                i_n = self.graph.neighbor_sizes[i] + 1
                j_n = self.graph.neighbor_sizes[j] + 1
                self.W[i][j] = 1 / max(i_n, j_n)
                # self.W[i][j] = 1 / i_n
                self.W[i][i] -= self.W[i][j]
    def run(self, local_models, node):
        remain_models = local_models[self.graph.neighbors[node]]
        remain_weight = self.W[node][self.graph.neighbors[node]]
        if self.exact_byz_cnt:
            estimate_byz_cnt = self.graph.byzantine_sizes[node]
        else:
            if self.Byz_cnt < 0:
                estimate_byz_cnt = self.graph.byzantine_sizes.max()
            else:
                estimate_byz_cnt = self.Byz_cnt
        for _ in range(estimate_byz_cnt):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean += self.W[node][node]*local_models[node]
            mean /= remain_weight.sum() + self.W[node][node]
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res += self.W[node][node]*local_models[node]
        res /= remain_weight.sum() + self.W[node][node]
        return res
    
class D_ios_equal_neigbor_weight(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='IOS_equal_neigbor_weight', graph=graph)
        node_size = graph.number_of_nodes()
        self.W = torch.eye(node_size, dtype=FEATURE_TYPE)
        max_degree = -1
        for i in range(node_size):
            if self.graph.neighbor_sizes[i] > max_degree:
                max_degree = self.graph.neighbor_sizes[i] + 1
        for i in range(node_size):
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i):
                    continue
                self.W[i][j] = 1 / max_degree
                # self.W[i][j] = 1 / i_n
                self.W[i][i] -= self.W[i][j]
    def run(self, local_models, node):
        remain_models = local_models[self.graph.neighbors[node]]
        remain_weight = self.W[node][self.graph.neighbors[node]]
        for _ in range(self.graph.byzantine_sizes[node]):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean += self.W[node][node]*local_models[node]
            mean /= remain_weight.sum() + self.W[node][node]
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res += self.W[node][node]*local_models[node]
        res /= remain_weight.sum() + self.W[node][node]
        
        # FABA
        # neighbor_models = self.all_neighbor_models(local_models, node)
        # local_model = local_models[node]
        # remain = neighbor_models
        # for _ in range(self.graph.byzantine_sizes[node]):
        #     mean = remain.mean(dim=0)
        #     # remove the largest 'byzantine_size' model
        #     distances = torch.tensor([
        #         torch.norm(model - mean) for model in remain
        #     ])
        #     remove_index = distances.argmax()
        #     remain = remain[torch.arange(remain.size(0)) != remove_index]
        # agg = remain.mean(dim=0)
        # hoenst_neighbor_size = self.graph.honest_sizes[node]
        # res2 = (agg * hoenst_neighbor_size + local_model) / (hoenst_neighbor_size + 1)
        return res
    
class D_bulyan(DecentralizedAggregation):
    def __init__(self, graph):
        self.byzantine_sizes = graph.byzantine_sizes
        super(D_bulyan, self).__init__(name='Bulyan', graph=graph)
    def run(self, local_models, node):
        local_model = local_models[node]
        agg = bulyan(local_model, byzantine_size=self.byzantine_sizes[node])
        return agg
        
class D_centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold=10):
        super().__init__(name=f'CC_tau={threshold}', graph=graph)
        self.memory = None
        self.threshold = threshold
    def run(self, local_models, node):
        if self.memory == None:
            self.memory = torch.zeros_like(local_models)
        diff = torch.zeros_like(self.memory[node])
        for n in self.graph.neighbors[node] + [node]:
            model = local_models[n]
            norm = (model - self.memory[node]).norm()
            if norm > self.threshold:
                diff += self.threshold * (model - self.memory[node]) / norm
            else:
                diff += model - self.memory[node]
        diff /= (self.graph.neighbor_sizes[node] + 1)
        self.memory[node] = self.memory[node] + diff
        return self.memory[node]
    
            
class D_self_centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold_selection='estimation', threshold=10):
        if threshold_selection == 'estimation':
            name = 'SCClip'
        elif threshold_selection == 'true':
            name = 'SCClip_T'
        elif threshold_selection == 'parameter':
            name = f'SCClip_tau={threshold}'
        else:
            raise ValueError('invalid threshold setting')
        super().__init__(name=name, graph=graph)
        self.W = MH_rule(graph)
        self.threshold = threshold
        self.threshold_selection = threshold_selection
    def get_threshold_estimate(self, local_models, node):
        # find the bottom-(honest-size) weights as the estimated threshold
        local_model = local_models[node]
        node_size = local_models.size(0)
        norm_list = torch.tensor([
            -(local_models[n]-local_model).norm()
            if n in self.graph.neighbors[node] and n != node else 1
            for n in range(node_size)
        ])
        
        honest_size = self.graph.honest_sizes[node]
        _, bottom_index = norm_list.topk(k=honest_size)
        top_index = [
            n for n in self.graph.neighbors[node] 
            if n not in bottom_index and n != node
        ]
        weighted_avg_norm = sum([
            self.W[node][n]*norm_list[n] for n in bottom_index
        ])
        cum_weight = sum([
            self.W[node][n] for n in top_index
        ])
        return torch.sqrt(weighted_avg_norm/cum_weight)
    def get_true_threshold(self, local_models, node):
        # find the bottom-(honest-size) weights as the estimated threshold
        local_model = local_models[node]
        
        weighted_avg_norm = sum([
            self.W[node][n]*(local_models[n]-local_model).norm()
            for n in self.graph.honest_neighbors[node]
        ])
        cum_weight = sum([
            self.W[node][n] for n in self.graph.byzantine_neighbors[node]
        ])
        return torch.sqrt(weighted_avg_norm/cum_weight)
    def run(self, local_models, node):
        if self.threshold_selection == 'estimation':
            threshold = self.get_threshold(local_models, node)
        elif self.threshold_selection == 'true':
            threshold = self.get_true_threshold(local_models, node)
        elif self.threshold_selection == 'parameter':
            threshold = self.threshold
        else:
            raise ValueError('invalid threshold setting')
        local_model = local_models[node]
        cum_diff = torch.zeros_like(local_model)
        for n in self.graph.neighbors[node]:
            model = local_models[n]
            diff = model - local_model
            norm = diff.norm()
            weight = self.W[node][n]
            if norm > threshold:
                cum_diff += weight * threshold * diff / norm
            else:
                cum_diff += weight * diff
        return local_model + cum_diff

"""
class DecentralizedAggregation():
    def __init__(self, name, graph, superparameter={}):
        self.name = name
        self.graph = graph
        # some aggregation may need global state of the system
        # this global state can be store in the global state dictionary
        self.global_state = {}
        self.superparam = superparameter
    def run(self, local_models, node):
        raise NotImplementedError
    def all_neighbor_models(self, local_models, node):
        return local_models[self.graph.neighbors[node]]
    def neighbor_models_and_itself(self, local_model, node):
        li = list(self.graph.neighbors[node]) + [node]
        return local_model[li]
    def __str__(self) -> str:
        return f'name[{self.name}]'

class D_bulyan(DecentralizedAggregation):
    def __init__(self, graph):
        self.byzantine_sizes = graph.byzantine_sizes
        super(D_bulyan, self).__init__(name='Bulyan', graph=graph)
    def run(self, local_models, node):
        local_model = local_models[node]
        agg = bulyan(local_model, byzantine_size=self.byzantine_sizes[node])
        return agg
    
class D_mean(DecentralizedAggregation):
    def __init__(self, graph):
        super(D_mean, self).__init__(name='mean', graph=graph)
    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return neighbor_models.mean(axis=0)
        
"""  

class D_performance_checking(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='Performance_Checking', graph=graph)

    @torch.no_grad()
    def get_accuracy(self, model, dataset, alpha=0.1):
        #print (f'alpha = {alpha}')
        samples_x, samples_y = dataset.features, dataset.targets
        sample_len = len(samples_x)
        random_indices = torch.randint(low=0, high=sample_len, size=(int(sample_len * alpha),))
        samples_x, samples_y = samples_x[random_indices], samples_y[random_indices]
        #print (f' type data = {type(samples_x)}, len = {len}, {samples_x[random_indices]}')

        predicted_y = model(samples_x)
        _, prediction_cls = torch.max(predicted_y.detach(), dim=1)
        accuracy = (prediction_cls == samples_y).sum().item()
        return accuracy /  len(samples_y) * 1.
    
    @torch.no_grad()
    def run(self, local_models, node, d_dataset=None, alpha=0.1):
        local_model = local_models[node]
        local_dataset = d_dataset[node] # sample 10% of it to test
        performance_list = [self.get_accuracy(local_model, local_dataset, alpha=alpha)]
        performance_list_index = [node]
        for nei_node in self.graph.neighbors[node]:
            performance_list.append(self.get_accuracy(local_models[nei_node], local_dataset, alpha=alpha))
            performance_list_index.append(nei_node)
        #:wlog.info(f'local perforamce {performance_list}, node = {node}, neighbors = {self.graph.neighbors[node]}')
        
        average_performance = sum(performance_list) / len(performance_list)
        aggregation_list = [node]
        for i in range(1, len(performance_list)):
            if performance_list[i] >= average_performance:
                aggregation_list.append(performance_list_index[i])
        local_models_vec = local_models.to_vectors()
        #log.info(f'aggregation list = {aggregation_list}')
        return local_models_vec[aggregation_list].mean(axis=0)
