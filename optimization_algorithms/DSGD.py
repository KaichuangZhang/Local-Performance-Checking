import random
import torch
import logging as log
from datasets.distributedDatasets import DistributedDataSets
from optimization_algorithms.DecentralizedByzantineEnvironment import DecentralizedByzantineEnvironment
from optimization_algorithms.DecentralizedMoldes import DecentralizedModels
from utils.tools import RANDOM_SEED
from optimization_algorithms.evaluate import test_by_nodes
import time

class DecentralizedSGD():
    def __init__(self,
                 graph,
                 task,
                 data_partition,
                 local_rounds,
                 LR_scheduler,
                 aggregation,
                 attack,
                 agg_alpha=0.1,
                 consensus_init=False,
                 seed=RANDOM_SEED,
                 device='cuda:0',
                 *args, **kw):
        self.graph = graph
        print (f'graph honest nodes = {self.graph.honest_nodes}')
        
        # task = model + dataset
        self.task = task
        self.model = task.model
        self.dataset = task.dataset
        self.data_partition = data_partition
        self.train_iter = task.get_train_iter
        self.test_iter = task.get_val_iter
        self.local_rounds = local_rounds
        self.LR_scheduler = LR_scheduler
        self.LR_scheduler.set_init_lr(self.task.super_params['lr'])
        self.rounds = self.task.super_params['rounds']
        self.display_interval = self.task.super_params['display_interval']
        self.loss_fn = self.task.loss_fn
        
        # aggragation
        self.aggregation = aggregation
        self.agg_alpha = agg_alpha
        # attack
        self.attack = attack

        # others
        self.seed = seed
        random.seed(RANDOM_SEED)
        self.rand_g = random
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def run(self):
        # models
        self.d_models = DecentralizedModels(self.model, self.graph.node_size, device=self.device)
        self.d_models.initilize_parameters(generator=self.rand_g)
        # datasets
        self.d_dataset = DistributedDataSets(dataset=self.dataset, 
                                           partition_cls=self.data_partition,
                                           nodes=self.graph.nodes,
                                           honest_nodes=self.graph.honest_nodes)
        # test dataset
        #self.test_dataset = self.dataset.get_val_set()
        self.d_test_dataset = DistributedDataSets(dataset=self.dataset.get_val_set(),
                                                  partition_cls=self.data_partition,
                                                  nodes=self.graph.nodes,
                                                  honest_nodes=self.graph.honest_nodes) 

        log.info(self.d_dataset)
        log.info(self.d_test_dataset)
        #log.info(self.test_dataset)
        
        # iteration for local datasets
        self.data_train_iters = []
        self.data_test_iters = []
        for node in self.graph.nodes:
            if node in self.graph.honest_nodes:
                train_iter = self.train_iter(self.d_dataset[node], rng=self.rand_g)
                # test_iter = self.test_iter(self.d_test_dataset[node], rng=self.rand_g)
                self.data_train_iters.append(train_iter)
                # self.data_test_iters.append(test_iter)
            else:
                self.data_train_iters.append(None)
                # self.data_test_iters.append(None)

        # training
        total_iterations = self.rounds * self.display_interval
        total_time = 0
        for i in range(total_iterations + 1):
            if i % self.display_interval == 0:
                log.info(f"""iteration[{i}/{total_iterations}({i/total_iterations})] - learning rate[{self.LR_scheduler.get_lr(i)}] - {self.test_by_nodes()}""")
            for node in self.graph.honest_nodes:
                self.d_models.active_model(node)
                model = self.d_models.workmodel
                for _ in range(self.local_rounds):
                    samples_x, samples_y = next(self.data_train_iters[node])
                    predicted_y = model(samples_x)
                    loss = self.loss_fn(predicted_y, samples_y)
                    model.zero_grad()
                    loss.backward()
                    
                    # backward
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.grad is not None:
                                param.data.mul_(1 - self.task.weight_decay * self.LR_scheduler.get_lr(i))
                                param.data.sub_(param.grad, alpha=self.LR_scheduler.get_lr(i))

            # aggregation
            d_models_vector = self.d_models.to_vectors()
            for node in self.graph.honest_nodes:
                # Byzantine attack
                byzantine_neighbors_size = self.graph.byzantine_sizes[node]
                if self.attack != None and byzantine_neighbors_size != 0:
                    self.attack.run(d_models_vector, node)
                # aggregation
                #gaussian_noise = torch.randn_like(param_bf_comm) * lr * self.noise_level
                if i == 0:
                    log.info(f'node[{node}] - neighbors{self.graph.neighbors[node]} byzantine neighbors{self.graph.byzantine_neighbors[node]}')
                # update this node
                if self.aggregation.name == 'Performance_Checking':
                    #log.info(f'update honest node {node}.')
                    self.d_models.update_model(d_models_vector[node], node)
                    # update the byzantine nodes
                    for byzantine_node in self.graph.byzantine_neighbors[node]:
                        #log.info(f'update byzantine node {byzantine_node}.')
                        self.d_models.update_model(d_models_vector[byzantine_node], byzantine_node)
                    start_time = time.time()
                    aggregation_res = self.aggregation.run(self.d_models, node, d_dataset=self.d_dataset, alpha=self.agg_alpha)
                    end_time = end_time = time.time()
                    total_time += end_time - start_time
                else:
                    aggregation_res = self.aggregation.run(d_models_vector, node)
                self.d_models.update_model(aggregation_res, node)
        log.info(f'Elapsed time: {total_time / total_iterations} seconds')

    @torch.no_grad()
    def test_by_nodes(self):
        test_info = {
            'average loss': 0,
            'average accuracy': 0,
            'total samples': 0,
            'test accuracy': [],
        }
        for node in self.graph.honest_nodes:
            self.d_models.active_model(node)
            model = self.d_models.workmodel
            dataset =  self.d_test_dataset[node]
            #dataset = self.test_dataset
            samples_x, samples_y = dataset.features, dataset.targets
            predicted_y = model(samples_x)
            # loss
            loss = self.loss_fn(predicted_y, samples_y).item()
            _, prediction_cls = torch.max(predicted_y.detach(), dim=1)
            accuracy = (prediction_cls == samples_y).sum().item()
            accuracy_ratio = accuracy / len(samples_y) * 1.
            if loss < 1e-5: loss = 0.0
            if node not in test_info:
                test_info[node] = {
                    'node': node,
                    'accuracy': accuracy_ratio,
                    'loss': loss,
                }
            test_info['average accuracy'] += accuracy
            test_info['average loss'] += loss
            test_info['total samples'] += len(samples_y)
            test_info['test accuracy'].append(accuracy / len(samples_y) * 100)
        test_info['average accuracy'] /= test_info['total samples']
        avg_loss = test_info['average loss'] / self.graph.honest_size
        test_info['average loss'] = avg_loss
        test_info['test accuracy variance'] = torch.var(torch.tensor(test_info['test accuracy']), unbiased=False).item()
        return test_info