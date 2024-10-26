'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 13:58:41
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-05-05 18:49:57
FilePath: /Research/mycode/my_research/DFLandFL/tasks/softmaxRegression.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
from functools import partial

import torch

from tasks.initialize import RandomInitialize
from utils.tools import adapt_model_type
from tasks.task import Task

class softmaxRegression_model(torch.nn.Module):
    def __init__(self, feature_dimension, num_classes):
        super(softmaxRegression_model, self).__init__()
        self.linear = torch.nn.Linear(in_features=feature_dimension,
                                      out_features=num_classes, bias=True)
    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.linear(features)

def softmax_regression_loss(predictions, targets):
    loss = torch.nn.functional.cross_entropy(
                predictions, targets.type(torch.long).view(-1))
    return loss

def softmax_regression_loss_v2(predictions, targets, p = 0):
    loss = torch.nn.functional.cross_entropy(
                predictions, targets.type(torch.long).view(-1))
    loss = 1. / (p + 1) * (loss ** (p + 1))
    return loss

def random_generator(dataset, batch_size=1, rng=random):
    while True:
        beg = rng.randint(0, len(dataset)-1)
        if beg+batch_size <= len(dataset):
            yield dataset[beg:beg+batch_size]
        else:
            features, targets = zip(dataset[beg:beg+batch_size],
                                    dataset[0:(beg+batch_size) % len(dataset)])
            yield torch.cat(features), torch.cat(targets)
        
def order_generator(dataset, batch_size=1, rng=random):
    beg = 0
    while beg < len(dataset):
        end = min(beg+batch_size, len(dataset))
        yield dataset[beg:end]
        beg += batch_size
    if end == len(dataset):
        end = 0
        
        
def full_generator(dataset, rng=random):
    while True:
        yield dataset[:]

class softmaxRegressionTask(Task):
    def __init__(self, dataset, batch_size=32):
        weight_decay = 0.01
        model = softmaxRegression_model(dataset.feature_dimension,
                                        dataset.num_classes)
        model = adapt_model_type(model)
        #loss_fn = softmax_regression_loss
        loss_fn = partial(softmax_regression_loss_v2)
        
        super_params = {
            'rounds': 100,
            'display_interval': 100,
            'batch_size': batch_size,
            'val_batch_size': 100,
            'lr': 9e-1,
        }
        get_train_iter = partial(random_generator,
                                 batch_size=super_params['batch_size'])
        get_val_iter = partial(order_generator,
                                 batch_size=super_params['val_batch_size']) # validation why dataset = dataset??
        super().__init__(weight_decay, dataset, model, loss_fn,
                         initialize_fn=RandomInitialize(),
                         get_train_iter=get_train_iter,
                         get_val_iter=get_val_iter,
                         super_params=super_params,
                         name=f'SR_{dataset.name}',
                         model_name='softmaxRegression')
     