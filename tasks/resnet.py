'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 13:58:41
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-04-01 23:22:01
FilePath: /mycode/my_research/DFLandFL/tasks/resnet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
import torch
import torch.nn as nn
from functools import partial
from torchvision import models
from tasks.task import Task
from utils.tools import adapt_model_type
from tasks.initialize import RandomInitialize

#resnet18 = models.resnet18()



def softmaxwithloss(predictions, targets):
    # print (f" predictions shape {predictions.shape},targets shape {targets.shape} ")
    loss = torch.nn.functional.cross_entropy(
                predictions, targets.type(torch.long).view(-1))
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
        

class ResnetTask(Task):
    def __init__(self, dataset, batch_size=128):
        weight_decay = 5e-4
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = adapt_model_type(model)
        loss_fn = softmaxwithloss

        super_params = {
            'rounds': 150,
            'display_interval': 10,
            'batch_size': batch_size,
            'val_batch_size': 1000,
            
            'lr':  0.1,
        }
        get_train_iter = partial(random_generator,
                                 batch_size=super_params['batch_size'])
        get_val_iter = partial(order_generator,
                                 batch_size=super_params['val_batch_size'])
        super().__init__(weight_decay, dataset, model, loss_fn,
                         initialize_fn=RandomInitialize(),
                         get_train_iter=get_train_iter,
                         get_val_iter=get_val_iter,
                         super_params=super_params,
                         name=f'Resnet_{dataset.name}',
                         model_name='Resnet')
        