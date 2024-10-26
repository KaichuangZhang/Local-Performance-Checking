import torch.nn as nn
import torch.nn.functional as F
import torch
from tasks.task import Task
from tasks.initialize import RandomInitialize
from utils.tools import adapt_model_type
from functools import partial
import random

class DNN_femnist(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def softmax_regression_loss(predictions, targets):
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
    if end == len(dataset):
        end = 0

class FEMNISTCNNTask(Task):
    def __init__(self, dataset, batch_size=64):
        weight_decay = 0.01
        model = DNN_femnist()
        model = adapt_model_type(model)
        #loss_fn = softmax_regression_loss
        loss_fn = partial(softmax_regression_loss)
        
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
                         model_name='FEMNISTCNNTask')
