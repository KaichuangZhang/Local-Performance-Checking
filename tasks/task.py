'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 13:58:41
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-03-27 23:44:59
FilePath: /Research/mycode/my_research/DFLandFL/tasks/task.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
class Task:
    def __init__(self, weight_decay, dataset, model, loss_fn,
                 get_train_iter=None, get_val_iter=None,
                 initialize_fn=None, super_params={},
                 name='', model_name=''):
        if model_name == '':
            model_name = model._get_name()
        if name == '':
            name = model_name + '_' + dataset.name
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.initialize_fn = initialize_fn
        self.get_train_iter = get_train_iter
        self.get_val_iter = get_val_iter
        self.super_params = super_params
        self.name = name
        self.model_name = model_name

    def __str__(self):
        return f'name [{self.name}] - model_name[{self.model_name}]'