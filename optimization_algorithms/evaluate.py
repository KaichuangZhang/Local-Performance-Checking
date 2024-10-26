'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-03-31 17:32:32
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-03-31 21:31:08
FilePath: /Research/mycode/my_research/DFLandFL/optimization_algorithms/evaluate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import logging as log

@torch.no_grad()
def test_by_nodes(d_models, d_test_dataset, data_test_iters, honest_nodes):
    for node in honest_nodes:
        d_models.active_model(node)
        model = d_models.workmodel
        dataset = d_test_dataset[node]
        for samples_x, samples_y in next(data_test_iters[node]):
            pass



    loss = 0
    accuracy = 0
    total_sample = 0
    #dist_models.update_params_torch()
    # evaluation
    # dist_models.activate_avg_model()
    # TODO: debug
    if node_list is None:
        node_list = range(dist_models.node_size)
    # dist_model.activate_avg_model(node_list=node_list)
    #for node in node_list:
    for index in range(len(node_list) + 1):
        if index == len(node_list):
            node = -1
            dist_models.activate_avg_model(node_list=node_list)
        else:
            node = node_list[index]
            #print (f'--->active model node {node}')
            dist_models.activate_model(node)
        model = dist_models.model
        #print (model)
        #print (f'node = {node}, paramters = {dist_models.params_vec[node]}')
        #print (f'node = {node}')
        #for parameters in model.parameters():
        #    print(parameters)
        local_model_loss = 0
        local_model_accuracy = 0
        local_samples = 0
        for features, targets in val_iter():
            #print (f"data dtype {features.dtype}, target dtype {targets.dtype}")
            model = model.to(use_device)
            predictions = model(features)
            #print (f"predict {predictions} predict shape {predictions.shape}, target {targets} target.shape {targets.shape}")
            temp_loss = loss_fn(predictions, targets, train=False).item()
            loss += temp_loss
            local_model_loss += temp_loss
            _, prediction_cls = torch.max(predictions.detach(), dim=1)
            temp_accuracy = (prediction_cls == targets).sum().item()
            accuracy += temp_accuracy
            local_model_accuracy += temp_accuracy
            local_samples_temp = len(targets)
            total_sample += local_samples_temp
            local_samples += local_samples_temp
        print (f'node {node}, local samples {local_samples}')
        print (f'node {node} : loss = {local_model_loss / local_samples}, accuracy = {local_model_accuracy / local_samples}')
    penalization = 0
    for param in model.parameters():
        penalization += weight_decay * param.norm()**2 / 2
    loss_avg = loss / total_sample + penalization
    accuracy_avg = accuracy / total_sample

    return loss_avg, accuracy_avg