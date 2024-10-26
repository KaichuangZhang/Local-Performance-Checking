'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-21 09:16:02
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-24 15:16:03
FilePath: /Research/mycode/my_research/DFLandCFL_difference/utils/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
import torch

TASK_TYPE =  ["DFL", "FL"]
RANDOM_SEED = 19930407

FEATURE_TYPE = torch.float64
CLASS_TYPE = torch.uint8
VALUE_TYPE = torch.float64


def random_rng(seed=10):
    return random.Random(seed)

def torch_rng(seed=10):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def adapt_model_type(model):
    if FEATURE_TYPE == torch.float64:
        return model.double()
    elif FEATURE_TYPE == torch.float32:
        return model.float()
    elif FEATURE_TYPE == torch.float16:
        return model.half()
