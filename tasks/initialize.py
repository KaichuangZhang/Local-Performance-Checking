'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-22 22:51:49
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-24 15:16:38
FilePath: /mycode/my_research/DFLandCFL_difference/tasks/initialize.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils.tools import torch_rng
import torch
from utils.tools import FEATURE_TYPE

class ZeroInitialize():
    def __call__(self, model, fix_init_model=False, seed=100):
        for param in model.parameters():
            param.data.zero_()
            param.data.add_(1)
    

class RandomInitialize():
    def __init__(self, scale=6):
        self.scale = scale
    def __call__(self, model, fix_init_model=False, seed=100):
        rng = torch_rng(seed=seed) if fix_init_model else None
        for param in model.parameters():
            init_param = self.scale * torch.randn(
                param.size(), dtype=FEATURE_TYPE, generator=rng
            )
            param.data.copy_(init_param)