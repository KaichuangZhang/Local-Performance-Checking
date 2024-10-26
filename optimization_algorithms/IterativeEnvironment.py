'''
Author: zhangkaichuang zhangkaichuang2022@163.com
Date: 2024-02-24 16:24:58
LastEditors: zhangkaichuang zhangkaichuang2022@163.com
LastEditTime: 2024-02-24 16:25:00
FilePath: /mycode/my_research/DFLandFL/optimization_algorithms/IterativeEnvironment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
class IterativeEnvironment():
    def __init__(self, name, lr, lr_ctrl=None,
                 rounds=10, display_interval=1000, total_iterations=None,
                 seed=None, fix_seed=False,
                 *args, **kw):
        '''
        display_interval: algorithm record the running information (e.g. loss,
                         accuracy) every 'display_interval' iterations
        rounds: totally 'rounds' information data will be records
        total_iterations: total iterations. We need to specify at least two of
                            arguments 'display_interval', 'rounds' and 
                            'total_iterations'.
                            'total_iterations' has to satisfy 
                            display_interval * rounds = total_iterations
        '''
        
        assert not fix_seed or seed != None
    
        # algorithm information
        self.name = name
        self.lr = lr
        self.seed = seed
        self.fix_seed = fix_seed
        
        # determine the runing time
        if rounds == None:
            rounds = total_iterations // display_interval
        elif display_interval == None:
            display_interval = total_iterations // rounds
        elif total_iterations == None:
            total_iterations = display_interval * rounds
        else:
            assert total_iterations == display_interval * rounds
        self.rounds = rounds
        self.display_interval = display_interval
        self.total_iterations = total_iterations
        
        # random number generator
        self.rng = random
        self.torch_rng = torch.default_generator
        
        # learning rate controller
        if lr_ctrl is None:
            self.lr_ctrl = constant_lr()
        else:
            self.lr_ctrl = lr_ctrl
        self.lr_ctrl.set_init_lr(lr)
            
    def construct_rng(self):
        # construct random number generator
        if self.fix_seed:
            self.rng = random_rng(self.seed)
            self.torch_rng = torch_rng(self.seed)
            
    def lr_path(self):
        return [self.lr_ctrl.get_lr(r * self.display_interval) 
                for r in range(self.rounds)]
        
    def set_params_suffix(self, params_show_names):
        # add suffix of parameters that is being tuned like: 
        # SGD_lr_0.01_momentum_0.9
        for params_code_name in params_show_names:
            params_value = self.__getattribute__(params_code_name)
            params_show_name = params_show_names[params_code_name]
            if params_show_name != '':
                self.name += f'_{params_show_name}={params_value}'
            else:
                self.name += f'_{params_value}'