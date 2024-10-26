
###
 # @Author: zhangkaichuang zhangkaichuang2022@163.com
 # @Date: 2024-02-21 09:17:11
 # @LastEditors: zhangkaichuang zhangkaichuang2022@163.com
 # @LastEditTime: 2024-07-11 23:57:58
 # @FilePath: /Research/mycode/my_research/DFLandCFL_difference/run.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

task_type="DFL"
network="ER"
network_sparsity=0.7
dataset="FMNIST"
data_partition="LabelSkew"
task="softmax_regression"
attack="gaussian"
aggregation="D_performance_checking"
#aggregation="scc"
lr_s="1_sqrt_k"
honest_size=10
byzantine_size=0
local_rounds=4
alpha=1.0
seed=44
python=/usr/bin/python3
$python main.py \
    --task-type=${task_type} \
    --network=${network} \
    --network-sparsity=${network_sparsity} \
    --dataset=${dataset} \
    --dataset-partition=${data_partition} \
    --task=${task} \
    --attack=${attack} \
    --learning-rate-scheduler="${lr_s}" \
    --aggregation=${aggregation} \
    --alpha=${alpha} \
    --honest-size=${honest_size} \
    --byzantine-size=${byzantine_size} \
    --local-rounds=${local_rounds} \
    --seed=${seed}
    #\
    #--log-file-path="./experiments/performance_checking/${task_type}_${task}_${network}_${network_sparsity}_${dataset}_${data_partition}_${attack}_${aggregation}_${local_rounds}"
