#!/bin/bash
read -p "Enter GPU_ID: " gpu_id
# IDs=(0 1 2 3 4 6 8 9)
IDs=(3)

for id in "${IDs[@]}";do
    # python main_T1T2QSM_GE_1iso.py --gpu_id=$gpu_id --K=10 --loupe=-1 --mc_fusion=0 --flag_train=0 --test_sub=$id
    # python main_T1T2QSM_GE_1iso.py --gpu_id=$gpu_id --K=10 --loupe=-2 --mc_fusion=0 --flag_train=0 --test_sub=$id
    python main_T1T2QSM_GE_1iso.py --gpu_id=$gpu_id --K=10 --loupe=-1 --mc_fusion=1 --flag_train=0 --test_sub=$id
    python main_T1T2QSM_GE_1iso.py --gpu_id=$gpu_id --K=10 --loupe=-2 --mc_fusion=1 --flag_train=0 --test_sub=$id
    done
