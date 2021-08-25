#!/bin/bash
read -p "Enter GPU_ID: " gpu_id
IDs=(0 1 2)
Ratios=(0.1)   # (0.1 0.15 0.2)

for id in "${IDs[@]}";do
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=1 --solver=1
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1
    
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.15
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.15
    
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.2
    python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1 --flag_unet=1
    python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1 --flag_unet=1
    
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=1 --K=10
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=1 --K=10

    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=0 --solver=0
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=0 --K=10 --prosp=1
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=0 --K=15 --prosp=1
    done
