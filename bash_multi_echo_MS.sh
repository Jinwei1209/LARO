#!/bin/bash
read -p "Enter GPU_ID: " gpu_id
# IDs=(0 1 2)
IDs=(0 1 2 3 4 5 6 7 8 9 10)


for id in "${IDs[@]}";do
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=1 --solver=1
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=1 --solver=1 --K=30
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=1 --solver=1 --K=30

    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=1 --K=30
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=1 --K=30

    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=0 --solver=0
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=0 --K=30
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=0 --K=30

    python main_multi_echo_MS.py --flag_train=0 --test_ID=$id --K=30 --loupe=0 --bcrnn=0 --solver=1 --necho=1
    done 
