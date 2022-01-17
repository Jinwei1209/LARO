#!/bin/bash
read -p "Enter GPU_ID: " gpu_id
IDs=(-2)  # (0 1 2 3)


for id in "${IDs[@]}";do

    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=0 --solver=0 --K=30 --samplingRatio=0.23  --flag_unet=0
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=0 --K=30 --samplingRatio=0.23  --flag_unet=0
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=0 --K=30 --samplingRatio=0.23  --flag_unet=0  # for sub=-2, use loupe=-1 to recon loupe=0

    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=1 --solver=1 --K=30 --samplingRatio=0.23  --flag_unet=1
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=1 --solver=1 --K=30 --samplingRatio=0.23 --flag_unet=1
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2  --bcrnn=1 --solver=1 --K=30 --samplingRatio=0.23 --flag_unet=1

    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=1 --K=30 --samplingRatio=0.23  --flag_unet=1
    # python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=1 --K=30 --samplingRatio=0.23  --flag_unet=1

    # for LLR recon
    python main_multi_echo_MS.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=1 --bcrnn=0 --solver=0 --K=2 --samplingRatio=0.23  --flag_unet=0
    done 
