#!/bin/bash
read -p "Enter GPU_ID: " gpu_id
# IDs=(0 1 2)  # (0 1 2)
IDs=(0 1 2 3 4 5 6 7 8 9)
Ratios=(0.1)   # (0.1 0.15 0.2)

for id in "${IDs[@]}";do
    
    # # ablation
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1  --flag_unet=1 --prosp=1
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1 --flag_unet=1 --prosp=1
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1 --flag_unet=1 --prosp=1
    # # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-3  --bcrnn=1 --solver=1 --K=10 --samplingRatio=0.1 --flag_unet=1
    
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=1 --K=10 --samplingRatio=0.1  --flag_unet=1
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=1 --K=10 --samplingRatio=0.1  --flag_unet=1

    # MoDL
    python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-2 --bcrnn=0 --solver=0 --K=15 --samplingRatio=0.1 --flag_unet=0 --prosp=1
    python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=-1 --bcrnn=0 --solver=0 --K=15 --samplingRatio=0.1 --flag_unet=0 --prosp=1 # bcrnn=0_loss=0_K=15_loupe=-1_ratio=0.1_solver=0_unet=0_last.pt for ID=1
    python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=0  --bcrnn=0 --solver=0 --K=15 --samplingRatio=0.1 --flag_unet=0 --prosp=1

    # # for LLR recon
    # python main_multi_echo_GE.py --gpu_id=$gpu_id --flag_train=0 --test_sub=$id --loupe=1 --bcrnn=0 --solver=0 --K=2 --samplingRatio=0.1  --flag_unet=0
    done
