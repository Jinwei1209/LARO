#!/bin/bash
read -p "Enter GPU_ID: " gpu_id
# LOUPEs=(-1 0 -2)
LOUPEs=(-2)
Ratios=(0.1)
Nums_ft_subs=(8)
# for loupe in "${LOUPEs[@]}";do  # for the other generalization error test without fine-tuning
for num_ft_subs in "${Nums_ft_subs[@]}";do  # for 075 fine-tuning test
    # python main_multi_echo_GE.py --flag_train=0 --test_sub=5 --loupe=$loupe --prosp=0  # for FA=25
    # python main_multi_echo_GE.py --flag_train=0 --test_sub=5 --loupe=$loupe --prosp=0 --necho=7  # for Necho=7
    # python main_multi_echo_GE.py --flag_train=0 --test_sub=5 --loupe=$loupe --prosp=0 --scanner=1 --normalization=1 --necho=10  # Siemens
    # python main_multi_echo_GE_train_CBIC_test_DHK.py --flag_train=0 --test_sub=5 --loupe=$loupe --prosp=0  # for DHK
    # python main_multi_echo_GE_train_1_test_075.py --flag_train=0 --test_sub=5 --loupe=$loupe --prosp=0 --samplingRatio=0.1  # for voxel size 0.75
    python main_multi_echo_GE_train_1_fine_tune_075.py --flag_train=0 --test_sub=5 --loupe=-2 --prosp=0 --samplingRatio=0.1  --num_ft_subs=$num_ft_subs  # for voxel size 0.75, fine-tuned on differen number of 0.75 subs
    done
