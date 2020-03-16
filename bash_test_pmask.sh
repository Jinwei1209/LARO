#!/bin/bash
read -p "Enter GPU_ID: " id
read -p "Enter Sampling Mask ID: " id_mask

python main_test_pmask.py --gpu_id=$id --flag_solver=-3 --K=10 --flag_fix=$id_mask
python main_test_pmask.py --gpu_id=$id --flag_solver=-1 --K=10 --flag_fix=$id_mask
python main_test_pmask.py --gpu_id=$id --flag_solver=0  --K=10 --flag_fix=$id_mask
python main_test_pmask.py --gpu_id=$id --flag_solver=2  --K=10 --flag_fix=$id_mask
