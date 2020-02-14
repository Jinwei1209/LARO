#!/bin/bash
read -p "Enter GPU_ID: " id

python main_test_pmask.py --gpu_id=$id --flag_solver=-3
python main_test_pmask.py --gpu_id=$id --flag_solver=-2
python main_test_pmask.py --gpu_id=$id --flag_solver=-1
python main_test_pmask.py --gpu_id=$id --flag_solver=0
python main_test_pmask.py --gpu_id=$id --flag_solver=1
python main_test_pmask.py --gpu_id=$id --flag_solver=2
