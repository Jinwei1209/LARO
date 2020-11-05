#!/bin/bash
which python
python main_multi_echo_GE.py --gpu_id=0 --flag_train=0 --echo_cat=1 --solver=1 --K=10 --loupe=-1 --samplingRatio=0.2
python main_multi_echo_GE.py --gpu_id=0 --flag_train=0 --echo_cat=1 --solver=1 --K=10 --loupe=0 --samplingRatio=0.2
python main_multi_echo_GE.py --gpu_id=0 --flag_train=0 --echo_cat=1 --solver=1 --K=10 --loupe=-1 --samplingRatio=0.15
python main_multi_echo_GE.py --gpu_id=0 --flag_train=0 --echo_cat=1 --solver=1 --K=10 --loupe=0 --samplingRatio=0.15
python main_multi_echo_GE.py --gpu_id=0 --flag_train=0 --echo_cat=1 --solver=1 --K=10 --loupe=-1 --samplingRatio=0.1
python main_multi_echo_GE.py --gpu_id=0 --flag_train=0 --echo_cat=1 --solver=1 --K=10 --loupe=0 --samplingRatio=0.1
