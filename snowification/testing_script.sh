#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/test.py' --model 'UnetResNet' --dataset 'cifar10' --time_steps 50 --test_type 'test_data' --extra_path 'trial' --save_folder_test '/users/harun.ergen/first_trial' --save_folder_train '/users/harun.ergen/first_trial' --exp_name 'trial' --forward_process_type 'Snow'
