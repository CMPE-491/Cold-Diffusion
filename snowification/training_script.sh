#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v5
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/train.py' --model 'UnetResNet' --dataset 'cifar10' --save_and_sample_every 99999 --time_steps 50 --train_steps 120 --sampling_routine x0_step_down --snow_level 1 --random_snow --save_folder './snow_cifar10_50_time_step_25_12_pgdm'  --exp_name 'trial' --forward_process_type 'Snow'
