#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/train.py' --resume_training --model 'UnetResNet' --dataset 'cifar10' --save_and_sample_every 999999 --time_steps 200 --train_steps 30500 --sampling_routine x0_step_down --snow_level 1 --random_snow --dataset_folder '/users/harun.ergen/cifar_10_train_dataset/clean' --save_folder '/users/harun.ergen/train_results/2_27_200ts' --forward_process_type 'Snow'
