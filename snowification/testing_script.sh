#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/test.py' --model 'UnetResNet' --dataset 'cifar10' --time_steps 50 --test_type 'test_data' --dataset_folder '/users/ahmet.susuz/cifar_10_test_dataset/clean' --save_folder_test '/users/ahmet.susuz/test_results/2_26_130k_clean' --save_folder_train '/users/ahmet.susuz/train_results/130k_epoch_50ts' --forward_process_type 'Snow'
