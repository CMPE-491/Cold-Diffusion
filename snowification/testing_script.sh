#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/april-image:v1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/test.py' --model 'UnetResNet' --dataset 'cifar10' --time_steps 50 --test_type 'test_data' --dataset_folder '/users/harun.ergen/cifar_10_test_dataset/clean' --save_folder_test '/users/harun.ergen/test_results/2_26_130k_clean' --save_folder_train '/users/harun.ergen/train_results/130k_epoch_50ts' --grad_folder '/users/harun.ergen/cifar_10_test_grads' --forward_process_type 'FGSM'
