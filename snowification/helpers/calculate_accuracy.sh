#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/april-image:v1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/helpers/calculate_accuracy.py' --folder_path '/users/harun.ergen/test_results/2_26_130k_clean/test_data/cleaned' --model_path '/users/harun.ergen/resnet18.pt' --dataset_folder '/users/harun.ergen/cifar_10_test_dataset/clean'
