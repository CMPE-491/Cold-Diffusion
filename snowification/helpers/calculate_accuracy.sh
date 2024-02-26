#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/helpers/calculate_accuracy.py' --folder_path '/users/harun.ergen/cifar_10_test_dataset/adv' --model_path '/users/harun.ergen/resnet18.pt'
