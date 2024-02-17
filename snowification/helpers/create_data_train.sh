#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/helpers/create_data.py' --root './cifar_10_train_dataset' --data_type 'train' --batch_size 32 --add_adversarial
