#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/helpers/create_data.py' --root '/users/ahmet.susuz/cifar_10_test_dataset' --model_path '/users/ahmet.susuz/cifar10_resnet18.pth' --data_type 'test' --batch_size 32 --add_adversarial
