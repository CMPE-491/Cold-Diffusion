#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/april-image:v1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/helpers/create_grads_fgsm.py' --root '/users/harun.ergen/cifar_10_train_grads' --model_path '/users/harun.ergen/resnet18.pt' --data_type 'train' --batch_size 32