#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v6
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 '/opt/python3/venv/base/train.py' --resume_training --model 'UnetResNet' --dataset 'cifar10' --save_and_sample_every 999999 --time_steps 50 --train_steps 10000 --sampling_routine x0_step_down --snow_level 1 --random_snow --dataset_folder '/users/ahmet.susuz/cifar_10_train_dataset/clean' --save_folder '/users/ahmet.susuz/train_results/3_29_50ts' --forward_process_type 'FGSM' --adv_model_path '/users/ahmet.susuz/resnet18.pt'
