#!/bin/bash
#SBATCH --container-image ghcr.io\#cmpe-491/first-image:v5
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G

source /opt/python3/venv/base/bin/activate
python3 -c 'import torch; print("Torch Version:", torch.__version__)'
