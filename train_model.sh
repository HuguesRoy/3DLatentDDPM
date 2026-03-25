#!/bin/bash
#SBATCH --job-name=LDM_train
#SBATCH --output=logs/train%j.out
#SBATCH --constraint=v100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=00:10:00

# Activate your conda environment
module load miniforge
conda activate mvae

export PROJECT_ROOT=/gpfswork/rech/krk/uqo89gi/projects/3DLatentDDPM/

CONFIG_PATH=/gpfswork/rech/krk/uqo89gi/projects/3DLatentDDPM
CONFIG_NAME=train_vp

# Move to the project root if needed
cd /gpfswork/rech/krk/uqo89gi/projects/3DLatentDDPM/src

# Run the Hydra training script
HYDRA_FULL_ERROR=1 python train.py --config-path $CONFIG_PATH --config-name $CONFIG_NAME split=1
