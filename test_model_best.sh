#!/bin/bash
#SBATCH --job-name=LDM_test
#SBATCH --output=logs/test%j.out
#SBATCH --constraint=v100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=00:30:00

# Activate your conda environment
module load miniforge
conda activate mvae

export PROJECT_ROOT=/gpfswork/rech/krk/uqo89gi/projects/3DLatentDDPM/

CONFIG_PATH=/gpfswork/rech/krk/uqo89gi/projects/3DLatentDDPM
CONFIG_NAME=test_vp3d_collab

# Move to the project root if needed
cd /gpfswork/rech/krk/uqo89gi/projects/3DLatentDDPM/src

JOB_ID=$SLURM_ARRAY_TASK_ID
TIME_STARS=(0.7 0.9)
TIME_STAR=${TIME_STARS[$JOB_ID]}

# Run the Hydra training script
echo HYDRA_FULL_ERROR=1 python test.py --config-path $CONFIG_PATH --config-name $CONFIG_NAME predictor.predictor_config.time_star=$TIME_STAR

HYDRA_FULL_ERROR=1 python test.py --config-path $CONFIG_PATH --config-name $CONFIG_NAME predictor.predictor_config.time_star=$TIME_STAR
