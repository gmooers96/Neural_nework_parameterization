#!/bin/bash

#SBATCH -A ees220005p
#SBATCH --job-name="gpu"
#SBATCH -o "outputs/gpu.%j.%N.out"
#SBATCH -p GPU-shared #could do GPU
#SBATCH --gpus=v100-16:1
#SBATCH -N 1
#SBATCH --export=ALL
#SBATCH -t 5:00:00 # max of 48 hours for GPU
#SBATCH --no-requeue

module purge

source /jet/home/gmooers/miniconda3/bin/activate CPU

cd /ocean/projects/ees220005p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/

python3 ml_train_nn_zarr_yaml.py /ocean/projects/ees220005p/gmooers/Githubs/Neural_nework_parameterization/NN_training/run_training/Improved_run_Experiments/Config_Files/config_100.yaml 