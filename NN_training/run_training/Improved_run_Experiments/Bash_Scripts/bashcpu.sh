#!/bin/bash

#SBATCH -A ees220005p
#SBATCH --job-name="cpu"
#SBATCH -o "outputs/cpu.%j.%N.out"
#SBATCH -p RM #could do RM-512, RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH -t 10:00:00 # max of 48 hours for GPU
#SBATCH --mem=200G
#SBATCH --no-requeue

module purge

source /jet/home/gmooers/miniconda3/bin/activate CPU

cd /ocean/projects/ees220005p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/

python3 ml_train_nn_zarr_yaml.py /ocean/projects/ees220005p/gmooers/Githubs/Neural_nework_parameterization/NN_training/run_training/Improved_run_Experiments/Config_Files/config_1.yaml 