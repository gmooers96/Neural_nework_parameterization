#!/bin/bash

#SBATCH -A ees220005p
#SBATCH --job-name="P4"
#SBATCH -o "outputs/P4.%j.%N.out"
#SBATCH -p RM-512 #could do RM-512, RM-shared, EM, RM
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH -t 15:00:00 # max of 48 hours for GPU
#SBATCH --mem=498G
#SBATCH --no-requeue


module purge

source /jet/home/gmooers/miniconda3/bin/activate CPU

cd /ocean/projects/ees220005p/gmooers/Githubs/Neural_nework_parameterization/NN_training/run_training/Training_data_Generators/

python3 build_simple_big_data_all_timesteps_Part_4.py

