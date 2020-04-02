#!/bin/bash
#SBATCH --job-name=RicardoScript
#SBATCH --output=RicardoScript.out
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000
#SBATCH --partition=cpu
cd ..
source seq_RNN_env/bin/activate
cd ./sequential_multiTask_RNN
python run_four_tasks.py
deactivate