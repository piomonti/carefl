#!/bin/bash
#
#SBATCH --job-name=carefl_cpu
#SBATCH --output=slurm_log/carefl_cpu.%A.out
#SBATCH --error=slurm_log/carefl_cpu.%A.err
#
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --time=0-12:00
#

if [ -z ${SLURM_ARRAY_TASK_ID} ]
then
        echo "SLURM array variable not set"
        python main.py "$@"
else
        echo "SLURM array variable is set"
        echo "${SLURM_ARRAY_TASK_ID}"
        python main.py --seed ${SLURM_ARRAY_TASK_ID} --n-sims 1 "$@"
fi

