#!/bin/bash
#$ -cwd
#$ -t 1-1000
#$ -j y    
#$ -pe smp 1 
#$ -l h_vmem=10G
# #$ -l highmem
#$ -l h_rt=240:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
# this bash script is supposed to be run from the subfolder outputs, so that the output goes directly there
cd ../../python_scripts/

python launch_UMST.py -ID ${SGE_TASK_ID} -rho 20 -eta 1 -starting_nu 11 -ending_nu 20 -Tmax 100000 -putTogether False -save_all True