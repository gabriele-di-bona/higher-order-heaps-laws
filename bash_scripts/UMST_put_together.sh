#!/bin/bash
#$ -cwd
#$ -t 1-10
#$ -j y    
#$ -pe smp 1 
#$ -l h_vmem=50G 
#$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
# this bash script is supposed to be run from the subfolder outputs_put_together, so that the output goes directly there
cd ../../python_scripts/

python launch_UMST.py -ID ${SGE_TASK_ID} -rho 20 -eta 1 -starting_nu 1 -ending_nu 10 -Tmax 100000 -putTogether True -save_all False