#!/bin/bash
#$ -cwd
#$ -t 1-2100 
#$ -j y    
#$ -pe smp 1 
#$ -l h_vmem=10G
# #$ -l highmem
#$ -l h_rt=24:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
# this bash script is supposed to be run from the subfolder outputs, so that the output goes directly there
cd ../../python_scripts/

python ERRW_SW.py \
    -ID ${SGE_TASK_ID} \
    -save_all False \
    -p 0.1 \
    -k 4 \
    -N 1000000 \
    -starting_dw "-1" \
    -ending_dw 1 \
    -num_dw 21 \
    -use_logspace True \
    -use_special_space False \
    -Tmax 100000 \
    -folder_ERRW SW \
