#!/bin/bash
#$ -cwd
#$ -t 1-250 # Last.fm: 890 # Project Gutenberg 2: 19637 # Semantic Scholar 3: 19000 # UMST 4: 2000 # ERRW 5: 150
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

python compute_entropies.py -ID ${SGE_TASK_ID} -number_reshuffles 10 -dataset_ID 5 -order False -calc_entropy_pairs_on_reshuffled_singletons False