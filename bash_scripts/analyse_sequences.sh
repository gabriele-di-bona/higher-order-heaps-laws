#!/bin/bash
#$ -cwd
#$ -t 1-890 # Last.fm: 890 # Project Gutenberg 2: 19637 # Semantic Scholar 3: 19000 # UMST 4: 2000 # ERRW 5: 150 # dir_ERRW 6: 16
#$ -j y    
#$ -pe smp 1 
#$ -l h_vmem=1G
# #$ -l highmem
#$ -l h_rt=1:0:0
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
# this bash script is supposed to be run from the subfolder outputs, so that the output goes directly there
cd ../../python_scripts/

python analyse_sequences.py -ID ${SGE_TASK_ID} -number_reshuffles 10 -dataset_ID 1 \
    --consider_temporal_order_in_tuples True --analyse_sequence_labels False \
    -save_all True \
    -rho 20 -starting_nu 1 -ending_nu 20 -eta 1 -Tmax 100000 \
    --folder_name_ERRW "SW_k4"
