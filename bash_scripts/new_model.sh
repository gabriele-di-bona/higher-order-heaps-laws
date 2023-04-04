#!/bin/bash
#$ -cwd
#$ -t 1-121#00
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

python new_model.py -ID ${SGE_TASK_ID} \
    -rho 10 \
    -starting_nu_1 0 -ending_nu_1 10 \
    -starting_nu_2 0 -ending_nu_2 20 -Tmax 100000 \
    -fraction_nu_2_cut_nu_1 2 \
    -N_0 1 -M_0 0 \
    -directed f \
    -do_non_overlapping_simulation f \
    -trigger_links_with_replacement t \
    -triggering_links_among_all_non_explored_links f \
    -look_among_all_non_explored_links_if_not_enough f \
    -trigger_links_with_new_nodes_if_not_enough f \
    -trigger_links_between_new_nodes_if_not_enough f \
    -putTogether t \
    -save_all False -save_raw_urn False -save_raw_sequence False -delete_files_put_together t \
    -do_prints f \
    