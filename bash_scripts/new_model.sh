#!/bin/bash
#$ -cwd
#$ -t 1-441#00
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
    -starting_nu_1 0 -ending_nu_1 20 \
    -starting_nu_2 0 -ending_nu_2 40 -Tmax 100000 \
    -fraction_nu_2_cut_nu_1 2 \
    -N_0 10 -M_0 0 \
    -directed f \
    -do_non_overlapping_simulation f \
    -trigger_links_with_replacement t \
    -triggering_links_among_all_non_explored_links f \
    -putTogether f \
    -save_all True -save_raw_urn True -save_raw_sequence True -delete_files_put_together f \
    -do_prints f \
    