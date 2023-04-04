import numpy as np
import pickle, joblib
import os
# Change directory to the root of the folder (this script was launched from the subfolder python_scripts)
# All utils presuppose that we are working from the root directory of the github folder
os.chdir("../")
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))
from datetime import datetime
import argparse
import glob
import fnmatch
import csv 
import gzip

# local utils
from analyse_higher_entropy import *
from analyse_sequence import *
from find_files_with_pattern import *    
    
    
# Beginning of MAIN
parser = argparse.ArgumentParser(description='Compute entropies on the sequences of labels in the data, both on singletons and pairs, and on randomized sequences.')


parser.add_argument("-ID", "--ID", type=int,
    help="The ID of the simulation, used to get the correct sequence [default 1]",
    default=1)

parser.add_argument("-number_reshuffles", "--number_reshuffles", type=int,
    help="Number of reshuffles to do check the randomized case of the entropy. [default 10]",
    default=10)

parser.add_argument("-dataset_ID", "--dataset_ID", type=int,
    help="The dataset to choose. 1 -> Last.fm, 2 -> Project Gutenberg, 3 -> Semantic Scholar, 4 -> UMST, 5 -> ERRW. [default 1]",
    default=1)

parser.add_argument("-order", "--consider_temporal_order_in_tuples", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If False, it considers AB and BA the same. [default True]",
    default=True)

parser.add_argument("-analyse_labels", "--analyse_sequence_labels", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If True, it considers the sequence of labels instead of the original sequence in all calculations. [default False]",
    default=False)

parser.add_argument("-save_all", "--save_all", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If True, it saves all results. [default True]",
    default=True)

# Parameters for UMST

parser.add_argument("-rho", "--rho", type=float,
    help="The reinforcement parameter. [default 10]",
    default=10)

parser.add_argument("-starting_nu", "--starting_nu", type=int,
    help="The initial triggering parameter (must be integer and positive). All nus will increase by one until ending_nu (extremes included). [default 1]",
    default=1)

parser.add_argument("-ending_nu", "--ending_nu", type=int,
    help="The ending triggering parameter (must be integer and positive). All nus will increase by one starting from starting_nu (extremes included). [default 20]",
    default=20)

parser.add_argument("-eta", "--eta", type=float,
    help="The semantic correlations parameters, must be between 0 and 1. [default 1]",
    default=1)

parser.add_argument("-Tmax", "--Tmax", type=int,
    help="The number of steps of the simulation. [default 100]",
    default=100)

# Parameter for ERRW

parser.add_argument("-folder_ERRW", "--folder_name_ERRW", type=str,
    help="Name of the subfolder in ~/data/ERRW/ to take the data from and do the analysis. [default 'SW']",
    default='SW')

arguments = parser.parse_args()
ID = arguments.ID - 1 # unfortunately I have to start this argument from 1 in shell. The -1 is added to start from 0
number_reshuffles = arguments.number_reshuffles
dataset_ID = arguments.dataset_ID
consider_temporal_order_in_tuples = arguments.consider_temporal_order_in_tuples
analyse_sequence_labels = arguments.analyse_sequence_labels
save_all = arguments.save_all
rho = arguments.rho
eta = arguments.eta
assert (0 <= eta  and eta <= 1), "eta is not within the correct bounds, with value %f"%eta
starting_nu = arguments.starting_nu
ending_nu = arguments.ending_nu
Tmax = arguments.Tmax
folder_name_ERRW = arguments.folder_name_ERRW

start = datetime.now()
    
    

# Get sequence of labels
if dataset_ID == 1: 
    # Last.fm
    data_folder = "./data/ocelma-dataset/lastfm-dataset-1K/"
    raw_folder = os.path.join(data_folder, "raw_sequences")
    if consider_temporal_order_in_tuples:
        analysis_folder = data_folder
    else:
        analysis_folder = os.path.join(data_folder,'analysis_tuples_without_order')
    # only consider those with at least 1000 elements
    with gzip.open(os.path.join(data_folder, "users_list_min_1000.pkl.gz"), "rb") as fp:
        users_list = joblib.load(fp)
    file_name = userid = users_list[ID]
    path = os.path.join(raw_folder,f"{file_name}.pkl.gz")
    print('Getting sequence from', path, flush=True)
    print('Getting sequence_labels from', path, flush=True)
    with gzip.open(path, "rb") as fp:
        user_dict = pickle.load(fp)
    sequence = user_dict['sequence_tracks']
    sequence_labels = user_dict['sequence_artists']
    if analyse_sequence_labels == True:
        sequence = sequence_labels
        save_all_file_path = os.path.join(analysis_folder, 'individual_results_artists', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'individual_results_artists_light', file_name + '.pkl')
    else:
        save_all_file_path = os.path.join(analysis_folder, 'individual_results', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'individual_results_light', file_name + '.pkl')
    save_entropies_file_path = os.path.join(analysis_folder, 'individual_results_entropy', file_name + '.pkl')
    
    
elif dataset_ID == 2: 
    # Project Gutenberg
    data_folder = './data/gutenberg/sequences_indexed/'
    analysis_folder = './data/gutenberg/analysis/'
    sequence_folder = os.path.join(data_folder,'sequences_words')
    sequence_labels_folder = os.path.join(data_folder,'sequences_stems')
    filepaths = sorted(find_pattern('*.pkl', sequence_folder))
    path = filepaths[ID]
    file_name = path[-path[::-1].index('/'):path.index('.pkl')]
    sequence_labels_path = os.path.join(sequence_labels_folder,f'{file_name}.tsv')
    print('Getting sequence from', path ,flush=True)
    print('Getting sequence_labels from', os.path.join(sequence_labels_folder, file_name + ".pkl") ,flush=True)
    with open(path, 'rb') as fp:
        sequence = joblib.load(fp)
    with open(os.path.join(sequence_labels_folder, file_name + ".pkl"), 'rb') as fp:
        sequence_labels = joblib.load(fp)
    if analyse_sequence_labels == True:
        sequence = sequence_labels
        save_all_file_path = os.path.join(analysis_folder, 'individual_results_stems', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'individual_results_stems_light', file_name + '.pkl')
    else:
        save_all_file_path = os.path.join(analysis_folder, 'individual_results', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'individual_results_light', file_name + '.pkl')
    save_entropies_file_path = os.path.join(analysis_folder, 'individual_results_entropy', file_name + '.pkl')
    
    
elif dataset_ID == 3: 
    # Semantic Scholar
    corpus_version = '2022-01-01'
    data_folder = os.path.join('./data/semanticscholar',corpus_version, 'data') # where you save the data
    if consider_temporal_order_in_tuples:
        analysis_folder = os.path.join('./data/semanticscholar',corpus_version, 'analysis')
    else:
        analysis_folder = os.path.join('./data/semanticscholar',corpus_version, 'analysis_tuples_without_order')
    with open(os.path.join(data_folder,'all_fieldsOfStudy.tsv'), 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        all_fieldsOfStudy = [] 
        for _ in tsv_output:
            all_fieldsOfStudy.append(_[0])
    num_journals_for_field = 1000
    fieldOfStudy = all_fieldsOfStudy[int(ID / num_journals_for_field)]
    ID = ID % num_journals_for_field
    sequence_folder = os.path.join(data_folder,'journals_fieldsOfStudy',fieldOfStudy)
    sequence_labels_folder = os.path.join(data_folder,'journals_fieldsOfStudy_stems',fieldOfStudy)
    sequence_folder_filepaths = sorted(find_pattern('*.tsv', sequence_folder))
    sequence_path = sequence_folder_filepaths[ID]
    file_name = sequence_path[-sequence_path[::-1].index('/'):sequence_path.index('.tsv')]
    sequence_labels_path = os.path.join(sequence_labels_folder,f'{file_name}.tsv')
    print('Getting sequence from', sequence_path ,flush=True)
    print('Getting sequence_labels from', sequence_labels_path ,flush=True)
    with open(sequence_path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence = [] 
        for _ in tsv_output:
            sequence.append(int(_[0]))
    with open(sequence_labels_path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence_labels = [] 
        for _ in tsv_output:
            sequence_labels.append(int(_[0]))
    if analyse_sequence_labels == True:
        sequence = sequence_labels
        save_all_file_path = os.path.join(analysis_folder, 'journals_fieldsOfStudy_stems', fieldOfStudy, file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'journals_fieldsOfStudy_stems_light', fieldOfStudy, file_name + '.pkl')
    else:
        save_all_file_path = os.path.join(analysis_folder, 'journals_fieldsOfStudy', fieldOfStudy, file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'journals_fieldsOfStudy_light', fieldOfStudy, file_name + '.pkl')
    save_entropies_file_path = os.path.join(analysis_folder, 'journals_fieldsOfStudy_entropy', fieldOfStudy,file_name + '.pkl')
    
    
elif dataset_ID == 4: 
    # UMST
    num_parameters = ending_nu - starting_nu + 1
    nu = starting_nu + ID % num_parameters
    run = int(ID/num_parameters)
    if consider_temporal_order_in_tuples:
        analysis_folder = f"./data/UMST/simulations/analysis/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/" 
    else:
        analysis_folder = f"./data/UMST/simulations/analysis_tuples_without_order/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/" 
    sequence_folder = f"./data/UMST/simulations/raw_sequences/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/" 
    sequence_labels_folder = f"./data/UMST/simulations/raw_sequences_labels/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/" 
    sequence_path = os.path.join(sequence_folder,f'{run}.tsv')
    file_name = sequence_path[-sequence_path[::-1].index('/'):sequence_path.index('.tsv')]
    sequence_labels_path = os.path.join(sequence_labels_folder, f'{file_name}.tsv')
    print('Getting sequence from', sequence_path ,flush=True)
    print('Getting sequence_labels from', sequence_labels_path ,flush=True)
    with open(sequence_path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence = [] 
        for _ in tsv_output:
            sequence.append(int(_[0]))
    with open(sequence_labels_path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence_labels = [] 
        for _ in tsv_output:
            sequence_labels.append(int(_[0]))
    if analyse_sequence_labels == True:
        sequence = sequence_labels
        save_all_file_path = os.path.join(analysis_folder, f'UMT_labels_run_{run}.pkl')
        save_light_file_path = os.path.join(analysis_folder, f'UMT_labels_light_run_{run}.pkl')
    else:
        save_all_file_path = os.path.join(analysis_folder, f'UMT_run_{run}.pkl')
        save_light_file_path = os.path.join(analysis_folder, f'UMT_light_run_{run}.pkl')
    save_entropies_file_path = os.path.join(analysis_folder, f'UMT_entropy_run_{run}.pkl')
    
    
elif dataset_ID == 5: 
    # ERRW
    # here sequence and sequence_labels are the same
    data_folder = os.path.join('./data/ERRW/',folder_name_ERRW,'raw_sequences')
    if consider_temporal_order_in_tuples:
        analysis_folder = os.path.join('./data/ERRW/',folder_name_ERRW,'analysis')
    else:
        analysis_folder = os.path.join('./data/ERRW/',folder_name_ERRW,'analysis_tuples_without_order')
    filepaths = sorted(find_pattern('*.txt', data_folder))
    path = filepaths[ID]
    print('Getting sequence from', path ,flush=True)
    print('Getting sequence_labels from', path ,flush=True)
    file_name = path[-path[::-1].index('/'):path.index('.txt')]
    with open(path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence = [] 
        for _ in tsv_output:
            sequence.append(int(_[0]))
    sequence_labels = sequence
    save_light_file_path = os.path.join(analysis_folder,'light_results',f'{file_name}.pkl')
    save_all_file_path = os.path.join(analysis_folder,'all_results',f'{file_name}.pkl')
    save_entropies_file_path = os.path.join(analysis_folder,'entropies_results',f'{file_name}.pkl')
   
    
elif dataset_ID == 6: 
    # directed ERRW
    # here sequence and sequence_labels are the same
    data_folder = os.path.join('./data/dir_ERRW/',folder_name_ERRW,'raw_sequences')
    if consider_temporal_order_in_tuples:
        analysis_folder = os.path.join('./data/dir_ERRW/',folder_name_ERRW,'analysis')
    else:
        analysis_folder = os.path.join('./data/dir_ERRW/',folder_name_ERRW,'analysis_tuples_without_order')
    filepaths = sorted(find_pattern('*.txt', data_folder))
    path = filepaths[ID]
    print('Getting sequence from', path ,flush=True)
    print('Getting sequence_labels from', path ,flush=True)
    file_name = path[-path[::-1].index('/'):path.index('.txt')]
    with open(path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence = [] 
        for _ in tsv_output:
            sequence.append(int(_[0]))
    sequence_labels = sequence
    save_light_file_path = os.path.join(analysis_folder,'light_results',f'{file_name}.pkl')
    save_all_file_path = os.path.join(analysis_folder,'all_results',f'{file_name}.pkl')
    save_entropies_file_path = os.path.join(analysis_folder,'entropies_results',f'{file_name}.pkl')
else:
    print('There is a problem with the dataset provided, only available options are 1, 2, 3, and 4.')
    exit()
    
    
    
# Analyse sequence

print("Starting analysis", flush=True)

result = analyse_sequence(
    sequence=sequence, 
    consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
    num_to_save=1000, 
    indices = [],
    use_D_as_D_indices = False, 
    D=None, 
    D2=None, 
    D3=None,
    D4=None, 
    sequence_labels = sequence_labels, 
    calculate_entropies_original = False, 
    calculate_entropies_labels = True,
    save_all = save_all, 
    save_all_file_path = save_all_file_path, 
    save_light_file_path = save_light_file_path, 
    save_entropies_file_path = save_entropies_file_path, 
    save_entropies_original_file_path = "./test_entropy_original.pkl",
    calculate_beta_loglogregr_indices = False,
    do_prints = True, 
    return_all = False
)

        
end = datetime.now()
print('Total time',end-start, flush=True)