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

parser.add_argument("-calc_entropy_pairs_on_reshuffled_singletons", "--calc_entropy_pairs_on_reshuffled_singletons", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If True, it also calculates the entropy of the sequence of pairs created from the randomized sequence of singletons. [default False]",
    default=False)

parser.add_argument("-order", "--consider_temporal_order_in_tuples", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If False, it considers AB and BA the same. [default True]",
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
calc_entropy_pairs_on_reshuffled_singletons = arguments.calc_entropy_pairs_on_reshuffled_singletons
consider_temporal_order_in_tuples = arguments.consider_temporal_order_in_tuples
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
    analysis_folder = data_folder
    folder = os.path.join(data_folder,'individual_results')
    filepaths = sorted(find_pattern('*.pkl', folder))
    path = filepaths[ID]
    with open(path, 'rb') as fp:
        tmp_result = joblib.load(fp)
    sequence_labels = tmp_result['sequence_labels']
    file_name = path[-path[::-1].index('/'):path.index('.pkl')]
    save_entropies_file_path = os.path.join(analysis_folder, 'individual_results_entropy',file_name + '.pkl')
elif dataset_ID == 2: 
    # Project Gutenberg
    data_folder = './data/gutenberg/sequences_indexed/'
    analysis_folder = './data/gutenberg/analysis/'
    folder = os.path.join(data_folder,'sequences_stems')
    filepaths = sorted(find_pattern('*.pkl', folder))
    path = filepaths[ID]
    with open(path, 'rb') as fp:
        sequence_labels = joblib.load(fp)
    file_name = path[-path[::-1].index('/'):path.index('.pkl')]
    save_entropies_file_path = os.path.join(analysis_folder, 'individual_results_entropy',file_name + '.pkl')
elif dataset_ID == 3: 
    # Semantic Scholar
    corpus_version = '2022-01-01'
    data_folder = os.path.join('./data/semanticscholar',corpus_version, 'data') # where you save the data
    analysis_folder = os.path.join('./data/semanticscholar',corpus_version, 'analysis')
    with open(os.path.join(data_folder,'all_fieldsOfStudy.tsv'), 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        all_fieldsOfStudy = [] 
        for _ in tsv_output:
            all_fieldsOfStudy.append(_[0])
    num_journals_for_field = 1000
    fieldOfStudy = all_fieldsOfStudy[int(ID / num_journals_for_field)]
    ID = ID % num_journals_for_field
    fieldsOfStudy_folder = os.path.join(data_folder,'journals_fieldsOfStudy_stems',fieldOfStudy)
    journals_filepaths = sorted(find_pattern('*.tsv', fieldsOfStudy_folder))
    if len(journals_filepaths) != num_journals_for_field:
        print(f'Error! Found {len(journals_filepaths)} journals for {fieldOfStudy} instead of {num_journals_for_field}')
    path = journals_filepaths[ID]
    sequence_labels = [] 
    with open(path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        for _ in tsv_output:
            sequence_labels.append(int(_[0]))
    file_name = path[-path[::-1].index('/'):path.index('.tsv')]
    save_entropies_file_path = os.path.join(analysis_folder, 'journals_fieldsOfStudy_entropy',fieldOfStudy,file_name + '.pkl')
elif dataset_ID == 4: 
    # UMST
    num_parameters = ending_nu - starting_nu + 1
    nu = starting_nu + ID % num_parameters
    run = int(ID/num_parameters)
    data_folder = f"./data/UMST/simulations/analysis/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/" 
    analysis_folder = data_folder
    path = os.path.join(data_folder,f'UMT_run_{run}.pkl')
    with open(path, 'rb') as fp:
        result = joblib.load(fp)
    sequence = result['sequence']
    sequence_labels = result['sequence_labels']
    save_entropies_file_path = os.path.join(analysis_folder,f'UMT_entropy_run_{run}.pkl')
elif dataset_ID == 5: 
    # ERRW
    data_folder = os.path.join('./data/ERRW',folder_name_ERRW,'raw_sequences')
    if consider_temporal_order_in_tuples:
        analysis_folder = os.path.join('./data/ERRW',folder_name_ERRW,'analysis')
    else:
        analysis_folder = os.path.join('./data/ERRW',folder_name_ERRW,'analysis_tuples_without_order')
    filepaths = sorted(find_pattern('*.txt', data_folder))
    path = filepaths[ID]
    with open(path, 'r', newline='\n') as fp:
        tsv_output = csv.reader(fp, delimiter='\n')
        sequence_labels = [] 
        for _ in tsv_output:
            sequence_labels.append(int(_[0]))
    file_name = path[-path[::-1].index('/'):path.index('.txt')]
    save_entropies_file_path = os.path.join(analysis_folder,'entropies_results',f'{file_name}.pkl')
else:
    print('There is a problem with the dataset provided, only available options are 1, 2, 3, and 4.')
    exit()
    
    
print('Getting data from', flush=True)
print(path, flush=True)
# Analyse sequence
result = analyse_sequence_higher_order_entropy(
    sequence = sequence_labels,
    consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
    index_original_sequence = False,
    number_reshuffles = number_reshuffles,
    calc_entropy_pairs_on_reshuffled_singletons = calc_entropy_pairs_on_reshuffled_singletons,
)


# Dump results
print('Dumping into', flush=True)
print(save_entropies_file_path, flush=True)
os.makedirs(os.path.dirname(save_entropies_file_path), exist_ok=True)
with open(save_entropies_file_path, "wb") as fp:
    pickle.dump(result, fp)

        
end = datetime.now()
print('Total time',end-start, flush=True)