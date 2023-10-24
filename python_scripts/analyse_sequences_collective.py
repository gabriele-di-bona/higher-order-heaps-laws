import numpy as np
import pickle
import os
# Change directory to the root of the folder (this script was launched from the subfolder python_scripts)
# All utils presuppose that we are working from the root directory of the github folder
os.chdir("../")
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
from datetime import datetime
import joblib
import gzip
import csv
import fnmatch
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import cld3
from iso639 import languages
import re
import random
import argparse

# local utils
from powerlaw_regr import *
from analyse_sequence import *
from find_files_with_pattern import *
    
# Beginning of MAIN
parser = argparse.ArgumentParser(description='Compute entropies on the sequences of labels in the data, both on singletons and pairs, and on randomized sequences.')


parser.add_argument("-ID", "--ID", type=int,
    help="The ID of the calculation, used as random seed [default 1]",
    default=1)

parser.add_argument("-dataset_ID", "--dataset_ID", type=int,
    help="The dataset to choose. 1 -> Last.fm, 2 -> Project Gutenberg, 3 -> Semantic Scholar, 4 -> UMST, 5 -> ERRW. [default 1]",
    default=1)

parser.add_argument("-analyse_labels", "--analyse_sequence_labels", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If True, it considers the sequence of labels instead of the original sequence in all calculations. [default False]",
    default=False)

parser.add_argument("-save_all", "--save_all", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If True, it saves all results. [default True]",
    default=True)

arguments = parser.parse_args()
random_ID = ID = arguments.ID - 1 # unfortunately I have to start this argument from 1 in shell. The -1 is added to start from 0
random.seed(ID)
file_name = str(ID)
dataset_ID = arguments.dataset_ID
analyse_sequence_labels = arguments.analyse_sequence_labels
save_all = arguments.save_all

start = datetime.now()
    
# Get sequence of labels
if dataset_ID == 1: 
    # Last.fm
    data_folder = "./data/ocelma-dataset/lastfm-dataset-1K/"
    raw_folder = os.path.join(data_folder, "raw_sequences")
    analysis_folder = data_folder
    # only consider those with at least 1000 elements
    with gzip.open(os.path.join(data_folder, "users_min_1000.pkl.gz"), "rb") as fp:
        users = joblib.load(fp)
    users_list = list(users.keys())
    users_list.sort()
    if analyse_sequence_labels == True:
        sequence = []
        random_users_list = users_list.copy()
        random.shuffle(random_users_list,)
        for userid in tqdm(random_users_list):
            sequence += list(users[userid]["sequence_artists"])
        sequence = np.array(sequence)
        save_all_file_path = os.path.join(analysis_folder, 'collective_results_artists', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'collective_results_artists_light', file_name + '_light.pkl')
    else:
        sequence = []
        random_users_list = users_list.copy()
        random.shuffle(random_users_list,)
        for userid in tqdm(random_users_list):
            sequence += list(users[userid]["sequence_tracks"])
        sequence = np.array(sequence)
        save_all_file_path = os.path.join(analysis_folder, 'collective_results', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'collective_results_light', file_name + '_light.pkl')
    
elif dataset_ID == 2: 
    # Project Gutenberg
    data_folder = './data/gutenberg/sequences_indexed/'
    analysis_folder = './data/gutenberg/analysis/'
    sequence_folder = os.path.join(data_folder,'sequences_words')
    sequence_labels_folder = os.path.join(data_folder,'sequences_stems')
    print('starting', datetime.now(), flush = True)
#     with gzip.open(os.path.join(analysis_folder, "all_books_preprocessed.pkl.gz"), "rb") as fp:
#         all_books = joblib.load(fp)
#     print(datetime.now(), flush = True)
    scratch_folder = '/data/scratch/ahw701/pairs_petralia/github/data/gutenberg/analysis/'

    if analyse_sequence_labels == True:
#         sequence_labels = []
#         randomized_list_all_book_keys = list(all_books.keys()).copy()
#         random.shuffle(randomized_list_all_book_keys)
#         for ID in tqdm(randomized_list_all_book_keys):
#             book = all_books[ID]
#             sequence_labels += list(book["sequence_stems"])
#         sequence_labels = np.array(sequence_labels)
        with open(os.path.join(scratch_folder, f"randomized_collective_sequences_labels", f'{random_ID}.pkl'), 'rb') as fp:
            sequence_labels = joblib.load(fp)
        sequence = sequence_labels
        save_all_file_path = os.path.join(analysis_folder, 'collective_results_stems', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'collective_results_stems_light', file_name + '_light.pkl')
    else:
#         sequence = []
#         randomized_list_all_book_keys = list(all_books.keys()).copy()
#         random.shuffle(randomized_list_all_book_keys)
#         for ID in randomized_list_all_book_keys:
#             book = all_books[ID]
#             sequence += list(book["sequence_words"])
#         sequence = np.array(sequence)
        with open(os.path.join(scratch_folder, f"randomized_collective_sequences", f'{random_ID}.pkl'), 'rb') as fp:
            sequence = joblib.load(fp)
        save_all_file_path = os.path.join(analysis_folder, 'collective_results', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'collective_results_light', file_name + '_light.pkl')
    print('finished getting sequence', datetime.now())

    
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
    num_journals = 1000
    
    # get randomized sequence
    scratch_folder = '/data/scratch/ahw701/pairs_petralia/github/data/semanticscholar/2022-01-01/analysis/'
    
    
    print(datetime.now())
    if analyse_sequence_labels == True:
        with open(os.path.join(scratch_folder, f"randomized_collective_sequences_labels", f'{random_ID}.pkl'), 'rb') as fp:
            sequence_labels = joblib.load(fp)
        sequence = sequence_labels
        save_all_file_path = os.path.join(analysis_folder, 'collective_results_stems', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'collective_results_stems_light', file_name + '_light.pkl')
#         sequence_labels = []
#         randomized_list_all_fields = all_fieldsOfStudy.copy()
#         random.shuffle(randomized_list_all_fields)
#         for fieldOfStudy in tqdm(randomized_list_all_fields):
#             fieldsOfStudy_folder = os.path.join(data_folder,'journals_fieldsOfStudy_stems',fieldOfStudy)
#             journals_filepaths = sorted(find_pattern('*.tsv', fieldsOfStudy_folder))
#             if len(journals_filepaths) != num_journals:
#                 print(f'Error! Found {len(journals_filepaths)} journals for {fieldOfStudy} instead of {num_journals}')
#                 break
#             random.shuffle(journals_filepaths)
#             for path in journals_filepaths:
#                 with open(path, 'r', newline='\n') as fp:
#                     tsv_output = csv.reader(fp, delimiter='\n')
#                     for _ in tsv_output:
#                         sequence_labels.append(int(_[0]))
#         sequence = np.array(sequence_labels)
#         print(datetime.now())
#         sequence = sequence_labels
#         save_all_file_path = os.path.join(analysis_folder, 'collective_journals_fieldsOfStudy_stems', file_name + '.pkl')
#         save_light_file_path = os.path.join(analysis_folder, 'collective_journals_fieldsOfStudy_stems_light', file_name + '_light.pkl')
    else:
        with open(os.path.join(scratch_folder, f"randomized_collective_sequences", f'{random_ID}.pkl'), 'rb') as fp:
            sequence = joblib.load(fp)
        save_all_file_path = os.path.join(analysis_folder, 'collective_results', file_name + '.pkl')
        save_light_file_path = os.path.join(analysis_folder, 'collective_results_light', file_name + '_light.pkl')
    print('finished getting sequence', datetime.now())
#         sequence = []
#         randomized_list_all_fields = all_fieldsOfStudy.copy()
#         random.shuffle(randomized_list_all_fields)
#         for fieldOfStudy in tqdm(randomized_list_all_fields):
#             fieldsOfStudy_folder = os.path.join(data_folder,'journals_fieldsOfStudy',fieldOfStudy)
#             journals_filepaths = sorted(find_pattern('*.tsv', fieldsOfStudy_folder))
#             if len(journals_filepaths) != num_journals:
#                 print(f'Error! Found {len(journals_filepaths)} journals for {fieldOfStudy} instead of {num_journals}')
#                 break
#             random.shuffle(journals_filepaths)
#             for path in journals_filepaths:
#                 with open(path, 'r', newline='\n') as fp:
#                     tsv_output = csv.reader(fp, delimiter='\n')
#                     for _ in tsv_output:
#                         sequence.append(int(_[0]))
#         sequence = np.array(sequence)
#         save_all_file_path = os.path.join(analysis_folder, 'collective_journals_fieldsOfStudy', file_name + '.pkl')
#         save_light_file_path = os.path.join(analysis_folder, 'collective_journals_fieldsOfStudy_light', file_name + '_light.pkl')
    print(datetime.now())
    
    
else:
    print('There is a problem with the dataset provided, only available options are 1, 2, 3, and 4.')
    exit()
    
    
    
# Analyse sequence

print("Starting analysis", flush=True)

result = analyse_sequence(
    sequence=sequence, 
#     consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
    num_to_save=1000, 
    indices = [],
    use_D_as_D_indices = False, 
    do_also_D2 = True, 
    do_also_D3 = True, 
    do_also_D4 = False, 
    D=None, 
    D2=None, 
    D3=None,
    D4=None, 
    sequence_labels = None, 
    calculate_entropies_original = False, 
    calculate_entropies_labels = False,
    save_all = save_all, 
    save_all_file_path = save_all_file_path, 
    save_light_file_path = save_light_file_path, 
    save_entropies_file_path = "./test_entropy.pkl",
    save_entropies_original_file_path = "./test_entropy_original.pkl",
    calculate_beta_loglogregr_indices = False,
    do_prints = True, 
    return_all = False
)

        
end = datetime.now()
print('Total time',end-start, flush=True)