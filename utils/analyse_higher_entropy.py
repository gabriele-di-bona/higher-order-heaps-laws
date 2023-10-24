import numpy as np
import pickle
import os
# # Change directory to the root of the folder (this script was launched from the subfolder python_scripts)
# # All utils presuppose that we are working from the root directory of the github folder
# os.chdir("../")
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))
from datetime import datetime

# local utils
from powerlaw_regr import *
from entropy import *
from index_sequence import *

def get_frequencies(
    sequence,
    get_ordered_keys_values = False
):
    freq_dict = {}
    for t,element in enumerate(sequence):
        try:
            freq_dict[element] += 1
        except KeyError:
            freq_dict[element] = 1
    ordered_freq_keys = np.array(list(freq_dict.keys()), dtype=np.int64)
    ordered_freq_values = np.array(list(freq_dict.values()), dtype=np.int64)
    if get_ordered_keys_values :
        return freq_dict, ordered_freq_keys, ordered_freq_values
    else:
        return freq_dict
    

def get_entropies_sequence(
    sequence,
    get_global_reshuffle_entropies = True,
    number_reshuffles = 1
):
    freq_dict, ordered_freq_keys, ordered_freq_values = get_frequencies(
        sequence,
        get_ordered_keys_values = True
    )
    # Calculate entropies
    sequence = np.array(sequence)
    entropies = entropyCalc(sequence, ordered_freq_keys, ordered_freq_values)
    dict_freq_list_entropies = get_dict_freq_list_entropies(entropies = entropies)
    if get_global_reshuffle_entropies:
        # Let's repeat on reshuffled sequence
        dict_freq_list_entropies_glob = {}
        for n in range(number_reshuffles):
            # TODO Add that repeat many times and average on frequencies
            reshuffled_sequence = np.array(sequence.copy())
            np.random.shuffle(reshuffled_sequence)
            entropies_glob = entropyCalc(reshuffled_sequence, ordered_freq_keys, ordered_freq_values)
            dict_freq_list_entropies_glob_tmp = get_dict_freq_list_entropies(entropies = entropies_glob)
            for f,entropies_tmp_list in dict_freq_list_entropies_glob_tmp.items():
                try:
                    dict_freq_list_entropies_glob[f] += entropies_tmp_list
                except KeyError:
                    dict_freq_list_entropies_glob[f] = entropies_tmp_list
        return dict_freq_list_entropies, dict_freq_list_entropies_glob
    else:
        return dict_freq_list_entropies

def analyse_sequence_higher_order_entropy(
    sequence,
    sequence_pairs = None, 
    consider_temporal_order_in_tuples=True, 
    index_original_sequence = False,
    number_reshuffles = 10,
    calc_entropy_pairs_on_reshuffled_singletons = False
):
    '''
        Usually the sequence is the provided sequence of labels.
        If consider_temporal_order_in_tuples == False, it considers AB and BA the same.
    '''
    results = {}
    if index_original_sequence:
        indexed_sequence = index_sequence(sequence)
    else:
        indexed_sequence = sequence
#     results['sequence'] = indexed_sequence
    if sequence_pairs is None:
        sequence_pairs, indexed_sequence_pairs = get_sequence_pairs(sequence, consider_temporal_order_in_tuples=consider_temporal_order_in_tuples)
    else:
        if index_original_sequence:
            indexed_sequence_pairs = index_sequence(sequence_pairs)
        else:
            indexed_sequence_pairs = sequence_pairs
#     results['sequence_pairs'] = indexed_sequence_pairs
    results["dict_freq_list_entropies"], results["dict_freq_list_entropies_glob"] = get_entropies_sequence(
        sequence,
        get_global_reshuffle_entropies = True,
        number_reshuffles = number_reshuffles,
    )
    results["dict_freq_list_entropies_pairs"], results["dict_freq_list_entropies_glob_pairs"] = get_entropies_sequence(
        indexed_sequence_pairs,
        get_global_reshuffle_entropies = True,
        number_reshuffles = number_reshuffles,
    )
    if calc_entropy_pairs_on_reshuffled_singletons:
        dict_freq_list_entropies_glob_pairs_reshuffling_singletons = {}
        for n in range(number_reshuffles):
            print(f'\tReshuffle n. {n}', flush=True)
            reshuffled_sequence = np.array(sequence.copy())
            np.random.shuffle(reshuffled_sequence)
            sequence_pairs, indexed_sequence_pairs = get_sequence_pairs(sequence, consider_temporal_order_in_tuples=consider_temporal_order_in_tuples)
            dict_freq_list_entropies_glob_tmp = get_entropies_sequence(
                indexed_sequence_pairs,
                get_global_reshuffle_entropies = False,
                number_reshuffles = 0,
            )
            for f,entropies_tmp_list in dict_freq_list_entropies_glob_tmp.items():
                try:
                    dict_freq_list_entropies_glob_pairs_reshuffling_singletons[f] += entropies_tmp_list
                except KeyError:
                    dict_freq_list_entropies_glob_pairs_reshuffling_singletons[f] = entropies_tmp_list
        results['dict_freq_list_entropies_glob_pairs_reshuffling_singletons'] = dict_freq_list_entropies_glob_pairs_reshuffling_singletons
    
    # Calculate average entropy difference
    # Singletons
    entropies_sorted_keys_singletons = sorted(list(results["dict_freq_list_entropies"].keys()))
    results["weighted_diff_entropies"], results["entropies_sorted_keys"], results["entropies_sorted_weights"], \
        results["mean_entropies"], results["mean_entropies_glob"] = \
        get_weighted_difference_entropies(
            results["dict_freq_list_entropies"], 
            results["dict_freq_list_entropies_glob"], 
            entropies_sorted_keys = entropies_sorted_keys_singletons,
            entropies_provided_is_dict_freq_entropies_list = True,
        )
    # Pairs
    entropies_sorted_keys_pairs = sorted(list(results["dict_freq_list_entropies_pairs"].keys()))
    results["weighted_diff_entropies_pairs"], results["entropies_sorted_keys_pairs"], results["entropies_sorted_weights_pairs"], \
        results["mean_entropies_pairs"], results["mean_entropies_glob_pairs"] = \
        get_weighted_difference_entropies(
            results["dict_freq_list_entropies_pairs"], 
            results["dict_freq_list_entropies_glob_pairs"], 
            entropies_sorted_keys = entropies_sorted_keys_pairs,
            entropies_provided_is_dict_freq_entropies_list = True,
        )
    # TODO Pairs with reshuffled sequence on singletons
    # This cannot be done right now because the frequencies are different in the pairs of a reshuffled sequence of singletons and on the original... 
    # Think of something
    return results
 