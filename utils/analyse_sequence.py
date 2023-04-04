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
from analyse_higher_entropy import *


   

def analyse_sequence(
    sequence=np.array([]), 
    sequence_pairs=None, 
    consider_temporal_order_in_tuples=True, 
    num_to_save=1000, 
    indices = [],
    use_D_as_D_indices = False, 
    D=None, 
    D2=None, 
    D3=None,
    D4=None, 
    sequence_labels = None, 
    calculate_entropies_original = False, 
    calculate_entropies_labels = True,
    number_reshuffles = 10, 
    save_all=False, 
    save_all_file_path = "./test.pkl", 
    save_light_file_path = "./test_light.pkl",
    save_entropies_file_path = "./test_entropy.pkl",
    save_entropies_original_file_path = "./test_entropy_original.pkl",
    calculate_beta_loglogregr_indices = False,
    do_prints = True, 
    return_all = False
):
    '''
        Computes and saves in a dictionary the array counting new elements, pairs, triplets and quadruplets.
        It also calculates the normalized Shannon Entropy and the exponents of the discovery arrays.
        
        Input:
            - sequence: time-ordered np.array containing the sequence of ids drawn (np.array(int))
                If None is provided, then it won't be used, and some analysis could be limited.
            - consider_temporal_order_in_tuples: if True, the pair AB is different from the pair BA. Else they are the same pair (bool)
            - num_to_save: at the end, the analysis is saved in a light dictionary,
                where only the indices related to a linspace and geomspace of length num_to_save are actually saved (int)
            - use_D_as_D_indices: if whole sequence or D or D2 or D3 or D4 are not present, 
                you can provide D_indices instead of D, etc. In this case indices have to be provided! (bool)
            - indices: useful only when use_D_as_D_indices == True. 
                It's a list of the indices where D_indices have been calculated, of the same length as D_indices etc. 
                It's made by the set of indices coming from a linspace and geomspace of length num_to_save (list)
            - calculate_entropies_original: if True it calculates the entropies on the sequence of IDs (bool)
            - calculate_entropies_labels: if True it calculates the entropies on the sequence of labels, i.e. mothers of IDs (bool)
            - number_reshuffles: number of times  entropies should be calculated on a reshuffle of the sequence (int)
            - D: array containing for each timestep the number of new elements in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - D: array containing for each timestep the number of new elements in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - D: array containing for each timestep the number of new elements in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - D: array containing for each timestep the number of new elements in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - sequence_labels: if calculate_entropies_labels==True, 
                this is used to calculate the entropies instead of sequence 
                and must be provided (None or np.array(int))
            - save_all: saves results of the simulation related to all indices, including the sequence of draws, of discovery, entropies (bool)
            - save_all_file_path: file path where to dump the results dict, if save_all == True (string, must be a valid path)
            - save_light_file_path: file path where to dump the light results dict (string, must be a valid path)
            - save_entropies_file_path: file path where to dump the entropies results dict (string, must be a valid path)
            - save_entropies_original_file_path: file path where to dump the entropies of the original sequence results dict (string, must be a valid path)
            - calculate_beta_loglogregr_indices: if True it calculates the power_law fit considering all points in the sequence with a linregress on loglogscale (bool)
            - do_prints: if False, it doesn't do any print in the function (bool)
            - return_all: if True, it returns the big dict results, otherwise results_light (bool)

        Output:
            - if save_all is True, saves all the sequences and analysis results in a pikled dictionary under save_all_root_dir
            - Saves the analysis results on a small subset of the indices on a light pickled dictionary, whose number is decided by num_to_save.
            - Returns the results and all calculations of Heaps' law and entropy (same dictionary saved with save_all=True)
    '''
    # safety check that the provided sequence is a numpy array and not a list
    sequence = np.array(sequence)
    sequence_labels = np.array(sequence_labels)
    if use_D_as_D_indices == False:
        Tmax = len(sequence)
        
    # Let's check if D, D2, D3 and D4 are correctly provided
    calculate_novelties = False
    if D is None or D2 is None or D3 is None or D4 is None:
        # One of the arrays of discoveries have not been provided
        calculate_novelties = True
    elif use_D_as_D_indices == False and (len(sequence) != len(D) or len(sequence) != len(D2) or len(sequence) != len(D3) or len(sequence) != len(D4)):
        # One of the arrays of discoveries is not of the same length of sequence, recalculating them...
        calculate_novelties = True
    if calculate_novelties == True:
        # Calculating D, D2, D3 and D4
        # first element is considered before the for loop
        D = np.zeros(Tmax, dtype=int) # counts the number of different elememts appeared in the sequence up to time t
        D[0] = 1
        D2 = np.zeros(Tmax, dtype=int) # counts the number of different pairs appeared in the sequence up to time t
        D3 = np.zeros(Tmax, dtype=int) # counts the number of different triplets appeared in the sequence up to time t
        D4 = np.zeros(Tmax, dtype=int) # counts the number of different quadruplets appeared in the sequence up to time t
        set_sequence = {sequence[0]} # set of all different IDs present in the sequence
        # in the following 3 lines dictionaries are nested, keys in first dict representing the first element of the different tuples, etc.
        set_pairs = {sequence[0]:set()} # dict of set of IDs representing all pairs present in the sequence
        set_triplets = {sequence[0]:{}} # dict of dict of set of IDs representing all pairs present in the sequence
        set_quadruplets = {sequence[0]:{}} # dict of dict of dict of set of IDs representing all pairs present in the sequence

        seq_freq = {sequence[0]:1} # The index is the ID, the number is the number of balls
        
        last_ID = -1 # ID in the sequence at t-1
        last_ID2 = -1 # ID in the sequence at t-2
        last_ID3 = -1 # ID in the sequence at t-3

        # Beginning of analysis of discoveries
        if do_prints == True:
            print("Starting creating D,D2,D3,D3 at", datetime.now(), flush=True)
        for t in range(1,Tmax):
            if do_prints == True and (Tmax >= 100) and (t%int(Tmax/100) == 0):
                print("Done %d/100 at"%(int(t/Tmax*100)), datetime.now(), flush=True)
            last_ID3 = last_ID2
            last_ID2 = last_ID
            last_ID = sequence[t-1]
            ID = sequence[t]
            
            if consider_temporal_order_in_tuples == True:
                # Check if there are new elements/pairs/triplets/quadruplets
                if ID not in set_sequence:
                    # This means that there is a new ID!
                    set_sequence.add(ID)
                    seq_freq[ID] = 1
                    D[t] = D[t-1]+1
                    # If the ID is new, it means that there is also a new pair, triplet and quadruplet
                    set_pairs[ID] = set()
                    set_triplets[ID] = {}
                    set_quadruplets[ID] = {}
                    set_pairs[last_ID].add(ID)
                    D2[t] = D2[t-1]+1
                    set_triplets[last_ID][ID] = set()
                    set_quadruplets[last_ID][ID] = {}
                    if last_ID2 != -1:
                        set_triplets[last_ID2][last_ID].add(ID)
                        D3[t] = D3[t-1]+1
                        set_quadruplets[last_ID2][last_ID][ID] = set()
                    if last_ID3 != -1:
                        set_quadruplets[last_ID3][last_ID2][last_ID].add(ID)
                        D4[t] = D4[t-1]+1
                else:
                    # ID is already present in the sequence, checking for pair
                    D[t] = D[t-1]
                    seq_freq[ID] += 1
                    if ID not in set_pairs[last_ID]:
                        # This means that there is a new pair, hence also triplet and quadruplet
                        set_pairs[last_ID].add(ID)
                        D2[t] = D2[t-1]+1
                        set_triplets[last_ID][ID] = set()
                        set_quadruplets[last_ID][ID] = {}
                        if last_ID2 != -1:
                            set_triplets[last_ID2][last_ID].add(ID)
                            D3[t] = D3[t-1]+1
                            set_quadruplets[last_ID2][last_ID][ID] = set()
                        if last_ID3 != -1:
                            set_quadruplets[last_ID3][last_ID2][last_ID].add(ID)
                            D4[t] = D4[t-1]+1
                    else:
                        # ID and last_ID:ID are present in sequence, checking for triplet
                        D2[t] = D2[t-1]
                        if last_ID2 != -1 and ID not in set_triplets[last_ID2][last_ID]:
                            # This means that there is a new triplet, hence also quadruplet
                            set_triplets[last_ID2][last_ID].add(ID)
                            D3[t] = D3[t-1]+1
                            set_quadruplets[last_ID2][last_ID][ID] = set()
                            if last_ID3 != -1:
                                set_quadruplets[last_ID3][last_ID2][last_ID].add(ID)
                                D4[t] = D4[t-1]+1
                        else:
                            # ID, last_ID:ID, last_ID2:last_ID:ID are present in sequence, checking for quadruplet
                            D3[t] = D3[t-1]
                            if last_ID3 != -1 and ID not in set_quadruplets[last_ID3][last_ID2][last_ID]:    
                                # This means that there is a new quadruplet
                                set_quadruplets[last_ID3][last_ID2][last_ID].add(ID)
                                D4[t] = D4[t-1]+1
                            else:
                                # not even a new quadruplet
                                D4[t] = D4[t-1]
            else:
                # Check if there are new singletons/pairs/triplets/quadruplets
                if ID in set_sequence:
                    # This means that the ID is old!
                    new_singleton = False
                    seq_freq[ID] = 1
                    # Check for pair
                    sorted_ID0 = sorted([ID,last_ID])[0]
                    sorted_ID1 = sorted([ID,last_ID])[1]
                    if sorted_ID0 in set_pairs and sorted_ID1 in set_pairs[sorted_ID0]:
                        # This means that the pair is old!
                        new_pair = False
                        # Check for triplet
                        sorted_ID0 = sorted([ID,last_ID,last_ID2])[0]
                        sorted_ID1 = sorted([ID,last_ID,last_ID2])[1]
                        sorted_ID2 = sorted([ID,last_ID,last_ID2])[2]
                        if sorted_ID0 in set_triplets and sorted_ID1 in set_triplets[sorted_ID0] and sorted_ID2 in set_triplets[sorted_ID0][sorted_ID1]:
                            # This means that the triplet is old!
                            new_triplet = False
                            # Check for quadruplet
                            sorted_ID0 = sorted([ID,last_ID,last_ID2,last_ID3])[0]
                            sorted_ID1 = sorted([ID,last_ID,last_ID2,last_ID3])[1]
                            sorted_ID2 = sorted([ID,last_ID,last_ID2,last_ID3])[2]
                            sorted_ID3 = sorted([ID,last_ID,last_ID2,last_ID3])[3]
                            if sorted_ID0 in set_quadruplets and sorted_ID1 in set_quadruplets[sorted_ID0] and sorted_ID2 in set_quadruplets[sorted_ID0][sorted_ID1] and sorted_ID3 in set_quadruplets[sorted_ID0][sorted_ID1][sorted_ID2]:
                                new_quadruplet = False
                            else:
                                new_quadruplet = True
                        else:
                            new_triplet = new_quadruplet = True
                    else:
                        new_pair = new_triplet = new_quadruplet = True
                else:
                    # If the ID is new, it means that there is also a new pair, triplet and quadruplet. 
                    new_singleton = new_pair = new_triplet = new_quadruplet = True
                
                D[t] = D[t-1]+int(new_singleton)
                D2[t] = D2[t-1]+int(new_pair)
                D3[t] = D3[t-1]+int(new_triplet)
                D4[t] = D4[t-1]+int(new_quadruplet)
                if new_singleton:
                    # update set singletons
                    set_sequence.add(ID)
                if new_pair:
                    # update set pairs
                    sorted_ID0 = sorted([ID,last_ID])[0]
                    sorted_ID1 = sorted([ID,last_ID])[1]
                    try:
                        set_pairs[sorted_ID0].add(sorted_ID1)
                    except:
                        set_pairs[sorted_ID0] = set([sorted_ID1])
                if new_triplet and last_ID2 != -1:
                    # update set triplets
                    sorted_ID0 = sorted([ID,last_ID,last_ID2])[0]
                    sorted_ID1 = sorted([ID,last_ID,last_ID2])[1]
                    sorted_ID2 = sorted([ID,last_ID,last_ID2])[2]
                    try:
                        try:
                            set_triplets[sorted_ID0][sorted_ID1].add(sorted_ID2)
                        except:
                            set_triplets[sorted_ID0][sorted_ID1] = set([sorted_ID2])
                    except:
                        set_triplets[sorted_ID0] = {sorted_ID1:set([sorted_ID2])}
                if new_quadruplet and last_ID3 != -1:
                    # update set quadruplets
                    sorted_ID0 = sorted([ID,last_ID,last_ID2,last_ID3])[0]
                    sorted_ID1 = sorted([ID,last_ID,last_ID2,last_ID3])[1]
                    sorted_ID2 = sorted([ID,last_ID,last_ID2,last_ID3])[2]
                    sorted_ID3 = sorted([ID,last_ID,last_ID2,last_ID3])[3]
                    try:
                        try:
                            try:
                                set_quadruplets[sorted_ID0][sorted_ID1][sorted_ID2].add(sorted_ID3)
                            except:
                                set_quadruplets[sorted_ID0][sorted_ID1][sorted_ID2] = set([sorted_ID3])
                        except:
                            set_quadruplets[sorted_ID0][sorted_ID1] = {sorted_ID2:set([sorted_ID3])}
                    except:
                        set_quadruplets[sorted_ID0] = {sorted_ID1:{sorted_ID2:set([sorted_ID3])}}
                    
        # End of analysis of discoveries
    
    if use_D_as_D_indices == False:
        # list of indices to save
        # indices will contain both a geomspace and a linspace of length num_to_save, starting from index 0 to the last index
        # ts will contain both a geomspace and a linspace of length num_to_save, starting from t=1 to the last index + 1
        # ts will be a np.array, while indices a list, useful for selecting the correct indices from D
        if Tmax > num_to_save*2:
            indices = set(np.geomspace(1,Tmax,num_to_save,dtype=int)-1).union(set(np.linspace(1,Tmax,num_to_save,dtype=int)-1))
        else:
            indices = set(np.arange(Tmax))
    indices = list(indices)
    indices.sort()
    ts = np.array(indices)+1
    if do_prints == True:
        print("Sorted indices to save at", datetime.now(), flush=True)
    # Now let's do only the geom indices, useful for loglog few points regression
    # set_geom contains the geometrically spanned indices from 1 to Tmax+1, diminished by 1 (so that first index is 0)
    # ts_geom contains the ts geometrically spanned, 
    # while indices_geom contains the indices inside ts_geom that are coming from a geomspace
    # useful for selecting the geomspace corrisponding indices in D_indices, D2, etc
    indices_geom = []
    set_geom = set(np.geomspace(1,indices[-1]+1,num_to_save,dtype=int)-1)
    ts_geom = list(np.array(list(set_geom))+1)
    ts_geom.sort()
    ts_geom = np.array(ts_geom)
    for i,x in enumerate(indices):
        if x in set_geom:
            indices_geom.append(i)
    if use_D_as_D_indices == False:
        D_indices = D[indices]
        D2_indices = D2[indices]
        D3_indices = D3[indices]
        D4_indices = D4[indices]
    else:
        D_indices = D
        D2_indices = D2
        D3_indices = D3
        D4_indices = D4
        
    if use_D_as_D_indices == False:
        results = { \
                    "D":D, "D2":D2, "D3":D3, "D4":D4, \
                    "ts":ts, "indices":indices, "ts_geom":ts_geom, "indices_geom":indices_geom, \
                    "D_indices":D_indices, "D2_indices":D2_indices, \
                    "D3_indices":D3_indices, "D4_indices":D4_indices \
                  }
        if sequence is not None:
            results["sequence"] = sequence
        if sequence_labels is not None:
            results["sequence_labels"] = sequence_labels
    else:
        results = { \
                    "ts":ts, "indices":indices, "ts_geom":ts_geom, "indices_geom":indices_geom, \
                    "D_indices":D_indices, "D2_indices":D2_indices, \
                    "D3_indices":D3_indices, "D4_indices":D4_indices \
                  }

    results_light = {}
    for k in ["ts", "indices", "ts_geom", "indices_geom", \
              "D_indices", "D2_indices", "D3_indices", "D4_indices"]:
        results_light[k] = results[k]


    # Start of analysis on simulation to save
    # Calculation entropies
    if calculate_entropies_labels == True:
        # Compute entropies on the sequence of labels (mothers) to show semantic correlations
        # sequence_labels must be provided!
        if sequence_labels is not None:
            if do_prints == True:
                print("Calculating entropies on sequence of labels at", datetime.now(), flush=True)
                if len(sequence) != len(sequence_labels):
                    print("Achtung: len(sequence) != len(sequence_labels), values are",len(sequence), len(sequence_labels), flush=True)
#             # Let's get the sequence of labels and their frequencies
#             labels_freq = {}
#             for t,label in enumerate(sequence_labels):
#                 if label not in labels_freq:
#                     labels_freq[label] = 1
#                 else:
#                     labels_freq[label] += 1
#             labels_freq_keys = list(labels_freq.keys())
#             labels_freq_values = list(labels_freq.values())
#             labels_freq_keys = np.array(labels_freq_keys, dtype=np.int64)
#             labels_freq_values = np.array(labels_freq_values, dtype=np.int64)
#             entropies = entropyCalc(sequence_labels, labels_freq_keys, labels_freq_values)
#             # Let's repeat on reshuffled sequence
#             if do_prints == True:
#                 print("Calculating now entropies on reshuffled sequence of labels at", datetime.now(), flush=True)
#             tmp_sequence_labels = sequence_labels.copy()
#             np.random.shuffle(tmp_sequence_labels)
#             entropies_glob = entropyCalc(tmp_sequence_labels, labels_freq_keys, labels_freq_values)
#             # Saving in results
#             results["entropies"] = entropies
#             results["entropies_glob"] = entropies_glob
#             if do_prints == True:
#                 print("Calculating now mean_entropies and weighted difference on sequence of labels at", datetime.now(), flush=True)
#             results["weighted_diff_entropies"], results["entropies_sorted_keys"], results["entropies_sorted_weights"], \
#                 results["mean_entropies"], results["mean_entropies_glob"] = \
#                 get_weighted_difference_entropies(entropies, entropies_glob)
#             results_light["weighted_diff_entropies"] = results["weighted_diff_entropies"]
            
            entropy_results = analyse_sequence_higher_order_entropy(
                sequence_labels,
                sequence_pairs = sequence_pairs,
                consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
                index_original_sequence = False,
                number_reshuffles = number_reshuffles,
                calc_entropy_pairs_on_reshuffled_singletons = False
            )
            for key,value in entropy_results.items():
                results[key] = value
                if 'weighted_diff_entropies' in key:
                    results_light[key] = value
            try:
                os.makedirs(os.path.dirname(save_entropies_file_path), exist_ok=True)
                with open(save_entropies_file_path,'wb') as fp:
                    pickle.dump(entropy_results,fp)
            except:
                print("Exception: couldn't save entropies labels into", save_entropies_file_path, flush=True)
        else:
            if do_prints == True:
                print("Couldn't calculate entropies on sequence of labels, sequence of labels not correctly provided!", flush=True)
                print("Calculating instead the entropies in the original sequence...", flush=True)
            calculate_entropies_original = True
    if calculate_entropies_original == True and sequence is not None:
        # Compute entropies on the sequence of IDs to show semantic correlations
        if do_prints == True:
            print("Calculating entropies on sequence of IDs at", datetime.now(), flush=True)
#         seq = sequence.copy()
#         ### DEPRECATED
#         # We can get the sequence frequencies from urn_freq, 
#         # because the number of balls of a ID i in the urn are 1 + rho * n_i,
#         # where n_i are the times it appeared in the sequence
# #         sequence_freq = urn_freq.copy()
# #         for i in sequence_freq.keys():
# #             sequence_freq[i] -= 1
# #             sequence_freq[i] = int(sequence_freq[i] / rho)
#         labels_freq = {}
#         for t,label in enumerate(sequence):
#             if label not in labels_freq:
#                 labels_freq[label] = 1
#             else:
#                 labels_freq[label] += 1
#         labels_freq_keys = list(labels_freq.keys())
#         labels_freq_values = list(labels_freq.values())
#         labels_freq_keys = np.array(labels_freq_keys, dtype=np.int64)
#         labels_freq_values = np.array(labels_freq_values, dtype=np.int64)
#         entropies = entropyCalc(sequence_labels, labels_freq_keys, labels_freq_values)
#         # Let's repeat on reshuffled sequence
#         if do_prints == True:
#             print("Calculating now entropies on reshuffled sequence of labels at", datetime.now(), flush=True)
#         tmp_sequence_labels = sequence_labels.copy()
#         np.random.shuffle(tmp_sequence_labels)
#         entropies_glob = entropyCalc(tmp_sequence_labels, labels_freq_keys, labels_freq_values)
#         # Saving in results
#         results["entropies"] = entropies
#         results["entropies_glob"] = entropies_glob
#         if do_prints == True:
#             print("Calculating now mean_entropies and weighted difference on sequence of IDs at", datetime.now(), flush=True)
#         results["weighted_diff_entropies_original"], results["entropies_sorted_keys_original"], results["entropies_sorted_weights_original"], \
#             results["mean_entropies_original"], results["mean_entropies_glob_original"] = \
#             get_weighted_difference_entropies(entropies, entropies_glob)
#         results_light["weighted_diff_entropies_original"] = results["weighted_diff_entropies_original"]
        
        entropy_original_results = analyse_sequence_higher_order_entropy(
            sequence,
            consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
            index_original_sequence = False,
            number_reshuffles = number_reshuffles,
            calc_entropy_pairs_on_reshuffled_singletons = False
        )
        for key,value in entropy_original_results.items():
            results[key+'_original_sequence'] = value
            if 'weighted_diff_entropies' in key:
                results_light[key+'_original_sequence'] = value
        
        try:
            os.makedirs(os.path.dirname(save_entropies_original_file_path), exist_ok=True)
            with open(save_entropies_original_file_path,'wb') as fp:
                pickle.dump(entropy_original_results,fp)
        except:
            print("Exception: couldn't save entropies original into", save_entropies_original_file_path, flush=True)
        
    

    if do_prints == True:
        print("Doing beta_mean_indices at", datetime.now(), flush=True)
    # Here the exponent is calculated as the slope in loglog scale given by only initial where it is 1 and final point
    # The suffix _indices is added because the calculation is done at final points given by the indices used to store ts
    beta_mean_indices = np.zeros(len(D_indices))
    beta2_mean_indices = np.zeros(len(D2_indices))
    beta3_mean_indices = np.zeros(len(D3_indices))
    beta4_mean_indices = np.zeros(len(D4_indices))
    for index,t in enumerate(ts):
        try:
            if t < 1:
                beta_mean_indices[index] = -1
            elif t == 1:
                beta_mean_indices[index] = 1
            else:
                beta_mean_indices[index] = np.log10(D_indices[index])/np.log10(t) # t because D[0] = 1, so first actual index is at t=1 (index = 0)
        except (ValueError,ZeroDivisionError):
            beta_mean_indices[index] = -1
        try:
            if t-1 < 1:
                beta2_mean_indices[index] = -1
            elif t-1 == 1:
                beta2_mean_indices[index] = 1
            else:
                beta2_mean_indices[index] = np.log10(D2_indices[index])/np.log10(t-1) # t-1 because D2[1] = 1, so first actual index is at t=2 (index = 1)
        except (ValueError,ZeroDivisionError):
            beta2_mean_indices[index] = -1
        try:
            if t-2 < 1:
                beta3_mean_indices[index] = -1
            elif t-2 == 1:
                beta3_mean_indices[index] = 1
            else:
                beta3_mean_indices[index] = np.log10(D3_indices[index])/np.log10(t-2) # t-2 because D3[2] = 1, so first actual index is at t=3 (index = 2)
        except (ValueError,ZeroDivisionError):
            beta3_mean_indices[index] = -1
        try:
            if t-3 < 1:
                beta4_mean_indices[index] = -1
            elif t-3 == 1:
                beta4_mean_indices[index] = 1
            else:
                beta4_mean_indices[index] = np.log10(D4_indices[index])/np.log10(t-3) # t-3 because D4[3] = 1, so first actual index is at t=4 (index = 3)
        except (ValueError,ZeroDivisionError):
            beta4_mean_indices[index] = -1
    results["beta_mean_indices"] = beta_mean_indices
    results["beta2_mean_indices"] = beta2_mean_indices
    results["beta3_mean_indices"] = beta3_mean_indices
    results["beta4_mean_indices"] = beta4_mean_indices
    for _ in ["beta_mean_indices", "beta2_mean_indices", "beta3_mean_indices", "beta4_mean_indices"]:
        results_light[_] = results[_]


    if use_D_as_D_indices == False and calculate_beta_loglogregr_indices == True: # Doing this doesn't make sense if we don't have all the steps
        if do_prints == True:
            print("Doing beta_loglogregr_indices at", datetime.now(), flush=True)
        # Here the exponent is calculated with a linregress fit in loglog scale considering ALL the time steps
        # The suffix _indices is added because the calculation is done at final points given by the indices used to store ts
        beta_loglogregr_indices = []
        beta2_loglogregr_indices = []
        beta3_loglogregr_indices = []
        beta4_loglogregr_indices = []
        for t in indices:
            beta_loglogregr_indices.append(powerLawRegr(D[:int(t)])[0])
            beta2_loglogregr_indices.append(powerLawRegr(D2[1:int(t)])[0])
            beta3_loglogregr_indices.append(powerLawRegr(D3[2:int(t)])[0])
            beta4_loglogregr_indices.append(powerLawRegr(D4[3:int(t)])[0])
        results["beta_loglogregr_indices"] = np.array(beta_loglogregr_indices)
        results["beta2_loglogregr_indices"] = np.array(beta2_loglogregr_indices)
        results["beta3_loglogregr_indices"] = np.array(beta3_loglogregr_indices)
        results["beta4_loglogregr_indices"] = np.array(beta4_loglogregr_indices)
        for _ in ["beta_loglogregr_indices", "beta2_loglogregr_indices", "beta3_loglogregr_indices", "beta4_loglogregr_indices"]:
            results_light[_] = results[_]


    if do_prints == True:
        print("Doing geom points loglogregr at", datetime.now(), flush=True)
#     # Here the exponent is calculated with a linregress fit in loglog scale considering only the geomspaced indices points
#     results["beta_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(0,i-100)+1:i+1], D_indices[indices_geom][max(0,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
#     results["beta2_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(1,i-100)+1:i+1]-1, D2_indices[indices_geom][max(1,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
#     results["beta3_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(2,i-100)+1:i+1]-2, D3_indices[indices_geom][max(2,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
#     results["beta4_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(3,i-100)+1:i+1]-3, D4_indices[indices_geom][max(3,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
    # We compute the power law fit using a lin regression in loglog scale with positive intercept using only the geomspaced indices points
    D_indices_geom = results['D_indices'][indices_geom]
    D2_indices_geom = results['D2_indices'][indices_geom]
    D3_indices_geom = results['D3_indices'][indices_geom]
    D4_indices_geom = results['D4_indices'][indices_geom]
    beta_loglogregr_indices_geom = []
    beta2_loglogregr_indices_geom = []
    beta3_loglogregr_indices_geom = []
    beta4_loglogregr_indices_geom = []
    intercept_loglogregr_indices_geom = []
    intercept2_loglogregr_indices_geom = []
    intercept3_loglogregr_indices_geom = []
    intercept4_loglogregr_indices_geom = []
    std_err_beta_loglogregr_indices_geom = []
    std_err_beta2_loglogregr_indices_geom = []
    std_err_beta3_loglogregr_indices_geom = []
    std_err_beta4_loglogregr_indices_geom = []
    std_err_intercept_loglogregr_indices_geom = []
    std_err_intercept2_loglogregr_indices_geom = []
    std_err_intercept3_loglogregr_indices_geom = []
    std_err_intercept4_loglogregr_indices_geom = []
    for i in range(len(ts_geom)):
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(0,i-100)+1:i+1], D_indices_geom[max(0,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta_loglogregr_indices_geom.append(slope)
        intercept_loglogregr_indices_geom.append(intercept)
        std_err_beta_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept_loglogregr_indices_geom.append(std_err[1])
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(1,i-100)+1:i+1]-1, D2_indices_geom[max(1,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta2_loglogregr_indices_geom.append(slope)
        intercept2_loglogregr_indices_geom.append(intercept)
        std_err_beta2_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept2_loglogregr_indices_geom.append(std_err[1])
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(2,i-100)+1:i+1]-2, D3_indices_geom[max(2,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta3_loglogregr_indices_geom.append(slope)
        intercept3_loglogregr_indices_geom.append(intercept)
        std_err_beta3_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept3_loglogregr_indices_geom.append(std_err[1])
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(3,i-100)+1:i+1]-3, D4_indices_geom[max(3,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta4_loglogregr_indices_geom.append(slope)
        intercept4_loglogregr_indices_geom.append(intercept)
        std_err_beta4_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept4_loglogregr_indices_geom.append(std_err[1])
    results["beta_loglogregr_indices_geom"] = np.array(beta_loglogregr_indices_geom)
    results["beta2_loglogregr_indices_geom"] = np.array(beta2_loglogregr_indices_geom)
    results["beta3_loglogregr_indices_geom"] = np.array(beta3_loglogregr_indices_geom)
    results["beta4_loglogregr_indices_geom"] = np.array(beta4_loglogregr_indices_geom)
    results["intercept_loglogregr_indices_geom"] = np.array(intercept_loglogregr_indices_geom)
    results["intercept2_loglogregr_indices_geom"] = np.array(intercept2_loglogregr_indices_geom)
    results["intercept3_loglogregr_indices_geom"] = np.array(intercept3_loglogregr_indices_geom)
    results["intercept4_loglogregr_indices_geom"] = np.array(intercept4_loglogregr_indices_geom)
    results["std_err_beta_loglogregr_indices_geom"] = np.array(std_err_beta_loglogregr_indices_geom)
    results["std_err_beta2_loglogregr_indices_geom"] = np.array(std_err_beta2_loglogregr_indices_geom)
    results["std_err_beta3_loglogregr_indices_geom"] = np.array(std_err_beta3_loglogregr_indices_geom)
    results["std_err_beta4_loglogregr_indices_geom"] = np.array(std_err_beta4_loglogregr_indices_geom)
    results["std_err_intercept_loglogregr_indices_geom"] = np.array(std_err_intercept_loglogregr_indices_geom)
    results["std_err_intercept2_loglogregr_indices_geom"] = np.array(std_err_intercept2_loglogregr_indices_geom)
    results["std_err_intercept3_loglogregr_indices_geom"] = np.array(std_err_intercept3_loglogregr_indices_geom)
    results["std_err_intercept4_loglogregr_indices_geom"] = np.array(std_err_intercept4_loglogregr_indices_geom)
    for _ in [
        "beta_loglogregr_indices_geom", 
        "beta2_loglogregr_indices_geom", 
        "beta3_loglogregr_indices_geom", 
        "beta4_loglogregr_indices_geom", 
        "intercept_loglogregr_indices_geom", 
        "intercept2_loglogregr_indices_geom", 
        "intercept3_loglogregr_indices_geom", 
        "intercept4_loglogregr_indices_geom", 
        "std_err_beta_loglogregr_indices_geom", 
        "std_err_beta2_loglogregr_indices_geom", 
        "std_err_beta3_loglogregr_indices_geom", 
        "std_err_beta4_loglogregr_indices_geom", 
        "std_err_intercept_loglogregr_indices_geom", 
        "std_err_intercept2_loglogregr_indices_geom", 
        "std_err_intercept3_loglogregr_indices_geom", 
        "std_err_intercept4_loglogregr_indices_geom", 
    ]:
        results_light[_] = results[_]


    # SAVE RESULTS IN A SUBFOLDER
    if do_prints == True:
        print("Saving at", datetime.now(), flush=True)
    tmp_dir = os.path.dirname(save_all_file_path)
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_dir = os.path.dirname(save_light_file_path)
    os.makedirs(tmp_dir, exist_ok=True)
    
    if save_all == True:
        with open(save_all_file_path,'wb') as fp:
            pickle.dump(results,fp)
    with open(save_light_file_path,'wb') as fp:
        pickle.dump(results_light,fp)
    
    if do_prints == True:
        print("Finished analysis at", datetime.now(), flush=True)
    if return_all == True:
        return results
    else:
        return results_light