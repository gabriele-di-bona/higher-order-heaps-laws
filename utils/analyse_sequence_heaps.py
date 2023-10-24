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
from tqdm import tqdm

# local utils
from powerlaw_regr import *
from entropy import *
from analyse_higher_entropy import *
from index_sequence import *


def compute_Heaps(
    sequence,
):
    """
        Compute Heaps of an indexed sequence.
    """
    curr_D = 0
    D = np.zeros(len(sequence))
    first_appear_sequence_dict = {}
    for t,x in enumerate(sequence):
        if x not in first_appear_sequence_dict:
            curr_D += 1
            first_appear_sequence_dict[x] = t
        D[t] = curr_D
    return D, first_appear_sequence_dict

def put_together_Heaps(
    ordered_dumped_D_files_paths = [],
    ordered_dumped_dict_files_paths = [],
    lengths_subsequences = [],
    Tmax = 0,
    compute_D_indices = False,
    indices = [],
    tmp_folder = "",
):
    previous_Tmax = 0
    for curr_file_id, file_path in enumerate(ordered_dumped_dict_files_paths):
        with open(file_path, 'rb') as fp:
            tmp_first_appear_sequence_dict = joblib.load(fp)
        to_dump_tmp_first_appear_sequence_dict = tmp_first_appear_sequence_dict.copy()
        for previous_file_path in ordered_dumped_dict_files_paths[:curr_file_id]:
            with open(file_path, 'rb') as fp:
                other_first_appear_sequence_dict = joblib.load(fp)
            for x, local_t in tmp_first_appear_sequence_dict.items():
                try:
                    _ = other_first_appear_sequence_dict[x]
                    del to_dump_tmp_first_appear_sequence_dict[x]
                except:
                    pass
            # update dict, so that you don't double check elements already found
            tmp_first_appear_sequence_dict = to_dump_tmp_first_appear_sequence_dict.copy()
        # dump only the times
        with open(os.path.join(tmp_folder, f'{curr_file_id}.pkl'), 'wb') as fp:
            joblib.dump(np.array(sorted(list(to_dump_tmp_first_appear_sequence_dict.values())), dtype = np.int) + previous_Tmax, fp)
        previous_Tmax += lengths_subsequences[curr_file_id]
        
    if compute_D_indices == False:
        D = np.zeros(Tmax)
        for curr_file_id range(len(ordered_dumped_dict_files_paths)):
            with open(os.path.join(tmp_folder, f'{curr_file_id}.pkl'), 'wb') as fp:
                novelty_times = joblib.load(fp)
            for novelty_time in novelty_times:
                D[novelty_time:] += 1
    else:
        D = np.zeros(len(indices))
        curr_D = 0
        curr_index_D = 0
        for curr_file_id range(len(ordered_dumped_dict_files_paths)):
            with open(os.path.join(tmp_folder, f'{curr_file_id}.pkl'), 'wb') as fp:
                novelty_times = joblib.load(fp)
            curr_index_novelty_times = 0
            curr_novelty_time = novelty_times[curr_index_novelty_times]
            while curr_index_novelty_times < len(novelty_times):
                if curr_novelty_time < indices[curr_index_D]:
                    curr_index_novelty_times += 1
                    curr_novelty_time = novelty_times[curr_index_novelty_times]
                    curr_D += 1
                else:
                    D[curr_index_D] = curr_D
                    curr_index_D += 1
                    bound_t = indices[curr_index_D]
            
    return D
    

def analyse_sequence(
    sequence=np.array([]), 
    sequence_pairs=None, 
    consider_temporal_order_in_tuples=True, 
    num_to_save=1000, 
    indices = [],
    use_D_as_D_indices = False, 
    D=None, 
    D2=None, 
    do_also_D2 = True,
    save_all=False, 
    save_all_file_path = "./test.pkl", 
    save_light_file_path = "./test_light.pkl",
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
            - D2: array containing for each timestep the number of new pairs in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - D3: array containing for each timestep the number of new triplets in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - D4: array containing for each timestep the number of new quadrupltes in the sequence up to that point (np.array(int)).
                If not provided (None), then it is calculated from sequence.
            - do_also_D2: considers also the 2nd order (bool)
            - do_also_D3: considers also the 3th order (bool)
            - do_also_D4: considers also the 4th order (bool)
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
        D = compute_Heaps(sequence, do_prints = do_prints) # counts the number of different elements appeared in the sequence up to time t
        if do_also_D2:
            sequence_pairs, indexed_sequence_pairs = get_sequence_pairs(
                sequence,
                consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
            )
            D2 = np.array([0]+list(compute_Heaps(indexed_sequence_pairs))) # counts the number of different pairs appeared in the sequence up to time t
            if do_also_D3:
                sequence_triplets, indexed_sequence_triplets = get_sequence_pairs(
                    indexed_sequence_pairs,
                    consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
                )
                D3 = np.array([0,0]+list(compute_Heaps(indexed_sequence_triplets, do_prints = do_prints))) # counts the number of different triplets appeared in the sequence up to time t
                if do_also_D4:
                    sequence_triplets, indexed_sequence_quadruplets = get_sequence_pairs(
                        indexed_sequence_triplets,
                        consider_temporal_order_in_tuples=consider_temporal_order_in_tuples, 
                    )
                    D4 = np.array([0,0,0]+list(compute_Heaps(indexed_sequence_quadruplets, do_prints = do_prints))) # counts the number of different quadruplets appeared in the sequence up to time t
                    
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
        if do_also_D2:
            D2_indices = D2[indices]
            if do_also_D3:
                D3_indices = D3[indices]
                if do_also_D4:
                    D4_indices = D4[indices]
    else:
        D_indices = D
        if do_also_D2:
            D2_indices = D2
            if do_also_D3:
                D3_indices = D3
                if do_also_D4:
                    D4_indices = D4
        
    if use_D_as_D_indices == False:
        results = { \
                    "D":D, \
#                     "D2":D2, "D3":D3, "D4":D4, \
                    "ts":ts, "indices":indices, "ts_geom":ts_geom, "indices_geom":indices_geom, \
                    "D_indices":D_indices, \
#                     "D2_indices":D2_indices, "D3_indices":D3_indices, "D4_indices":D4_indices \
                  }
        if do_also_D2:
            results['D2'] = D2
            results['D2_indices'] = D2_indices
            if do_also_D3:
                results['D3'] = D3
                results['D3_indices'] = D3_indices
                if do_also_D4:
                    results['D4'] = D4
                    results['D4_indices'] = D4_indices
        if sequence is not None:
            results["sequence"] = sequence
        if sequence_labels is not None:
            results["sequence_labels"] = sequence_labels
    else:
        results = { \
                    "ts":ts, "indices":indices, "ts_geom":ts_geom, "indices_geom":indices_geom, \
                    "D_indices":D_indices, \
#                     "D2_indices":D2_indices, "D3_indices":D3_indices, "D4_indices":D4_indices \
                  }
        if do_also_D2:
            results['D2'] = D2
            results['D2_indices'] = D2_indices
            if do_also_D3:
                results['D3'] = D3
                results['D3_indices'] = D3_indices
                if do_also_D4:
                    results['D4'] = D4
                    results['D4_indices'] = D4_indices

    results_light = {}
    for k in ["ts", "indices", "ts_geom", "indices_geom", \
              "D_indices", "D2_indices", "D3_indices", "D4_indices"]:
        try:
            results_light[k] = results[k]
        except:
            pass


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
    if do_also_D2:
        beta2_mean_indices = np.zeros(len(D2_indices))
        if do_also_D3:
            beta3_mean_indices = np.zeros(len(D3_indices))
            if do_also_D4:
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
        if do_also_D2:
            try:
                if t-1 < 1:
                    beta2_mean_indices[index] = -1
                elif t-1 == 1:
                    beta2_mean_indices[index] = 1
                else:
                    beta2_mean_indices[index] = np.log10(D2_indices[index])/np.log10(t-1) # t-1 because D2[1] = 1, so first actual index is at t=2 (index = 1)
            except (ValueError,ZeroDivisionError):
                beta2_mean_indices[index] = -1
            if do_also_D3:
                try:
                    if t-2 < 1:
                        beta3_mean_indices[index] = -1
                    elif t-2 == 1:
                        beta3_mean_indices[index] = 1
                    else:
                        beta3_mean_indices[index] = np.log10(D3_indices[index])/np.log10(t-2) # t-2 because D3[2] = 1, so first actual index is at t=3 (index = 2)
                except (ValueError,ZeroDivisionError):
                    beta3_mean_indices[index] = -1
                if do_also_D4:
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
    if do_also_D2:
        results["beta2_mean_indices"] = beta2_mean_indices
        if do_also_D3:
            results["beta3_mean_indices"] = beta3_mean_indices
            if do_also_D3:
                results["beta4_mean_indices"] = beta4_mean_indices
    for _ in ["beta_mean_indices", "beta2_mean_indices", "beta3_mean_indices", "beta4_mean_indices"]:
        try:
            results_light[_] = results[_]
        except:
            pass


    if use_D_as_D_indices == False and calculate_beta_loglogregr_indices == True: # Doing this doesn't make sense if we don't have all the steps
        if do_prints == True:
            print("Doing beta_loglogregr_indices at", datetime.now(), flush=True)
        # Here the exponent is calculated with a linregress fit in loglog scale considering ALL the time steps
        # The suffix _indices is added because the calculation is done at final points given by the indices used to store ts
        beta_loglogregr_indices = []
        if do_also_D2:
            beta2_loglogregr_indices = []
            if do_also_D3:
                beta3_loglogregr_indices = []
                if do_also_D4:
                    beta4_loglogregr_indices = []
        for t in indices:
            beta_loglogregr_indices.append(powerLawRegr(D[:int(t)])[0])
            if do_also_D2:
                beta2_loglogregr_indices.append(powerLawRegr(D2[1:int(t)])[0])
                if do_also_D3:
                    beta3_loglogregr_indices.append(powerLawRegr(D3[2:int(t)])[0])
                    if do_also_D4:
                        beta4_loglogregr_indices.append(powerLawRegr(D4[3:int(t)])[0])
        results["beta_loglogregr_indices"] = np.array(beta_loglogregr_indices)
        if do_also_D2:
            results["beta2_loglogregr_indices"] = np.array(beta2_loglogregr_indices)
            if do_also_D3:
                results["beta3_loglogregr_indices"] = np.array(beta3_loglogregr_indices)
                if do_also_D4:
                    results["beta4_loglogregr_indices"] = np.array(beta4_loglogregr_indices)
        for _ in ["beta_loglogregr_indices", "beta2_loglogregr_indices", "beta3_loglogregr_indices", "beta4_loglogregr_indices"]:
            try:
                results_light[_] = results[_]
            except:
                pass


    if do_prints == True:
        print("Doing geom points loglogregr at", datetime.now(), flush=True)
#     # Here the exponent is calculated with a linregress fit in loglog scale considering only the geomspaced indices points
#     results["beta_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(0,i-100)+1:i+1], D_indices[indices_geom][max(0,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
#     results["beta2_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(1,i-100)+1:i+1]-1, D2_indices[indices_geom][max(1,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
#     results["beta3_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(2,i-100)+1:i+1]-2, D3_indices[indices_geom][max(2,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
#     results["beta4_loglogregr_indices_geom"] = np.array([powerLawRegrPoints(ts_geom[max(3,i-100)+1:i+1]-3, D4_indices[indices_geom][max(3,i-100)+1:i+1])[0] for i in range(len(ts_geom))])
    # We compute the power law fit using a lin regression in loglog scale with positive intercept using only the geomspaced indices points
    D_indices_geom = results['D_indices'][indices_geom]
    beta_loglogregr_indices_geom = []
    intercept_loglogregr_indices_geom = []
    std_err_beta_loglogregr_indices_geom = []
    std_err_intercept_loglogregr_indices_geom = []
    for i in range(len(ts_geom)):
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(0,i-100)+1:i+1], D_indices_geom[max(0,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta_loglogregr_indices_geom.append(slope)
        intercept_loglogregr_indices_geom.append(intercept)
        std_err_beta_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept_loglogregr_indices_geom.append(std_err[1])
    results["beta_loglogregr_indices_geom"] = np.array(beta_loglogregr_indices_geom)
    results["intercept_loglogregr_indices_geom"] = np.array(intercept_loglogregr_indices_geom)
    results["std_err_beta_loglogregr_indices_geom"] = np.array(std_err_beta_loglogregr_indices_geom)
    results["std_err_intercept_loglogregr_indices_geom"] = np.array(std_err_intercept_loglogregr_indices_geom)
    
    D2_indices_geom = results['D2_indices'][indices_geom]
    beta2_loglogregr_indices_geom = []
    intercept2_loglogregr_indices_geom = []
    std_err_beta2_loglogregr_indices_geom = []
    std_err_intercept2_loglogregr_indices_geom = []
    for i in range(len(ts_geom)):
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(1,i-100)+1:i+1]-1, D2_indices_geom[max(1,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta2_loglogregr_indices_geom.append(slope)
        intercept2_loglogregr_indices_geom.append(intercept)
        std_err_beta2_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept2_loglogregr_indices_geom.append(std_err[1])
    results["beta2_loglogregr_indices_geom"] = np.array(beta2_loglogregr_indices_geom)
    results["intercept2_loglogregr_indices_geom"] = np.array(intercept2_loglogregr_indices_geom)
    results["std_err_beta2_loglogregr_indices_geom"] = np.array(std_err_beta2_loglogregr_indices_geom)
    results["std_err_intercept2_loglogregr_indices_geom"] = np.array(std_err_intercept2_loglogregr_indices_geom)
    
    D3_indices_geom = results['D3_indices'][indices_geom]
    beta3_loglogregr_indices_geom = []
    intercept3_loglogregr_indices_geom = []
    std_err_beta3_loglogregr_indices_geom = []
    std_err_intercept3_loglogregr_indices_geom = []
    for i in range(len(ts_geom)):
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(2,i-100)+1:i+1]-2, D3_indices_geom[max(2,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta3_loglogregr_indices_geom.append(slope)
        intercept3_loglogregr_indices_geom.append(intercept)
        std_err_beta3_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept3_loglogregr_indices_geom.append(std_err[1])
    results["beta3_loglogregr_indices_geom"] = np.array(beta3_loglogregr_indices_geom)
    results["intercept3_loglogregr_indices_geom"] = np.array(intercept3_loglogregr_indices_geom)
    results["std_err_beta3_loglogregr_indices_geom"] = np.array(std_err_beta3_loglogregr_indices_geom)
    results["std_err_intercept3_loglogregr_indices_geom"] = np.array(std_err_intercept3_loglogregr_indices_geom)
    
    D4_indices_geom = results['D4_indices'][indices_geom]
    beta4_loglogregr_indices_geom = []
    intercept4_loglogregr_indices_geom = []
    std_err_beta4_loglogregr_indices_geom = []
    std_err_intercept4_loglogregr_indices_geom = []
    for i in range(len(ts_geom)):
        slope, intercept, std_err = powerLawRegrPoints(ts_geom[max(3,i-100)+1:i+1]-3, D4_indices_geom[max(3,i-100)+1:i+1], intercept_left_lim = 0, intercept_right_lim = np.inf)
        beta4_loglogregr_indices_geom.append(slope)
        intercept4_loglogregr_indices_geom.append(intercept)
        std_err_beta4_loglogregr_indices_geom.append(std_err[0])
        std_err_intercept4_loglogregr_indices_geom.append(std_err[1])
    results["beta4_loglogregr_indices_geom"] = np.array(beta4_loglogregr_indices_geom)
    results["intercept4_loglogregr_indices_geom"] = np.array(intercept4_loglogregr_indices_geom)
    results["std_err_beta4_loglogregr_indices_geom"] = np.array(std_err_beta4_loglogregr_indices_geom)
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
        try:
            results_light[_] = results[_]
        except:
            pass


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