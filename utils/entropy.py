import numpy as np
from numba import njit


@njit
def entropyCalc(seq, labels_freq_keys, labels_freq_values):
    '''
    For every different label in the sequence, compute the normalized Shannon entropy of that label in the sequence (see Tria 2014).
    
    Input:
        - seq: list or 1d np.array containing all the ids corresponding to the labels
        - labels_freq_keys: list of int64 containing the label ids in the sequence
        - labels_freq_values: list of int64 containing the frequency of the label in the sequence, ordered like labels_freq_keys
    
    Output:
        - Returns a list of lists of three elements, respectively the label, the frequency f and the respective renormalized entropy.
    '''
    entropies = [[1.,1,float(i)] for i in range(0)]
    ordLabels = set()
    for t in range(len(seq)): 
        label = int(seq[t])
        if (label in ordLabels) == False and label != -1:
            ordLabels.add(label)
            subseq = seq[t:] # consider only the subsequency starting from first appearance of the label
            index = np.where(labels_freq_keys == label)[0][0]
            f = labels_freq_values[index]
            l = len(subseq)
            entropy = 0.
            if f > 1:
                for i in range(f):
                    freq = 0
                    for j in range(round(l/f*i),round(l/f*(i+1))):
                        if subseq[j] == label:
                            freq += 1
                    if freq != 0:
                        entropy -= freq/f * np.log10(freq/f)
                entropy /= np.log10(f) # normalization
                entropies.append([label,f,entropy])
    return entropies


def get_dict_freq_list_entropies(entropies = []):
    '''
        Transforms the list of entropies created in entropyCalc into a {frequency:[entropies]} dictionary.
        
        Input:
            - entropies: list of 3-tuples [[label,f,entropy]]
        
        Output:
            - dict_freq_list_entropies: {frequency:[entropies]} dictionary.
    '''
    dict_freq_list_entropies = {}
    for label,f,entropy in entropies:
        try:
            dict_freq_list_entropies[f].append(entropy)
        except KeyError:
            dict_freq_list_entropies[f] = [entropy]
    return dict_freq_list_entropies


def get_weighted_difference_entropies(
    entropies, 
    entropies_glob, 
    entropies_sorted_keys = None,
    entropies_provided_is_dict_freq_entropies_list = False,
):
    '''
        Calculates the weighted difference between the entropies values 
            and the corresponding ones calculated on the reshuffled sequence.
        The difference is calculated by considering every label separately.
        This function also calculates the average at each frequency, 
            also providing the number of elements at each frequency (weight).
        
        Input:
            - entropies: list of lists of 3 elements, respectively the label, 
                its frequency in the sequence, and the calculated entropy on the original sequence
            - entropies_glob: list of lists of 3 elements, respectively the label, 
                its frequency in the sequence, and the calculated entropy on the reshuffled sequence
            - entropies_sorted_keys: list of sorted frequencies from which the entropy is calculated.
                If None, it is created from entropies on the spot.
            - entropies_provided_is_dict_freq_entropies_list: if True, then considers entropy the same as dict_freq_list_entropies (also for _glob).
                If True, then the dictionary of frequency to entropies list is calculated on the spot from entropy (result of entropyCalc).
        
        Output:
            - weighted_diff: weighted difference between entropies and entropies_glob
            - entropies_sorted_keys: all the frequencies present in the sequence, increasingly ordered
            - entropies_sorted_weights: normalized number of labels with the same frequency, 
                where frequencies are taken with the same order as entropies_sorted_keys
            - mean_entropies: dict where keys are frequencies and values are the average entropy for all labels with that frequency,
                as calculated from entropies
            - mean_entropies_glob dict where keys are frequencies and values are the average entropy for all labels with that frequency,
                as calculated from entropies_glob
    '''
#     # OLD
#     mean_entropies = {}
#     mean_entropies_glob = {}
#     weighted_diff = 0
#     total_count = 0
#     for index,f,ent in entropies:
#         if f not in mean_entropies:
#             mean_entropies[f] = [ent]
#         else:
#             mean_entropies[f].append(ent)
#         total_count += 1
#         weighted_diff -= ent
#     for index,f,ent in entropies_glob:
#         if f not in mean_entropies_glob:
#             mean_entropies_glob[f] = [ent]
#         else:
#             mean_entropies_glob[f].append(ent)
#         weighted_diff += ent
#     if total_count > 0:
#         weighted_diff /= total_count
    if entropies_provided_is_dict_freq_entropies_list == False:
        dict_freq_list_entropies = get_dict_freq_list_entropies(entropies = entropies)
        dict_freq_list_entropies_glob = get_dict_freq_list_entropies(entropies = entropies_glob)
    else:
        dict_freq_list_entropies = entropies
        dict_freq_list_entropies_glob = entropies_glob
    mean_entropies = {}
    mean_entropies_glob = {}
    if entropies_sorted_keys == None:
        entropies_sorted_keys = sorted(list(dict_freq_list_entropies.keys()))
    entropies_sorted_weights = []
    weighted_diff = 0
    total_count = 0
    for f in entropies_sorted_keys:
        try:
            entropies_list = dict_freq_list_entropies[f]
            mean_entropies[f] = np.mean(entropies_list)
            mean_entropies_glob[f] = np.mean(dict_freq_list_entropies_glob[f])
            num_labels = len(entropies_list)
            entropies_sorted_weights.append(num_labels)
            weighted_diff += (mean_entropies_glob[f] - mean_entropies[f]) * num_labels
            total_count += num_labels
        except Exception as e:
            print(f'ERROR for frequency {f} in entropy difference calculations: {e}')
    if total_count > 0:
        entropies_sorted_weights = list(np.array(entropies_sorted_weights)/total_count)
        weighted_diff /= total_count
    return weighted_diff, entropies_sorted_keys, entropies_sorted_weights, mean_entropies, mean_entropies_glob

