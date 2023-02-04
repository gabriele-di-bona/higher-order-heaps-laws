import numpy as np
import random
import pickle
import os
# # Change directory to the root of the folder (this script was launched from the subfolder python_scripts)
# # All utils presuppose that we are working from the root directory of the github folder
# os.chdir("../")
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))
from datetime import datetime
from numpy.random import rand, choice
import argparse

# local utils
from analyse_sequence import *


def UMST(rho=5, nu=4,eta=1,Tmax=10000, run=0,num_to_save=1000, 
             calculate_entropies_original = False, calculate_entropies_labels = True,
             save_all=False, save_all_file_path = "./test.pkl", save_light_file_path = "./test_light.pkl",
             save_entropies_file_path = "./test_entropies.pkl",
             save_entropies_original_file_path = "./test_entropies_original.pkl",):
    '''
        Runs a simulation of the UMST (urn model with semantic triggering) with the parameters given (see Tria 2014).

        Input:
            - rho: reinforcement parameter (float). 
                At every turn add rho copies of drawn ball
            - nu: triggering parameter (int). 
                Every time a discovery is made, add nu+1 new colors in the urn
            - eta: semantic correlations parameter (float, 0 <= eta <= 1). 
                At every turn multiply temporarily the weight of all non neighbors of last drawn color by eta
            - Tmax: number of steps of the simulation (int)
            - run: Id of the run of the simulation, to avoid overwriting when doing multiple runs with same parameters
            - num_to_save: at the end of the simulation, the analysis is saved in a light dictionary,
                where only the indices related to a linspace and geomspace of length num_to_save are actually saved (int)
            - calculate_entropies_original: if True it calculates the entropies on the sequence of colors (bool)
            - calculate_entropies_labels: if True it calculates the entropies on the sequence of labels, i.e. mothers of colors (bool)
            - save_all: saves results of the simulation related to all indices, including the sequence of draws, of discovery, entropies
            - save_all_file_path: file path where to dump the results dict, if save_all == True. Subdirectories are created if not present
            - save_light_file_path: file path where to dump the light results dict. Subdirectories are created if not present

        Output:
            - if save_all is True, saves all the sequences and analysis results in a pikled dictionary
            - Saves the analysis results on a small subset of the indices on a light pickled dictionary, whose number is decided by num_to_save.
            - Returns the urn and all calculations of Heaps' law and entropy (same dictionary saved with save_all=True)
    '''
    assert type(nu) == int, "nu is not an integer, it's of type %s!"%(str(type(nu)))
    assert (eta >= 0 and eta <= 1), "eta is not between 0 and 1, it's %f!"%(eta)
    assert type(Tmax) == int, "Tmax is not an integer, it's of type %s!"%(str(type(Tmax)))

    # INITIALISATION
    # The following initialisation supposes that the for starts from t=1
    # All colors are indexed with increased order of appearance, starting from color 0
    # -1 is used for non assigned value
    sequence = -np.ones(Tmax, dtype=int)
    # At t=0 we draw ball 0
    sequence[0] = 0
    D = np.zeros(Tmax, dtype=int) # counts the number of different elememts appeared in the sequence up to time t
    D[0] = 1 
    D2 = np.zeros(Tmax, dtype=int) # counts the number of different pairs appeared in the sequence up to time t
    D3 = np.zeros(Tmax, dtype=int) # counts the number of different triplets appeared in the sequence up to time t
    D4 = np.zeros(Tmax, dtype=int) # counts the number of different quadruplets appeared in the sequence up to time t
    set_sequence = {0} # set of all different colors present in the sequence
    # in the following 3 lines dictionaries are nested, keys in first dict representing the first element of the different tuples, etc.
    set_pairs = {0:set()} # dict of set of colors representing all pairs present in the sequence
    set_triplets = {0:{}} # dict of dict of set of colors representing all pairs present in the sequence
    set_quadruplets = {0:{}} # dict of dict of dict of set of colors representing all pairs present in the sequence

    # On first step add rho copies of 0 and his nu+1 children
    urn_freq = {0: 1+rho} # The index is the color, the number is the number of balls
    if eta < 1 and eta > 0:
        tmp_urn_freq = {0: 1+rho} # it is used to draw with semantics, changing the weights
    if eta < 1 or calculate_entropies_labels == True:
        # The semantic structure created is a tree and is stored through these 3 dicts
        urn_mothers = {0:-1} # 0 does not have a mother. MEMENTO: Only 0 has -1, which has to be discarded
        urn_brothers = {0:-1} # 0 does not have a mother. MEMENTO: Only 0 has -1, which has to be discarded
        urn_children = {0:1} # only save the index of the first child. All children are from value to value+nu (total of nu+1)# add the children
    max_id_urn = 0
    for i in range(max_id_urn+1,max_id_urn+nu+2):
        urn_freq[i] = 1
        if eta > 0 and eta < 1:
            tmp_urn_freq[i] = 1
        if eta < 1 or calculate_entropies_labels == True:
            urn_mothers[i] = 0 
            urn_brothers[i] = 1
    max_id_urn += nu+1
    urn_tot = 1+rho+nu+1 # keeps track of the total number of balls in the urn
    

    last_color = -1 # color in the sequence at t-1
    last_color2 = -1 # color in the sequence at t-2
    last_color3 = -1 # color in the sequence at t-3

    # Beginning of simulation
    print("Starting simulation at", datetime.now(), flush=True)
    for t in range(1,Tmax):
        if (Tmax >= 100) and (t%int(Tmax/100) == 0):
            print("Done %d/100, rho=%.5f, nu=%d, eta=%.5f, Tmax=%d, run=%d at"%(int(t/Tmax*100), rho, nu, eta, Tmax, run), datetime.now(), flush=True)
        last_color3 = last_color2
        last_color2 = last_color
        last_color = sequence[t-1]

        # Color sampling
        if eta == 1:
            # Draw a random ball from the urn
            ball_tmp = random.randint(0,urn_tot-1) # -1 is because random.randint includes both extremes
            color = 0
            sum_tmp = urn_freq[0]
            while ball_tmp >= sum_tmp:
                color += 1
                sum_tmp += urn_freq[color]
        else:
            # find all neighbors of last_color
            neighbors = {last_color}
            if last_color != 0:
                # 0 is root, so doesn't have mother or brothers
                neighbors = neighbors.union([urn_brothers[last_color]+i for i in range(nu+1)])
                neighbors.add(urn_mothers[last_color])
            neighbors = neighbors.union([urn_children[last_color]+i for i in range(nu+1)])

            if eta == 0:
                # in this case it can only move only to one of the neighbors
                weights_neighbors = {}
                for tmp in neighbors:
                    # copy frequency in urn
                    weights_neighbors[int(tmp)] = urn_freq[int(tmp)]
                if len(weights_neighbors) == 0: # This shouldn't happen
                    print(f"At step {t} from last_color={last_color} there are no colors a max dist 1 in the urn", flush=True)
                # draw random ball from weights_neighbors
                weights_neighbors_sum = sum(weights_neighbors.values())
                tmp_rand = rand()*weights_neighbors_sum
                tmp_sum = 0
                for i, j in weights_neighbors.items():
                    tmp_sum += j
                    if tmp_sum > tmp_rand:
                        color = i
                        break
                else:
                    # Did not enter in the if, number was too large so pick the last one
                    print("(rho=%.3f,nu=%d,eta=%.5f)"%(rho,nu,eta),"At step", t, "user", user, "got overflow", flush=True)
                    color = i
            else:
                # Adjust neighbors weight temporarily to enhance semantics correlation
                for tmp in neighbors:
                    tmp_urn_freq[tmp] = urn_freq[tmp] / eta
                # draw a ball
                adjusted_urn_sum = sum(tmp_urn_freq.values())
                tmp_rand = rand()*adjusted_urn_sum
                tmp_sum = 0
                for i, j in tmp_urn_freq.items():
                    tmp_sum += j
                    if tmp_sum > tmp_rand:
                        color = i
                        break
                else:
                    # Did not enter in the if, number was too large so pick the last one
                    print("(rho=%.3f,nu=%d,eta=%.5f)"%(rho,nu,eta),"At step", t, "user", user, "got overflow", flush=True)
                    color = i
                # adjust tmp_urn_freq back to the real frequencies in the urn
                for tmp in neighbors:
                    tmp_urn_freq[tmp] = urn_freq[int(tmp)]

        # Update sequence after draw 
        sequence[t] = color
        urn_freq[color] += rho # reinforcement
        if eta > 0 and eta < 1:
            tmp_urn_freq[color] += rho
        urn_tot += rho
        
        # Check if there are new elements/pairs/triplets/quadruplets
        if color not in set_sequence:
            # This means that there is a new color!
            # Triggering of new elements in the urn
            for i in range(1,nu+2):
                urn_freq[max_id_urn + i] = 1
                if eta < 1 and eta > 0:
                    tmp_urn_freq[max_id_urn + i] = 1 
                if eta < 1 or calculate_entropies_labels == True:
                    # add new links in the tree (triggering)
                    urn_mothers[max_id_urn + i] = color
                    urn_brothers[max_id_urn + i] = max_id_urn + 1
                    urn_children[color] = max_id_urn + 1    
            max_id_urn += nu+1
            urn_tot += nu+1
            set_sequence.add(color)
            D[t] = D[t-1]+1

            # If the color is new, it means that there is also a new pair, triplet and quadruplet
            set_pairs[color] = set()
            set_triplets[color] = {}
            set_quadruplets[color] = {}
            set_pairs[last_color].add(color)
            D2[t] = D2[t-1]+1
            set_triplets[last_color][color] = set()
            set_quadruplets[last_color][color] = {}
            if last_color2 != -1:
                set_triplets[last_color2][last_color].add(color)
                D3[t] = D3[t-1]+1
                set_quadruplets[last_color2][last_color][color] = set()
            if last_color3 != -1:
                set_quadruplets[last_color3][last_color2][last_color].add(color)
                D4[t] = D4[t-1]+1
        else:
            # color is already present in the sequence, checking for pair
            D[t] = D[t-1]
            if color not in set_pairs[last_color]:
                # This means that there is a new pair, hence also triplet and quadruplet
                set_pairs[last_color].add(color)
                D2[t] = D2[t-1]+1
                set_triplets[last_color][color] = set()
                set_quadruplets[last_color][color] = {}
                if last_color2 != -1:
                    set_triplets[last_color2][last_color].add(color)
                    D3[t] = D3[t-1]+1
                    set_quadruplets[last_color2][last_color][color] = set()
                if last_color3 != -1:
                    set_quadruplets[last_color3][last_color2][last_color].add(color)
                    D4[t] = D4[t-1]+1
            else:
                # color and last_color:color are present in sequence, checking for triplet
                D2[t] = D2[t-1]
                if last_color2 != -1 and color not in set_triplets[last_color2][last_color]:
                    # This means that there is a new triplet, hence also quadruplet
                    set_triplets[last_color2][last_color].add(color)
                    D3[t] = D3[t-1]+1
                    set_quadruplets[last_color2][last_color][color] = set()
                    if last_color3 != -1:
                        set_quadruplets[last_color3][last_color2][last_color].add(color)
                        D4[t] = D4[t-1]+1
                else:
                    # color, last_color:color, last_color2:last_color:color are present in sequence, checking for quadruplet
                    D3[t] = D3[t-1]
                    if last_color3 != -1 and color not in set_quadruplets[last_color3][last_color2][last_color]:    
                        # This means that there is a new quadruplet
                        set_quadruplets[last_color3][last_color2][last_color].add(color)
                        D4[t] = D4[t-1]+1
                    else:
                        # not even a new quadruplet
                        D4[t] = D4[t-1]
    # End of simulation                    
    print("Done 100/100, rho=%.3f, nu=%d, run=%d at"%(rho, nu, run), datetime.now(), flush=True)

    
    
    if calculate_entropies_labels == True:
        sequence_labels = sequence.copy()
        for t,color in enumerate(sequence):
            mother = urn_mothers[color]
            sequence_labels[t] = mother
    else:
        sequence_labels = None
    
    results = analyse_sequence(sequence=sequence, num_to_save=num_to_save, find_novelties = False, 
                               use_D_as_D_indices = False, indices = [],
                               calculate_entropies_original = calculate_entropies_original, 
                               calculate_entropies_labels = calculate_entropies_labels,
                               D=D, D2=D2, D3=D3, D4=D4, sequence_labels = sequence_labels,
                               save_all=False, save_all_file_path = save_all_file_path, 
                               save_light_file_path = save_light_file_path,
                               save_entropies_file_path = save_entropies_file_path,
                               save_entropies_original_file_path = save_entropies_original_file_path,
                               calculate_beta_loglogregr_indices = False,
                               do_prints = True, return_all = True)
    
    if save_all == True:
        # SAVE RESULTS IN A SUBFOLDER
        print("Saving all at", datetime.now(), flush=True)
        results["urn_freq"] = urn_freq
        results["urn_mothers"] = urn_mothers
        results["set_sequence"] = set_sequence
        results["set_pairs"] = set_pairs
        results["set_triplets"] = set_triplets
        results["set_quadruplets"] = set_quadruplets

        with open(save_all_file_path,'wb') as fp:
            pickle.dump(results,fp)
    
    print("Finished UMST analysis at", datetime.now(), flush=True)
    return results

    