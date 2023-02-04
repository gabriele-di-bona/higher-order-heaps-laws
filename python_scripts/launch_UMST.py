import numpy as np
import pickle
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

# local utils
from powerlaw_regr import *
from UMST import *

    
    
    
# Beginning of MAIN
parser = argparse.ArgumentParser(description='Run multiple UMST simulations with different parameters and put them together, with complete analysis.')


parser.add_argument("-ID", "--ID", type=int,
    help="The ID of the simulation, used to get the correct parameters and run [default 1]",
    default=1)

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

parser.add_argument("-putTogether", "--putTogether", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If False it does the simulation, otherwise recovers all runs with the same parameters and analyse the average. [default False]",
    default=False)

parser.add_argument("-save_all", "--saveAll", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y']),
    help="If False it saves values at certain indices, otherwise it saves all. [default False]",
    default=False)

arguments = parser.parse_args()
ID = arguments.ID - 1 # unfortunately I have to start this argument from 1 in shell. The -1 is added to start from 0
rho = arguments.rho
eta = arguments.eta
assert (0 <= eta  and eta <= 1), "eta is not within the correct bounds, with value %f"%eta
starting_nu = arguments.starting_nu
ending_nu = arguments.ending_nu
Tmax = arguments.Tmax
putTogether = arguments.putTogether
save_all = arguments.saveAll


# I'm doing 100 runs per each set of parameters, with nu going from 1 to rho-1

# PARAMETERS
num_parameters = ending_nu - starting_nu + 1
nu = starting_nu + ID % num_parameters
run = int(ID/num_parameters)
num_to_save = 1000


workdir = f"./data/UMST/simulations/analysis/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/"


    
if putTogether == False:
    # RUN THE SIMULATION
#     if run < 10:
#         # Saving the big file in scratch only for the first 10 runs, otherwise this will saturate memory
#         save_all=True
#     else:
#         save_all=False
    print("rho=%.5f, nu=%d, eta=%.5f, Tmax=%d, run=%d"%(rho,nu,eta,Tmax,run))
    
    save_all_file_path = os.path.join(workdir, f"UMT_run_{run}.pkl")
    save_light_file_path = os.path.join(workdir, f"UMT_light_run_{run}.pkl") 
    save_entropies_file_path = os.path.join(workdir, f"UMT_entropy_run_{run}.pkl") 
    save_entropies_original_file_path = os.path.join(workdir, f"UMT_entropy_original_run_{run}.pkl") 
    
    if (save_all == False and os.path.exists(save_light_file_path) == False) or (save_all == True and (os.path.exists(save_all_file_path) == False or os.path.exists(save_light_file_path) == False)) :
        # This way if the simulation is already done it's not repeated
        # Some simulations with too low nu with respect to rho and eta might give sequences where no labels are repeated, 
        # which gives an error on the calculation (at least a frequency muuust be greater than 1!), so those simulations are not saved.
        # Moreover, you can use this if else if you want to add other successive runs in a second moment, without doing again all already done
        results = UMST(rho=rho, nu=nu, eta=eta,Tmax=Tmax, run=run,num_to_save=num_to_save, 
                 calculate_entropies_original = False, calculate_entropies_labels = True,
                 save_all=save_all, save_all_file_path = save_all_file_path, save_light_file_path = save_light_file_path,
                 save_entropies_file_path=save_entropies_file_path, save_entropies_original_file_path=save_entropies_original_file_path)
        
        if save_all == True:
            # SAVE ORIGINAL SEQUENCES!
            sequence = results['sequence']
            sequence_labels = results['sequence_labels']
            os.makedirs(os.path.dirname(f"./data/UMST/simulations/raw_sequences/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/f'{run}.txt"), exist_ok=True)
            f = open(f"./data/UMST/simulations/raw_sequences/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/f'{run}.txt", "a")
            f.writelines([str(i) for i in sequence])
            f.close()
            os.makedirs(os.path.dirname(f"./data/UMST/simulations/raw_sequences_labels/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/{run}.txt"), exist_ok=True)
            f = open(f"./data/UMST/simulations/raw_sequences_labels/rho_{rho:.5f}/nu_{nu}/Tmax_{Tmax}/eta_{eta:.5f}/{run}.txt", "a")
            f.writelines([str(i) for i in sequence_labels])
            f.close()
    else:
        print("File already present")
    
    
else:
    # PUT TOGETHER DIFFERENT RUNS WITH SAME PARAMETERS AND DO ANALYSIS
    # Let's use the light results pickles
    save_all_file_path = os.path.join(workdir, "average_UMT_results.pkl")
    save_light_file_path = os.path.join(workdir, "average_UMT_light_results.pkl") 

    # Let's create the average sequence!
    print("Loading all done simulations at", datetime.now(), flush=True)
    if save_all == True:
        # Use save_all files, i.e. f"UMT_run_{run}.pkl"
        paths = glob.glob(workdir+'*UMT_run*.pkl')
    else:
        paths = glob.glob(workdir+'*UMT_light_run_*.pkl')
    if len(paths) == 0:
        print(f'No simulations for rho = {rho:.5f}, nu = {nu}, Tmax = {Tmax}, eta = {eta:.5f}')
        exit()
    with open(paths[0],'rb') as fp:
        results = pickle.load(fp)
    if results['ts'][-1] == Tmax - 1 or results['D_indices'][0] > 1:
        # adjust the results from the old setup to the new one
        right_len_indices = len(results['indices'])+1
        right_len_indices_geom = len(results['indices_geom'])+1
        for _ in results.keys():
            if _ in ['indices','indices_geom']:
                results[_] = [1] + list(np.array(results[_])+1)
            elif _ in ['ts', 'ts_geom']:
                results[_] = np.array([1] + list(np.array(results[_])+1))
            elif _ in ['D_indices']:
                results[_] = np.array([1] + list(np.array(results[_])))
            elif _ in ['D2_indices', 'D3_indices','D4_indices']:
                results[_] = np.array([0] + list(np.array(results[_])))
            else:
                try:
                    if 'geom' in _ and len(results[_]) == right_len_indices_geom - 1:
                        results[_] = np.array(list(results[_])+results[_][-1])
                    elif len(results[_]) == right_len_indices - 1:
                        results[_] = np.array(list(results[_])+results[_][-1])
                    else:
                        print('problem with', _, 'in', paths[0])
                except TypeError:
                    pass
    else:
        right_len_indices = len(results['indices'])
        right_len_indices_geom = len(results['indices_geom'])
    for _ in list(results.keys()):
        if _ == "indices" or _ == "indices_geom":
            continue
        if type(results[_]) in [list, type(np.array([]))]:
            results[_+"_list_finals"] = [results[_][-1]]
            results[_+"_all"] = [results[_]]
        else:
            results[_+"_list"] = [results[_]]
    counted = 1
    for path in paths[1:]:
        try:
            with open(path,'rb') as fp:
                results_tmp = pickle.load(fp)
            if results_tmp['ts'][-1] == Tmax - 1 or results_tmp['D_indices'][0] > 1:
                # adjust the results from the old setup to the new one
                for _ in results_tmp.keys():
                    if _ in ['indices','indices_geom']:
                        results_tmp[_] = [1] + list(np.array(results_tmp[_])+1)
                    elif _ in ['ts', 'ts_geom']:
                        results_tmp[_] = np.array([1] + list(np.array(results_tmp[_])+1))
                    elif _ in ['D_indices']:
                        results_tmp[_] = np.array([1] + list(np.array(results_tmp[_])))
                    elif _ in ['D2_indices', 'D3_indices','D4_indices']:
                        results_tmp[_] = np.array([0] + list(np.array(results_tmp[_])))
                    else:
                        try:
                            if 'geom' in _ and len(results[_]) == right_len_indices_geom - 1:
                                results_tmp[_] = np.array(list(results_tmp[_])+results_tmp[_][-1])
                            elif len(results[_]) == right_len_indices - 1:
                                results_tmp[_] = np.array(list(results_tmp[_])+results_tmp[_][-1])
                            else:
                                print('problem with', _, 'in', path)
                        except TypeError:
                            pass
            for _ in results_tmp.keys():
                try:
                    if _ == "indices" or _ == "indices_geom":
                        continue
                    results[_] += results_tmp[_]
                    if type(results[_]) in [list, type(np.array([]))]:
                        results[_+"_list_finals"].append(results_tmp[_][-1])
                        results[_+"_all"].append(results_tmp[_])
                    else:
                        results[_+"_list"].append(results_tmp[_])
                except:
                    pass
            del results_tmp
            counted += 1
        except FileNotFoundError:
            pass

    print(f"Found {counted} simulations with nu {nu}, rho {rho}, eta {eta} and Tmax {Tmax}.",flush=True)
    ts = np.array(results["ts"]/counted, dtype = int)
    ts_geom = np.array(results["ts_geom"]/counted, dtype = int)
    indices = results["indices"]
    indices_geom = results["indices_geom"]

    average_results = {"ts":ts, "indices":indices, "ts_geom":ts_geom, "indices_geom":indices_geom}
    for _ in results.keys():
        try:
            if _ == "indices" or _ == "indices_geom" or _ == "ts" or _ == "ts_geom":
                continue
            if "_list" in _:
                average_results[_] = results[_]
            else:
                average_results["average_"+_] = results[_]/counted
        except:
            pass

    # Now let's do the analysis on the average D_indices
    # Unfortunately we can't calculate the entropy on the average, since we don't have a sequence
    D_indices = average_results["average_D_indices"]
    D2_indices = average_results["average_D2_indices"]
    D3_indices = average_results["average_D3_indices"]
    D4_indices = average_results["average_D4_indices"]
    indices = average_results["indices"]
    if indices[0] == 1:
        indices = list(np.array(indices)-1) # first index is 0, which correspond to t = 1
    results_tmp = analyse_sequence(sequence=None, num_to_save=num_to_save, find_novelties = False, 
                             use_D_as_D_indices = True, indices = indices,
                             calculate_entropies_original = False, calculate_entropies_labels = False,
                             D=D_indices, D2=D2_indices, D3=D3_indices, D4=D4_indices, 
                             sequence_labels = None,
                             save_all=False, save_all_file_path = save_all_file_path, 
                             save_light_file_path = save_light_file_path,
                             do_prints = True, return_all = True)

    for _ in results_tmp.keys():
        average_results[_] = results_tmp[_]
    for _ in average_results.keys():
        results[_] = average_results[_]

    if save_all == False:
        with open(save_light_file_path,'wb') as fp:
            pickle.dump(average_results,fp)
    else:
        with open(save_all_file_path,'wb') as fp:
            pickle.dump(results,fp)
    
    # END ANALYSIS