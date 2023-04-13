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
from math import log
import csv

# local utils
from powerlaw_regr import *
from new_model import *
from analyse_higher_entropy import index_sequence


start = datetime.now()
    
    
    
# Beginning of MAIN
parser = argparse.ArgumentParser(description='Run multiple UMST simulations with different parameters and put them together, with complete analysis.')


parser.add_argument("-ID", "--ID", type=int,
    help="The ID of the simulation, used to get the correct parameters and run [default 1]",
    default=1)

parser.add_argument("-rho", "--rho", type=float,
    help="The reinforcement parameter. [default 10]",
    default=10)

parser.add_argument("-starting_nu_1", "--starting_nu_1", type=int,
    help="The initial triggering parameter (must be integer and positive). All nus will increase by one until ending_nu (extremes included). [default 1]",
    default=1)

parser.add_argument("-ending_nu_1", "--ending_nu_1", type=int,
    help="The ending triggering parameter (must be integer and positive). All nus will increase by one starting from starting_nu (extremes included). [default 10]",
    default=10)

parser.add_argument("-starting_nu_2", "--starting_nu_2", type=int,
    help="The initial triggering parameter (must be integer and positive). All nus will increase by one until ending_nu (extremes included). [default 1]",
    default=1)

parser.add_argument("-ending_nu_2", "--ending_nu_2", type=int,
    help="The ending triggering parameter (must be integer and positive). All nus will increase by one starting from starting_nu (extremes included). [default 10]",
    default=10)

parser.add_argument("-fraction_nu_2_cut_nu_1", "--fraction_nu_2_cut_nu_1", type=float,
    help="The reinforcement parameter. [default 10]",
    default=2)

parser.add_argument("-Tmax", "--Tmax", type=int,
    help="The number of steps of the simulation. [default 1000]",
    default=1000)

parser.add_argument("-do_non_overlapping_simulation", "--do_non_overlapping_simulation", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="Impose overlapping links if true. [default True]",
    default=False)

parser.add_argument("-trigger_links_with_replacement", "--trigger_links_with_replacement", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="Impose overlapping links if true. [default True]",
    default=True)

parser.add_argument("-triggering_links_among_all_non_explored_links", "--triggering_links_among_all_non_explored_links", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="Impose overlapping links if true. [default True]",
    default=True)

parser.add_argument("-N_0", "--N_0", type=int,
    help="Initial number of nodes. [default 2]",
    default=2)

parser.add_argument("-M_0", "--M_0", type=int,
    help="Initial number of links. [default 2]",
    default=2)

parser.add_argument("-putTogether", "--putTogether", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If False it does the simulation, otherwise recovers all runs with the same parameters and analyse the average. [default False]",
    default=False)

parser.add_argument("-save_all", "--saveAll", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If False it saves values at certain indices, otherwise it saves all. [default False]",
    default=False)

parser.add_argument("-save_raw_sequence", "--save_raw_sequence", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If True it saves the raw sequence created by the model. [default True]",
    default=True)

parser.add_argument("-save_raw_urn", "--save_raw_urn", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If True it saves the raw urn created by the model. [default True]",
    default=True)

parser.add_argument("-delete_files_put_together", "--delete_files_put_together", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If True it saves the raw urn created by the model. [default True]",
    default=True)

parser.add_argument("-directed", "--directed", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If True it saves the raw urn created by the model. [default True]",
    default=True)

parser.add_argument("-do_prints", "--do_prints", type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y']),
    help="If True it saves the raw urn created by the model. [default True]",
    default=False)


arguments = parser.parse_args()
ID = arguments.ID - 1 # unfortunately I have to start this argument from 1 in shell. The -1 is added to start from 0
rho = arguments.rho
starting_nu_1 = arguments.starting_nu_1
ending_nu_1 = arguments.ending_nu_1
starting_nu_2 = arguments.starting_nu_2
ending_nu_2 = arguments.ending_nu_2
fraction_nu_2_cut_nu_1 = arguments.fraction_nu_2_cut_nu_1
Tmax = arguments.Tmax
do_non_overlapping_simulation = arguments.do_non_overlapping_simulation
trigger_links_with_replacement = arguments.trigger_links_with_replacement
triggering_links_among_all_non_explored_links = arguments.triggering_links_among_all_non_explored_links
N_0 = arguments.N_0
M_0 = arguments.M_0
putTogether = arguments.putTogether
save_all = arguments.saveAll
save_raw_sequence = arguments.save_raw_sequence
save_raw_urn = arguments.save_raw_urn
delete_files_put_together = arguments.delete_files_put_together
directed = arguments.directed
do_prints = arguments.do_prints

# I'm doing 100 runs per each set of parameters, with nu going from 1 to rho-1

# PARAMETERS
if fraction_nu_2_cut_nu_1 == 0:
    num_parameters_nu_1 = ending_nu_1 - starting_nu_1 + 1
    nu_1 = starting_nu_1 + ID % num_parameters_nu_1
    num_parameters_nu_2 = ending_nu_2 - starting_nu_2 + 1
    nu_2 = starting_nu_2 + int(ID/num_parameters_nu_1) % num_parameters_nu_2
    run = int(ID/num_parameters_nu_1/num_parameters_nu_2)
    num_parameters = num_parameters_nu_1 * num_parameters_nu_2
else:
    parameters = []
    for nu_1 in range(starting_nu_1, ending_nu_1 + 1):
        for nu_2 in range(starting_nu_2, ending_nu_2 + 1):
            if nu_2 <= fraction_nu_2_cut_nu_1 * nu_1:
                parameters.append([nu_1, nu_2])
    num_parameters = len(parameters)
    run = int(ID/num_parameters)
    ID_num_parameters = ID % num_parameters
    nu_1 = parameters[ID_num_parameters][0]
    nu_2 = parameters[ID_num_parameters][1]
    

print(f'num_parameters = {num_parameters}', flush=True)

num_to_save = 1000

main_dir = f'./data/new_model/'
if directed:
    main_dir += 'directed/'
    
if do_non_overlapping_simulation:
    main_dir += 'do_non_overlapping_simulation/'

if trigger_links_with_replacement:
    main_dir += 'trigger_links_with_replacement/'
    
if triggering_links_among_all_non_explored_links:
    main_dir += 'triggering_links_among_all_non_explored_links/'

work_dir = os.path.join(main_dir, f"simulations/analysis/rho_{rho:.5f}/nu_1{nu_1}/nu_2{nu_2}/Tmax_{Tmax}/N_0_{N_0}/M_0_{M_0}/")
raw_urn_dir = os.path.join(main_dir, f"simulations/raw_urn/rho_{rho:.5f}/nu_1{nu_1}/nu_2{nu_2}/Tmax_{Tmax}/N_0_{N_0}/M_0_{M_0}/")
raw_sequence_dir = os.path.join(main_dir, f"simulations/raw_sequence/rho_{rho:.5f}/nu_1{nu_1}/nu_2{nu_2}/Tmax_{Tmax}/N_0_{N_0}/M_0_{M_0}/")


    
if putTogether == False:
    # RUN THE SIMULATION
    print("rho=%.5f, nu_1=%d, nu_2=%d, Tmax=%d, run=%d, directed=%s, do_non_overlapping_simulation=%s, trigger_links_with_replacement=%s, triggering_links_among_all_non_explored_links=%s"%(rho,nu_1,nu_2,Tmax,run,str(directed), str(do_non_overlapping_simulation), str(trigger_links_with_replacement), str(triggering_links_among_all_non_explored_links)), flush=True)
    
    save_all_file_path = os.path.join(work_dir, f"run_{run}.pkl")
    save_light_file_path = os.path.join(work_dir, f"light_run_{run}.pkl") 
    save_entropies_file_path = os.path.join(work_dir, f"entropy_run_{run}.pkl") 
    save_entropies_original_file_path = os.path.join(work_dir, f"entropy_original_run_{run}.pkl") 
    raw_urn_file_path = os.path.join(raw_urn_dir, f"{run}.pkl")
    raw_sequence_file_path = os.path.join(raw_sequence_dir, f"{run}.tsv")
    
    if (save_all == False and os.path.exists(save_light_file_path) == False) or (save_all == True and (os.path.exists(save_all_file_path) == False or os.path.exists(save_light_file_path) == False)) :
        new_model = new_model(
            rho = rho, 
            nu_1 = nu_1, 
            nu_2 = nu_2, 
            N_0 = N_0, 
            M_0 = M_0, 
            Tmax = Tmax, 
            directed = directed,
            do_non_overlapping_simulation = do_non_overlapping_simulation,
            trigger_links_with_replacement = trigger_links_with_replacement,
            triggering_links_among_all_non_explored_links = triggering_links_among_all_non_explored_links,
            do_prints = do_prints,
        )
        
        t = len(new_model.sequence_D1)
        print(t)
        D1 = new_model.sequence_D1[-1]
        print(D1)
        D2 = new_model.sequence_D2[-1]
        print(D2)
        try:
            print('nu_1/rho', nu_1/rho, flush=True)
            print('nu_2/rho', nu_2/rho, flush=True)
            try:
                print('nu_1/nu_2', nu_1/nu_2, flush=True)
            except:
                pass
            print('beta1', log(D1) / log(t), flush=True)
            print('beta1 new', lin_regr_with_stats(np.log(np.arange(t)+1), np.log(np.array(new_model.sequence_D1)), intercept_left_lim = 0, intercept_right_lim = np.inf, get_more_statistcs = False, alpha_for_conf_interv = 0.99), flush=True)
            print('beta2', log(D2) / log(t), flush=True)
            print('beta2 new', lin_regr_with_stats(np.log(np.arange(t)+1), np.log(np.array(new_model.sequence_D2)), intercept_left_lim = 0, intercept_right_lim = np.inf, get_more_statistcs = False, alpha_for_conf_interv = 0.99), flush=True)
        except Exception as e:
            print(e, flush=True)
        if save_raw_urn == True:
            os.makedirs(os.path.dirname(raw_urn_file_path), exist_ok=True)
            with open(raw_urn_file_path, "wb") as fp:
                pickle.dump(new_model, fp)
        
        
        
        counter_times_not_enough = new_model.counter_times_not_enough
        sequence = []
        sequence_extractions = new_model.sequence_extractions
        sequence.append(sequence_extractions[0][0])
        for link in sequence_extractions:
            sequence.append(link[1])

        if do_non_overlapping_simulation == False:
            if save_raw_sequence == True:
                os.makedirs(os.path.dirname(raw_sequence_file_path), exist_ok=True)
                with open(raw_sequence_file_path, "w", newline='\n') as f_output:
                    tsv_output = csv.writer(f_output, delimiter='\n')
                    tsv_output.writerow(sequence)

            # Analyse sequence

            print("Starting analysis", flush=True)

            result = analyse_sequence(
                sequence=sequence, 
                consider_temporal_order_in_tuples=True, 
                num_to_save=1000, 
                indices = [],
                use_D_as_D_indices = False, 
                D=None, 
                D2=None, 
                D3=None,
                D4=None, 
                sequence_labels = sequence, 
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
        else:
            sequence_extractions = new_model.sequence_extractions
            sequence = []
            sequence_extractions = new_model.sequence_extractions
#             sequence.append(sequence_extractions[0][0])
            for link in sequence_extractions:
                sequence.append(link[1])
            indexed_sequence_links = index_sequence(sequence_extractions)
            D1 = np.array(new_model.sequence_D1)
            D2 = np.array(new_model.sequence_D2)
            

            if save_raw_sequence == True:
                os.makedirs(os.path.dirname(raw_sequence_file_path), exist_ok=True)
                with open(raw_sequence_file_path, "w", newline='\n') as f_output:
                    tsv_output = csv.writer(f_output, delimiter='\n')
                    tsv_output.writerow(sequence_extractions)

            # Analyse sequence

            print("Starting analysis", flush=True)
            
            result = analyse_sequence(
                sequence=sequence, 
                sequence_pairs=indexed_sequence_links, 
                consider_temporal_order_in_tuples=True, 
                num_to_save=1000, 
                indices = [],
                use_D_as_D_indices = False, 
                D=D1, 
                D2=D2, 
                D3=D2,
                D4=D2, 
                sequence_labels = sequence, 
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
        result['counter_times_not_enough'] = counter_times_not_enough
        print('counter_times_not_enough', counter_times_not_enough, flush=True)
        with open(save_light_file_path, 'wb') as fp:
            pickle.dump(result, fp)
    else:
        print("File already present")
    
    
else:
    # PUT TOGETHER DIFFERENT RUNS WITH SAME PARAMETERS AND DO ANALYSIS
    # Let's use the light results pickles
    save_all_file_path = os.path.join(work_dir, "average_results.pkl")
    save_light_file_path = os.path.join(work_dir, "average_light_results.pkl") 
    save_entropies_file_path = os.path.join(work_dir, f"average_entropy.pkl") 
    try:
        os.remove(save_all_file_path)
    except:
        pass
    try:
        os.remove(save_light_file_path)
    except:
        pass
    try:
        os.remove(save_entropies_file_path)
    except:
        pass
#     save_all_file_path = os.path.join(work_dir, f"run_{run}.pkl")
#     save_light_file_path = os.path.join(work_dir, f"light_run_{run}.pkl") 
#     save_entropies_file_path = os.path.join(work_dir, f"entropy_run_{run}.pkl") 
#     save_entropies_original_file_path = os.path.join(work_dir, f"entropy_original_run_{run}.pkl") 
#     raw_urn_file_path = os.path.join(raw_urn_dir, f"{run}.pkl")
#     raw_sequence_file_path = os.path.join(raw_sequence_dir, f"{run}.tsv")
    
    # Let's create the average sequence!
    print("Loading all done simulations at", datetime.now(), 'from', work_dir, flush=True)
    if save_all == True:
        # Use save_all files, i.e. f"UMT_run_{run}.pkl"
        paths = glob.glob(os.path.join(work_dir, "run*.pkl"))
    else:
        paths = glob.glob(os.path.join(work_dir, "light_run_*.pkl"))
    
    if len(paths) != 100:
        print('paths', paths, flush=True)
    found_one = False
    while len(paths) > 0 and found_one == False:
        print(len(paths), found_one, paths[0], flush=True)
        try:
            with open(paths[0],'rb') as fp:
                results = pickle.load(fp)
            found_one = True
        except:
            if delete_files_put_together:
                os.remove(paths[0])
            del paths[0]
            print('Problem... Total paths remaining', len(paths), flush=True)
    if found_one == False:
        print(f'No simulations for chosen parameters')
        exit()
    else:
        print('Found one! Total paths remaining', len(paths), flush=True)
    
#     with open(paths[0],'rb') as fp:
#         results = pickle.load(fp)
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

    print(f"Found {counted} simulations.",flush=True)
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
    results_tmp = analyse_sequence(sequence=None, num_to_save=num_to_save, 
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
    
    
    entropies_results = {}
    entropies_results['mean_entropies'] = {}
    entropies_results['mean_entropies_glob'] = {}
    entropies_results['mean_entropies_pairs'] = {}
    entropies_results['mean_entropies_glob_pairs'] = {}
    entropies_results['weighted_diff_entropies'] = []
    entropies_results['weighted_diff_entropies_pairs'] = []
    paths = glob.glob(os.path.join(work_dir, "*entropy_run_*.pkl"))
    for path in paths:
        with open(path, 'rb') as fp:
            entropy_result = pickle.load(fp)
        for i_order,order in enumerate(['', '_pairs']):
            n_order = i_order + 1
            for freq,value in entropy_result[f'mean_entropies{order}'].items():
                try:
                    entropies_results[f'mean_entropies{order}'][freq].append(value)
                except KeyError:
                    entropies_results[f'mean_entropies{order}'][freq] = [value]
            for freq,value in entropy_result[f'mean_entropies_glob{order}'].items():
                try:
                    entropies_results[f'mean_entropies_glob{order}'][freq].append(value)
                except KeyError:
                    entropies_results[f'mean_entropies_glob{order}'][freq] = [value]
        entropies_results['weighted_diff_entropies'].append(entropy_result['weighted_diff_entropies'])
        entropies_results['weighted_diff_entropies_pairs'].append(entropy_result['weighted_diff_entropies_pairs'])
    with open(save_entropies_file_path,'wb') as fp:
        pickle.dump(entropies_results,fp)
    # END ANALYSIS

    
    if delete_files_put_together:
        print('Deleting single files', flush=True)
        paths = glob.glob(os.path.join(work_dir, "run*.pkl"))
        for path in paths:
            os.remove(path)
        paths = glob.glob(os.path.join(work_dir, "light_run_*.pkl"))
        for path in paths:
            os.remove(path)
        paths = glob.glob(os.path.join(work_dir, "entropy_run_*.pkl"))
        for path in paths:
            os.remove(path)
        paths = glob.glob(os.path.join(work_dir, "entropy_original_run_*.pkl"))
        for path in paths:
            os.remove(path)
        paths = glob.glob(os.path.join(raw_urn_dir, f"{run}.pkl"))
        for path in paths:
            os.remove(path)
        paths = glob.glob(os.path.join(raw_sequence_dir, f"{run}.tsv"))
        for path in paths:
            os.remove(path)
end = datetime.now()
print('Total time',end-start, flush=True)
    