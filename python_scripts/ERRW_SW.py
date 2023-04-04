import networkx as nx
import numpy as np
import random
import os
# Change directory to the root of the folder (this script was launched from the subfolder python_scripts)
# All utils presuppose that we are working from the root directory of the github folder
os.chdir('../')
from datetime import datetime
import argparse
import pickle
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))

# local utils
from analyse_higher_entropy import *
from analyse_sequence import *
from find_files_with_pattern import *    


# MODEL DEFINITION
def run_ERRW(G, dw, T, n0=None):
    """
    Running the ERRW.

    Parameters
    ----------
    G : networkx Graph
        Underlying structure.
    dw : float
        Edge-reinforcement.
    T : int
        Number of steps of the walk.
    n0: object (default=None)
        Seed node (has to be in G).
    
    Returns
    -------
    S : list
        List of visited nodes.
    """
    
    def get_next_node(G, initial_node):
        neighbors = list(G.neighbors(initial_node))
        weights = np.array([G[initial_node][neighbor]['weight'] for neighbor in neighbors])
        norm_weights = 1.*weights/sum(weights)
        next_node = np.random.choice(neighbors, p=norm_weights)
        return next_node

    #Setting the weights to the initial value
    nx.set_edge_attributes(G, values=1, name='weight')
    #Setting the seed node
    if n0==None:
        #Random seed
        walker_position = random.choice(list(G.nodes))
    else:
        #Provided seed
        walker_position = n0
    
    #Here I will store the walk sequence
    S = []    
    S.append(walker_position)

    for t in range(T):
        #Choosing the next node proportionally to the weights
        next_node = get_next_node(G, walker_position)
        #Reinforcing the weight of the traversed edge
        G[walker_position][next_node]['weight'] += dw
        walker_position = next_node
        S.append(walker_position)
        
    return S






# Beginning of MAIN
parser = argparse.ArgumentParser(description='Compute entropies on the sequences of labels in the data, both on singletons and pairs, and on randomized sequences.')


parser.add_argument("-ID", "--ID", type=int,
    help="The ID of the simulation, used to get the correct sequence [default 1]",
    default=1)

parser.add_argument("-number_reshuffles", "--number_reshuffles", type=int,
    help="Number of reshuffles to do check the randomized case of the entropy. [default 10]",
    default=10)

parser.add_argument("-order", "--consider_temporal_order_in_tuples", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y', 't']),
    help="If False, it considers AB and BA the same. [default True]",
    default=True)

parser.add_argument("-analyse_labels", "--analyse_sequence_labels", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y', 't']),
    help="If True, it considers the sequence of labels instead of the original sequence in all calculations. [default False]",
    default=False)

parser.add_argument("-save_all", "--save_all", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y', 't']),
    help="If True, it saves all results. [default True]",
    default=True)

# Parameter for ERRW

parser.add_argument("-p", "--p", type=float,
    help="The probability parameter p of the Watts-Strogatz graph model to randomly rewire the links. [default 0.1]",
    default=0.1)

parser.add_argument("-k", "--K", type=int,
    help="The initial degree of each node for the Watts-Strogatz graph model. [default 0.1]",
    default=0.1)

parser.add_argument("-N", "--N", type=int,
    help="The number of nodes of the graph. [default 1000000]",
    default=int(1e6))

parser.add_argument("-starting_dw", "--starting_dw", type=float,
    help="The starting reinforcement parameter. If use_logspace is True, this is the starting dw. [default 0]",
    default=0)

parser.add_argument("-ending_dw", "--ending_dw", type=float,
    help="The ending reinforcement parameter (included). If use_logspace is True, this is the ending dw. [default 10]",
    default=10)

parser.add_argument("-num_dw", "--num_dw", type=int,
    help="The number of different dw to create the arrsys of dw (extremes included). [default 21]",
    default=21)

parser.add_argument("-use_logspace", "--use_logspace", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y', 't']),
    help="If True, it uses a logspace for dw. [default False]",
    default=False)

parser.add_argument("-use_special_space", "--use_special_space", type=lambda x: (str(x).lower() in ['true','1', 'yes', 'y', 't']),
    help="If True, it uses a special space for dw. [default False]",
    default=False)

parser.add_argument("-Tmax", "--Tmax", type=int,
    help="The number of steps of the simulation. [default 100]",
    default=int(1e5))

parser.add_argument("-folder_ERRW", "--folder_name_ERRW", type=str,
    help="Name of the subfolder in ~/data/ERRW/ to take the data from and do the analysis. [default 'SW']",
    default='SW')

arguments = parser.parse_args()

ID = arguments.ID - 1 # unfortunately I have to start this argument from 1 in shell. The -1 is added to start from 0
number_reshuffles = arguments.number_reshuffles
consider_temporal_order_in_tuples = arguments.consider_temporal_order_in_tuples
analyse_sequence_labels = arguments.analyse_sequence_labels
save_all = arguments.save_all
p = arguments.p
K = arguments.K
N = arguments.N
starting_dw = arguments.starting_dw
ending_dw = arguments.ending_dw
num_dw = arguments.num_dw
use_logspace = arguments.use_logspace
use_special_space = arguments.use_special_space
Tmax = arguments.Tmax
folder_name_ERRW = arguments.folder_name_ERRW


start = datetime.now()
print('Starting at', datetime.now(), flush=True)

if use_special_space:
    mean_dw = 2 # THIS IS FOUND BY ATTEMPTS, it is the dw such that on average beta is 0.5
    desired_num_indices = int((num_dw + 1) / 2)
    for current_attempt in range(1,(num_dw-1)*2):
        indices = sorted(set(np.geomspace(1,(num_dw-1)*2+1,current_attempt, dtype = int)-1))
        if len(indices) == desired_num_indices:
            break
    dws = list(np.geomspace(mean_dw,starting_dw, (num_dw-1)*2+1)[indices])[::-1] + list(np.geomspace(mean_dw,ending_dw, (num_dw-1)*2+1)[indices[1:]])
    folder_name_ERRW += "_special_space"
#     mean_dw = 2 # THIS IS FOUND BY ATTEMPTS, it is the dw such that on average beta is 0.5
#     desired_num_indices = int((num_dw + 1) / 2)
#     for current_attempt in range(1,(num_dw-1)*3):
#         indices = sorted(set(np.geomspace(1,(num_dw-1)*3+1,current_attempt, dtype = int)-1))
#         if len(indices) == desired_num_indices:
#             break
#     dws = list(np.geomspace(mean_dw,starting_dw, int((num_dw+1)/2))) + list(np.geomspace(mean_dw,ending_dw, int((num_dw+1)/2)))[1:]
#     folder_name_ERRW += "_special_space"
elif use_logspace:
    dws = np.logspace(starting_dw,ending_dw,num_dw)
    folder_name_ERRW += "_logspace"
else:
    dws = np.linspace(starting_dw,ending_dw,num_dw)


num_parameters = num_dw
index = ID % num_parameters
run = int(ID/num_parameters)
dw = dws[index]


main_dir = f'./data/ERRW/{folder_name_ERRW}_k_{str(K)}_T_{Tmax}/'
raw_sequences_dir = os.path.join(main_dir, 'raw_sequences')
file_name = 'Seq_SW_p%.3f_dw%.5f_simID%d'%(p, dw, run)
raw_sequence_file_name = os.path.join(raw_sequences_dir, f'{file_name}.txt')
os.makedirs(raw_sequences_dir, exist_ok = True)

print('p', p, flush=True)
print('K', K, flush=True)
print('N', N, flush=True)
print('dw', dw, flush=True)
print('Tmax', Tmax, flush=True)
print('run', run, flush=True)
        
# Create graph
G = nx.connected_watts_strogatz_graph(n=N, k=K, p=p, tries=500)
print('Graph created at', datetime.now(), flush=True)

# Run ERRW
sequence = run_ERRW(G, dw, Tmax, n0=None)

print('ERRW complete at', datetime.now(), flush=True)

# Saving sequence
file = open(raw_sequence_file_name,'w')
for s in sequence:
    file.write("%s\n"%s)
file.close()


        
    
# START ANALYSIS 

if consider_temporal_order_in_tuples:
    analysis_folder = os.path.join(main_dir,'analysis')
else:
    analysis_folder = os.path.join(main_dir,'analysis_tuples_without_order')

save_light_file_path = os.path.join(analysis_folder,'light_results',f'{file_name}.pkl')
save_all_file_path = os.path.join(analysis_folder,'all_results',f'{file_name}.pkl')
save_entropies_file_path = os.path.join(analysis_folder,'entropies_results',f'{file_name}.pkl')


# Analyse sequence

print('Starting analysis at', datetime.now(), flush=True)

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

        
end = datetime.now()
print('Total time',end-start, flush=True)

