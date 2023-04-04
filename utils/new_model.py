import numpy as np
import random
import pickle
import os
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0, os.path.join(os.getcwd(),'utils'))
from datetime import datetime
from numpy.random import rand, choice

# local utils
from analyse_sequence import *

class new_model:
    ############## INITIALIZE CLASS FUNCTION ##############
    def __init__(
        self, 
        rho = 10, 
        nu_1 = 5, 
        nu_2 = 7, 
        N_0 = 1, 
        M_0 = 0, 
        Tmax = int(1e4),
        directed = False,
        do_non_overlapping_simulation = False,
        trigger_links_with_replacement = True,
        triggering_links_among_all_non_explored_links = False,
        do_prints = False,
    ):
        """
            rho: Whenever a link is extracted, it is reinforced with rho copies (float greater than 0).
            
            nu_1: Whenever a link with a new node is visited for the first time, add nu_1 + 1 nodes and link them to the last node extracted. (integer greater or equal than 0)
            
            nu_2: Whenever a new link is visited for the first time, add nu_2 new links between the last node and other nodes already visited. (integer greater or equal than 0)
            
            N_0 is the number of initial nodes (integer greater than 0). This has been overriden using particular initialization procedures. Check initialization procedures.
            
            M_0 is the number of initial links, added randomly. Whenever initialized, the link has weight 1 (integer greater than 0).  This has been overriden using particular initialization procedures.
            
            Tmax is the number of time steps to do (integer greater than 0)
            
            directed: makes the network directed or undirecetd accordingly (boolean, default: False). 
            
            do_non_overlapping_simulation: if True, you pick random links in the whole network, not just from outgoing links of the current node (boolean, default: False). 
            
            trigger_links_with_replacement: if True, new links between already explored nodes are added randomly, allowing replacement, otherwise they are chosen from those not already triggered (boolean, default: False). 
            
            triggering_links_among_all_non_explored_links: if True, new links between already explored nodes are added in the whole network, not just among outgoing links (boolean, default: False). 
            
            do_prints: if True, activates all prints for debug (boolean, default: False). 
        """
        self.do_prints = do_prints
        assert type(rho) in [float, int], "rho is not a float, it's of type %s!"%(str(type(rho)))
        assert rho > 0, "rho is not positive, is %s!"%(str(rho))
        assert type(nu_1) == int, "nu_1 is not an integer, it's of type %s!"%(str(type(nu_1)))
        assert nu_1 >= 0, "nu_1 is not positive, is %s!"%(str(nu_1))
        assert type(nu_2) == int, "nu_2 is not an integer, it's of type %s!"%(str(type(nu_2)))
        assert nu_2 >= 0, "nu_2 is not positive, is %s!"%(str(nu_2))
        assert type(N_0) == int, "N_0 is not an integer, it's of type %s!"%(str(type(N_0)))
        assert type(M_0) == int, "N_0 is not an integer, it's of type %s!"%(str(type(M_0)))
        assert type(Tmax) == int, "Tmax is not an integer, it's of type %s!"%(str(type(Tmax)))
        assert Tmax > 0, "Tmax is not positive, is %s!"%(str(Tmax))
        self.rho = rho
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.N_0 = N_0
        self.M_0 = M_0
        self.Tmax = Tmax
        self.do_non_overlapping_simulation = do_non_overlapping_simulation
        self.trigger_links_with_replacement = trigger_links_with_replacement
        self.triggering_links_among_all_non_explored_links = triggering_links_among_all_non_explored_links
        self.urn = {} # this dict structure is {node1: {node2: freq}}
        self.total_weight_urn = 0
        self.sequence_extractions = []
        self.sequence_D1 = []
        self.sequence_D2 = []
        self.counter_times_not_enough = 0
        self.new_node_index = 0 # this is also the counter of the existing nodes in the urn
        self.directed = directed
        self.nodes_triggered = set()
        self.nodes_explored = set()
        self.links_triggered = {}
        self.links_explored = {}
        if self.N_0 == 1 and self.M_0 == 0:
            # Start from one node considered explored
            self.trigger_node()
        elif self.N_0 > 1 and self.N_0 < 100 and self.M_0 == 0: 
            self.star_of_stars_initialization()
        elif self.N_0 > 1 and self.M_0 == 0: 
            self.star_of_stars_of_stars_initialization()
        elif self.N_0 > 1 and self.M_0 >= self.N_0:
            self.ring_and_random_initialization()
        else:
            print('Values of N_0 and M_0 NOT SUPPORTED, i.e.,', self.N_0, self.M_0, flush=True)
            print('Exiting')
            exit()
            
        # start simulation
        print(f'Running model for {self.Tmax} steps', flush=True)
        for t in range(self.Tmax):
            self.t = t
            if t % max(1,int(self.Tmax/100)) == 0:
                print(f'\t{str(t/self.Tmax*100)}% completed', flush=True)
            self.next_step()
    
    
    ############## INITIALIZE NETWORK FUNCTIONS ##############
    
    
    def star_of_stars_initialization(self):
        '''
            This initialization is called when 1 < N_0 < 100 and M_0 == 0.
            
            It creates and trigger a root node with nu_1 + 1 children, and continues triggering these children. 
            The triggered nodes do not trigger anymore and can accept links as if they were already explored. 
            However, they still increase D1 when first visited in the random walk.
            
            POSSIBLE MODIFICATION: instead of triggering the nu_1 + 1 children, use the number N_0 >= 1 of nodes to trigger
            and for each one each time trigger a random leaf, not necessarily at level 1 of the tree
        '''
        print(f'Initializing network with star of stars considering {self.N_0} nodes explored...', flush=True)
        print(f'Creating a triggered star from a root with nu_1 + 1 stars-leaves', flush=True)
        num_added_links = 0
        # create root
        root = self.get_new_node_index() # this is 0
        first_children = self.trigger_node(triggering_node = root)
        self.links_explored[root] = set()
        self.links_triggered[root] = set()
        for first_child in first_children:
            # create second layer star, all unexplored
            second_children = self.trigger_node(triggering_node = first_child)
            self.links_explored[first_child] = set()
            self.links_triggered[first_child] = set()


    
    def star_of_stars_of_stars_initialization(self):
        '''
            This initialization is called when N_0 >= 100 and M_0 == 0.
            
            It creates and trigger a root node with nu_1 + 1 children, and continues triggering these children and their own children.
            The triggered nodes do not trigger anymore and can accept links as if they were already explored. 
            However, they still increase D1 when first visited in the random walk.
        '''
        print(f'Initializing network with star of stars considering {self.N_0} nodes explored...', flush=True)
        print(f'Creating a triggered star from a root with nu_1 + 1 stars-leaves', flush=True)
        num_added_links = 0
        # create root
        root = self.get_new_node_index() # this is 0
        first_children = self.trigger_node(triggering_node = root)
        self.links_explored[root] = set()
        self.links_triggered[root] = set()
        second_children = []
        for first_child in first_children:
            # create second layer star, all unexplored
            tmp_second_children = self.trigger_node(triggering_node = first_child)
            second_children += tmp_second_children
            self.links_explored[first_child] = set()
            self.links_triggered[first_child] = set()
        for second_child in second_children:
            # create second layer star, all unexplored
            tmp_third_children = self.trigger_node(triggering_node = second_child)
            self.links_explored[second_child] = set()
            self.links_triggered[second_child] = set()


    def ring_and_random_initialization(self):
        '''
        FUNCTION DEPRECATED, REWRITE. 
        POSSIBLE MODIFICATION: Substitute with a random ER or SW network where a percentage of nodes and links has been already explored.
        
            This initialization is called when M_0 > 0.
            
            It creates a ring between N_0 nodes, and continues adding links until triggering these children and their own children.
            
            In the current implementation, no nodes are considered explored.
            
        '''
        # DEPRECATED
        print(f'Initializing urn with {self.N_0} different nodes and {self.M_0} links...', flush=True)
        print(f'First add a ring between all nodes', flush=True)
        num_added_links = 0
        if self.directed:
            initial_node = self.get_new_node_index()
            first_node = initial_node
            possible_links = []
            for _ in range(1,self.N_0):
                second_node = self.get_new_node_index() # this should be _
                self.add_link(first_node = first_node, second_node = second_node, weight = 1)
                num_added_links += 1
                for _2 in range(self.N_0):
                    if _2 != second_node and _2 != first_node:
                        possible_links.append((first_node, _2))
                first_node = second_node
            self.add_link(first_node = first_node, second_node = initial_node, weight = 1)
            for _2 in range(self.N_0):
                if _2 != initial_node and _2 != first_node:
                    possible_links.append((first_node, _2))
            num_added_links += 1
            print(f'Now adding other random nodes', flush=True)
            print(possible_links)
            print(f'self.M_0', self.M_0, len(possible_links), num_added_links, flush=True)
            assert self.M_0 <= len(possible_links) + num_added_links, "M_0 is too big (maximum is %d), is %s!"%(len(possible_links) + num_added_links, str(self.M_0))
            random_links = random.sample(possible_links, self.M_0 - num_added_links)
            for first_node, second_node in random_links:
                self.add_link(first_node = first_node, second_node = second_node, weight = 1)
        else:
            initial_node = self.get_new_node_index()
            first_node = initial_node
            possible_links = []
            for _ in range(1,self.N_0):
                second_node = self.get_new_node_index() # this should be _
                self.add_link(first_node = first_node, second_node = second_node, weight = 1)
                num_added_links += 1
                for _2 in range(_, self.N_0):
                    if _2 != second_node and _2 != first_node:
                        possible_links.append((first_node, _2))
                first_node = second_node
            if self.N_0 > 2:
                self.add_link(first_node = first_node, second_node = initial_node, weight = 1)
                num_added_links += 1
            print(f'Now adding other random nodes', flush=True)
            print(possible_links)
            print(f'self.M_0', self.M_0, len(possible_links), num_added_links, flush=True)
            assert int(self.M_0/2) <= len(possible_links) + num_added_links, "M_0 is too big (maximum is %d), is %s!"%(len(possible_links) + num_added_links, str(self.M_0))
            random_links = random.sample(possible_links, int(self.M_0/2) - num_added_links)
            for first_node, second_node in random_links:
                self.add_link(first_node = first_node, second_node = second_node, weight = 1)

    ############## MAIN FUNCTION ##############

   
    def next_step(self):
        '''
            Make a time step, moving to a new node and updating the network and all variables.
        '''
        # get new link
        found_new_node = False
        if self.do_non_overlapping_simulation or len(self.sequence_extractions) == 0:
            # If not given, randomly pick a link in the network, with no limitation on the first node.
            first_node = None
        else:
            first_node = self.sequence_extractions[-1][1]
        if len(self.sequence_extractions) == 0 and self.M_0 == 0:
            # if M_0 is null, we are using a star-like or tree-like structure. So start explorign from root, indexed as 0.
            first_node = 0
            
        first_node, second_node = self.extract_random_link(first_node = first_node, second_node = None)
        if self.do_prints:
            print('EXPLORING', first_node,second_node)
        
        # check novelty at order 1 of first node
        found_new_node = max(found_new_node, 1-self.is_node_explored(first_node))
        if found_new_node:
            self.trigger_node(triggering_node = first_node)
            self.nodes_explored.add(first_node)
        
        # check novelty at order 1 of second node
        found_new_node = max(found_new_node, 1-self.is_node_explored(second_node))
        if found_new_node:
            self.trigger_node(triggering_node = second_node)
            self.nodes_explored.add(second_node)
        
        if self.do_prints:
            print('new_node?',found_new_node, flush=True)
        # update D1
        try:
            self.sequence_D1.append(self.sequence_D1[-1] + int(found_new_node))
        except IndexError:
            # D1 is empty
            self.sequence_D1.append(int(found_new_node))
        
        # check novelty at order 2
        found_new_link = not(self.is_link_explored(first_node, second_node))
        if found_new_link:
            # trigger new links
            try:
                self.links_explored[first_node].add(second_node)
            except KeyError:
                self.links_explored[first_node] = set([second_node])
            if self.directed == False:
                try:
                    self.links_explored[second_node].add(first_node)
                except KeyError:
                    self.links_explored[second_node] = set([first_node])
            self.trigger_link(first_node_triggering_link = first_node, second_node_triggering_link = second_node)
        
        if self.do_prints:
            print('new_link?',found_new_link, flush=True)
        # update D2
        try:
            self.sequence_D2.append(self.sequence_D2[-1] + int(found_new_link))
        except IndexError:
            # D2 is empty
            self.sequence_D2.append(int(found_new_link))
            
        # reinforcement
        self.add_link(first_node = first_node, second_node = second_node, weight = self.rho)
        
        # update sequence
        self.sequence_extractions.append((first_node,second_node))
        
        
    ############## QUERY NETWORK FUNCTIONS ##############

    
    def is_node_explored(self, node):
        '''
            Returns True if node has been already explored, i.e., appeared in the sequence during the RW.
        '''
        is_explored = node in self.nodes_explored
        return is_explored
    
    def is_node_triggered(self, node):
        '''
            Returns True if node has been already triggered, either in the initialization or later.
        '''
        is_triggered = node in self.nodes_triggered
        return is_triggered
    
    def is_link_explored(self, first_node, second_node):
        '''
            Returns True if the link (first_node, second_node) has been already explored, i.e., appeared in the sequence during the RW.
        '''
        is_explored = False
        try:
            if second_node in self.links_explored[first_node]:
                is_explored = True
        except KeyError:
            pass
        return is_explored
    
    def is_link_triggered(self, first_node, second_node):
        '''
            Returns True if the link (first_node, second_node) has been already triggered, either in the initialization or later.
        '''
        is_triggered = False
        try:
            if second_node in self.links_triggered[first_node]:
                is_triggered = True
        except KeyError:
            pass
        return is_triggered
    
    def get_unexplored_links_with_triggered_nodes(self, first_node = None):
        '''
            Returns the set of all tuples (first_node, second_node) indicating all links not yet explored (i.e., appeared in the sequence) between triggered nodes. 
            If first_node is None, it looks among all possible links, otherwise only among its outgoing links.
            
            ACHTUNG: self-loops are not allowed. Change code accordingly if you want to add this feature.
        '''
        unexplored_links_with_triggered_nodes = set()
        if self.do_prints:
            print('Getting not triggered links from', first_node, flush=True)
        if first_node is None:
            for tmp_node in self.nodes_triggered:
                unexplored_links_with_triggered_nodes = unexplored_links_with_triggered_nodes.union(self.get_unexplored_links_with_triggered_nodes(first_node = tmp_node))
            for (first_tmp_node, second_tmp_nodes) in unexplored_links_with_triggered_nodes:
                if first_tmp_node == second_tmp_nodes:
                    unexplored_links_with_triggered_nodes.remove((first_tmp_node, second_tmp_nodes))
        else:
            try:
                tmp_neighbors = self.nodes_triggered.difference(self.links_explored[first_node])
            except KeyError as e:
                print('ERROR get_unexplored_links_with_triggered_nodes', e, first_node,)
                tmp_neighbors = self.nodes_triggered.copy()
            tmp_neighbors.remove(first_node)
            for tmp_node in tmp_neighbors:
                if self.directed:
                    unexplored_links_with_triggered_nodes.add((first_node, tmp_node))
                else:
                    unexplored_links_with_triggered_nodes.add(tuple(sorted([first_node, tmp_node])))
        return unexplored_links_with_triggered_nodes
                
    def get_not_triggered_links_with_triggered_nodes(self, first_node = None):
        '''
            Returns the set of all tuples (first_node, second_node) indicating all links not yet triggered (i.e., appeared in the sequence or considered triggered during the initialization) between triggered nodes. 
            If first_node is None, it looks among all possible links, otherwise only among its outgoing links.
            
            ACHTUNG: self-loops are not allowed. Change code accordingly if you want to add this feature.
        '''
        not_triggered_links_with_triggered_nodes = set()
        if self.do_prints:
            print('Getting not triggered links from', first_node, flush=True)
        if first_node is None:
            for tmp_node in self.nodes_triggered:
                not_triggered_links_with_triggered_nodes = not_triggered_links_with_triggered_nodes.union(self.get_not_triggered_links_with_triggered_nodes(first_node = tmp_node))
            for (first_tmp_node, second_tmp_nodes) in not_triggered_links_with_triggered_nodes:
                if first_tmp_node == second_tmp_nodes:
                    not_triggered_links_with_triggered_nodes.remove((first_tmp_node, second_tmp_nodes))
        else:
            try:
                tmp_neighbors = self.nodes_triggered.difference(self.links_explored[first_node])
            except KeyError as e:
                print('ERROR get_unexplored_links_with_triggered_nodes', e, first_node,)
                tmp_neighbors = self.nodes_triggered.copy()
            tmp_neighbors.remove(first_node)
            for tmp_node in tmp_neighbors:
                if self.directed:
                    not_triggered_links_with_triggered_nodes.add((first_node, tmp_node))
                else:
                    not_triggered_links_with_triggered_nodes.add(tuple(sorted([first_node, tmp_node])))
        return not_triggered_links_with_triggered_nodes
                
    
    def extract_random_link(self, first_node = None, second_node = None):
        '''
            Randomly returns a link from the network with probabilities depending on the weights of the link.
            If first_node is None, it looks among all possible links, otherwise only among its outgoing links.
        '''
        assert second_node is None, 'ERROR: FEATURE NOT IMPLEMENTED'
        if first_node is None and second_node is None:
            random_weight = random.random() * self.total_weight_urn
            counter_weight = 0
            for first_node, first_node_dict in self.urn.items():
                for second_node, freq in first_node_dict.items():
                    counter_weight += freq
                    if random_weight < counter_weight:
                        chosen_first_node = first_node
                        chosen_second_node = second_node
                        break
                if random_weight < counter_weight:
                    break
        elif first_node is not None and second_node is None:
            chosen_first_node = first_node
            first_node_dict = self.urn[first_node]
            weight_filtered_urn = np.sum(list(first_node_dict.values()))
            random_weight = random.random() * weight_filtered_urn
            counter_weight = 0
            for second_node, freq in first_node_dict.items():
                counter_weight += freq
                if random_weight < counter_weight:
                    chosen_second_node = second_node
                    break
        return (chosen_first_node, chosen_second_node)
    
    
    ############## MODIFY NETWORK FUNCTIONS ##############

    
    def add_link(self, first_node = None, second_node = None, weight = 1, recursion_depth = 0, admit_self_loops = False):
        '''
            Add a link from first_node to second_node with the given weight. If network is undirected, also adds the link in the other direction.
            The urn is also updated.
            If one of the extremes is not given, a new one is created.
        '''
        if first_node is None:
            first_node = self.get_new_node_index()
        if second_node is None:
            second_node = self.get_new_node_index()
        if admit_self_loops == True or (admit_self_loops == False and first_node != second_node):
            try:
                try:
                    self.urn[first_node][second_node] += weight
                except KeyError:
                    self.urn[first_node][second_node] = weight
            except KeyError:
                self.urn[first_node] = {second_node : weight}
            self.total_weight_urn += weight
            if self.directed == False and recursion_depth == 0 and first_node != second_node:
                # add also the same link in the other direction
                self.add_link(first_node = second_node, second_node = first_node, weight = weight, recursion_depth = 1)
            if self.do_prints:
                print(f'Added weight {weight} to link ({first_node},{second_node})', flush=True)
    
    def get_new_node_index(self):
        self.new_node_index += 1
        return self.new_node_index - 1
    
    def trigger_node(self, triggering_node = None):
        '''
            This function triggers then given node, creating nu_1 + 1 new nodes and linking them to the given node with unitary weight. 
            If node not given, a new one is created and triggered.
            Once triggered, the node cannot be triggered again.
            It does not automatically explore (D_1 += 1) the node
        '''
        if triggering_node is None:
            triggering_node = self.get_new_node_index()
        children = []
        if self.is_node_triggered(triggering_node):
            if self.do_prints:
                print(f'{triggering_node} already triggered', flush=True)
        else:
            if self.do_prints:
                print(f'Creating a star from node {triggering_node} with nu_1 + 1 new nodes', flush=True)
            self.nodes_triggered.add(triggering_node)
            for _ in range(self.nu_1 + 1):
                child = self.get_new_node_index()
                children.append(child)
                self.add_link(first_node = triggering_node, second_node = child, weight = 1)
        return children
            
    def trigger_link(self, first_node_triggering_link, second_node_triggering_link):
        """
            This function triggers the given link, creating nu_2 new unexplored links between explored nodes. 
            Once triggered, the link cannot be triggered again.
            It does not automatically explore (D_2 += 1) the link.
        """
        if self.is_link_triggered(first_node_triggering_link, second_node_triggering_link):
            if self.do_prints:
                print(f'{(first_node_triggering_link, second_node_triggering_link)} already triggered', flush=True)
        else:
            try:
                self.links_triggered[first_node_triggering_link].add(second_node_triggering_link)
            except KeyError:
                self.links_triggered[first_node_triggering_link] = set([second_node_triggering_link])
        # first trigger within the actual
        # look for outgoing new neighbors to add
        num_to_trigger = self.nu_2
        num_triggered = 0
        if self.triggering_links_among_all_non_explored_links == False:
            first_node_for_search = second_node_triggering_link
        else:
            first_node_for_search = None
        if self.trigger_links_with_replacement:
            set_possible_pairs_to_trigger = self.get_unexplored_links_with_triggered_nodes(first_node = first_node_for_search)
            if len(set_possible_pairs_to_trigger) > 0:
                pairs_to_trigger = random.choices(list(set_possible_pairs_to_trigger), k=num_to_trigger)
            else:
                pairs_to_trigger = []
        else:
            set_possible_pairs_to_trigger = self.get_not_triggered_links_with_triggered_nodes(first_node = first_node_for_search)
            if len(set_possible_pairs_to_trigger) > num_to_trigger:
                pairs_to_trigger = random.sample(list(set_possible_pairs_to_trigger), num_to_trigger)
            else:
                pairs_to_trigger = list(set_possible_pairs_to_trigger)
        if self.do_prints:
            print('set_possible_pairs_to_trigger:', set_possible_pairs_to_trigger, flush=True)
            print('pairs_to_trigger:', pairs_to_trigger, flush=True)
        for tmp_first_node, tmp_second_node in pairs_to_trigger:
            num_triggered += 1
            self.add_link(first_node = tmp_first_node, second_node = tmp_second_node, weight = 1)
        if num_triggered < num_to_trigger:
            self.counter_times_not_enough += 1
            print('\t\tadded only', num_triggered, 'out of', num_to_trigger, 'to trigger', flush=True)
        else:
            if self.do_prints:
                print('\t\tadded', num_triggered, 'out of', num_to_trigger, 'to trigger', flush=True)
