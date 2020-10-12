import sys
sys.path.append('../')
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from random import choice

import copy
import utility.cache as ch
import structure_graph.conditional_intensity_matrix as condim
import structure_graph.network_graph as ng
import estimators.parameters_estimator as pe
import estimators.structure_estimator as se
import structure_graph.sample_path as sp
import structure_graph.structure as st
import estimators.fam_score_calculator as fam_score

from utility.decorators import timing



#from numba import njit

import multiprocessing
from multiprocessing import Pool



'''
#TODO: Create a parent class StructureEstimator and Two Subclasses (Score-Based and Constraint-Based)
#TODO: Evaluate if it could be better to change list_edges to set for improve the performance
'''

class StructureScoreBasedEstimator(se.StructureEstimator):
    """
    Has the task of estimating the network structure given the trajectories in samplepath by
    using a score based approach.

    """

    def __init__(self, sample_path: sp.SamplePath):
        super().__init__(sample_path)


    @timing
    def estimate_structure(self, max_parents:int = None, iterations_number:int= 40, patience:int = None ):
        """
        Compute the score-based algorithm to find the optimal structure

        Parameters:
            max_parents: maximum number of parents for each variable. If None, disabled
            iterations_number: maximum number of optimization algorithm's iteration
            patience: number of iteration without any improvement before to stop the search.If None, disabled
        Returns:
            void

        """
        'Save the true edges structure in tuples'
        true_edges = copy.deepcopy(self.sample_path.structure.edges)
        true_edges = list(map(tuple, true_edges))

        'Remove all the edges from the structure'   
        self.sample_path.structure.clean_structure_edges()

        estimate_parents = self.estimate_parents

        n_nodes= len(self.nodes)
        
        l_max_parents= [max_parents] * n_nodes
        l_iterations_number = [iterations_number] * n_nodes
        l_patience = [patience] * n_nodes


        'get the number of CPU'
        cpu_count = multiprocessing.cpu_count()

        'Estimate the best parents for each node'
        with multiprocessing.Pool(processes=cpu_count) as pool:
            list_edges_partial = pool.starmap(estimate_parents, zip(self.nodes,l_max_parents,l_iterations_number,l_patience))
            #list_edges_partial = [estimate_parents(n,max_parents,iterations_number,patience) for n in self.nodes]
            #list_edges_partial = p.map(estimate_parents, self.nodes)

        'Concatenate all the edges list'
        list_edges =  list(itertools.chain.from_iterable(list_edges_partial))

        print('-------------------------')

        'TODO: Pensare a un modo migliore -- set difference sembra non funzionare '
        n_missing_edges = 0
        n_added_fake_edges = 0

        for estimate_edge in list_edges:
            if not estimate_edge in true_edges:
                n_added_fake_edges += 1
        
        for true_edge in true_edges:
            if not true_edge in list_edges:
                n_missing_edges += 1
                print(true_edge)


        print(f"n archi reali non trovati: {n_missing_edges}")
        print(f"n archi non reali aggiunti: {n_added_fake_edges}")
        print(true_edges)
        print(list_edges)
    

    def estimate_parents(self,node_id:str, max_parents:int = None, iterations_number:int= 40, patience:int = 10 ):
        """
        Use the FamScore of a node in order to find the best parent nodes
        Parameters:
            node_id: current node's id
            max_parents: maximum number of parents for each variable. If None, disabled
            iterations_number: maximum number of optimization algorithm's iteration
            patience: number of iteration without any improvement before to stop the search.If None, disabled
        Returns:
            A list of the best edges for the currente node
        """
        
        'Create the graph for the single node'
        graph = ng.NetworkGraph(self.sample_path.structure)

        other_nodes =  [node for node in self.sample_path.structure.nodes_labels if node != node_id]
        actual_best_score = self.get_score_from_graph(graph,node_id)

        patince_count = 0
        for i in range(iterations_number):
            'choose a new random edge'
            current_new_parent = choice(other_nodes)
            current_edge =  (current_new_parent,node_id)
            added = False
            parent_removed = None 
            

            if graph.has_edge(current_edge):
                graph.remove_edges([current_edge])
            else:
                'check the max_parents constraint'
                if max_parents is not None:
                    parents_list = graph.get_parents_by_id(node_id)
                    if len(parents_list) >= max_parents :
                        parent_removed = (choice(parents_list), node_id)
                        graph.remove_edges([parent_removed])
                graph.add_edges([current_edge])
                added = True
            
            current_score =  self.get_score_from_graph(graph,node_id)

            if current_score > actual_best_score:
                'update current best score' 
                actual_best_score = current_score
                patince_count = 0
            else:
                'undo the last update'
                if added:
                    graph.remove_edges([current_edge])
                    'If a parent was removed, add it again to the graph'
                    if parent_removed is not None:
                        graph.add_edges([parent_removed])
                else:
                    graph.add_edges([current_edge])
                'update patience count'
                patince_count += 1

            if patience is not None and patince_count > patience:
                break

        print(f"finito variabile: {node_id}")
        return graph.edges


    def get_score_from_graph(self,graph: ng.NetworkGraph,node_id:str):
        """
        Use the FamScore of a node in order to find the best parent nodes
        Parameters:
           node_id: current node's id
           graph: current graph to be computed 
        Returns:
            The FamSCore for this graph structure
        """

        'inizialize the graph for a single node'
        graph.fast_init(node_id) 

        params_estimation = pe.ParametersEstimator(self.sample_path, graph)

        'Inizialize and compute parameters for node'
        params_estimation.fast_init(node_id)
        SoCims = params_estimation.compute_parameters_for_node(node_id)

        'calculate the FamScore for the node'
        fam_score_obj = fam_score.FamScoreCalculator()

        score = fam_score_obj.get_fam_score(SoCims.actual_cims)
        
        #print(f" lo score per {node_id} risulta: {score} ")
        return score 


    def generate_possible_sub_sets_of_size(self, u: typing.List, size: int, parent_label: str):
        """
        Creates a list containing all possible subsets of the list u of size size,
        that do not contains a the node identified by parent_label.

        Parameters:
            u: the list of nodes
            size: the size of the subsets
            parent_label: the nodes to exclude in the subsets generation
        Returns:
            a Map Object containing a list of lists

        """
        list_without_test_parent = u[:]
        list_without_test_parent.remove(parent_label)
        return map(list, itertools.combinations(list_without_test_parent, size))


