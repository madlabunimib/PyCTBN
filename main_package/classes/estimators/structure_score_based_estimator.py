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
import optimizers.hill_climbing_search as hill
import optimizers.tabu_search as tabu

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
    def estimate_structure(self, max_parents:int = None, iterations_number:int= 40,
                         patience:int = None, tabu_length:int = None, tabu_rules_duration:int = 5,
                         optimizer: str = 'hill' ):
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
        true_edges = set(map(tuple, true_edges))

        'Remove all the edges from the structure'   
        self.sample_path.structure.clean_structure_edges()

        estimate_parents = self.estimate_parents

        n_nodes= len(self.nodes)
        
        l_max_parents= [max_parents] * n_nodes
        l_iterations_number = [iterations_number] * n_nodes
        l_patience = [patience] * n_nodes
        l_tabu_length = [tabu_length] * n_nodes
        l_tabu_rules_duration = [tabu_rules_duration] * n_nodes
        l_optimizer = [optimizer] * n_nodes


        'get the number of CPU'
        cpu_count = multiprocessing.cpu_count()

        #cpu_count = 1

        'Estimate the best parents for each node'
        with multiprocessing.Pool(processes=cpu_count) as pool:
            list_edges_partial = pool.starmap(estimate_parents, zip(
                                                                self.nodes,
                                                                l_max_parents,
                                                                l_iterations_number,
                                                                l_patience,
                                                                l_tabu_length,
                                                                l_tabu_rules_duration,
                                                                l_optimizer))
            # list_edges_partial = [estimate_parents(n,max_parents,iterations_number,patience,tabu_length,tabu_rules_duration,optimizer) for n in self.nodes]
            #list_edges_partial = p.map(estimate_parents, self.nodes)
            #list_edges_partial= estimate_parents('Q',max_parents,iterations_number,patience,tabu_length,tabu_rules_duration,optimizer) 

        'Concatenate all the edges list'
        set_list_edges =  set(itertools.chain.from_iterable(list_edges_partial))

        #print('-------------------------')


        'calculate precision and recall'
        n_missing_edges = 0
        n_added_fake_edges = 0

        try:
            n_added_fake_edges = len(set_list_edges.difference(true_edges))

            n_missing_edges = len(true_edges.difference(set_list_edges))

            n_true_positive = len(true_edges) - n_missing_edges

            precision = n_true_positive / (n_true_positive + n_added_fake_edges)

            recall = n_true_positive / (n_true_positive + n_missing_edges)


            # print(f"n archi reali non trovati: {n_missing_edges}")
            # print(f"n archi non reali aggiunti: {n_added_fake_edges}")
            print(true_edges)
            print(set_list_edges)
            print(f"precision: {precision} ")
            print(f"recall: {recall} ")
        except Exception as e: 
            print(f"errore: {e}")

        return set_list_edges
    

    def estimate_parents(self,node_id:str, max_parents:int = None, iterations_number:int= 40,
                            patience:int = 10, tabu_length:int = None, tabu_rules_duration:int=5, 
                            optimizer:str = 'hill' ):
        """
        Use the FamScore of a node in order to find the best parent nodes
        Parameters:
            node_id: current node's id
            max_parents: maximum number of parents for each variable. If None, disabled
            iterations_number: maximum number of optimization algorithm's iteration
            patience: number of iteration without any improvement before to stop the search.If None, disabled
            tabu_length: maximum lenght of the data structures used in the optimization process
            tabu_rules_duration: number of iterations in which each rule keeps its value 
            optimzer: name of the optimizer algorithm. Possible values: 'hill' (Hill climbing),'tabu' (tabu search)
        Returns:
            A list of the best edges for the currente node
        """

        "choose the optimizer algotithm"
        if optimizer == 'tabu':
            optimizer = tabu.TabuSearch(
                        node_id = node_id,
                        structure_estimator = self,
                        max_parents = max_parents,
                        iterations_number = iterations_number,
                        patience = patience,
                        tabu_length = tabu_length,
                        tabu_rules_duration = tabu_rules_duration)
        else: #if optimizer == 'hill':
            optimizer = hill.HillClimbing(
                                    node_id = node_id,
                                    structure_estimator = self,
                                    max_parents = max_parents,
                                    iterations_number = iterations_number,
                                    patience = patience)

        "call the optmizer's function that calculates the current node's parents"
        return optimizer.optimize_structure()

    
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




