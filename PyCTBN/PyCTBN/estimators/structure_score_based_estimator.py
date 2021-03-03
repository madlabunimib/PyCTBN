
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from random import choice

import concurrent.futures

import copy

from ..structure_graph.conditional_intensity_matrix import ConditionalIntensityMatrix
from ..structure_graph.network_graph import NetworkGraph
from .parameters_estimator import ParametersEstimator
from .structure_estimator import StructureEstimator
from ..structure_graph.sample_path import SamplePath
from ..structure_graph.structure import Structure
from .fam_score_calculator import FamScoreCalculator
from ..optimizers.hill_climbing_search import HillClimbing
from ..optimizers.tabu_search import TabuSearch


import multiprocessing
from multiprocessing import Pool




class StructureScoreBasedEstimator(StructureEstimator):
    """
    Has the task of estimating the network structure given the trajectories in samplepath by
    using a score based approach and differt kinds of optimization algorithms.

    :param sample_path: the _sample_path object containing the trajectories and the real structure
    :type sample_path: SamplePath
    :param tau_xu: hyperparameter over the CTBN’s q parameters, default to 0.1
    :type tau_xu: float, optional
    :param alpha_xu: hyperparameter over the CTBN’s q parameters, default to 1
    :type alpha_xu: float, optional
    :param known_edges: List of known edges, default to []
    :type known_edges: List, optional

    """

    def __init__(self, sample_path: SamplePath, tau_xu:int=0.1, alpha_xu:int = 1,known_edges: typing.List= []):
        super().__init__(sample_path,known_edges)
        self.tau_xu=tau_xu
        self.alpha_xu=alpha_xu


    def estimate_structure(self, max_parents:int = None, iterations_number:int= 40,
                         patience:int = None, tabu_length:int = None, tabu_rules_duration:int = None,
                         optimizer: str = 'tabu',disable_multiprocessing:bool= False ):
        """
        Compute the score-based algorithm to find the optimal structure

        :param max_parents: maximum number of parents for each variable. If None, disabled, default to None
        :type max_parents: int, optional
        :param iterations_number: maximum number of optimization algorithm's iteration, default to 40
        :type iterations_number: int, optional
        :param patience: number of iteration without any improvement before to stop the search.If None, disabled, default to None
        :type patience: int, optional
        :param tabu_length: maximum lenght of the data structures used in the optimization process, default to None
        :type tabu_length: int, optional
        :param tabu_rules_duration: number of iterations in which each rule keeps its value, default to None
        :type tabu_rules_duration: int, optional
        :param optimizer: name of the optimizer algorithm. Possible values: 'hill' (Hill climbing),'tabu' (tabu search), defualt to 'tabu'
        :type optimizer: string, optional
        :param disable_multiprocessing: true if you desire to disable the multiprocessing operations, default to False
        :type disable_multiprocessing: Boolean, optional
        """
        'Save the true edges structure in tuples'
        true_edges = copy.deepcopy(self._sample_path.structure.edges)
        true_edges = set(map(tuple, true_edges))

        'Remove all the edges from the structure'   
        self._sample_path.structure.clean_structure_edges()

        estimate_parents = self.estimate_parents

        n_nodes= len(self._nodes)
        
        l_max_parents= [max_parents] * n_nodes
        l_iterations_number = [iterations_number] * n_nodes
        l_patience = [patience] * n_nodes
        l_tabu_length = [tabu_length] * n_nodes
        l_tabu_rules_duration = [tabu_rules_duration] * n_nodes
        l_optimizer = [optimizer] * n_nodes


        'get the number of CPU'
        cpu_count = multiprocessing.cpu_count()
        print(f"CPU COUNT: {cpu_count}")

        if disable_multiprocessing:
            cpu_count = 1

        



        #with get_context("spawn").Pool(processes=cpu_count) as pool:
        #with multiprocessing.Pool(processes=cpu_count) as pool:

        'Estimate the best parents for each node'
        if disable_multiprocessing:
            list_edges_partial = [estimate_parents(n,max_parents,iterations_number,patience,tabu_length,tabu_rules_duration,optimizer) for n in self._nodes]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                list_edges_partial = executor.map(estimate_parents, 
                                                            self._nodes,
                                                            l_max_parents,
                                                            l_iterations_number,
                                                            l_patience,
                                                            l_tabu_length,
                                                            l_tabu_rules_duration,
                                                            l_optimizer)


        'Concatenate all the edges list'
        set_list_edges =  set(itertools.chain.from_iterable(list_edges_partial))


        'calculate precision and recall'
        n_missing_edges = 0
        n_added_fake_edges = 0

        try:
            n_added_fake_edges = len(set_list_edges.difference(true_edges))

            n_missing_edges = len(true_edges.difference(set_list_edges))

            n_true_positive = len(true_edges) - n_missing_edges

            precision = n_true_positive / (n_true_positive + n_added_fake_edges)

            recall = n_true_positive / (n_true_positive + n_missing_edges)
        
            print(true_edges)
            print(set_list_edges)
            print(f"precision: {precision} ")
            print(f"recall: {recall} ")
        except Exception as e: 
            print(f"errore: {e}")


        'Update the graph'
        self._complete_graph = nx.DiGraph()
        self._complete_graph.add_edges_from(set_list_edges)

        return set_list_edges
    

    def estimate_parents(self,node_id:str, max_parents:int = None, iterations_number:int= 40,
                            patience:int = 10, tabu_length:int = None, tabu_rules_duration:int=5, 
                            optimizer:str = 'hill' ):
        """
        Use the FamScore of a node in order to find the best parent nodes
        
        :param node_id: current node's id
        :type node_id: string
        :param max_parents: maximum number of parents for each variable. If None, disabled, default to None
        :type max_parents: int, optional
        :param iterations_number: maximum number of optimization algorithm's iteration, default to 40
        :type iterations_number: int, optional
        :param patience: number of iteration without any improvement before to stop the search.If None, disabled, default to None
        :type patience: int, optional
        :param tabu_length: maximum lenght of the data structures used in the optimization process, default to None
        :type tabu_length: int, optional
        :param tabu_rules_duration: number of iterations in which each rule keeps its value, default to None
        :type tabu_rules_duration: int, optional
        :param optimizer: name of the optimizer algorithm. Possible values: 'hill' (Hill climbing),'tabu' (tabu search), defualt to 'tabu'
        :type optimizer: string, optional

        :return: A list of the best edges for the currente node
        :rtype: List
        """

        "choose the optimizer algotithm"
        if optimizer == 'tabu':
            optimizer = TabuSearch(
                        node_id = node_id,
                        structure_estimator = self,
                        max_parents = max_parents,
                        iterations_number = iterations_number,
                        patience = patience,
                        tabu_length = tabu_length,
                        tabu_rules_duration = tabu_rules_duration)
        else: #if optimizer == 'hill':
            optimizer = HillClimbing(
                                    node_id = node_id,
                                    structure_estimator = self,
                                    max_parents = max_parents,
                                    iterations_number = iterations_number,
                                    patience = patience)

        "call the optmizer's function that calculates the current node's parents"
        return optimizer.optimize_structure()

    
    def get_score_from_graph(self,
                            graph: NetworkGraph,
                            node_id:str):
        """
        Get the FamScore of a node 
        
        :param node_id: current node's id
        :type node_id: string
        :param graph: current graph to be computed
        :type graph: class:'NetworkGraph'


        :return: The FamSCore for this graph structure
        :rtype: float
        """

        'inizialize the graph for a single node'
        graph.fast_init(node_id) 

        params_estimation = ParametersEstimator(self._sample_path.trajectories, graph)

        'Inizialize and compute parameters for node'
        params_estimation.fast_init(node_id)
        SoCims = params_estimation.compute_parameters_for_node(node_id)

        'calculate the FamScore for the node'
        fam_score_obj = FamScoreCalculator()

        score = fam_score_obj.get_fam_score(SoCims.actual_cims,tau_xu = self.tau_xu,alpha_xu=self.alpha_xu)
        
        #print(f" lo score per {node_id} risulta: {score} ")
        return score 
