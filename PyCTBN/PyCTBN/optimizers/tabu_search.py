
import itertools
import json
import typing

import networkx as nx
import numpy as np

from random import choice,sample

from abc import ABC


from .optimizer import Optimizer
from ..estimators.structure_estimator import StructureEstimator
from ..structure_graph.network_graph import NetworkGraph

import queue


class TabuSearch(Optimizer):
    """
    Optimizer class that implement Tabu Search


    :param node_id: current node's id
    :type node_id: string
    :param structure_estimator: a structure estimator object with the information about the net
    :type structure_estimator: class:'StructureEstimator' 
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

    
    """
    def __init__(self,
                node_id:str,
                structure_estimator: StructureEstimator,
                max_parents:int = None,
                iterations_number:int= 40,
                patience:int = None,
                tabu_length:int = None,
                tabu_rules_duration = None
                ):
        """
        Constructor
        """
        super().__init__(node_id, structure_estimator)
        self.max_parents = max_parents
        self.iterations_number = iterations_number
        self.patience = patience
        self.tabu_length = tabu_length
        self.tabu_rules_duration = tabu_rules_duration


    def optimize_structure(self) -> typing.List:
        """
        Compute Optimization process for a structure_estimator by using a Hill Climbing Algorithm

        :return: the estimated structure for the node
        :rtype: List
        """
        print(f"tabu search is processing the structure of {self.node_id}")

        'Create the graph for the single node'
        graph = NetworkGraph(self.structure_estimator._sample_path.structure)

        'get the index for the current node'
        node_index = self.structure_estimator._sample_path._structure.get_node_indx(self.node_id)

        'list of prior edges'
        prior_parents = set()

        'Add the edges from prior knowledge'
        for i in range(len(self.structure_estimator._removable_edges_matrix)):
            if not self.structure_estimator._removable_edges_matrix[i][node_index]:
                parent_id= self.structure_estimator._sample_path._structure.get_node_id(i)
                prior_parents.add(parent_id)

                'Add the node to the starting structure'
                graph.add_edges([(parent_id, self.node_id)])



        'get all the possible parents'
        other_nodes =  set([node for node in 
                                            self.structure_estimator._sample_path.structure.nodes_labels if
                                                                                            node != self.node_id and
                                                                                            not prior_parents.__contains__(node)])

        'calculate the score for the node without parents'
        actual_best_score = self.structure_estimator.get_score_from_graph(graph,self.node_id)


        'initialize tabu_length and tabu_rules_duration if None'
        if self.tabu_length is None:
            self.tabu_length = len(other_nodes)

        if self.tabu_rules_duration is None:
            self.tabu_rules_duration = len(other_nodes)

        'inizialize the data structures'
        tabu_set = set()
        tabu_queue = queue.Queue()

        patince_count = 0
        tabu_count = 0 
        for i in range(self.iterations_number):
            
            current_possible_nodes = other_nodes.difference(tabu_set)

            'choose a new random edge according to tabu restiction'
            if(len(current_possible_nodes) > 0):
                current_new_parent = sample(current_possible_nodes,k=1)[0]
            else:
                current_new_parent = tabu_queue.get()
                tabu_set.remove(current_new_parent)


            
            current_edge =  (current_new_parent,self.node_id)
            added = False
            parent_removed = None 

            if graph.has_edge(current_edge):
                graph.remove_edges([current_edge])
            else:
                'check the max_parents constraint'
                if self.max_parents is not None:
                    parents_list = graph.get_parents_by_id(self.node_id)
                    if len(parents_list) >= self.max_parents :
                        parent_removed = (choice(parents_list), self.node_id)
                        graph.remove_edges([parent_removed])
                graph.add_edges([current_edge])
                added = True
           
            current_score =  self.structure_estimator.get_score_from_graph(graph,self.node_id)

            if current_score > actual_best_score:
                'update current best score' 
                actual_best_score = current_score
                patince_count = 0
                'update tabu list'


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
                
            
            if tabu_queue.qsize() >= self.tabu_length:
                    current_removed = tabu_queue.get()
                    tabu_set.remove(current_removed)
            'Add the node on the tabu list'
            tabu_queue.put(current_new_parent)
            tabu_set.add(current_new_parent)

            tabu_count += 1

            'Every tabu_rules_duration step remove an item from the tabu list '
            if tabu_count % self.tabu_rules_duration == 0:
                if tabu_queue.qsize() > 0:
                    current_removed = tabu_queue.get()
                    tabu_set.remove(current_removed)
                    tabu_count = 0
                else:
                    tabu_count = 0

            if self.patience is not None and patince_count > self.patience:
                break

        print(f"finito variabile: {self.node_id}")
        return graph.edges