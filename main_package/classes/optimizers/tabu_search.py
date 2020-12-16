import sys
sys.path.append('../')
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from random import choice,sample

from abc import ABC


from optimizers.optimizer import Optimizer
from estimators import structure_estimator as se
import structure_graph.network_graph as ng

import queue


class TabuSearch(Optimizer):
    """
    Optimizer class that implement Hill Climbing Search
    
    """
    def __init__(self,
                node_id:str,
                structure_estimator: se.StructureEstimator,
                max_parents:int = None,
                iterations_number:int= 40,
                patience:int = None,
                tabu_length:int = None,
                tabu_rules_duration = None
                ):
        """
        Compute Optimization process for a structure_estimator

        Parameters:
            node_id: the node label
            structure_estimator: a structure estimator object with the information about the net
            max_parents: maximum number of parents for each variable. If None, disabled
            iterations_number: maximum number of optimization algorithm's iteration
            patience: number of iteration without any improvement before to stop the search.If None, disabled
            tabu_length: maximum lenght of the data structures used in the optimization process
            tabu_rules_duration: number of iterations in which each rule keeps its value 

        """
        super().__init__(node_id, structure_estimator)
        self.max_parents = max_parents
        self.iterations_number = iterations_number
        self.patience = patience
        self.tabu_length = tabu_length
        self.tabu_rules_duration = tabu_rules_duration


    def optimize_structure(self) -> typing.List:
        """
        Compute Optimization process for a structure_estimator

        Parameters:

        Returns:
            the estimated structure for the node

        """
        print(f"tabu search is processing the structure of {self.node_id}")

        'Create the graph for the single node'
        graph = ng.NetworkGraph(self.structure_estimator.sample_path.structure)

        'get all the possible parents'
        other_nodes =  set([node for node in self.structure_estimator.sample_path.structure.nodes_labels if node != self.node_id])

        'calculate the score for the node without parents'
        actual_best_score = self.structure_estimator.get_score_from_graph(graph,self.node_id)


        'initialize tabu_length and tabu_rules_duration if None'
        if self.tabu_length is None:
            self.tabu_length = len(other_nodes)

        if self.tabu_rules_duration is None:
            self.tabu_tabu_rules_durationength = len(other_nodes)

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
            #print('**************************')
            current_score =  self.structure_estimator.get_score_from_graph(graph,self.node_id)


            # print("-------------------------------------------")
            # print(f"Current new parent: {current_new_parent}")
            # print(f"Current score: {current_score}")
            # print(f"Current best score: {actual_best_score}")
            # print(f"tabu list : {str(tabu_set)} length: {len(tabu_set)}")
            # print(f"tabu queue : {str(tabu_queue)} length: {tabu_queue.qsize()}")
            # print(f"graph edges: {graph.edges}")

            # print("-------------------------------------------")
            # input()
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