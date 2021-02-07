import sys
sys.path.append('../')
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from random import choice

from abc import ABC


from optimizers.optimizer import Optimizer
from estimators import structure_estimator as se
import structure_graph.network_graph as ng


class HillClimbing(Optimizer):
    """
    Optimizer class that implement Hill Climbing Search
    

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


    
    """
    def __init__(self,
                node_id:str,
                structure_estimator: se.StructureEstimator,
                max_parents:int = None,
                iterations_number:int= 40,
                patience:int = None
                ):
        """
        Constructor
        """
        super().__init__(node_id, structure_estimator)
        self.max_parents = max_parents
        self.iterations_number = iterations_number
        self.patience = patience
        


    def optimize_structure(self) -> typing.List:
        """
        Compute Optimization process for a structure_estimator by using a Hill Climbing Algorithm

        :return: the estimated structure for the node
        :rtype: List
        """

        #'Create the graph for the single node'
        graph = ng.NetworkGraph(self.structure_estimator._sample_path.structure)

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
        other_nodes =  [node for node in 
                                            self.structure_estimator._sample_path.structure.nodes_labels if
                                                                                            node != self.node_id and
                                                                                            not prior_parents.__contains__(node)]
        
        actual_best_score = self.structure_estimator.get_score_from_graph(graph,self.node_id)

        patince_count = 0
        for i in range(self.iterations_number):
            'choose a new random edge'
            current_new_parent = choice(other_nodes)
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

            if self.patience is not None and patince_count > self.patience:
                break

        print(f"finito variabile: {self.node_id}")
        return graph.edges