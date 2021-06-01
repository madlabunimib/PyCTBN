
# License: MIT License


import itertools
import json
import typing

import networkx as nx
import numpy as np

from random import choice

from abc import ABC

import copy


from .optimizer import Optimizer
from ..estimators.structure_estimator import StructureEstimator
from ..structure_graph.network_graph import NetworkGraph


class ConstraintBasedOptimizer(Optimizer):
    """
    Optimizer class that implement a CTPC Algorithm

    :param node_id: current node's id
    :type node_id: string
    :param structure_estimator: a structure estimator object with the information about the net
    :type structure_estimator: class:'StructureEstimator' 
    :param tot_vars_count: number of variables in the dataset
    :type tot_vars_count: int
    """
    def __init__(self,
                node_id:str,
                structure_estimator: StructureEstimator,
                tot_vars_count:int
                ):
        """
        Constructor
        """
        super().__init__(node_id, structure_estimator)
        self.tot_vars_count = tot_vars_count
        


    def optimize_structure(self):
        """
        Compute Optimization process for a structure_estimator by using a CTPC Algorithm

        :return: the estimated structure for the node
        :rtype: List
        """
        print("##################TESTING VAR################", self.node_id)

        graph = NetworkGraph(self.structure_estimator._sample_path.structure)

        other_nodes =  [node for node in self.structure_estimator._sample_path.structure.nodes_labels if node != self.node_id]
        
        for possible_parent in other_nodes:
            graph.add_edges([(possible_parent,self.node_id)])

        
        u = other_nodes
        child_states_numb = self.structure_estimator._sample_path.structure.get_states_number(self.node_id)
        b = 0
        while b < len(u):
            parent_indx = 0
            while parent_indx < len(u):
                removed = False
                test_parent = u[parent_indx]
               	i = self.structure_estimator._sample_path.structure.get_node_indx(test_parent)
                j = self.structure_estimator._sample_path.structure.get_node_indx(self.node_id)
                if self.structure_estimator._removable_edges_matrix[i][j]:
                    S = StructureEstimator.generate_possible_sub_sets_of_size(u, b, test_parent)
                    for parents_set in S:
	                    if self.structure_estimator.complete_test(test_parent, self.node_id, parents_set, child_states_numb, self.tot_vars_count,i,j):
	                        graph.remove_edges([(test_parent, self.node_id)])
	                        u.remove(test_parent)
	                        removed = True
	                        break
                if not removed:
                    parent_indx += 1
            b += 1
        self.structure_estimator._cache.clear()
        return graph.edges