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

import copy


from optimizers.optimizer import Optimizer
from estimators import structure_estimator as se
import structure_graph.network_graph as ng


class ConstraintBasedOptimizer(Optimizer):
    """
    Optimizer class that implement Hill Climbing Search
    
    """
    def __init__(self,
                node_id:str,
                structure_estimator: se.StructureEstimator,
                tot_vars_count:int
                ):
        """
        Compute Optimization process for a structure_estimator

        """
        super().__init__(node_id, structure_estimator)
        self.tot_vars_count = tot_vars_count
        


    def optimize_structure(self):
        """
        Compute Optimization process for a structure_estimator

        Parameters:

        Returns:
            the estimated structure for the node

        """
        print("##################TESTING VAR################", self.node_id)

        graph = ng.NetworkGraph(self.structure_estimator._sample_path.structure)

        other_nodes =  [node for node in self.structure_estimator._sample_path.structure.nodes_labels if node != self.node_id]
        
        for possible_parent in other_nodes:
            graph.add_edges([(possible_parent,self.node_id)])

        
        u = other_nodes
        #tests_parents_numb = len(u)
        #complete_frame = self.complete_graph_frame
        #test_frame = complete_frame.loc[complete_frame['To'].isin([self.node_id])]
        child_states_numb = self.structure_estimator._sample_path.structure.get_states_number(self.node_id)
        b = 0
        while b < len(u):
            parent_indx = 0
            while parent_indx < len(u):
                removed = False
                S = self.structure_estimator.generate_possible_sub_sets_of_size(u, b, u[parent_indx])
                test_parent = u[parent_indx]
                for parents_set in S:
                    if self.structure_estimator.complete_test(test_parent, self.node_id, parents_set, child_states_numb, self.tot_vars_count):
                        graph.remove_edges([(test_parent, self.node_id)])
                        u.remove(test_parent)
                        removed = True
                        break
                if not removed:
                    parent_indx += 1
            b += 1
        self.structure_estimator.cache.clear()
        return graph.edges
