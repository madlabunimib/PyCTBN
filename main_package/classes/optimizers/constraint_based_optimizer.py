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
        u = list(self.structure_estimator.complete_graph.predecessors(self.node_id))
        #tests_parents_numb = len(u)
        #complete_frame = self.complete_graph_frame
        #test_frame = complete_frame.loc[complete_frame['To'].isin([self.node_id])]
        child_states_numb = self.structure_estimator.sample_path.structure.get_states_number(self.node_id)
        b = 0
        while b < len(u):
            #for parent_id in u:
            parent_indx = 0
            while parent_indx < len(u):
                removed = False
                #if not list(self.structure_estimator.generate_possible_sub_sets_of_size(u, b, u[parent_indx])):
                    #break
                S = self.structure_estimator.generate_possible_sub_sets_of_size(u, b, u[parent_indx])
                #print("U Set", u)
                #print("S", S)
                test_parent = u[parent_indx]
                #print("Test Parent", test_parent)
                for parents_set in S:
                    #print("Parent Set", parents_set)
                    #print("Test Parent", test_parent)
                    if self.structure_estimator.complete_test(test_parent, self.node_id, parents_set, child_states_numb, self.tot_vars_count):
                        #print("Removing EDGE:", test_parent, self.node_id)
                        self.structure_estimator.complete_graph.remove_edge(test_parent, self.node_id)
                        u.remove(test_parent)
                        removed = True
                        break
                    #else:
                        #parent_indx += 1
                if not removed:
                    parent_indx += 1
            b += 1
        self.structure_estimator.cache.clear()