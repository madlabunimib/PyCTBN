
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from random import choice

import cache as ch
import conditional_intensity_matrix as condim
import network_graph as ng
import parameters_estimator as pe
import sample_path as sp
import structure as st
import fam_score_calculator as fam_score


'''
#TODO: Insert maximum number of parents
#TODO: Evaluate if it's better to start from a complete or an empty graph 
#TODO: Create a parent class StructureEstimator and Two Subclasses (Score-Based and Constraint-Based)
'''

class StructureScoreBasedEstimator:
    """
    Has the task of estimating the network structure given the trajectories in samplepath by
    using a score based approach.

    :sample_path: the sample_path object containing the trajectories and the real structure

    :nodes: the nodes labels
    :nodes_vals: the nodes cardinalities
    :nodes_indxs: the nodes indexes
    :complete_graph: the complete directed graph built using the nodes labels in nodes
    :cache: the cache object
    """

    def __init__(self, sample_path: sp.SamplePath):
        self.sample_path = sample_path
        self.nodes = np.array(self.sample_path.structure.nodes_labels)
        self.nodes_vals = self.sample_path.structure.nodes_values
        self.nodes_indxs = self.sample_path.structure.nodes_indexes
        self.complete_graph = self.build_complete_graph(self.sample_path.structure.nodes_labels)
        self.cache = ch.Cache()

    def build_complete_graph(self, node_ids: typing.List):
        """
        Builds a complete directed graph (no self loops) given the nodes labels in the list node_ids:

        Parameters:
            node_ids: the list of nodes labels
        Returns:
            a complete Digraph Object
        """
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(node_ids)
        complete_graph.add_edges_from(itertools.permutations(node_ids, 2))
        return complete_graph



    def estimate_structure(self):
        """
        Compute the score-based algorithm to find the optimal structure

        Parameters:
           node_id: the label of the node
        Returns:
            void

        """
        estimate_parents = self.estimate_parents
        'Estimate the best parents for each node'
        #[estimate_parents(n) for n in self.nodes]
        estimate_parents('X')
    
    def estimate_parents(self,node_id:str):
        """
        Use the FamScore of a node in order to find the best parent nodes

        Parameters:
            void
        Returns:
            void

        """
        'Create the graph for the single node'
        graph = ng.NetworkGraph(self.sample_path.structure)
        'inizialize the graph for a single node'
        graph.fast_init(node_id)

        params_estimation = pe.ParametersEstimator(self.sample_path, graph)

        'Inizialize and compute parameters for node'
        params_estimation.fast_init(node_id)
        SoCims = params_estimation.compute_parameters_for_node(node_id)

        'Get the node\'s parents list'
        parents = graph.get_parents_by_id(node_id)

        values = graph.get_states_number(parents[0])

        print(f" actual_cims {len(SoCims.actual_cims)} padri {len(parents)} ")

        fam_score_obj = fam_score.FamScoreCalculator()

        score = fam_score_obj.get_fam_score(SoCims.actual_cims)
        
        '''mask = np.array([True,True])

        cims = SoCims.filter_cims_with_mask(mask,[1,1])

       # print(f"-----{len(SoCims.transition_matrices)}-------")
        print(f"{cims[0].state_transition_matrix}")

        cims = SoCims.filter_cims_with_mask(mask,[0,0])

        print(f"---parents {len(parents)}---------")
        print(f"{cims[0].state_transition_matrix}")
        '''

       
        


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

    def save_results(self):
        """
        Save the estimated Structure to a .json file

        Parameters:
            void
        Returns:
            void
        """
        res = json_graph.node_link_data(self.complete_graph)
        name = self.sample_path.importer.file_path.rsplit('/',1)[-1]
        #print(name)
        name = 'results_' + name
        with open(name, 'w+') as f:
            json.dump(res, f)


    def remove_diagonal_elements(self, matrix):
        m = matrix.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = matrix.strides
        return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

