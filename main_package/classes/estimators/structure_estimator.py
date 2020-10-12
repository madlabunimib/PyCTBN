import sys
sys.path.append('../')
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from abc import ABC

import utility.cache as ch
import structure_graph.conditional_intensity_matrix as condim
import structure_graph.network_graph as ng
import estimators.parameters_estimator as pe
import structure_graph.sample_path as sp
import structure_graph.structure as st


class StructureEstimator(ABC):
    """
    Has the task of estimating the network structure given the trajectories in samplepath.

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
        name = '../results_' + name
        with open(name, 'w+') as f:
            json.dump(res, f)


    def remove_diagonal_elements(self, matrix):
        m = matrix.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = matrix.strides
        return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

