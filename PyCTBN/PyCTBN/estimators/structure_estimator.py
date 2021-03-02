
import itertools
import json
import typing

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from abc import ABC
import os
import abc

from ..utility.cache import Cache
from ..structure_graph.conditional_intensity_matrix import ConditionalIntensityMatrix
from ..structure_graph.network_graph import NetworkGraph
from .parameters_estimator import ParametersEstimator
from ..structure_graph.sample_path import SamplePath
from ..structure_graph.structure import Structure


class StructureEstimator(object):
    """Has the task of estimating the network structure given the trajectories in ``samplepath``.

    :param sample_path: the _sample_path object containing the trajectories and the real structure
    :type sample_path: SamplePath
    :_nodes: the nodes labels
    :_nodes_vals: the nodes cardinalities
    :_nodes_indxs: the nodes indexes
    :_complete_graph: the complete directed graph built using the nodes labels in ``_nodes``
    """

    def __init__(self, sample_path: SamplePath, known_edges: typing.List = None):
        self._sample_path = sample_path
        self._nodes = np.array(self._sample_path.structure.nodes_labels)
        self._nodes_vals = self._sample_path.structure.nodes_values
        self._nodes_indxs = self._sample_path.structure.nodes_indexes
        self._removable_edges_matrix = self.build_removable_edges_matrix(known_edges)
        self._complete_graph = StructureEstimator.build_complete_graph(self._sample_path.structure.nodes_labels)
        

    def build_removable_edges_matrix(self, known_edges: typing.List):
        """Builds a boolean matrix who shows if a edge could be removed or not, based on prior knowledge given:

        :param known_edges: the list of nodes labels
        :type known_edges: List
        :return: a boolean matrix
        :rtype: np.ndarray
        """
        tot_vars_count = self._sample_path.total_variables_count
        complete_adj_matrix = np.full((tot_vars_count, tot_vars_count), True)
        if known_edges:
            for edge in known_edges:
                i = self._sample_path.structure.get_node_indx(edge[0])
                j = self._sample_path.structure.get_node_indx(edge[1])
                complete_adj_matrix[i][j] = False
        return complete_adj_matrix

    @staticmethod
    def build_complete_graph(node_ids: typing.List) -> nx.DiGraph:
        """Builds a complete directed graph (no self loops) given the nodes labels in the list ``node_ids``:

        :param node_ids: the list of nodes labels
        :type node_ids: List
        :return: a complete Digraph Object
        :rtype: networkx.DiGraph
        """
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(node_ids)
        complete_graph.add_edges_from(itertools.permutations(node_ids, 2))
        return complete_graph


    @staticmethod
    def generate_possible_sub_sets_of_size( u: typing.List, size: int, parent_label: str):
        """Creates a list containing all possible subsets of the list ``u`` of size ``size``,
        that do not contains a the node identified by ``parent_label``.

        :param u: the list of nodes
        :type u: List
        :param size: the size of the subsets
        :type size: int
        :param parent_label: the node to exclude in the subsets generation
        :type parent_label: string
        :return: an Iterator Object containing a list of lists
        :rtype: Iterator
        """
        list_without_test_parent = u[:]
        list_without_test_parent.remove(parent_label)
        return map(list, itertools.combinations(list_without_test_parent, size))

    def save_results(self) -> None:
        """Save the estimated Structure to a .json file in the path where the data are loaded from.
        The file is named as the input dataset but the `results_` word is appended to the results file.
        """
        res = json_graph.node_link_data(self._complete_graph)
        name = self._sample_path._importer.file_path.rsplit('/', 1)[-1]
        name = name.split('.', 1)[0]
        name += '_' + str(self._sample_path._importer.dataset_id())
        name += '.json'
        file_name = 'results_' + name
        with open(file_name, 'w') as f:
            json.dump(res, f)


    #def remove_diagonal_elements(self, matrix):
    #   m = matrix.shape[0]
    #    strided = np.lib.stride_tricks.as_strided
    #    s0, s1 = matrix.strides
    #    return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)


    @abc.abstractmethod
    def estimate_structure(self) -> typing.List:
        """Abstract method to estimate the structure

        :return: List of estimated edges
        :rtype: Typing.List
        """
        pass

    
    def adjacency_matrix(self) -> np.ndarray:
        """Converts the estimated structure ``_complete_graph`` to a boolean adjacency matrix representation.

        :return: The adjacency matrix of the graph ``_complete_graph``
        :rtype: numpy.ndArray
        """
        return nx.adj_matrix(self._complete_graph).toarray().astype(bool)

    def spurious_edges(self) -> typing.List:
        """Return the spurious edges present in the estimated structure, if a prior net structure is present in
            ``_sample_path.structure``.

        :return: A list containing the spurious edges
        :rtype: List
        """
        if not self._sample_path.has_prior_net_structure:
            return []
        real_graph = nx.DiGraph()
        real_graph.add_nodes_from(self._sample_path.structure.nodes_labels)
        real_graph.add_edges_from(self._sample_path.structure.edges)
        return nx.difference(real_graph, self._complete_graph).edges

    def save_plot_estimated_structure_graph(self, file_path: str) -> None:  
            """Plot the estimated structure in a graphical model style, use .png extension.
            Spurious edges are colored in red if a prior structure is present.

            :param file_path: path to save the file to
            :type: string
            """
            graph_to_draw = nx.DiGraph()
            spurious_edges = self.spurious_edges()
            non_spurious_edges = list(set(self._complete_graph.edges) - set(spurious_edges))
            edges_colors = ['red' if edge in spurious_edges else 'black' for edge in self._complete_graph.edges]
            graph_to_draw.add_edges_from(spurious_edges)
            graph_to_draw.add_edges_from(non_spurious_edges)
            pos = nx.spring_layout(graph_to_draw, k=0.5*1/np.sqrt(len(graph_to_draw.nodes())), iterations=50,scale=10)
            options = {
                "node_size": 2000,
                "node_color": "white",
                "edgecolors": "black",
                'linewidths':2,
                "with_labels":True,
                "font_size":13,
                'connectionstyle': 'arc3, rad = 0.1',
                "arrowsize": 15,
                "arrowstyle": '<|-',
                "width": 1,
                "edge_color":edges_colors,
            }

            nx.draw(graph_to_draw, pos, **options)
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("off")
            plt.savefig(file_path)
            plt.clf()
            print("Estimated Structure Plot Saved At: ", os.path.abspath(file_path))





