import typing as ty

import numpy as np


class Structure:
    """Contains all the infos about the network structure(nodes labels, nodes caridinalites, edges, indexes)

    :param nodes_labels_list: the symbolic names of the variables
    :type nodes_labels_list: List
    :param nodes_indexes_arr: the indexes of the nodes
    :type nodes_indexes_arr: numpy.ndArray
    :param nodes_vals_arr: the cardinalites of the nodes
    :type nodes_vals_arr: numpy.ndArray
    :param edges_list: the edges of the network
    :type edges_list: List
    :param total_variables_number: the total number of variables in the net
    :type total_variables_number: int
    """

    def __init__(self, nodes_labels_list: ty.List, nodes_indexes_arr: np.ndarray, nodes_vals_arr: np.ndarray,
                 edges_list: ty.List, total_variables_number: int):
        """Constructor Method
        """
        self._nodes_labels_list = nodes_labels_list
        self._nodes_indexes_arr = nodes_indexes_arr
        self._nodes_vals_arr = nodes_vals_arr
        self._edges_list = edges_list
        self._total_variables_number = total_variables_number

    @property
    def edges(self) -> ty.List:
        return self._edges_list

    @property
    def nodes_labels(self) -> ty.List:
        return self._nodes_labels_list

    @property
    def nodes_indexes(self) -> np.ndarray:
        return self._nodes_indexes_arr

    @property
    def nodes_values(self) -> np.ndarray:
        return self._nodes_vals_arr

    @property
    def total_variables_number(self) -> int:
        return self._total_variables_number

    def get_node_id(self, node_indx: int) -> str:
        """Given the ``node_index`` returns the node label.

        :param node_indx: the node index
        :type node_indx: int
        :return: the node label
        :rtype: string
        """
        return self._nodes_labels_list[node_indx]

    def get_node_indx(self, node_id: str) -> int:
        """Given the ``node_index`` returns the node label.

        :param node_id: the node label
        :type node_id: string
        :return: the node index
        :rtype: int
        """
        pos_indx = self._nodes_labels_list.index(node_id)
        return self._nodes_indexes_arr[pos_indx]

    def get_positional_node_indx(self, node_id: str) -> int:
        return self._nodes_labels_list.index(node_id)

    def get_states_number(self, node: str) -> int:
        """Given the node label ``node`` returns the cardinality of the node.

        :param node: the node label
        :type node: string
        :return: the node cardinality
        :rtype: int
        """
        pos_indx = self._nodes_labels_list.index(node)
        return self._nodes_vals_arr[pos_indx]

    def __repr__(self):
        return "Variables:\n" + str(self._nodes_labels_list) +"\nValues:\n"+ str(self._nodes_vals_arr) +\
               "\nEdges: \n" + str(self._edges_list)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Structure):
            return set(self._nodes_labels_list) == set(other._nodes_labels_list) and \
                   np.array_equal(self._nodes_vals_arr, other._nodes_vals_arr) and \
                   np.array_equal(self._nodes_indexes_arr, other._nodes_indexes_arr) and \
                   self._edges_list == other._edges_list

        return NotImplemented

