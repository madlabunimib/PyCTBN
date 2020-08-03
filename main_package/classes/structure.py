import typing as ty
import numpy as np


class Structure:
    """
    Contains all the infos about the network structure(nodes names, nodes caridinalites, edges...)

    :nodes_labels_list: the symbolic names of the variables
    :nodes_indexes_arr: the indexes of the nodes
    :nodes_vals_arr: the cardinalites of the nodes
    :edges_list: the edges of the network
    :total_variables_number: the total number of variables in the net
    """

    def __init__(self, nodes_label_list: ty.List, node_indexes_arr: np.array, nodes_vals_arr: np.array,
                 edges_list: ty.List, total_variables_number: int):
        self._nodes_labels_list = nodes_label_list
        self._nodes_indexes_arr = node_indexes_arr
        self._nodes_vals_arr = nodes_vals_arr
        self._edges_list = edges_list
        self._total_variables_number = total_variables_number

    @property
    def edges(self):
        #records = self.structure_frame.to_records(index=False)
        #edges_list = list(records)
        return self._edges_list

    @property
    def nodes_labels(self):
        return self._nodes_labels_list

    @property
    def nodes_indexes(self):
        return self._nodes_indexes_arr

    @property
    def nodes_values(self):
        return self._nodes_vals_arr

    @property
    def total_variables_number(self):
        return self._total_variables_number

    def get_node_id(self, node_indx: int):
        return self._nodes_labels_list[node_indx]

    def get_node_indx(self, node_id: str):
        pos_indx = self._nodes_labels_list.index(node_id)
        return self._nodes_indexes_arr[pos_indx]

    def get_positional_node_indx(self, node_id: str):
        return self._nodes_labels_list.index(node_id)

    def get_states_number(self, node: str):
        pos_indx = self._nodes_labels_list.index(node)
        return self._nodes_vals_arr[pos_indx]

    def get_states_number_by_indx(self, node_indx: int):
        #print(self.value_label)
        #print("Node indx", node_indx)
        return self._nodes_vals_arr[node_indx]

    def __repr__(self):
        return "Variables:\n" + str(self._nodes_labels_list) +"\nValues:\n"+ str(self._nodes_vals_arr) +\
               "\nEdges: \n" + str(self._edges_list)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Structure):
            return set(self._nodes_labels_list) == set(other._nodes_labels_list) and \
                   np.array_equal(self._nodes_vals_arr, other._nodes_vals_arr) and \
                   np.array_equal(self._nodes_indexes_arr, other._nodes_indexes_arr) and \
                   set(self._edges_list) == set(other._edges_list)

        return NotImplemented

