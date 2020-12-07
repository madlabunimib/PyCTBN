
import typing
import networkx as nx
import numpy as np

from .structure import Structure


class NetworkGraph:
    """Abstracts the infos contained in the Structure class in the form of a directed graph.
    Has the task of creating all the necessary filtering and indexing structures for parameters estimation

    :param graph_struct: the ``Structure`` object from which infos about the net will be extracted
    :type graph_struct: Structure
    :_graph: directed graph
    :_aggregated_info_about_nodes_parents: a structure that contains all the necessary infos
        about every parents of the node of which all the indexing and filtering structures will be constructed.
    :_time_scalar_indexing_structure: the indexing structure for state res time estimation
    :_transition_scalar_indexing_structure: the indexing structure for transition computation
    :_time_filtering: the columns filtering structure used in the computation of the state res times
    :_transition_filtering: the columns filtering structure used in the computation of the transition
        from one state to another
    :_p_combs_structure: all the possible parents states combination for the node of interest
    """

    def __init__(self, graph_struct: Structure):
        """Constructor Method
        """
        self._graph_struct = graph_struct
        self._graph = nx.DiGraph()
        self._aggregated_info_about_nodes_parents = None
        self._time_scalar_indexing_structure = None
        self._transition_scalar_indexing_structure = None
        self._time_filtering = None
        self._transition_filtering = None
        self._p_combs_structure = None

    def fast_init(self, node_id: str) -> None:
        """Initializes all the necessary structures for parameters estimation of the node identified by the label
        node_id

        :param node_id: the label of the node
        :type node_id: string
        """
        self.add_nodes(self._graph_struct.nodes_labels)
        self.add_edges(self._graph_struct.edges)
        self._aggregated_info_about_nodes_parents = self.get_ordered_by_indx_set_of_parents(node_id)
        p_indxs = self._aggregated_info_about_nodes_parents[1]
        p_vals = self._aggregated_info_about_nodes_parents[2]
        self._time_scalar_indexing_structure = self.build_time_scalar_indexing_structure_for_a_node(node_id,
                                                                                                    p_vals)
        self._transition_scalar_indexing_structure = self.build_transition_scalar_indexing_structure_for_a_node(node_id,
                                                                                                                p_vals)
        node_indx = self.get_node_indx(node_id)
        self._time_filtering = self.build_time_columns_filtering_for_a_node(node_indx, p_indxs)
        self._transition_filtering = self.build_transition_filtering_for_a_node(node_indx, p_indxs)
        self._p_combs_structure = self.build_p_comb_structure_for_a_node(p_vals)

    def add_nodes(self, list_of_nodes: typing.List) -> None:
        """Adds the nodes to the ``_graph`` contained in the list of nodes ``list_of_nodes``.
        Sets all the properties that identify a nodes (index, positional index, cardinality)

        :param list_of_nodes: the nodes to add to ``_graph``
        :type list_of_nodes: List
        """
        nodes_indxs = self._graph_struct.nodes_indexes
        nodes_vals = self._graph_struct.nodes_values
        pos = 0
        for id, node_indx, node_val in zip(list_of_nodes, nodes_indxs, nodes_vals):
            self._graph.add_node(id, indx=node_indx, val=node_val, pos_indx=pos)
            pos += 1

    def add_edges(self, list_of_edges: typing.List) -> None:
        """Add the edges to the ``_graph`` contained in the list ``list_of_edges``.

        :param list_of_edges: the list containing of tuples containing the edges
        :type list_of_edges: List
        """
        self._graph.add_edges_from(list_of_edges)

    def get_ordered_by_indx_set_of_parents(self, node: str) -> typing.Tuple:
        """Builds the aggregated structure that holds all the infos relative to the parent set of the node, namely
        (parents_labels, parents_indexes, parents_cardinalities).

        :param node: the label of the node
        :type node: string
        :return: a tuple containing all the parent set infos
        :rtype: Tuple
        """
        parents = self.get_parents_by_id(node)
        nodes = self._graph_struct.nodes_labels
        d = {v: i for i, v in enumerate(nodes)}
        sorted_parents = sorted(parents, key=lambda v: d[v])
        get_node_indx = self.get_node_indx
        p_indxes = [get_node_indx(node) for node in sorted_parents]
        p_values = [self.get_states_number(node) for node in sorted_parents]
        return (sorted_parents, p_indxes, p_values)

    def build_time_scalar_indexing_structure_for_a_node(self, node_id: str, parents_vals: typing.List) -> np.ndarray:
        """Builds an indexing structure for the computation of state residence times values.

        :param node_id: the node label
        :type node_id: string
        :param parents_vals: the caridinalites of the node's parents
        :type parents_vals: List
        :return: The time indexing structure
        :rtype: numpy.ndArray
        """
        T_vector = np.array([self.get_states_number(node_id)])
        T_vector = np.append(T_vector, parents_vals)
        T_vector = T_vector.cumprod().astype(np.int)
        return T_vector

    def build_transition_scalar_indexing_structure_for_a_node(self, node_id: str, parents_vals: typing.List) \
            -> np.ndarray:
        """Builds an indexing structure for the computation of state transitions values.

        :param node_id: the node label
        :type node_id: string
        :param parents_vals: the caridinalites of the node's parents
        :type parents_vals: List
        :return: The transition indexing structure
        :rtype: numpy.ndArray
        """
        node_states_number = self.get_states_number(node_id)
        M_vector = np.array([node_states_number,
                             node_states_number])
        M_vector = np.append(M_vector, parents_vals)
        M_vector = M_vector.cumprod().astype(np.int)
        return M_vector

    def build_time_columns_filtering_for_a_node(self, node_indx: int, p_indxs: typing.List) -> np.ndarray:
        """
        Builds the necessary structure to filter the desired columns indicated by ``node_indx`` and ``p_indxs``
        in the dataset.
        This structute will be used in the computation of the state res times.
        :param node_indx: the index of the node
        :type node_indx: int
        :param p_indxs: the indexes of the node's parents
        :type p_indxs: List
        :return: The filtering structure for times estimation
        :rtype: numpy.ndArray
        """
        return np.append(np.array([node_indx], dtype=np.int), p_indxs).astype(np.int)

    def build_transition_filtering_for_a_node(self, node_indx: int, p_indxs: typing.List) -> np.ndarray:
        """Builds the necessary structure to filter the desired columns indicated by ``node_indx`` and ``p_indxs``
        in the dataset.
        This structure will be used in the computation of the state transitions values.
        :param node_indx: the index of the node
        :type node_indx: int
        :param p_indxs: the indexes of the node's parents
        :type p_indxs: List
        :return: The filtering structure for transitions estimation
        :rtype: numpy.ndArray
        """
        nodes_number = self._graph_struct.total_variables_number
        return np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int)

    def build_p_comb_structure_for_a_node(self, parents_values: typing.List) -> np.ndarray:
        """
        Builds the combinatorial structure that contains the combinations of all the values contained in
        ``parents_values``.

        :param parents_values: the cardinalities of the nodes
        :type parents_values: List
        :return: A numpy matrix containing a grid of the combinations
        :rtype: numpy.ndArray
        """
        tmp = []
        for val in parents_values:
            tmp.append([x for x in range(val)])
        if len(parents_values) > 0:
            parents_comb = np.array(np.meshgrid(*tmp)).T.reshape(-1, len(parents_values))
            if len(parents_values) > 1:
                tmp_comb = parents_comb[:, 1].copy()
                parents_comb[:, 1] = parents_comb[:, 0].copy()
                parents_comb[:, 0] = tmp_comb
        else:
            parents_comb = np.array([[]], dtype=np.int)
        return parents_comb

    def get_parents_by_id(self, node_id) -> typing.List:
        """Returns a list of labels of the parents of the node ``node_id``

        :param node_id: the node label
        :type node_id: string
        :return: a List of labels of the parents
        :rtype: List
        """
        return list(self._graph.predecessors(node_id))

    def get_states_number(self, node_id) -> int:
        return self._graph.nodes[node_id]['val']

    def get_node_indx(self, node_id) -> int:
        return nx.get_node_attributes(self._graph, 'indx')[node_id]

    def get_positional_node_indx(self, node_id) -> int:
        return self._graph.nodes[node_id]['pos_indx']

    @property
    def nodes(self) -> typing.List:
        return self._graph_struct.nodes_labels

    @property
    def edges(self) -> typing.List:
        return list(self._graph.edges)

    @property
    def nodes_indexes(self) -> np.ndarray:
        return self._graph_struct.nodes_indexes

    @property
    def nodes_values(self) -> np.ndarray:
        return self._graph_struct.nodes_values

    @property
    def time_scalar_indexing_strucure(self) -> np.ndarray:
        return self._time_scalar_indexing_structure

    @property
    def time_filtering(self) -> np.ndarray:
        return self._time_filtering

    @property
    def transition_scalar_indexing_structure(self) -> np.ndarray:
        return self._transition_scalar_indexing_structure

    @property
    def transition_filtering(self) -> np.ndarray:
        return self._transition_filtering

    @property
    def p_combs(self) -> np.ndarray:
        return self._p_combs_structure

    """
    ##############These Methods are actually unused but could become useful in the near future################

    def init_graph(self):
        self.add_nodes(self._nodes_labels)
        self.add_edges(self._graph_struct.edges)
        self._aggregated_info_about_nodes_parents = self.get_ord_set_of_par_of_all_nodes()
        self._fancy_indexing = self.build_fancy_indexing_structure(0)
        self.build_scalar_indexing_structures()
        self.build_time_columns_filtering_structure()
        self.build_transition_columns_filtering_structure()
        self._p_combs_structure = self.build_p_combs_structure()

    def build_time_columns_filtering_structure(self):
        nodes_indxs = self._nodes_indexes
        self._time_filtering = [np.append(np.array([node_indx], dtype=np.int), p_indxs).astype(np.int)
            for node_indx, p_indxs in zip(nodes_indxs, self._fancy_indexing)]

    def build_transition_columns_filtering_structure(self):
        nodes_number = self._graph_struct.total_variables_number
        nodes_indxs = self._nodes_indexes
        self._transition_filtering = [np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int)
                                      for node_indx, p_indxs in zip(nodes_indxs,
                                                                    self._fancy_indexing)]

    def build_scalar_indexing_structures(self):
        parents_values_for_all_nodes = self.get_ordered_by_indx_parents_values_for_all_nodes()
        build_transition_scalar_indexing_structure_for_a_node = \
            self.build_transition_scalar_indexing_structure_for_a_node
        build_time_scalar_indexing_structure_for_a_node = self.build_time_scalar_indexing_structure_for_a_node
        aggr = [(build_transition_scalar_indexing_structure_for_a_node(node_id, p_vals),
                 build_time_scalar_indexing_structure_for_a_node(node_id, p_vals))
                                                      for node_id, p_vals in
                                                      zip(self._nodes_labels,
                                                          parents_values_for_all_nodes)]
        self._transition_scalar_indexing_structure = [i[0] for i in aggr]
        self._time_scalar_indexing_structure = [i[1] for i in aggr]

    def build_p_combs_structure(self):
        parents_values_for_all_nodes = self.get_ordered_by_indx_parents_values_for_all_nodes()
        p_combs_struct = [self.build_p_comb_structure_for_a_node(p_vals) for p_vals in parents_values_for_all_nodes]
        return p_combs_struct

    def get_ord_set_of_par_of_all_nodes(self):
        get_ordered_by_indx_set_of_parents = self.get_ordered_by_indx_set_of_parents
        result = [get_ordered_by_indx_set_of_parents(node) for node in self._nodes_labels]
        return result

    def get_ordered_by_indx_parents_values_for_all_nodes(self):
        pars_values = [i[2] for i in self._aggregated_info_about_nodes_parents]
        return pars_values

    def build_fancy_indexing_structure(self, start_indx):
        if start_indx > 0:
            pass
        else:
            fancy_indx = [i[1] for i in self._aggregated_info_about_nodes_parents]
            return fancy_indx
    """