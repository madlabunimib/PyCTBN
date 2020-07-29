
import networkx as nx
import numpy as np



class NetworkGraph():
    """
    Rappresenta il grafo che contiene i nodi e gli archi presenti nell'oggetto Structure graph_struct.
    Ogni nodo contine la label node_id, al nodo è anche associato un id numerico progressivo indx che rappresenta la posizione
    dei sui valori nella colonna indx della traj

    :graph_struct: l'oggetto Structure da cui estrarre i dati per costruire il grafo graph
    :graph: il grafo

    """

    def __init__(self, graph_struct):
        self.graph_struct = graph_struct
        self.graph = nx.DiGraph()
        self._nodes_indexes = self.graph_struct.list_of_nodes_indexes()
        self._nodes_labels = self.graph_struct.list_of_nodes_labels()
        self.aggregated_info_about_nodes_parents = None
        self._fancy_indexing = None
        self._time_scalar_indexing_structure = None
        self._transition_scalar_indexing_structure = None
        self._time_filtering = None
        self._transition_filtering = None

    def init_graph(self):
        self.add_nodes(self.graph_struct.list_of_nodes_labels())
        self.add_edges(self.graph_struct.list_of_edges())
        self.aggregated_info_about_nodes_parents = self.get_ord_set_of_par_of_all_nodes()
        self._fancy_indexing = self.build_fancy_indexing_structure(0)
        self.build_scalar_indexing_structures()
        #self.build_time_scalar_indexing_structure()
        self.build_time_columns_filtering_structure()
        #self.build_transition_scalar_indexing_structure()
        self.build_transition_columns_filtering_structure()

    def add_nodes(self, list_of_nodes):
        #self.graph.add_nodes_from(list_of_nodes)
        set_node_attr = nx.set_node_attributes
        nodes_indxs = self.graph_struct.list_of_nodes_indexes()
        nodes_vals = self.graph_struct.nodes_values()
        pos = 0
        for id, node_indx, node_val in zip(list_of_nodes, nodes_indxs, nodes_vals):
            self.graph.add_node(id, indx=node_indx, val=node_val, pos_indx=pos)
            pos += 1
            #set_node_attr(self.graph, {id:node_indx}, 'indx')

    def add_edges(self, list_of_edges):
        self.graph.add_edges_from(list_of_edges)

    def get_ordered_by_indx_set_of_parents(self, node):
        parents = self.get_parents_by_id(node)
        nodes = self.get_nodes()
        sorted_parents = [x for _, x in sorted(zip(nodes, parents))]
        #p_indxes= []
        #p_values = []
        get_node_indx = self.get_node_indx
        get_states_number_by_indx = self.get_states_number_by_indx
        p_indxes = [get_node_indx(node) for node in sorted_parents]
        p_values = [get_states_number_by_indx(indx) for indx in p_indxes]
        """for n in parents:
            #indx = self.graph_struct.get_node_indx(n)

            #print(indx)
            #ordered_set[n] = indx
            node_indx = self.get_node_indx(n)
            p_indxes.append(node_indx)
            #p_values.append(self.graph_struct.get_states_number(n))
            p_values.append(self.get_states_number_by_indx(node_indx))"""
        #ordered_set = (sorted_parents, p_indxes, p_values)
        return (sorted_parents, p_indxes, p_values)

    def get_ord_set_of_par_of_all_nodes(self):
        #result = []
        #for node in self._nodes_labels:
            #result.append(self.get_ordered_by_indx_set_of_parents(node))
        get_ordered_by_indx_set_of_parents = self.get_ordered_by_indx_set_of_parents
        result = [get_ordered_by_indx_set_of_parents(node) for node in self._nodes_labels]
        return result

    """def get_ordered_by_indx_parents_values(self, node):
        parents_values = []
        parents = self.get_ordered_by_indx_set_of_parents(node)
        for n in parents:
            parents_values.append(self.graph_struct.get_states_number(n))
        return parents_values"""

    def get_ordered_by_indx_parents_values_for_all_nodes(self):
        """result = []
        for node in self._nodes_labels:
            result.append(self.get_ordered_by_indx_parents_values(node))
        return result"""
        pars_values = [i[2] for i in self.aggregated_info_about_nodes_parents]
        return pars_values

    def get_states_number_of_all_nodes_sorted(self):
        #states_number_list = []
        #for node in self._nodes_labels:
            #states_number_list.append(self.get_states_number(node))
        get_states_number = self.get_states_number
        states_number_list = [get_states_number(node) for node in self._nodes_labels]
        return states_number_list

    def build_fancy_indexing_structure(self, start_indx):
        """list_of_parents_list = self.get_ord_set_of_par_of_all_nodes()
        #print(list_of_parents_list)
        index_structure = []
        for i, list_of_parents in enumerate(list_of_parents_list):
            indexes_for_a_node = []
            for j, node in enumerate(list_of_parents):
                indexes_for_a_node.append(self.get_node_indx(node) + start_indx)
            index_structure.append(np.array(indexes_for_a_node, dtype=np.int))
        #print(index_structure)
        return index_structure"""
        if start_indx > 0:
            pass
        else:
            fancy_indx = [i[1] for i in self.aggregated_info_about_nodes_parents]
            return fancy_indx


    def build_time_scalar_indexing_structure_for_a_node(self, node_indx, parents_indxs):
        #print(node_indx)
        #print("Parents_id", parents_indxs)
        #T_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        get_states_number_by_indx = self.graph_struct.get_states_number_by_indx
        T_vector = np.array([get_states_number_by_indx(node_indx)])
        #print(T_vector)
        #T_vector = np.append(T_vector, [get_states_number_by_indx(x) for x in parents_indxs])
        T_vector = np.append(T_vector, parents_indxs)
        #print(T_vector)
        T_vector = T_vector.cumprod().astype(np.int)
        return T_vector
        #print(T_vector)

    def build_time_scalar_indexing_structure(self):
        #parents_indexes_list = self._fancy_indexing
        """for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing):
                self._time_scalar_indexing_structure.append(
                    self.build_time_scalar_indexing_structure_for_a_node(node_indx, p_indxs))"""
        build_time_scalar_indexing_structure_for_a_node = self.build_time_scalar_indexing_structure_for_a_node
        self._time_scalar_indexing_structure = [build_time_scalar_indexing_structure_for_a_node(node_indx, p_indxs)
                                                for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(),
                                                                              self.get_ordered_by_indx_parents_values_for_all_nodes())]

    def build_transition_scalar_indexing_structure_for_a_node(self, node_indx, parents_indxs):
        #M_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1],
                             #self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        node_states_number = self.get_states_number_by_indx(node_indx)
        get_states_number_by_indx = self.graph_struct.get_states_number_by_indx
        M_vector = np.array([node_states_number,
                             node_states_number])
        #M_vector = np.append(M_vector, [get_states_number_by_indx(x) for x in parents_indxs])
        M_vector = np.append(M_vector, parents_indxs)
        M_vector = M_vector.cumprod().astype(np.int)
        return M_vector

    def build_transition_scalar_indexing_structure(self):
        #parents_indexes_list = self._fancy_indexing
        """for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing):
            self._transition_scalar_indexing_structure.append(
                self.build_transition_scalar_indexing_structure_for_a_node(node_indx, p_indxs))"""
        build_transition_scalar_indexing_structure_for_a_node = self.build_transition_scalar_indexing_structure_for_a_node
        self._transition_scalar_indexing_structure = \
            [build_transition_scalar_indexing_structure_for_a_node(node_indx, p_indxs)
                                                      for node_indx, p_indxs in
                                                      zip(self.graph_struct.list_of_nodes_indexes(),
                                                          self.get_ordered_by_indx_parents_values_for_all_nodes())]

    def build_time_columns_filtering_structure(self):
        #parents_indexes_list = self._fancy_indexing
        """for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing):
                self._time_filtering.append(np.append(np.array([node_indx], dtype=np.int), p_indxs).astype(np.int))"""
        self._time_filtering = [np.append(np.array([node_indx], dtype=np.int), p_indxs).astype(np.int)
            for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing)]

    def build_transition_columns_filtering_structure(self):
        #parents_indexes_list = self._fancy_indexing
        nodes_number = self.graph_struct.total_variables_number
        """for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing):
            self._transition_filtering.append(np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int))"""
        self._transition_filtering = [np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int)
                                      for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(),
                                                                    self._fancy_indexing)]

    def build_scalar_indexing_structures(self):
        parents_values_for_all_nodes = self.get_ordered_by_indx_parents_values_for_all_nodes()
        build_transition_scalar_indexing_structure_for_a_node = self.build_transition_scalar_indexing_structure_for_a_node
        build_time_scalar_indexing_structure_for_a_node = self.build_time_scalar_indexing_structure_for_a_node
        aggr = [(build_transition_scalar_indexing_structure_for_a_node(node_indx, p_indxs),
                 build_time_scalar_indexing_structure_for_a_node(node_indx, p_indxs))
                                                      for node_indx, p_indxs in
                                                      zip(self.graph_struct.list_of_nodes_indexes(),
                                                          parents_values_for_all_nodes)]
        self._transition_scalar_indexing_structure = [i[0] for i in aggr]
        self._time_scalar_indexing_structure = [i[1] for i in aggr]

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_edges(self):
        return list(self.graph.edges)

    def get_nodes_sorted_by_indx(self):
        return self.graph_struct.list_of_nodes_labels()

    def get_parents_by_id(self, node_id):
        return list(self.graph.predecessors(node_id))

    def get_states_number(self, node_id):
        #return self.graph_struct.get_states_number(node_id)
        return self.graph.nodes[node_id]['val']

    def get_states_number_by_indx(self, node_indx):
        return self.graph_struct.get_states_number_by_indx(node_indx)

    def get_node_by_index(self, node_indx):
        return self.graph_struct.get_node_id(node_indx)

    def get_node_indx(self, node_id):
        return nx.get_node_attributes(self.graph, 'indx')[node_id]
        #return self.graph_struct.get_node_indx(node_id)

    def get_positional_node_indx(self, node_id):
        return self.graph.nodes[node_id]['pos_indx']

    @property
    def time_scalar_indexing_strucure(self):
        return self._time_scalar_indexing_structure

    @property
    def time_filtering(self):
        return self._time_filtering

    @property
    def transition_scalar_indexing_structure(self):
        return self._transition_scalar_indexing_structure

    @property
    def transition_filtering(self):
        return self._transition_filtering

    """def remove_node(self, node_id):
        node_indx = self.get_node_indx(node_id)
        self.graph_struct.remove_node(node_id)
        self.graph.remove_node(node_id)
        del self._fancy_indexing[node_indx]
        del self._time_filtering[node_indx]
        del self._nodes_labels[node_indx]
        del self._transition_scalar_indexing_structure[node_indx]
        del self._transition_filtering[node_indx]
        del self._time_scalar_indexing_structure[node_indx]
        del self.aggregated_info_about_nodes_parents[node_indx]
        del self._nodes_indexes[node_indx]"""

