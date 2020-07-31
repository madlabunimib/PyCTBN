
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
        self._nodes_values = self.graph_struct.nodes_values()
        self.aggregated_info_about_nodes_parents = None
        self._fancy_indexing = None
        self._time_scalar_indexing_structure = None
        self._transition_scalar_indexing_structure = None
        self._time_filtering = None
        self._transition_filtering = None
        self._p_combs_structure = None

    def init_graph(self):
        self.add_nodes(self._nodes_labels)
        self.add_edges(self.graph_struct.list_of_edges())
        self.aggregated_info_about_nodes_parents = self.get_ord_set_of_par_of_all_nodes()
        self._fancy_indexing = self.build_fancy_indexing_structure(0)
        self.build_scalar_indexing_structures()
        self.build_time_columns_filtering_structure()
        self.build_transition_columns_filtering_structure()
        self._p_combs_structure = self.build_p_combs_structure()
    #ATTENZIONE LIST_OF_NODES DEVE ESSERE COERENTE CON L?ORDINAMENTO DEL DS
    def add_nodes(self, list_of_nodes):
        #self.graph.add_nodes_from(list_of_nodes)
        nodes_indxs = self._nodes_indexes
        nodes_vals = self.graph_struct.nodes_values()
        pos = 0
        #print("LIST OF NODES", list_of_nodes)
        for id, node_indx, node_val in zip(list_of_nodes, nodes_indxs, nodes_vals):
            self.graph.add_node(id, indx=node_indx, val=node_val, pos_indx=pos)
            pos += 1
            #set_node_attr(self.graph, {id:node_indx}, 'indx')

    def add_edges(self, list_of_edges):
        self.graph.add_edges_from(list_of_edges)

    def get_ordered_by_indx_set_of_parents(self, node):
        parents = self.get_parents_by_id(node)
        #print("PARENTS", parents)
        nodes = self.get_nodes()
        #print("NODES", nodes)
        d = {v: i for i, v in enumerate(nodes)}
        sorted_parents = sorted(parents, key=lambda v: d[v])
        #sorted_parents = [x for _, x in sorted(zip(nodes, parents))]
        #print("SORTED PARENTS IN GRAPH",sorted_parents)
        #p_indxes= []
        #p_values = []
        get_node_indx = self.get_node_indx
        p_indxes = [get_node_indx(node) for node in sorted_parents]
        #p_indxes.sort()
        p_values = [self.get_states_number(node) for node in sorted_parents]
        #print("P INDXS", p_indxes)
        #print("P VALS", p_values)
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
        #get_states_number = self.get_states_number
        #states_number_list = [get_states_number(node) for node in self._nodes_labels]
        return self._nodes_values

    def build_fancy_indexing_structure(self, start_indx):
        if start_indx > 0:
            pass
        else:
            fancy_indx = [i[1] for i in self.aggregated_info_about_nodes_parents]
            return fancy_indx


    def build_time_scalar_indexing_structure_for_a_node(self, node_id, parents_indxs):
        T_vector = np.array([self.get_states_number(node_id)])
        T_vector = np.append(T_vector, parents_indxs)
        T_vector = T_vector.cumprod().astype(np.int)
        # print(T_vector)
        return T_vector


    def build_transition_scalar_indexing_structure_for_a_node(self, node_id, parents_indxs):
        #M_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1],
                             #self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        node_states_number = self.get_states_number(node_id)
        #get_states_number_by_indx = self.graph_struct.get_states_number_by_indx
        M_vector = np.array([node_states_number,
                             node_states_number])
        #M_vector = np.append(M_vector, [get_states_number_by_indx(x) for x in parents_indxs])
        M_vector = np.append(M_vector, parents_indxs)
        M_vector = M_vector.cumprod().astype(np.int)
        return M_vector

    def build_time_columns_filtering_structure(self):
        #parents_indexes_list = self._fancy_indexing
        """for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing):
                self._time_filtering.append(np.append(np.array([node_indx], dtype=np.int), p_indxs).astype(np.int))"""
        nodes_indxs = self.graph_struct.list_of_nodes_indexes()
        #print("FINDXING", self._fancy_indexing)
        #print("Nodes Indxs", nodes_indxs)
        self._time_filtering = [np.append(np.array([node_indx], dtype=np.int), p_indxs).astype(np.int)
            for node_indx, p_indxs in zip(nodes_indxs, self._fancy_indexing)]

    def build_transition_columns_filtering_structure(self):
        #parents_indexes_list = self._fancy_indexing
        nodes_number = self.graph_struct.total_variables_number
        """for node_indx, p_indxs in zip(self.graph_struct.list_of_nodes_indexes(), self._fancy_indexing):
            self._transition_filtering.append(np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int))"""
        nodes_indxs = self.graph_struct.list_of_nodes_indexes()
        self._transition_filtering = [np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int)
                                      for node_indx, p_indxs in zip(nodes_indxs,
                                                                    self._fancy_indexing)]

    def build_scalar_indexing_structures(self):
        parents_values_for_all_nodes = self.get_ordered_by_indx_parents_values_for_all_nodes()
        build_transition_scalar_indexing_structure_for_a_node = self.build_transition_scalar_indexing_structure_for_a_node
        build_time_scalar_indexing_structure_for_a_node = self.build_time_scalar_indexing_structure_for_a_node
        aggr = [(build_transition_scalar_indexing_structure_for_a_node(node_indx, p_indxs),
                 build_time_scalar_indexing_structure_for_a_node(node_indx, p_indxs))
                                                      for node_indx, p_indxs in
                                                      zip(self._nodes_labels,
                                                          parents_values_for_all_nodes)]
        self._transition_scalar_indexing_structure = [i[0] for i in aggr]
        self._time_scalar_indexing_structure = [i[1] for i in aggr]

    def build_p_combs_structure(self):
        parents_values_for_all_nodes = self.get_ordered_by_indx_parents_values_for_all_nodes()
        p_combs_struct = [self.build_p_comb_structure_for_a_node(p_vals) for p_vals in parents_values_for_all_nodes]
        return p_combs_struct

    def build_p_comb_structure_for_a_node(self, parents_values):
        tmp = []
        for val in parents_values:
            tmp.append([x for x in range(val)])
        #print("TIMP", tmp)
        if len(parents_values) > 0:
            parents_comb = np.array(np.meshgrid(*tmp)).T.reshape(-1, len(parents_values))
            #print("PArents COmb", parents_comb)
            if len(parents_values) > 1:
                tmp_comb = parents_comb[:, 1].copy()
                #print(tmp_comb)
                parents_comb[:, 1] = parents_comb[:, 0].copy()
                parents_comb[:, 0] = tmp_comb
        else:
            parents_comb = np.array([[]], dtype=np.int)
        return parents_comb

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

    @property
    def p_combs(self):
        return self._p_combs_structure


