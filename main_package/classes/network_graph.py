import os
import sample_path as sp
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
        self._fancy_indexing = None
        self._time_scalar_indexing_structure = []
        self._transition_scalar_indexing_structure = []
        self._time_filtering = []
        self._transition_filtering = []

    def init_graph(self):
        self.add_nodes(self.graph_struct.list_of_nodes_labels())
        self.add_edges(self.graph_struct.list_of_edges())
        self._fancy_indexing = self.build_fancy_indexing_structure(0)
        self.build_time_scalar_indexing_structure()
        self.build_time_columns_filtering_structure()
        self.build_transition_scalar_indexing_structure()
        self.build_transition_columns_filtering_structure()

    def add_nodes(self, list_of_nodes):
        for indx, id in enumerate(list_of_nodes):
            self.graph.add_node(id)
            nx.set_node_attributes(self.graph, {id:indx}, 'indx')

    def add_edges(self, list_of_edges):
        self.graph.add_edges_from(list_of_edges)

    def get_ordered_by_indx_set_of_parents(self, node):
        ordered_set = {}
        parents = self.get_parents_by_id(node)
        for n in parents:
            indx = self._nodes_labels.index(n)
            ordered_set[n] = indx
        ordered_set = {k: v for k, v in sorted(ordered_set.items(), key=lambda item: item[1])}
        return list(ordered_set.keys())

    def get_ord_set_of_par_of_all_nodes(self):
        result = []
        for node in self._nodes_labels:
            result.append(self.get_ordered_by_indx_set_of_parents(node))
        return result

    def get_ordered_by_indx_parents_values(self, node):
        parents_values = []
        parents = self.get_ordered_by_indx_set_of_parents(node)
        for n in parents:
            parents_values.append(self.graph_struct.get_states_number(n))
        return parents_values

    def get_ordered_by_indx_parents_values_for_all_nodes(self):
        result = []
        for node in self._nodes_labels:
            result.append(self.get_ordered_by_indx_parents_values(node))
        return result

    def get_states_number_of_all_nodes_sorted(self):
        states_number_list = []
        for node in self._nodes_labels:
            states_number_list.append(self.get_states_number(node))
        return states_number_list

    def build_fancy_indexing_structure(self, start_indx):
        list_of_parents_list = self.get_ord_set_of_par_of_all_nodes()
        index_structure = []
        for i, list_of_parents in enumerate(list_of_parents_list):
            indexes_for_a_node = []
            for j, node in enumerate(list_of_parents):
                indexes_for_a_node.append(self.get_node_indx(node) + start_indx)
            index_structure.append(np.array(indexes_for_a_node, dtype=np.int))
        return index_structure

    def build_time_scalar_indexing_structure_for_a_node(self, node_id, parents_id):
        #print(parents_id)
        T_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        #print(T_vector)
        T_vector = np.append(T_vector, [self.graph_struct.variables_frame.iloc[x, 1] for x in parents_id])
        #print(T_vector)
        T_vector = T_vector.cumprod().astype(np.int)
        return T_vector
        #print(T_vector)

    def build_time_scalar_indexing_structure(self):
        parents_indexes_list = self._fancy_indexing
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            if p_indxs.size == 0:
                self._time_scalar_indexing_structure.append(np.array([self.get_states_number_by_indx(node_indx)], dtype=np.int))
            else:
                self._time_scalar_indexing_structure.append(
                    self.build_time_scalar_indexing_structure_for_a_node(node_indx, p_indxs))

    def build_transition_scalar_indexing_structure_for_a_node(self, node_id, parents_id):
        M_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1],
                             self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        M_vector = np.append(M_vector, [self.graph_struct.variables_frame.iloc[x, 1] for x in parents_id])
        M_vector = M_vector.cumprod().astype(np.int)
        return M_vector

    def build_transition_scalar_indexing_structure(self):
        parents_indexes_list = self._fancy_indexing
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            self._transition_scalar_indexing_structure.append(
                self.build_transition_scalar_indexing_structure_for_a_node(node_indx, p_indxs))

    def build_time_columns_filtering_structure(self):
        parents_indexes_list = self._fancy_indexing
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            if p_indxs.size == 0:
                self._time_filtering.append(np.append(p_indxs, np.array([node_indx], dtype=np.int)))
            else:
                self._time_filtering.append(np.append(np.array([node_indx], dtype=np.int), p_indxs))

    def build_transition_columns_filtering_structure(self):
        parents_indexes_list = self._fancy_indexing
        nodes_number = len(parents_indexes_list)
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            self._transition_filtering.append(np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int))

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_nodes_sorted_by_indx(self):
        return self.graph_struct.list_of_nodes

    def get_parents_by_id(self, node_id):
       return list(self.graph.predecessors(node_id))

    def get_states_number(self, node_id):
        return self.graph_struct.get_states_number(node_id)

    def get_states_number_by_indx(self, node_indx):
        return self.graph_struct.get_states_number_by_indx(node_indx)

    def get_node_by_index(self, node_indx):
        return self.graph_struct.get_node_id(node_indx)

    def get_node_indx(self, node_id):
        return nx.get_node_attributes(self.graph, 'indx')[node_id]

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

    



######Veloci Tests#######
"""os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'
s1 = sp.SamplePath(path)
s1.build_trajectories()
s1.build_structure()

g1 = NetworkGraph(s1.structure)
g1.init_graph()
print(g1.transition_scalar_indexing_structure)
print(g1.transition_filtering)
print(g1.time_scalar_indexing_strucure)
print(g1.time_filering)


#print(g1.build_fancy_indexing_structure(0))
#print(g1.get_states_number_of_all_nodes_sorted())
g1.build_scalar_indexing_structure()
print(g1.scalar_indexing_structure)
print(g1.build_columns_filtering_structure())
g1.build_transition_scalar_indexing_structure()
print(g1.transition_scalar_indexing_structure)
g1.build_transition_columns_filtering_structure()
print(g1.transition_filtering)

[array([3, 9]), array([ 3,  9, 27]), array([ 3,  9, 27, 81])]
[array([3, 0]), array([4, 1, 2]), array([5, 2, 0, 1])]"""

