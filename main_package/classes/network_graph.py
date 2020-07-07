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
        self.scalar_indexing_structure = []
        self.transition_scalar_indexing_structure = []
        self.filtering_structure = []
        self.transition_filtering = []

    def init_graph(self):
        self.add_nodes(self.graph_struct.list_of_nodes())
        self.add_edges(self.graph_struct.list_of_edges())
        self.build_scalar_indexing_structure()
        self.build_columns_filtering_structure()
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
            indx = self.graph_struct.get_node_indx(n)
            ordered_set[n] = indx
        {k: v for k, v in sorted(ordered_set.items(), key=lambda item: item[1])}
        return list(ordered_set.keys())

    def get_ord_set_of_par_of_all_nodes(self):
        result = []
        for node in self.get_nodes():
            result.append(self.get_ordered_by_indx_set_of_parents(node))
        return result

    def get_ordered_by_indx_parents_values(self, node):
        parents_values = []
        parents = self.get_parents_by_id(node)
        parents.sort() #Assumo che la structure rifletta l'ordine delle colonne del dataset
        for n in parents:
            parents_values.append(self.graph_struct.get_states_number(n))
        return parents_values

    def get_ordered_by_indx_parents_values_for_all_nodes(self):
        result = []
        for node in self.get_nodes(): #TODO bisogna essere sicuri che l'ordine sia coerente con quello del dataset serve un metodo get_nodes_sort_by_indx
            result.append(self.get_ordered_by_indx_parents_values(node))
        return result

    def get_states_number_of_all_nodes_sorted(self):
        states_number_list = []
        for node in self.get_nodes(): #TODO SERVE UN get_nodes_ordered!!!!!!
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

    def build_scalar_indexing_structure_for_a_node(self, node_id, parents_id):
        print(parents_id)
        T_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        print(T_vector)
        T_vector = np.append(T_vector, [self.graph_struct.variables_frame.iloc[x, 1] for x in parents_id])
        print(T_vector)
        T_vector = T_vector.cumprod().astype(np.int)
        return T_vector
        print(T_vector)

    def build_scalar_indexing_structure(self):
        parents_indexes_list = self.build_fancy_indexing_structure(0)
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            if p_indxs.size == 0:
                self.scalar_indexing_structure.append(np.array([self.get_states_number_by_indx(node_indx)], dtype=np.int))
            else:
                self.scalar_indexing_structure.append(
                    self.build_scalar_indexing_structure_for_a_node(node_indx, p_indxs))

    def build_transition_scalar_indexing_structure_for_a_node(self, node_id, parents_id):
        M_vector = np.array([self.graph_struct.variables_frame.iloc[node_id, 1],
                             self.graph_struct.variables_frame.iloc[node_id, 1].astype(np.int)])
        M_vector = np.append(M_vector, [self.graph_struct.variables_frame.iloc[x, 1] for x in parents_id])
        M_vector = M_vector.cumprod().astype(np.int)
        return M_vector

    def build_transition_scalar_indexing_structure(self):
        parents_indexes_list = self.build_fancy_indexing_structure(0)
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            """if p_indxs.size == 0:
                self.scalar_indexing_structure.append(
                    np.array([self.get_states_number_by_indx(node_indx)], dtype=np.int))
            else:"""
            self.transition_scalar_indexing_structure.append(
                self.build_transition_scalar_indexing_structure_for_a_node(node_indx, p_indxs))

    def build_columns_filtering_structure(self):
        parents_indexes_list = self.build_fancy_indexing_structure(0)
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            if p_indxs.size == 0:
                self.filtering_structure.append(np.append(p_indxs, np.array([node_indx], dtype=np.int)))
            else:
                self.filtering_structure.append(np.append(np.array([node_indx], dtype=np.int), p_indxs))

    def build_transition_columns_filtering_structure(self):
        parents_indexes_list = self.build_fancy_indexing_structure(0)
        nodes_number = len(parents_indexes_list)
        for node_indx, p_indxs in enumerate(parents_indexes_list):
            #if p_indxs.size == 0:
                #self.filtering_structure.append(np.append(p_indxs, np.array([node_indx], dtype=np.int)))
            #else:
                self.transition_filtering.append(np.array([node_indx + nodes_number, node_indx, *p_indxs], dtype=np.int))


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


    



######Veloci Tests#######
"""os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'
s1 = sp.SamplePath(path)
s1.build_trajectories()
s1.build_structure()

g1 = NetworkGraph(s1.structure)
g1.init_graph()
print(g1.graph.number_of_nodes())
print(g1.graph.number_of_edges())

print(nx.get_node_attributes(g1.graph, 'indx')['X'])
for node in g1.get_parents_by_id('Z'):
  #  print(g1.get_node_by_index(node))
    print(node)
print(g1.get_ordered_by_indx_parents_values_for_all_nodes())
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

