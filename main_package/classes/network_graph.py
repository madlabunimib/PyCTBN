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

    def init_graph(self):
        self.add_nodes(self.graph_struct.list_of_nodes())
        self.add_edges(self.graph_struct.list_of_edges())

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
            index_structure.append(indexes_for_a_node)
        return index_structure

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_nodes_sorted_by_indx(self):
        return self.graph_struct.list_of_nodes

    def get_parents_by_id(self, node_id):
       return list(self.graph.predecessors(node_id))

    def get_states_number(self, node_id):
        return self.graph_struct.get_states_number(node_id)

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
print(g1.build_fancy_indexing_structure())
print(g1.get_states_number_of_all_nodes_sorted())"""



