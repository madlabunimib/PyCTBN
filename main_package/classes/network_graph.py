
import networkx as nx



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

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_parents_by_id(self, node_id):
       return list(self.graph.predecessors(node_id))

    def get_states_number(self):
        return self.graph_struct.get_states_number()

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
    print(g1.get_node_by_index(node))
    print(node)
print(g1.get_ordered_by_indx_set_of_parents('Z'))
print(g1.get_ord_set_of_par_of_all_nodes())"""



