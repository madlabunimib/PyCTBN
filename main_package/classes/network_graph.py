
import sample_path as sp
import networkx as nx
import os


class NetworkGraph():
    """
    Rappresenta un grafo dinamico con la seguente struttura:

        :sample_path: le traiettorie/a da cui costruire il grafo
        :graph: la struttura dinamica che definisce il grafo

    """

    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.graph = nx.DiGraph()


    def init_graph(self):
        self.sample_path.build_trajectories()
        self.sample_path.build_structure()
        self.add_nodes(self.sample_path.structure.list_of_node_ids())
        self.add_edges(self.sample_path.structure.list_of_edge_ids())

    def add_nodes(self, list_of_node_ids):
        for indx, id in enumerate(list_of_node_ids):
            #print(indx, id)
            self.graph.add_node(id)
            nx.set_node_attributes(self.graph, {id:indx}, 'indx')
        #for node in list(self.graph.nodes):
            #print(node)

    def add_edges(self, list_of_edges):
        self.graph.add_edges_from(list_of_edges)

    def get_parents_by_id(self, node_id):
       return list(self.graph.predecessors(node_id))

    def get_node(self, node_id):
        to_find = nd.Node(node_id)
        for node in self.graph.nodes:
            if node == to_find:
                return node


    



######Veloci Tests#######
os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'
s1 = sp.SamplePath(path)

g1 = NetworkGraph(s1)
g1.init_graph()
print(g1.graph.number_of_nodes())
print(g1.graph.number_of_edges())

print(nx.get_node_attributes(g1.graph, 'indx')['X'])
for node in g1.get_parents_by_id('X'):
    print(g1.sample_path.structure.get_node_indx(node))
    print(node)



