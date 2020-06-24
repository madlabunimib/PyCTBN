import numpy as np
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
        print(self.sample_path.structure.list_of_edges())
        self.add_edges(self.sample_path.structure.list_of_edges())

    def add_edges(self, list_of_edges):
        self.graph.add_edges_from(list_of_edges)


    



######Veloci Tests#######
os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'
s1 = sp.SamplePath(path)

g1 = NetworkGraph(s1)
g1.init_graph()
print(g1.graph)
