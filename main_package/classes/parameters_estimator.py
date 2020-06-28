import network_graph as ng
import sample_path as sp
import os
import amalgamated_cims as acims


class ParametersEstimator:

    def __init__(self, sample_path, net_graph):
        self.sample_path = sample_path
        self.net_graph = net_graph
        self.amalgamated_cims_struct = None

    def init_amalgamated_cims_struct(self):
        self.amalgamated_cims_struct = acims.AmalgamatedCims(self.net_graph.get_states_number(),
                                                     self.net_graph.get_nodes(),
                                                             self.net_graph.get_ord_set_of_par_of_all_nodes())

# Simple Test #
os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'

s1 = sp.SamplePath(path)
s1.build_trajectories()
s1.build_structure()

g1 = ng.NetworkGraph(s1.structure)
g1.init_graph()

pe = ParametersEstimator(s1, g1)
pe.init_amalgamated_cims_struct()
print(pe.amalgamated_cims_struct.get_set_of_cims('X').get_cims_number())
print(pe.amalgamated_cims_struct.get_set_of_cims('Y').get_cims_number())
