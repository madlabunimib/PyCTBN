import os
import time as tm
from line_profiler import LineProfiler
import numpy as np

import network_graph as ng
import sample_path as sp
import amalgamated_cims as acims


class ParametersEstimator:

    def __init__(self, sample_path, net_graph):
        self.sample_path = sample_path
        self.net_graph = net_graph
        self.fancy_indexing_structure = self.net_graph.build_fancy_indexing_structure(1)
        self.amalgamated_cims_struct = None

    def init_amalgamated_cims_struct(self):
        self.amalgamated_cims_struct = acims.AmalgamatedCims(self.net_graph.get_states_number_of_all_nodes_sorted(),
                                                     self.net_graph.get_nodes(),
                                                    self.net_graph.get_ordered_by_indx_parents_values_for_all_nodes())

    def parameters_estimation(self):
        #print("Starting computing")
        #t0 = tm.time()
        for indx, trajectory in enumerate(self.sample_path.trajectories):
            self.parameters_estimation_single_trajectory(trajectory.get_trajectory())
            #print("Finished Trajectory number", indx)
        #t1 = tm.time() - t0
        #print("Elapsed Time ", t1)

    def parameters_estimation_single_trajectory(self, trajectory):
        tr_len = trajectory.shape[0]
        row_length = trajectory.shape[1]
        print(tr_len)
        print(row_length)
        t0 = tm.time()
        for indx, row in enumerate(trajectory):
           """ #if int(trajectory[indx][1]) == -1:
                #break
            if indx == tr_len - 2:
                break
            if trajectory[indx + 1][1] != -1:
                transition = self.find_transition(trajectory[indx], trajectory[indx + 1], row_length)
                which_node = transition[0]
            # print(which_node)
                which_matrix = self.which_matrix_to_update(row, transition[0])
                which_element = transition[1]
                self.amalgamated_cims_struct.update_state_transition_for_matrix(which_node, which_matrix, which_element)

            #changed_node = which_node
            if int(trajectory[indx][0]) == 0:
                time = trajectory[indx + 1][0]
            #time = self.compute_time_delta(trajectory[indx], trajectory[indx + 1])
            which_element = transition[1][0]
            self.amalgamated_cims_struct.update_state_residence_time_for_matrix(which_node, which_matrix, which_element,
                                                                                time)

            for node_indx in range(0, 3):
                if node_indx != transition[0]:
                    # print(node)
                    which_node = node_indx
                    which_matrix = self.which_matrix_to_update(row, node_indx)
                    which_element = int(row[node_indx + 1])
                    # print("State res time element " + str(which_element) + node)
                    # print("State res time matrix indx" + str(which_matrix))
                    self.amalgamated_cims_struct.update_state_residence_time_for_matrix(which_node, which_matrix,
                                                                                        which_element, time)
        t1 = tm.time() - t0
        print("Elapsed Time ", t1)"""

    def find_transition(self, current_row, next_row, row_length):
        for indx in range(1, row_length):
            if current_row[indx] != next_row[indx]:
                return [indx - 1, (current_row[indx], next_row[indx])]

    def compute_time_delta(self, current_row, next_row):
        return next_row[0] - current_row[0]

    def which_matrix_to_update(self, current_row, node_indx): # produce strutture {'X':1, 'Y':2} dove X e Y sono i parent di node_id
       return current_row[self.fancy_indexing_structure[node_indx]]





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
print(pe.amalgamated_cims_struct.get_set_of_cims(0).get_cims_number())
print(pe.amalgamated_cims_struct.get_set_of_cims(1).get_cims_number())
print(pe.amalgamated_cims_struct.get_set_of_cims(2).get_cims_number())
#pe.parameters_estimation_single_trajectory(pe.sample_path.trajectories[0].get_trajectory())
lp = LineProfiler()
lp_wrapper = lp(pe.parameters_estimation_single_trajectory)
lp_wrapper(pe.sample_path.trajectories.get_trajectory())
lp.print_stats()
#pe.parameters_estimation()
"""for matrix in pe.amalgamated_cims_struct.get_set_of_cims(1).actual_cims:
    print(matrix.state_residence_times)
    print(matrix.state_transition_matrix)
    matrix.compute_cim_coefficients()
    print(matrix.cim)"""

