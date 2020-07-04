import os
import time as tm
from line_profiler import LineProfiler


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
        print("Starting computing")
        t0 = tm.time()
        for trajectory in self.sample_path.trajectories:
            #tr_length = trajectory.size()
            self.parameters_estimation_single_trajectory(trajectory.get_trajectory())
            #print("Finished Trajectory number", indx)
        t1 = tm.time() - t0
        print("Elapsed Time ", t1)

    def parameters_estimation_single_trajectory(self, trajectory):

        row_length = trajectory.shape[1]
        for indx, row in enumerate(trajectory[:-1]):
            self.compute_sufficient_statistics_for_row(trajectory[indx], trajectory[indx + 1], row_length)

    def compute_sufficient_statistics_for_row(self, current_row, next_row, row_length):
        #time = self.compute_time_delta(current_row, next_row)
        time = current_row[0]
        for indx in range(1, row_length):
            if current_row[indx] != next_row[indx] and next_row[indx] != -1:
                transition = [indx - 1, (current_row[indx], next_row[indx])]
                which_node = transition[0]
                which_matrix = self.which_matrix_to_update(current_row, transition[0])
                which_element = transition[1]
                self.amalgamated_cims_struct.update_state_transition_for_matrix(which_node, which_matrix, which_element)
                which_element = transition[1][0]
                self.amalgamated_cims_struct.update_state_residence_time_for_matrix(which_node, which_matrix,
                                                                                    which_element,
                                                                                    time)
            else:
                which_node = indx - 1
                which_matrix = self.which_matrix_to_update(current_row, which_node)
                which_element = current_row[indx]
                self.amalgamated_cims_struct.update_state_residence_time_for_matrix(
                    which_node, which_matrix, which_element, time)


    def find_transition(self, current_row, next_row, row_length):
        for indx in range(1, row_length):
            if current_row[indx] != next_row[indx]:
                return [indx - 1, (current_row[indx], next_row[indx])]


    def compute_time_delta(self, current_row, next_row):
        return next_row[0] - current_row[0]

    def which_matrix_to_update(self, current_row, node_indx):
       return tuple(current_row.take(self.fancy_indexing_structure[node_indx]))





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
pe.parameters_estimation()
"""lp = LineProfiler()
lp.add_function(pe.compute_sufficient_statistics_for_row)   # add additional function to profile
lp_wrapper = lp(pe.parameters_estimation_single_trajectory)
#lp_wrapper = lp(pe.parameters_estimation)
lp_wrapper(pe.sample_path.trajectories[0].get_trajectory())
lp.print_stats()"""
for matrix in pe.amalgamated_cims_struct.get_set_of_cims(1).actual_cims:
    print(matrix.state_residence_times)
    print(matrix.state_transition_matrix)
    matrix.compute_cim_coefficients()
    print(matrix.cim)

