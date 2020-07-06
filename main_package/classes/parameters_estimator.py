import os
import time as tm
from line_profiler import LineProfiler
from multiprocessing import Process

import numba as nb
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

    def which_matrix_to_update(self, current_row, node_indx):
        #print(type(self.fancy_indexing_structure[node_indx]))
        return tuple(current_row.take(self.fancy_indexing_structure[node_indx]))
        #return tuple(ParametersEstimator.taker(current_row, self.fancy_indexing_structure[node_indx]))

    def parameters_estimation_for_variable_multiple_parents(self, node_indx, times, transitions ,variable_values, parents_values):
        #print(times)
        #print(variable_values)
        #print(parents_values)

        #print("Starting computing")
        #t0 = tm.time()
        for indx, row in enumerate(variable_values):
            time = times[indx]
            which_matrix = tuple(parents_values[indx])  # questo è un vettore
            current_state = variable_values[indx]
            if transitions[indx] == 1:
                prev_state = variable_values[indx - 1]
                transition = [node_indx, (prev_state, current_state)]
                #which_node = transition[0]
                which_element = transition[1]
                self.amalgamated_cims_struct.update_state_transition_for_matrix(node_indx, which_matrix, which_element)
            #which_element = current_state
            self.amalgamated_cims_struct.update_state_residence_time_for_matrix(node_indx, which_matrix,
                                                                                    current_state,
                                                                                    time)

    def parameters_estimation_for_variable_single_parent(self, node_indx, times, transitions, variable_values,
                                                                parents_values):
            for indx, row in enumerate(variable_values):
                time = times[indx]
                which_matrix = parents_values[indx]  # Avendo un solo parent questo è uno scalare
                current_state = variable_values[indx]
                #which_matrix = ParametersEstimator.taker(parents_values, indx)
                # print(which_matrix.dtype)
                if transitions[indx] == 1:
                    prev_state = variable_values[indx - 1]
                    transition = [node_indx, (prev_state, current_state)]
                    which_element = transition[1]
                    self.amalgamated_cims_struct.update_state_transition_for_matrix(node_indx, which_matrix,
                                                                                    which_element)
                which_element = current_state
                self.amalgamated_cims_struct.update_state_residence_time_for_matrix(node_indx, which_matrix,
                                                                                    which_element,time)

    def parameters_estimation_for_variable_no_parent(self, node_indx, times, transitions,variable_values):

        for indx, row in enumerate(variable_values):
            time = times[indx]

            which_matrix = 0
            current_state = variable_values[indx]
            """if transitions[indx] == 1:
                prev_state = variable_values[indx - 1]
                #current_state = variable_values[indx]
                transition = [node_indx, (prev_state, current_state)]

                which_element = transition[1]
                self.amalgamated_cims_struct.update_state_transition_for_matrix(node_indx, which_matrix,
                                                                                            which_element)"""
            which_element = current_state
            self.amalgamated_cims_struct.update_state_residence_time_for_matrix(node_indx, which_matrix,
                                                                                            which_element,
                                                                                            time)

    def parameters_estimation_for_variable_no_parent_in_place(self, node_indx, times, transitions, variable_values):
            state_trans_matrix = np.zeros(shape=(3,3), dtype=np.int)
            state_res_time_array = np.zeros(shape=(3), dtype=np.float)
            for indx, row in enumerate(variable_values):
                time = times[indx]
                #which_matrix = 0
                current_state = variable_values[indx]
                if transitions[indx] == 1:
                    prev_state = variable_values[indx - 1]
                    #current_state = variable_values[indx]
                    transition = [node_indx, (prev_state, current_state)]

                    which_element = transition[1]
                    #self.amalgamated_cims_struct.update_state_transition_for_matrix(node_indx, which_matrix,
                                                                                                #which_element)
                    state_trans_matrix[which_element] += 1
                which_element = current_state
                #self.amalgamated_cims_struct.update_state_residence_time_for_matrix(node_indx, which_matrix,
                                                                                    #which_element,
                                                                                    #time)
                state_res_time_array[which_element] += time

    def parameters_estimation_for_variable_single_parent_in_place(self, node_indx, times, transitions, variable_values,
                                                                parents_values,values_tuple):
            state_res_time_dim = values_tuple[1:]

            state_trans_matricies = np.zeros(shape=27, dtype=np.int)
            state_res_time_array = np.zeros(shape=9, dtype=np.float)
            state_transition_indx = np.array(values_tuple, dtype=np.int)
            for indx, row in enumerate(variable_values):
                time = times[indx]
                #which_matrix = np.ravel_multi_index(parents_values[indx], )  # Avendo un solo parent questo è uno scalare
                #current_state = variable_values[indx]
                #which_matrix = ParametersEstimator.taker(parents_values, indx)
                state_transition_indx[0] = parents_values[indx]
                state_transition_indx[1] = variable_values[indx]
                # print(which_matrix.dtype)
                if transitions[indx] == 1:
                    state_transition_indx[2] = variable_values[indx - 1]

                    #transition = [node_indx, (prev_state, current_state)]
                    #which_element = transition[1]
                    #self.amalgamated_cims_struct.update_state_transition_for_matrix(node_indx, which_matrix,
                                                                                    #which_element)
                    scalar_indx = np.ravel_multi_index(state_transition_indx, values_tuple)
                    print("State Transition", scalar_indx)
                    state_trans_matricies[scalar_indx] += 1
                scalar_indx = np.ravel_multi_index(state_transition_indx[:-1], state_res_time_dim)
                print("Res Time",scalar_indx)
                state_res_time_array[scalar_indx] += time
                #which_element = current_state
                #self.amalgamated_cims_struct.update_state_residence_time_for_matrix(node_indx, which_matrix,
                                                                                    #which_element,time)
        #t1 = tm.time() - t0
        #print("Elapsed Time ", t1)



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
print(np.shape(s1.trajectories[0].transitions)[0])
#pe.parameters_estimation_for_variable(0, pe.sample_path.trajectories[0].get_trajectory()[:, 0],
                                     # pe.sample_path.trajectories[0].get_trajectory()[:, 1], [])
#pe.parameters_estimation_single_trajectory(pe.sample_path.trajectories[0].get_trajectory())
#pe.parameters_estimation()
lp = LineProfiler()
#lp.add_function(pe.compute_sufficient_statistics_for_row)   # add additional function to profile
#lp_wrapper = lp(pe.parameters_estimation_single_trajectory)
#lp_wrapper = lp(pe.parameters_estimation)
#lp_wrapper(pe.sample_path.trajectories[0].get_trajectory())
#lp.print_stats()

#lp_wrapper = lp(pe.parameters_estimation_for_variable)
#lp_wrapper(2, pe.sample_path.trajectories[0].get_times(),
                                      #pe.sample_path.trajectories[0].get_trajectory()[:, 2],
           #pe.sample_path.trajectories[0].get_trajectory()[:, [0,1]])


"""lp_wrapper = lp(pe.parameters_estimation_for_variable_single_parent)
lp_wrapper(1, pe.sample_path.trajectories[0].get_times(),
                                      pe.sample_path.trajectories[0].get_trajectory()[:, 1],
           pe.sample_path.trajectories[0].get_trajectory()[:, 2])
lp.print_stats()

#print( pe.sample_path.trajectories[0].get_trajectory()[:, [1,2]])
for matrix in pe.amalgamated_cims_struct.get_set_of_cims(1).actual_cims:
    print(matrix.state_residence_times)
    print(matrix.state_transition_matrix)
    matrix.compute_cim_coefficients()
    print(matrix.cim)"""

"""lp_wrapper = lp(pe.parameters_estimation_for_variable_no_parent_in_place)
lp_wrapper(0, pe.sample_path.trajectories[0].get_times(), pe.sample_path.trajectories[0].transitions[:, 0],
           pe.sample_path.trajectories[0].get_trajectory()[:, 0] )
lp.print_stats()

lp_wrapper = lp(pe.parameters_estimation_for_variable_single_parent)
lp_wrapper(1, pe.sample_path.trajectories[0].get_times(), pe.sample_path.trajectories[0].transitions[:, 1],
           pe.sample_path.trajectories[0].get_trajectory()[:,1], pe.sample_path.trajectories[0].get_trajectory()[:,2] )
lp.print_stats()
lp_wrapper = lp(pe.parameters_estimation_for_variable_multiple_parents)
lp_wrapper(2, pe.sample_path.trajectories[0].get_times(), pe.sample_path.trajectories[0].transitions[:, 2],
           pe.sample_path.trajectories[0].get_trajectory()[:,2], pe.sample_path.trajectories[0].get_trajectory()[:, [0,1]] )
lp.print_stats()"""

lp_wrapper = lp(pe.parameters_estimation_for_variable_single_parent_in_place)
lp_wrapper(1, pe.sample_path.trajectories[0].get_times(), pe.sample_path.trajectories[0].transitions[:, 1],
           pe.sample_path.trajectories[0].get_trajectory()[:,1], pe.sample_path.trajectories[0].get_trajectory()[:,2], (3,3,3) )
lp.print_stats()