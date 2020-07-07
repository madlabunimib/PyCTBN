import os
import time as tm
from line_profiler import LineProfiler

import numba as nb
import numpy as np
import network_graph as ng
import sample_path as sp
import amalgamated_cims as acims


class ParametersEstimator:

    def __init__(self, sample_path, net_graph):
        self.sample_path = sample_path
        self.net_graph = net_graph
        #self.scalar_indexes_converter = self.net_graph.
        #self.columns_filtering_structure = self.net_graph.filtering_structure
        #self.transition_scalar_index_converter = self.net_graph.transition_scalar_indexing_structure
        #self.transition_filtering = self.net_graph.transition_filtering
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
            self.compute_sufficient_statistics_for_trajectory(trajectory.times, trajectory.actual_trajectory, trajectory.transitions, row_length)

    def compute_sufficient_statistics_for_trajectory(self, times, traj_values, traj_transitions, row_length):
        #time = self.compute_time_delta(current_row, next_row)
        #time = current_row[0]
        print(times)
        print(traj_values)
        print(traj_transitions)
        for row in traj_transitions:
            time = times[0]
            for indx in range(0, row_length):
                if row[indx] == 1:
                    which_node = indx
                    transition = [which_node, (traj_values[indx - 1], traj_values[indx])]
                    which_matrix = self.which_matrix_to_update(row, which_node)
                    which_element = transition[1]
                    self.amalgamated_cims_struct.update_state_transition_for_matrix(which_node, which_matrix, which_element)
                    which_element = transition[1][0]
                    self.amalgamated_cims_struct.update_state_residence_time_for_matrix(which_node, which_matrix,
                                                                                    which_element,
                                                                                    time)
            else:
                which_node = indx
                which_matrix = self.which_matrix_to_update(row, which_node)
                which_element = row[indx]
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

    def compute_parameters(self):
        for node_indx, set_of_cims in enumerate(self.amalgamated_cims_struct.sets_of_cims):
            self.compute_state_res_time_for_node(node_indx, self.sample_path.trajectories.times,
                                                 self.sample_path.trajectories.trajectory,
                                                 self.net_graph.time_filtering[node_indx],
                                                 self.net_graph.time_scalar_indexing_strucure[node_indx],
                                                 set_of_cims.state_residence_times)
            self.compute_state_transitions_for_a_node(node_indx,
                                                      self.sample_path.trajectories.complete_trajectory,
                                                      self.net_graph.transition_filtering[node_indx],
                                                      self.net_graph.transition_scalar_indexing_structure[node_indx],
                                                      set_of_cims.transition_matrices)
            set_of_cims.build_cims(set_of_cims.state_residence_times, set_of_cims.transition_matrices)



    def compute_state_res_time_for_node(self, node_indx, times, trajectory, cols_filter, scalar_indexes_struct, T):
        #print(times.size)
        #print(trajectory)
        #print(cols_filter)
        #print(scalar_indexes_struct)
        #print(T)
        T[:] = np.bincount(np.sum(trajectory[:, cols_filter] * scalar_indexes_struct / scalar_indexes_struct[0], axis=1)
                           .astype(np.int), \
                           times,
                           minlength=scalar_indexes_struct[-1]).reshape(-1, T.shape[1])
        #print("Done This NODE", T)

    def compute_state_residence_time_for_all_nodes(self):
        for node_indx, set_of_cims in enumerate(self.amalgamated_cims_struct.sets_of_cims):
            self.compute_state_res_time_for_node(node_indx, self.sample_path.trajectories[0].get_times(),
                self.sample_path.trajectories[0].get_trajectory(), self.columns_filtering_structure[node_indx],
                    self.scalar_indexes_converter[node_indx], set_of_cims.state_residence_times)


    def compute_state_transitions_for_a_node(self, node_indx, trajectory, cols_filter, scalar_indexing, M):
        #print(node_indx)
        #print(trajectory)
        #print(cols_filter)
        #print(scalar_indexing)
        #print(M)
        diag_indices = np.array([x * M.shape[1] + x % M.shape[1] for x in range(M.shape[0] * M.shape[1])],
                                dtype=np.int64)
        trj_tmp = trajectory[trajectory[:, int(trajectory.shape[1] / 2) + node_indx].astype(np.int) >= 0]
        #print(trj_tmp)
        #print("Summing", np.sum(trj_tmp[:, cols_filter] * scalar_indexing / scalar_indexing[0], axis=1).astype(np.int))
        #print(M.shape[1])
        #print(M.shape[2])

        M[:] = np.bincount(np.sum(trj_tmp[:, cols_filter] * scalar_indexing / scalar_indexing[0], axis=1).astype(np.int),
                           minlength=scalar_indexing[-1]).reshape(-1, M.shape[1], M.shape[2])
        M_raveled = M.ravel()
        M_raveled[diag_indices] = 0
        #print(M_raveled)
        M_raveled[diag_indices] = np.sum(M, axis=2).ravel()
        #print(M_raveled)

        #print(M)



    def compute_state_transitions_for_all_nodes(self):
        for node_indx, set_of_cims in enumerate(self.amalgamated_cims_struct.sets_of_cims):
            self.compute_state_transitions_for_a_node(node_indx, self.sample_path.trajectories[0].get_complete_trajectory(),
                 self.transition_filtering[node_indx],
                    self.transition_scalar_index_converter[node_indx], set_of_cims.transition_matrices)


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
lp = LineProfiler()

"""[[2999.2966 2749.2298 3301.5975]
 [3797.1737 3187.8345 2939.2009]
 [3432.224  3062.5402 4530.9028]]

[[ 827.6058  838.1515  686.1365]
 [1426.384  2225.2093 1999.8528]
 [ 745.3068  733.8129  746.2347]
 [ 520.8113  690.9502  853.4022]
 [1590.8609 1853.0021 1554.1874]
 [ 637.5576  643.8822  654.9506]
 [ 718.7632  742.2117  998.5844]
 [1811.984  1598.0304 2547.988 ]
 [ 770.8503  598.9588  984.3304]]

lp_wrapper = lp(pe.compute_state_residence_time_for_all_nodes)
lp_wrapper()
lp.print_stats()

#pe.compute_state_residence_time_for_all_nodes()
print(pe.amalgamated_cims_struct.sets_of_cims[0].state_residence_times)

[[[14472,  3552, 10920],
        [12230, 25307, 13077],
        [ 9707, 14408, 24115]],

       [[22918,  6426, 16492],
        [10608, 16072,  5464],
        [10746, 11213, 21959]],

       [[23305,  6816, 16489],
        [ 3792, 19190, 15398],
        [13718, 18243, 31961]]])
        
        Raveled [14472  3552 10920 12230 25307 13077  9707 14408 24115 22918  6426 16492
 10608 16072  5464 10746 11213 21959 23305  6816 16489  3792 19190 15398
 13718 18243 31961]"""

lp_wrapper = lp(pe.compute_parameters)
lp_wrapper()
#for variable in pe.amalgamated_cims_struct.sets_of_cims:
    #for cond in variable.get_cims():
        #print(cond.cim)
print(pe.amalgamated_cims_struct.get_cims_of_node(1,[2]))
lp.print_stats()