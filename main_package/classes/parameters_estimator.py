import os

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
        self.amalgamated_cims_struct = None

    def init_amalgamated_cims_struct(self):
        self.amalgamated_cims_struct = acims.AmalgamatedCims(self.net_graph.get_states_number_of_all_nodes_sorted(),
                                                     self.net_graph.get_nodes(),
                                                    self.net_graph.get_ordered_by_indx_parents_values_for_all_nodes())


    def compute_parameters(self):
        #print(self.net_graph.get_nodes())
        #print(self.amalgamated_cims_struct.sets_of_cims)
        #enumerate(zip(self.net_graph.get_nodes(), self.amalgamated_cims_struct.sets_of_cims))
        for indx, aggr in enumerate(zip(self.net_graph.get_nodes(), self.amalgamated_cims_struct.sets_of_cims)):
            #print(self.net_graph.time_filtering[indx])
            #print(self.net_graph.time_scalar_indexing_strucure[indx])
            self.compute_state_res_time_for_node(self.net_graph.get_node_indx(aggr[0]), self.sample_path.trajectories.times,
                                                 self.sample_path.trajectories.trajectory,
                                                 self.net_graph.time_filtering[indx],
                                                 self.net_graph.time_scalar_indexing_strucure[indx],
                                                 aggr[1].state_residence_times)
            #print(self.net_graph.transition_filtering[indx])
            #print(self.net_graph.transition_scalar_indexing_structure[indx])
            self.compute_state_transitions_for_a_node(self.net_graph.get_node_indx(aggr[0]),
                                                      self.sample_path.trajectories.complete_trajectory,
                                                      self.net_graph.transition_filtering[indx],
                                                      self.net_graph.transition_scalar_indexing_structure[indx],
                                                      aggr[1].transition_matrices)
            aggr[1].build_cims(aggr[1].state_residence_times, aggr[1].transition_matrices)



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
"""os.getcwd()
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

[[2999.2966 2749.2298 3301.5975]
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
 13718 18243 31961]

lp_wrapper = lp(pe.compute_parameters)
lp_wrapper()
#for variable in pe.amalgamated_cims_struct.sets_of_cims:
    #for cond in variable.get_cims():
        #print(cond.cim)
print(pe.amalgamated_cims_struct.get_cims_of_node(1,[2]))
lp.print_stats()"""

