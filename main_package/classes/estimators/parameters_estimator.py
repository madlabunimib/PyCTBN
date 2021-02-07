import sys
sys.path.append('../')
import numpy as np

import structure_graph.network_graph as ng
import structure_graph.sample_path as sp
import structure_graph.set_of_cims as sofc
import structure_graph.sets_of_cims_container as acims


class ParametersEstimator:
    """Has the task of computing the cims of particular node given the trajectories and the net structure
    in the graph ``_net_graph``.

    :param trajectories:  the trajectories
    :type trajectories: Trajectory
    :param net_graph: the net structure
    :type net_graph: NetworkGraph
    :_single_set_of_cims: the set of cims object that will hold the cims of the node
    """

    def __init__(self, sample_path: sp.SamplePath, net_graph: ng.NetworkGraph):
        """Constructor Method
        """
        self.sample_path = sample_path
        self.net_graph = net_graph
        self.sets_of_cims_struct = None
        self.single_set_of_cims = None

    def fast_init(self, node_id: str):
        """Initializes all the necessary structures for the parameters estimation for the node ``node_id``.

        :param node_id: the node label
        :type node_id: string
        """
        p_vals = self.net_graph.aggregated_info_about_nodes_parents[2]
        node_states_number = self.net_graph.get_states_number(node_id)
        self.single_set_of_cims = sofc.SetOfCims(node_id, p_vals, node_states_number, self.net_graph.p_combs)

    def compute_parameters_for_node(self, node_id: str) -> sofc.SetOfCims:
        """Compute the CIMS of the node identified by the label ``node_id``.

        :param node_id: the node label
        :type node_id: string
        :return: A SetOfCims object filled with the computed CIMS
        :rtype: SetOfCims
        """
        node_indx = self.net_graph.get_node_indx(node_id)
        state_res_times = self.single_set_of_cims._state_residence_times
        transition_matrices = self.single_set_of_cims._transition_matrices
        trajectory = self.sample_path.trajectories.trajectory
        self.compute_state_res_time_for_node(node_indx, self.sample_path.trajectories.times,
                                             trajectory,
                                             self.net_graph.time_filtering,
                                             self.net_graph.time_scalar_indexing_strucure,
                                             state_res_times)
        self.compute_state_transitions_for_a_node(node_indx,
                                                  self.sample_path.trajectories.complete_trajectory,
                                                  self.net_graph.transition_filtering,
                                                  self.net_graph.transition_scalar_indexing_structure,
                                                  transition_matrices)
        self.single_set_of_cims.build_cims(state_res_times, transition_matrices)
        return self.single_set_of_cims

    def compute_state_res_time_for_node(self, node_indx: int, times: np.ndarray, trajectory: np.ndarray,
                                        cols_filter: np.ndarray, scalar_indexes_struct: np.ndarray, T: np.ndarray):
        """Compute the state residence times for a node and fill the matrix ``T`` with the results

        :param node_indx: the index of the node
        :type node_indx: int
        :param times: the times deltas vector
        :type times: numpy.array
        :param trajectory: the trajectory
        :type trajectory: numpy.ndArray
        :param cols_filter: the columns filtering structure
        :type cols_filter: numpy.array
        :param scalar_indexes_struct: the indexing structure
        :type scalar_indexes_struct: numpy.array
        :param T: the state residence times vectors
        :type T: numpy.ndArray
        """
        T[:] = np.bincount(np.sum(trajectory[:, cols_filter] * scalar_indexes_struct / scalar_indexes_struct[0], axis=1)
                           .astype(np.int), \
                           times,
                           minlength=scalar_indexes_struct[-1]).reshape(-1, T.shape[1])

    def compute_state_transitions_for_a_node(self, node_indx, trajectory, cols_filter, scalar_indexing, M):
        """Compute the state residence times for a node and fill the matrices ``M`` with the results.

        :param node_indx: the index of the node
        :type node_indx: int
        :param trajectory: the trajectory
        :type trajectory: numpy.ndArray
        :param cols_filter: the columns filtering structure
        :type cols_filter: numpy.array
        :param scalar_indexing: the indexing structure
        :type scalar_indexing: numpy.array
        :param M: the state transitions matrices
        :type M: numpy.ndArray
        """
        diag_indices = np.array([x * M.shape[1] + x % M.shape[1] for x in range(M.shape[0] * M.shape[1])],
                                dtype=np.int64)
        trj_tmp = trajectory[trajectory[:, int(trajectory.shape[1] / 2) + node_indx].astype(np.int) >= 0]
        M[:] = np.bincount(np.sum(trj_tmp[:, cols_filter] * scalar_indexing / scalar_indexing[0], axis=1).astype(np.int),
                           minlength=scalar_indexing[-1]).reshape(-1, M.shape[1], M.shape[2])
        M_raveled = M.ravel()
        M_raveled[diag_indices] = 0
        M_raveled[diag_indices] = np.sum(M, axis=2).ravel()

    def init_sets_cims_container(self):
        self.sets_of_cims_struct = acims.SetsOfCimsContainer(self.net_graph.nodes,
                                                             self.net_graph.nodes_values,
                                                             self.net_graph.get_ordered_by_indx_parents_values_for_all_nodes(),
                                                             self.net_graph.p_combs)

    def compute_parameters(self):
        #print(self.net_graph.get_nodes())
        #print(self.amalgamated_cims_struct.sets_of_cims)
        #enumerate(zip(self.net_graph.get_nodes(), self.amalgamated_cims_struct.sets_of_cims))
        for indx, aggr in enumerate(zip(self.net_graph.nodes, self.sets_of_cims_struct.sets_of_cims)):
            #print(self.net_graph.time_filtering[indx])
            #print(self.net_graph.time_scalar_indexing_strucure[indx])
            self.compute_state_res_time_for_node(self.net_graph.get_node_indx(aggr[0]), self.sample_path.trajectories.times,
                                                 self.sample_path.trajectories.trajectory,
                                                 self.net_graph.time_filtering[indx],
                                                 self.net_graph.time_scalar_indexing_strucure[indx],
                                                 aggr[1]._state_residence_times)
            #print(self.net_graph.transition_filtering[indx])
            #print(self.net_graph.transition_scalar_indexing_structure[indx])
            self.compute_state_transitions_for_a_node(self.net_graph.get_node_indx(aggr[0]),
                                                      self.sample_path.trajectories.complete_trajectory,
                                                      self.net_graph.transition_filtering[indx],
                                                      self.net_graph.transition_scalar_indexing_structure[indx],
                                                      aggr[1]._transition_matrices)
            aggr[1].build_cims(aggr[1]._state_residence_times, aggr[1]._transition_matrices)








