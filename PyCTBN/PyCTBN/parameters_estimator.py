
import numpy as np

from .trajectory import Trajectory
from .set_of_cims import SetOfCims
from .network_graph import NetworkGraph


class ParametersEstimator:
    """Has the task of computing the cims of particular node given the trajectories and the net structure
    in the graph ``_net_graph``.

    :param trajectories:  the trajectories
    :type trajectories: Trajectory
    :param net_graph: the net structure
    :type net_graph: NetworkGraph
    :_single_set_of_cims: the set of cims object that will hold the cims of the node
    """

    def __init__(self, net_graph: NetworkGraph):
        """Constructor Method
        """
        #self._times = times
        #self._trajectories = trajectories
        self._net_graph = net_graph
        self._single_set_of_cims = None

    def fast_init(self, node_id: str) -> None:
        """Initializes all the necessary structures for the parameters estimation for the node ``node_id``.

        :param node_id: the node label
        :type node_id: string
        """
        p_vals = self._net_graph._aggregated_info_about_nodes_parents[2]
        node_states_number = self._net_graph.get_states_number(node_id)
        self._single_set_of_cims = SetOfCims(node_id, p_vals, node_states_number, self._net_graph.p_combs)

    def compute_parameters_for_node(self, node_id: str, times, trajectories) -> SetOfCims:
        """Compute the CIMS of the node identified by the label ``node_id``.

        :param node_id: the node label
        :type node_id: string
        :return: A SetOfCims object filled with the computed CIMS
        :rtype: SetOfCims
        """
        node_indx = self._net_graph.get_node_indx(node_id)
        state_res_times = self._single_set_of_cims._state_residence_times
        transition_matrices = self._single_set_of_cims._transition_matrices
        self.compute_state_res_time_for_node(node_indx, times,
                                             trajectories,
                                             self._net_graph.time_filtering,
                                             self._net_graph.time_scalar_indexing_strucure,
                                             state_res_times)
        self.compute_state_transitions_for_a_node(node_indx,
                                                  trajectories,
                                                  self._net_graph.transition_filtering,
                                                  self._net_graph.transition_scalar_indexing_structure,
                                                  transition_matrices)
        self._single_set_of_cims.build_cims(state_res_times, transition_matrices)
        return self._single_set_of_cims

    @staticmethod
    def compute_state_res_time_for_node(node_indx: int, times: np.ndarray, trajectory: np.ndarray,
                                        cols_filter: np.ndarray, scalar_indexes_struct: np.ndarray,
                                        T: np.ndarray) -> None:
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

    @staticmethod
    def compute_state_transitions_for_a_node(node_indx: int, trajectory: np.ndarray, cols_filter: np.ndarray,
                                             scalar_indexing: np.ndarray, M: np.ndarray):
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

    """
    ##############These Methods are actually unused but could become useful in the near future################

    def init_sets_cims_container(self):
        self.sets_of_cims_struct = acims.SetsOfCimsContainer(self._net_graph.nodes,
                                                             self._net_graph.nodes_values,
                                                             self._net_graph.
                                                             get_ordered_by_indx_parents_values_for_all_nodes(),
                                                             self._net_graph.p_combs)

    def compute_parameters(self):
        for indx, aggr in enumerate(zip(self._net_graph.nodes, self.sets_of_cims_struct.sets_of_cims)):
            self.compute_state_res_time_for_node(self._net_graph.get_node_indx(aggr[0]), self.sample_path.trajectories.times,
                                                 self.sample_path.trajectories.trajectory,
                                                 self._net_graph.time_filtering[indx],
                                                 self._net_graph.time_scalar_indexing_strucure[indx],
                                                 aggr[1]._state_residence_times)
            self.compute_state_transitions_for_a_node(self._net_graph.get_node_indx(aggr[0]),
                                                      self.sample_path.trajectories.complete_trajectory,
                                                      self._net_graph.transition_filtering[indx],
                                                      self._net_graph.transition_scalar_indexing_structure[indx],
                                                      aggr[1]._transition_matrices)
            aggr[1].build_cims(aggr[1]._state_residence_times, aggr[1]._transition_matrices)
    """








