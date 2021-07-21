
# License: MIT License



import typing

import numpy as np

from .conditional_intensity_matrix import ConditionalIntensityMatrix


class SetOfCims(object):
    """Aggregates all the CIMS of the node identified by the label _node_id.

    :param node_id: the node label
    :type node_ind: string
    :param parents_states_number: the cardinalities of the parents
    :type parents_states_number: List
    :param node_states_number: the caridinality of the node
    :type node_states_number: int
    :param p_combs: the p_comb structure bound to this node
    :type p_combs: numpy.ndArray
    :_state_residence_time: matrix containing all the state residence time vectors for the node
    :_transition_matrices: matrix containing all the transition matrices for the node
    :_actual_cims: the cims of the node
    """

    def __init__(self, node_id: str, parents_states_number: typing.List, node_states_number: int, p_combs: np.ndarray,
        cims: np.ndarray = None):
        """Constructor Method
        """
        self._node_id = node_id
        self._parents_states_number = parents_states_number
        self._node_states_number = node_states_number
        self._actual_cims = []
        self._state_residence_times = None
        self._transition_matrices = None
        self._p_combs = p_combs

        if cims is not None:
            self._actual_cims = cims
        else:
            self.build_times_and_transitions_structures()

    def build_times_and_transitions_structures(self) -> None:
        """Initializes at the correct dimensions the state residence times matrix and the state transition matrices.
        """
        if not self._parents_states_number:
            self._state_residence_times = np.zeros((1, self._node_states_number), dtype=np.float)
            self._transition_matrices = np.zeros((1, self._node_states_number, self._node_states_number), dtype=np.int)
        else:
            self._state_residence_times = \
                np.zeros((np.prod(self._parents_states_number), self._node_states_number), dtype=np.float)
            self._transition_matrices = np.zeros([np.prod(self._parents_states_number), self._node_states_number,
                                                  self._node_states_number], dtype=np.int)

    def build_cims(self, state_res_times: np.ndarray, transition_matrices: np.ndarray) -> None:
        """Build the ``ConditionalIntensityMatrix`` objects given the state residence times and transitions matrices.
        Compute the cim coefficients.The class member ``_actual_cims`` will contain the computed cims.

        :param state_res_times: the state residence times matrix
        :type state_res_times: numpy.ndArray
        :param transition_matrices: the transition matrices
        :type transition_matrices: numpy.ndArray
        """
        for state_res_time_vector, transition_matrix in zip(state_res_times, transition_matrices):
            cim_to_add = ConditionalIntensityMatrix(state_residence_times = state_res_time_vector, state_transition_matrix = transition_matrix)
            cim_to_add.compute_cim_coefficients()
            self._actual_cims.append(cim_to_add)
        self._actual_cims = np.array(self._actual_cims)
        self._transition_matrices = None
        self._state_residence_times = None

    def filter_cims_with_mask(self, mask_arr: np.ndarray, comb: typing.List) -> np.ndarray:
        """Filter the cims contained in the array ``_actual_cims`` given the boolean mask ``mask_arr`` and the index
        ``comb``.

        :param mask_arr: the boolean mask that indicates which parent to consider
        :type mask_arr: numpy.array
        :param comb: the state/s of the filtered parents
        :type comb: numpy.array
        :return: Array of ``ConditionalIntensityMatrix`` objects
        :rtype: numpy.array
        """
        if mask_arr.size <= 1:
            return self._actual_cims
        else:
            flat_indxs = np.argwhere(np.all(self._p_combs[:, mask_arr] == comb, axis=1)).ravel()
            return np.array(self._actual_cims)[flat_indxs.astype(int)]

    @property
    def actual_cims(self) -> np.ndarray:
        return self._actual_cims

    @property
    def p_combs(self) -> np.ndarray:
        return self._p_combs

    def get_cims_number(self):
        return len(self._actual_cims)




