import typing

import numpy as np

import conditional_intensity_matrix as cim


class SetOfCims:
    """
    Aggregates all the CIMS of the node identified by the label _node_id.

    :_node_id: the node label
    :_parents_states_number: the cardinalities of the parents
    :_node_states_number: the caridinality of the node
    :_p_combs: the relative p_comb structure
    :state_residence_time: matrix containing all the state residence time vectors for the node
    :_transition_matrices: matrix containing all the transition matrices for the node
    :_actual_cims: the cims of the node
    """

    def __init__(self, node_id: str, parents_states_number: typing.List, node_states_number: int, p_combs: np.ndarray):
        """
        Parameters:
        :_node_id: the node label
        :_parents_states_number: the cardinalities of the parents
        :_node_states_number: the caridinality of the node
        :_p_combs: the relative p_comb structure

        """
        self._node_id = node_id
        self._parents_states_number = parents_states_number
        self._node_states_number = node_states_number
        self._actual_cims = []
        self._state_residence_times = None
        self._transition_matrices = None
        self._p_combs = p_combs
        self.build_times_and_transitions_structures()

    def build_times_and_transitions_structures(self):
        """
        Initializes at the correct dimensions the state residence times matrix and the state transition matrices

        Parameters:
            :void
        Returns:
            :void
        """
        if not self._parents_states_number:
            self._state_residence_times = np.zeros((1, self._node_states_number), dtype=np.float)
            self._transition_matrices = np.zeros((1, self._node_states_number, self._node_states_number), dtype=np.int)
        else:
            self._state_residence_times = \
                np.zeros((np.prod(self._parents_states_number), self._node_states_number), dtype=np.float)
            self._transition_matrices = np.zeros([np.prod(self._parents_states_number), self._node_states_number,
                                                  self._node_states_number], dtype=np.int)

    def build_cims(self, state_res_times: np.ndarray, transition_matrices: np.ndarray):
        """
        Build the ConditionalIntensityMatrix object given the state residence times and transitions matrices.
        Compute the cim coefficients.

        Parameters:
            :state_res_times: the state residence times matrix
            :_transition_matrices: the transition matrices
        Returns:
            :void
        """
        for state_res_time_vector, transition_matrix in zip(state_res_times, transition_matrices):
            cim_to_add = cim.ConditionalIntensityMatrix(state_res_time_vector, transition_matrix)
            cim_to_add.compute_cim_coefficients()
            self._actual_cims.append(cim_to_add)
        self._actual_cims = np.array(self._actual_cims)
        self._transition_matrices = None
        self._state_residence_times = None

    def filter_cims_with_mask(self, mask_arr: np.ndarray, comb: typing.List) -> np.ndarray:
        """
        Filter the cims contained in the array _actual_cims given the boolean mask mask_arr and the index comb.
        Parameters:
            :mask_arr: the boolean mask
            :comb: the indexes of the selected cims

        Returns:
            :Array of ConditionalIntensityMatrix
        """
        if mask_arr.size <= 1:
            return self._actual_cims
        else:
            flat_indxs = np.argwhere(np.all(self._p_combs[:, mask_arr] == comb, axis=1)).ravel()
            return self._actual_cims[flat_indxs]

    @property
    def actual_cims(self) -> np.ndarray:
        return self._actual_cims

    @property
    def p_combs(self) -> np.ndarray:
        return self._p_combs

    def get_cims_number(self):
        return len(self._actual_cims)



