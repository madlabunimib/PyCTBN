import typing

import numpy as np

import conditional_intensity_matrix as cim


class SetOfCims:
    """
    Aggregates all the CIMS of the node identified by the label node_id.

    :node_id: the node label
    :parents_states_number: the cardinalities of the parents
    :node_states_number: the caridinality of the node
    :p_combs: the relative p_comb structure

    :state_residence_time: matrix containing all the state residence time vectors for the node
    :transition_matrices: matrix containing all the transition matrices for the node
    :actaul_cims: the cims of the node
    """

    def __init__(self, node_id: str, parents_states_number: typing.List, node_states_number: int, p_combs: np.ndarray):
        self.node_id = node_id
        self.parents_states_number = parents_states_number
        self.node_states_number = node_states_number
        self.actual_cims = []
        self.state_residence_times = None
        self.transition_matrices = None
        self.p_combs = p_combs
        self.build_times_and_transitions_structures()

    def build_times_and_transitions_structures(self):
        """
        Initializes at the correct dimensions the state residence times matrix and the state transition matrices

        Parameters:
            void
        Returns:
            void
        """
        if not self.parents_states_number:
            self.state_residence_times = np.zeros((1, self.node_states_number), dtype=np.float)
            self.transition_matrices = np.zeros((1,self.node_states_number, self.node_states_number), dtype=np.int)
        else:
            self.state_residence_times = \
                np.zeros((np.prod(self.parents_states_number), self.node_states_number), dtype=np.float)
            self.transition_matrices = np.zeros([np.prod(self.parents_states_number), self.node_states_number,
                                                 self.node_states_number], dtype=np.int)

    def build_cims(self, state_res_times: typing.List, transition_matrices: typing.List):
        """
        Build the ConditionalIntensityMatrix object given the state residence times and transitions matrices.
        Compute the cim coefficients.

        Parameters:
            state_res_times: the state residence times matrix
            transition_matrices: the transition matrices
        Returns:
            void
        """
        for state_res_time_vector, transition_matrix in zip(state_res_times, transition_matrices):
            cim_to_add = cim.ConditionalIntensityMatrix(state_res_time_vector, transition_matrix)
            cim_to_add.compute_cim_coefficients()
            self.actual_cims.append(cim_to_add)
        self.actual_cims = np.array(self.actual_cims)
        self.transition_matrices = None
        self.state_residence_times = None

    def filter_cims_with_mask(self, mask_arr: np.ndarray, comb: typing.List) -> np.ndarray:
        """
        Filter the cims contained in the array actual_cims given the boolean mask mask_arr and the index comb.
        Parameters:
            mask_arr: the boolean mask
            comb: the indexes of the selected cims

        Returns:
            Array of ConditionalIntensityMatrix
        """
        if mask_arr.size <= 1:
            return self.actual_cims
        else:
            flat_indxs = np.argwhere(np.all(self.p_combs[:, mask_arr] == comb, axis=1)).ravel()
            return self.actual_cims[flat_indxs]

    @property
    def get_cims(self):
        return self.actual_cims

    def get_cims_number(self):
        return len(self.actual_cims)
"""
    def get_cim(self, index):
        flat_index = self.indexes_converter(index)
        return self.actual_cims[flat_index]

    def indexes_converter(self, indexes):
        assert len(indexes) == len(self.parents_states_number)
        vector_index = 0
        if not indexes:
            return vector_index
        else:
            for indx, value in enumerate(indexes):
                vector_index = vector_index*self.parents_states_number[indx] + indexes[indx]
            return vector_index"""


