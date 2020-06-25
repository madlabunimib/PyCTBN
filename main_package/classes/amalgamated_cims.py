import conditional_intensity_matrix as cim
import numpy as np


class AmalgamatedCims:

    def __init__(self, states_number,list_of_keys, list_of_matrices_dims):
        self.actual_cims = {}
        self.init_cims_structure(list_of_keys, list_of_matrices_dims)
        self.states_per_variable = states_number

    def init_cims_structure(self, keys, dims):
        for key, dim in (keys, dims):
            self.actual_cims[key] = np.empty(dim, dtype=cim.ConditionalIntensityMatrix)
        for key in self.actual_cims.keys():
            for indx in range(len(self.actual_cims[key])):
                self.actual_cims[key][indx] = cim.ConditionalIntensityMatrix(self.states_per_variable)

    def compute_matrix_indx(self, row, col):
        return self.state_per_variable * row + col
