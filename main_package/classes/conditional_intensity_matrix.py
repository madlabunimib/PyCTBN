import numpy as np


class ConditionalIntensityMatrix:

    def __init__(self, dimension, state_residence_times, state_transition_matrix):
        self.state_residence_times = state_residence_times
        self.state_transition_matrix = state_transition_matrix
        #self.cim = np.zeros(shape=(dimension, dimension), dtype=float)
        self.cim = self.state_transition_matrix.astype(np.float)

    def update_state_transition_count(self, element_indx):
        #print(element_indx)
        #self.state_transition_matrix[element_indx[0]][element_indx[1]] += 1
        self.state_transition_matrix[element_indx] += 1

    def update_state_residence_time_for_state(self, state, time):
        #print("Time updating In state", state, time)
        self.state_residence_times[state] += time

    def compute_cim_coefficients(self):
        np.fill_diagonal(self.cim, self.cim.diagonal() * -1)
        self.cim = ((self.cim.T + 1) / (self.state_residence_times + 1)).T

    def __repr__(self):
        return 'CIM:\n' + str(self.cim)

