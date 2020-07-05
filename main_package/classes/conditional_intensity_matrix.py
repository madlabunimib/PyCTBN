import numpy as np


class ConditionalIntensityMatrix:

    def __init__(self, dimension):
        self.state_residence_times = np.zeros(shape=dimension)
        self.state_transition_matrix = np.zeros(shape=(dimension, dimension), dtype=int)
        self.cim = np.zeros(shape=(dimension, dimension), dtype=float)

    def update_state_transition_count(self, element_indx):
        #print(element_indx)
        self.state_transition_matrix[element_indx[0]][element_indx[1]] += 1

    def update_state_residence_time_for_state(self, state, time):
        #print("Time updating In state", state, time)
        self.state_residence_times[state] += time

    def compute_cim_coefficients(self):
        for i, row in enumerate(self.state_transition_matrix):
            row_sum = 0.0
            for j, elem in enumerate(row):
                rate_coefficient = elem / self.state_residence_times[i]
                self.cim[i][j] = rate_coefficient
                row_sum = row_sum + rate_coefficient
            self.cim[i][i] = -1 * row_sum

