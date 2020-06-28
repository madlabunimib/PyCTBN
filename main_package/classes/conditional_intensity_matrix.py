import numpy as np


class ConditionalIntensityMatrix:

    def __init__(self, dimension):
        self.state_residence_times = np.zeros(shape=(1, dimension))
        self.state_transition_matrix = np.zeros(shape=(dimension, dimension), dtype=int)
        self.cim = np.zeros(shape=(dimension, dimension), dtype=float)

    def update_state_transition_count(self, positions_list):
        self.state_transition_matrix[positions_list[0]][positions_list[1]] = self.state_transition_matrix[positions_list[0]][positions_list[1]] + 1

