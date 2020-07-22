import numpy as np


class ConditionalIntensityMatrix:

    def __init__(self, state_residence_times, state_transition_matrix):
        self._state_residence_times = state_residence_times
        self._state_transition_matrix = state_transition_matrix
        #self.cim = np.zeros(shape=(dimension, dimension), dtype=float)
        self._cim = self.state_transition_matrix.astype(np.float)

    def compute_cim_coefficients(self):
        np.fill_diagonal(self._cim, self._cim.diagonal() * -1)
        #print(self._cim)
        self._cim = ((self._cim.T + 1) / (self._state_residence_times + 1)).T
        #np.fill_diagonal(self.state_transition_matrix, 0)

    @property
    def state_residence_times(self):
        return self._state_residence_times

    @property
    def state_transition_matrix(self):
        return self._state_transition_matrix

    @property
    def cim(self):
        return self._cim

    def __repr__(self):
        return 'CIM:\n' + str(self.cim)

